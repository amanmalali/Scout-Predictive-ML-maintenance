#!/usr/bin/env python3
# census_calibration_metrics_uce_h2_csv.py
# -----------------------------------------------------------
# Census metrics (no plots):
#   • ECE (confidence-binned, equal-width)
#   • UCE_H2 (equal-width bins over H2 in [0,1], compares empirical error to h(H2))
#   • EU separability (incorrect - correct) gaps in mean/median H2-disagreement
# Writes a single CSV covering IID ("train") and DRIFT ("val") splits
# across available techniques (No TS, Prob-TS global/per-member, Unc-Cal, FBTS).
# -----------------------------------------------------------

import argparse, pathlib, pickle, warnings, csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import LBFGS

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = "cpu"
EPS = 1e-12

# ------------------------- Defaults -------------------------
DATA_DIR   = "./census/data"
RAW_MODEL  = "./census/saved_models/sense_model_v2.pkl"
BEST_DIR   = "./census/saved_models"
OUT_DIR    = "./ts_eval_census/metrics"

# TS bounds (keep your Census defaults; adjust if desired)
T_MIN, T_MAX = 0.5, 7.0

# Binning (match CIFAR metrics)
ECE_BINS = 15
UCE_BINS = 15

# ----------------------------- Data I/O -----------------------------
def load_split(data_dir: str, tag: str):
    """
    census/data layout:
      - sense_census_train_x.npy / _y.npy
      - sense_census_val_x.npy   / _y.npy
      - calib_over_x.npy         / _y.npy
    """
    base = f"{data_dir}/calib_over" if tag == "calib_over" else f"{data_dir}/sense_census_{tag}"
    X = np.load(f"{base}_x.npy")
    y = np.load(f"{base}_y.npy")
    return X.astype(np.float32), torch.as_tensor(y, device=DEVICE).long()

def to_tensor(x): 
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)

# --------------------- Model helpers ---------------------
def _maybe_load(path):
    p = pathlib.Path(path)
    if not p.exists(): return None
    with open(p, "rb") as f: return pickle.load(f)

def logits_for(model, X_np, mode=None):
    if model is None: return None
    if mode is None:  # raw model API
        return model.gen_logits(X_np)
    out = model.inf(X_np, apply_temp=True, mode=mode)
    return out.detach() if isinstance(out, torch.Tensor) else out

# ---------------- Prob/logit handling (robust) ----------------
def _as_probabilities(L: torch.Tensor) -> torch.Tensor:
    """Convert logits-or-probs to probabilities; detect probs to avoid double-softmax."""
    L = to_tensor(L)
    if L.dim() not in (2,3): 
        raise ValueError(f"Expected 2D/3D, got {L.shape}")
    sums = L.sum(-1)
    prob_like = torch.all((L >= -1e-6) & (L <= 1.0 + 1e-6)) and torch.allclose(
        sums, torch.ones_like(sums), atol=1e-3
    )
    return L if prob_like else F.softmax(L, dim=-1)

def softmax_probs(logits):
    L = to_tensor(logits)
    P = _as_probabilities(L)
    return P.mean(1).detach() if P.dim() == 3 else P.detach()

# --------------------- Uncertainty utils -----------------
def collision_entropy(p):                 # H2 total uncertainty (Rényi-2) in bits
    p = p.clamp_min(EPS)
    return -torch.log2((p**2).sum(-1))

def epi_uncert(logits):                   # H2(E[p]) − E[H2(p)]  (≥ 0)
    t = to_tensor(logits)
    if t.dim() != 3:
        return torch.zeros(len(t), device=t.device)
    p_k = _as_probabilities(t)            # (N,K,C)
    return (collision_entropy(p_k.mean(1)) -
            collision_entropy(p_k).mean(1)).clamp(min=0.)

def h_collision_binary(u):
    """Analytical error curve from H2 for binary C=2:
       h(u) = 0.5 * (1 - sqrt(2^(1-u) - 1)), u ∈ [0,1]."""
    u = to_tensor(u).clamp(0.0, 1.0)
    arg = (2.0**(1.0 - u)) - 1.0
    arg = torch.clamp(arg, min=0.0)
    return 0.5 * (1.0 - torch.sqrt(arg))

# ---------------- Probability TS ----------------
def nll_of_ensemble(L_scaled, y):
    L = to_tensor(L_scaled)
    Pbar = _as_probabilities(L).mean(1) if L.dim() == 3 else _as_probabilities(L)
    return -torch.log(Pbar[torch.arange(len(y), device=L.device), y].clamp_min(EPS)).mean()

def learn_Tg(logits_cal, y_cal):
    w = torch.nn.Parameter(torch.tensor(0.0, device=DEVICE))
    opt = LBFGS([w], lr=0.1, max_iter=60, line_search_fn="strong_wolfe")
    Lc = to_tensor(logits_cal)
    def closure():
        opt.zero_grad()
        T = T_MIN + (T_MAX - T_MIN) * torch.sigmoid(w)
        loss = nll_of_ensemble(Lc / T, y_cal)
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        T = T_MIN + (T_MAX - T_MIN) * torch.sigmoid(w)
    return float(T.cpu())

def learn_Tv(log3d_cal, y_cal):
    Lc = to_tensor(log3d_cal)
    if Lc.dim() != 3: return None
    K = Lc.shape[1]
    w = torch.nn.Parameter(torch.zeros(K, device=DEVICE))
    opt = LBFGS([w], lr=0.1, max_iter=80, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        T = T_MIN + (T_MAX - T_MIN) * torch.sigmoid(w)  # (K,)
        loss = nll_of_ensemble(Lc / T.view(1, K, 1), y_cal)
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        T = T_MIN + (T_MAX - T_MIN) * torch.sigmoid(w)
    return T.detach().cpu()

def apply_prob_T(logits, Tg=None, Tv=None):
    L = to_tensor(logits)
    if Tg is not None:
        return (L / float(Tg)).cpu().numpy()
    if Tv is not None:
        assert L.dim() == 3 and L.shape[1] == len(Tv), "Tv size mismatch"
        return (L / to_tensor(Tv).view(1, -1, 1)).cpu().numpy()
    return L.cpu().numpy()

# ----------------- Standard metrics ---------------------
def ece_confidence(probs, labels, M=15):
    """
    Standard ECE (Guo et al., 2017): equal-width bins over confidence=max prob.
    Weighted |acc - conf| per bin; last bin closed.
    """
    p = probs if isinstance(probs, torch.Tensor) else to_tensor(probs)
    y = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels, dtype=torch.long, device=p.device)
    conf, pred = p.max(dim=-1)
    acc = (pred == y).float()
    edges = torch.linspace(0.0, 1.0, M + 1, device=p.device, dtype=p.dtype)
    ece = torch.tensor(0.0, device=p.device, dtype=p.dtype)
    for m in range(M):
        lo, hi = edges[m], edges[m+1]
        mask = (conf >= lo) & ((conf < hi) if m < M-1 else (conf <= hi))
        if mask.any():
            w = mask.float().mean()
            ece += w * (acc[mask].mean() - conf[mask].mean()).abs()
    return float(ece.item())

def uce_h2_analytical(probs, labels, M=15):
    """
    Laves-style UCE over collision entropy H2 (binary only):
      bins: equal-width over u in [0,1],
      residual per bin: | empirical error rate - h_collision_binary(mean_u) |,
      UCE = sum_bin (w_bin * residual_bin).
    Returns NaN if C != 2.
    """
    p = probs if isinstance(probs, torch.Tensor) else to_tensor(probs)
    y = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels, dtype=torch.long, device=p.device)
    C = p.shape[-1]
    if C != 2:
        return float("nan")

    u = collision_entropy(p).clamp(0.0, 1.0)  # H2 ∈ [0,1] for binary
    err = (p.argmax(-1) != y).float()
    edges = torch.linspace(0.0, 1.0, M + 1, device=p.device, dtype=p.dtype)

    uce = torch.tensor(0.0, device=p.device, dtype=p.dtype)
    for m in range(M):
        lo, hi = edges[m], edges[m+1]
        mask = (u >= lo) & ((u < hi) if m < M-1 else (u <= hi))
        if mask.any():
            w = mask.float().mean()
            mean_u = u[mask].mean()
            err_rate = err[mask].mean()
            hval = h_collision_binary(mean_u)
            uce += w * (err_rate - hval).abs()
    return float(uce.item())

def eu_separability(logits, labels):
    """
    EU separability = differences (incorrect - correct) in mean and median EU.
    EU via collision-entropy disagreement.
    """
    y = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels, dtype=torch.long, device=DEVICE)
    p = softmax_probs(logits)
    pred = p.argmax(-1)
    correct = (pred == y)
    eu = epi_uncert(logits)

    eu_corr = eu[correct]
    eu_inc  = eu[~correct]
    if eu_corr.numel() == 0 or eu_inc.numel() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    mean_gap   = float(eu_inc.mean().item() - eu_corr.mean().item())
    median_gap = float(eu_inc.median().item() - eu_corr.median().item())
    return mean_gap, median_gap, float(eu_corr.mean().item()), float(eu_inc.mean().item()), float(eu_corr.median().item()), float(eu_inc.median().item())

# ------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser(description="Census calibration metrics (ECE, UCE_H2, EU separability)")
    ap.add_argument("--data_dir",   default=DATA_DIR)
    ap.add_argument("--raw_model",  default=RAW_MODEL)
    ap.add_argument("--best_dir",   default=BEST_DIR)
    ap.add_argument("--out_dir",    default=OUT_DIR)
    ap.add_argument("--ece_bins",   type=int, default=ECE_BINS)
    ap.add_argument("--uce_bins",   type=int, default=UCE_BINS)
    # Optional: restrict techniques shown (comma-separated of the human-readable names below)
    ap.add_argument("--methods",    default="No TS,Prob-TS (global),Prob-TS (per-member),Unc-Cal (global),Unc-Cal (ensemble),FBTS (feature-TS)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models (raw required; others optional)
    model_raw = _maybe_load(args.raw_model)
    if model_raw is None:
        raise RuntimeError(f"Raw model not found: {args.raw_model}")

    mdl_unc_g = _maybe_load(pathlib.Path(args.best_dir, "sense_model_global.pkl"))
    mdl_unc_e = (_maybe_load(pathlib.Path(args.best_dir, "sense_model_ensemble.pkl")) or
                 _maybe_load(pathlib.Path(args.best_dir, "sense_model_ens.pkl")))
    mdl_fbts  = _maybe_load(pathlib.Path(args.best_dir, "sense_model_fbts.pkl"))

    # Data
    X_iid,   y_iid   = load_split(args.data_dir, "train")       # IID
    X_drift, y_drift = load_split(args.data_dir, "val")         # DRIFT
    X_cal,   y_cal   = load_split(args.data_dir, "calib_over")  # calibration

    # Learn prob-TS on calibration
    raw_cal = logits_for(model_raw, X_cal, mode=None)
    Tg = learn_Tg(raw_cal, y_cal)
    Tv = learn_Tv(raw_cal, y_cal)  # may be None

    # Build logits per split & technique (use the same human-readable keys as CIFAR script)
    def build_logits(split):
        X = X_iid if split == "IID" else X_drift
        d = {}
        base = logits_for(model_raw, X, mode=None)
        d["No TS"]                 = base
        d["Prob-TS (global)"]      = apply_prob_T(base, Tg=Tg) if base is not None else None
        if Tv is not None and base is not None:
            d["Prob-TS (per-member)"] = apply_prob_T(base, Tv=Tv)
        d["Unc-Cal (global)"]      = logits_for(mdl_unc_g, X, mode="global")   if mdl_unc_g else None
        d["Unc-Cal (ensemble)"]    = logits_for(mdl_unc_e, X, mode="ensemble") if mdl_unc_e else None
        d["FBTS (feature-TS)"]     = logits_for(mdl_fbts,  X, mode="feat")     if mdl_fbts  else None
        return d

    schemes_iid   = build_logits("IID")
    schemes_drift = build_logits("DRIFT")

    # Optionally filter methods
    wanted = [s.strip() for s in args.methods.split(",") if s.strip()]
    if wanted:
        schemes_iid   = {k:v for k,v in schemes_iid.items() if k in wanted and v is not None}
        schemes_drift = {k:v for k,v in schemes_drift.items() if k in wanted and v is not None}
    else:
        schemes_iid   = {k:v for k,v in schemes_iid.items()   if v is not None}
        schemes_drift = {k:v for k,v in schemes_drift.items() if v is not None}

    # ------------------- Compute metrics --------------------
    rows = []
    def add_rows_for_split(schemes, y, split_name):
        for name, logits in schemes.items():
            probs = softmax_probs(logits)  # (N,C)
            n = probs.shape[0]

            ece   = ece_confidence(probs, y, M=args.ece_bins)
            uce_h2 = uce_h2_analytical(probs, y, M=args.uce_bins)

            eu_gap_mean, eu_gap_median, eu_mean_corr, eu_mean_inc, eu_med_corr, eu_med_inc = eu_separability(logits, y)

            error_prev = float((probs.argmax(-1) != y).float().mean().item())

            rows.append({
                "dataset": "Census",
                "split": split_name,
                "scheme": name,
                "n": int(n),
                "ece_bins": int(args.ece_bins),
                "uce_h2_bins": int(args.uce_bins),
                "ECE": float(ece),
                "UCE_H2": float(uce_h2),
                "EU_gap_mean": eu_gap_mean,
                "EU_gap_median": eu_gap_median,
                "EU_mean_correct": eu_mean_corr,
                "EU_mean_incorrect": eu_mean_inc,
                "EU_median_correct": eu_med_corr,
                "EU_median_incorrect": eu_med_inc,
                "error_prevalence": error_prev,
            })

    add_rows_for_split(schemes_iid,   y_iid,   "IID")
    add_rows_for_split(schemes_drift, y_drift, "DRIFT")

    # ------------------- Write CSV -----------------------
    csv_path = pathlib.Path(out_dir) / "census_calibration_metrics_uce_h2.csv"
    fieldnames = [
        "dataset","split","scheme","n","ece_bins","uce_h2_bins",
        "ECE","UCE_H2",
        "EU_gap_mean","EU_gap_median",
        "EU_mean_correct","EU_mean_incorrect",
        "EU_median_correct","EU_median_incorrect",
        "error_prevalence"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"✓ Saved metrics CSV: {csv_path.resolve()}")

if __name__ == "__main__":
    main()
