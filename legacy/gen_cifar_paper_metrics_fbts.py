#!/usr/bin/env python3
# calibration_metrics_uce_h2_csv.py
# -----------------------------------------------------------
# Outputs one CSV with:
#   • ECE (Guo et al. 2017): equal-width bins over confidence=max prob
#   • UCE_H2 (this work, Laves-style binning): equal-width bins over H2(p) in [0,1],
#       comparing empirical error to the analytical curve h(u) derived from H2 (binary only)
#   • EU separability: gaps (incorrect - correct) in mean/median H2-disagreement
# No plots. Handles No TS / Prob-TS (global/per-member) and optional FBTS/Unc-Cal models if present.
# -----------------------------------------------------------

import argparse, pathlib, pickle, warnings, csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import LBFGS

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------- DEFAULT PATHS -----------------------
DATA_DIR  = "./cifar10/data"
RAW_MODEL = "./cifar10/saved_models/sense_model_vit_cifar.pkl"
BEST_DIR  = "./cifar10/saved_models/"
OUT_DIR   = "./ts_eval_cifar/metrics"

CALIB_X = "./cifar10/data/calib_un_x.npy"
CALIB_Y = "./cifar10/data/calib_un_y.npy"

# ----------------------- DEFAULTS ----------------------------
ECE_BINS      = 15         # standard
UCE_BINS      = 15         # equal-width bins over H2 in [0,1]
T_MIN, T_MAX  = 0.4, 7.0
SEED          = 0
EPS           = 1e-12

# ----------------------- IO helpers -------------------
def load_split(tag, data_dir, calib_x, calib_y):
    if tag == "calib_un":
        x = np.load(calib_x); y = np.load(calib_y)
    elif tag == "calib_over":
        x = np.load(f"{data_dir}/calib_over_x.npy")
        y = np.load(f"{data_dir}/calib_over_y.npy")
    else:
        x = np.load(f"{data_dir}/sense_cifar_{tag}_x.npy")
        y = np.load(f"{data_dir}/sense_cifar_{tag}_y.npy")
    return x, torch.as_tensor(y).long()

def to_tensor(a, device=None):
    t = torch.as_tensor(a, dtype=torch.float32)
    return t.to(device) if device is not None else t

# ---------------- Probabilities & Uncertainty ----------
def softmax_probs(logits):
    L = to_tensor(logits)
    if L.dim() == 3:                      # (N,K,C)
        return F.softmax(L, dim=-1).mean(1).detach()  # predictive probs
    return F.softmax(L, dim=-1).detach()  # (N,C)

def collision_entropy(p):                 # H2 total uncertainty in bits
    p = p.clamp_min(EPS)
    return -torch.log2((p**2).sum(-1))

def epi_uncert(logits):                   # H2(E[p]) - E[H2(p)]  (≥ 0)
    L = to_tensor(logits)
    if L.dim() != 3:
        return torch.zeros(L.shape[0])
    p_k = F.softmax(L, dim=-1)            # (N,K,C)
    return (collision_entropy(p_k.mean(1)) - collision_entropy(p_k).mean(1)).clamp_min(0.)

# ---- Analytical error curve from collision entropy (binary C=2):
# Given u = H2(p) = -log2(p0^2 + p1^2), the Bayes error e = min(p0,p1)
# satisfies e = 0.5 * (1 - sqrt(2^(1-u) - 1))  in [0, 0.5].
def h_collision_binary(u):
    u = to_tensor(u).clamp(0.0, 1.0)      # for binary, H2 ∈ [0,1]
    # guard for tiny negatives due to rounding:
    arg = (2.0**(1.0 - u)) - 1.0
    arg = torch.clamp(arg, min=0.0)
    return 0.5 * (1.0 - torch.sqrt(arg))

# ---------------- Probability TS training --------------
def nll_of_ensemble(logits_scaled, y):
    L = to_tensor(logits_scaled)
    Pbar = F.softmax(L, dim=-1).mean(1) if L.dim() == 3 else F.softmax(L, dim=-1)
    return -torch.log(Pbar[torch.arange(len(y)), y].clamp_min(EPS)).mean()

def learn_Tg(logits_cal, y_cal):
    w = torch.nn.Parameter(torch.tensor(0.0))
    opt = LBFGS([w], lr=0.01, max_iter=60, line_search_fn="strong_wolfe")
    Lc = to_tensor(logits_cal).detach()
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
    Lc = to_tensor(log3d_cal).detach()
    if Lc.dim() != 3: return None
    K = Lc.shape[1]
    w = torch.nn.Parameter(torch.zeros(K))
    opt = LBFGS([w], lr=0.01, max_iter=80, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        T = T_MIN + (T_MAX - T_MIN) * torch.sigmoid(w)
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
        return (L / Tv.view(1, -1, 1)).cpu().numpy()
    return L.cpu().numpy()

# ----------------- Standard metrics ---------------------
def ece_confidence(probs, labels, M=15):
    """
    Standard ECE (Guo et al., 2017): equal-width bins over confidence=max prob.
    Weighted |acc - conf| per bin; last bin closed.
    """
    p = probs if isinstance(probs, torch.Tensor) else torch.tensor(probs, dtype=torch.float32)
    y = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
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
    p = probs if isinstance(probs, torch.Tensor) else torch.tensor(probs, dtype=torch.float32)
    y = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
    C = p.shape[-1]
    if C != 2:
        return float("nan")  # analytical curve h(u) below is binary-only

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
    y = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
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

# ------------------- Method filtering -------------------
def parse_methods_option(s):
    if s is None or s.strip() == "":
        return None
    items = [t.strip() for t in s.split(",") if t.strip()]
    out = {}
    for it in items:
        if "=" in it:
            k, v = it.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[it] = it
    return out

def filter_and_rename_schemes(schemes, include_map):
    if include_map is None:
        return schemes, {k: k for k in schemes.keys()}
    filt, name_map = {}, {}
    for k, v in schemes.items():
        if k in include_map:
            filt[include_map[k]] = v
            name_map[include_map[k]] = k
    return filt, name_map

# ------------------- Main -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-model", default=RAW_MODEL)
    parser.add_argument("--best-dir",  default=BEST_DIR)
    parser.add_argument("--outdir",    default=OUT_DIR)
    parser.add_argument("--save-prefix", default="")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--calib-x", default=CALIB_X)
    parser.add_argument("--calib-y", default=CALIB_Y)
    parser.add_argument("--ece-bins", type=int, default=ECE_BINS)
    parser.add_argument("--uce-bins", type=int, default=UCE_BINS)
    parser.add_argument("--methods", type=str, default="No TS=No TS,Prob-TS (global)=Prob-TS (global),Prob-TS (per-member)=Prob-TS (per-member),FBTS (feature-TS)=FBTS (feature-TS)")
    args = parser.parse_args()

    np.random.seed(SEED); torch.manual_seed(SEED)
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    best_dir = pathlib.Path(args.best_dir)

    # Load base model and splits
    with open(args.raw_model, "rb") as f:
        sense = pickle.load(f)

    Xt_np, yt = load_split("test",  args.data_dir, args.calib_x, args.calib_y)   # IID
    Xv_np, yv = load_split("val",   args.data_dir, args.calib_x, args.calib_y)   # DRIFT
    Xc_np, yc = load_split("calib_un", args.data_dir, args.calib_x, args.calib_y)

    raw_test = sense.gen_logits(Xt_np)
    raw_val  = sense.gen_logits(Xv_np)
    raw_cal  = sense.gen_logits(Xc_np)

    # Learn probability TS (on calibration set)
    Tg = learn_Tg(raw_cal, yc)
    Tv = learn_Tv(raw_cal, yc)  # per-member (may be None)

    # Optionally load Unc-Cal / FBTS models (if present)
    def maybe_load(name):
        p = best_dir / name
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
        return None

    m_unc_global   = maybe_load("sense_model_global.pkl")
    m_unc_ensemble = maybe_load("sense_model_ensemble.pkl")
    m_fbts         = maybe_load("sense_model_fbts.pkl")

    # Build logits per scheme
    def logits_by_scheme(X, base_logits):
        d = {
            "No TS": base_logits,
            "Prob-TS (global)": apply_prob_T(base_logits, Tg=Tg),
        }
        if Tv is not None:
            d["Prob-TS (per-member)"] = apply_prob_T(base_logits, Tv=Tv)
        if m_unc_global is not None:
            d["Unc-Cal (global)"] = m_unc_global.inf(X, apply_temp=True, mode="global").detach()
        if m_unc_ensemble is not None:
            d["Unc-Cal (ensemble)"] = m_unc_ensemble.inf(X, apply_temp=True, mode="ensemble").detach()
        if m_fbts is not None:
            d["FBTS (feature-TS)"] = m_fbts.inf(X, apply_temp=True, mode="feat").detach()
        return d

    schemes_iid_full   = logits_by_scheme(Xt_np, raw_test)
    schemes_drift_full = logits_by_scheme(Xv_np, raw_val)

    include_map = parse_methods_option(args.methods)
    schemes_iid,  _ = filter_and_rename_schemes(schemes_iid_full, include_map)
    schemes_drift, _ = filter_and_rename_schemes(schemes_drift_full, include_map)

    # ------------------- Compute metrics --------------------
    rows = []
    def add_rows_for_split(schemes, y, split_name):
        for name, logits in schemes.items():
            probs = softmax_probs(logits)  # (N,C)
            n = probs.shape[0]

            # Standard ECE (confidence bins)
            ece = ece_confidence(probs, y, M=args.ece_bins)

            # UCE over collision entropy vs analytical curve (binary only)
            uce_h2 = uce_h2_analytical(probs, y, M=args.uce_bins)

            # EU separability
            eu_gap_mean, eu_gap_median, eu_mean_corr, eu_mean_inc, eu_med_corr, eu_med_inc = eu_separability(logits, y)

            error_prev = float((probs.argmax(-1) != y).float().mean().item())

            rows.append({
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

    add_rows_for_split(schemes_iid,   yt, "IID")
    add_rows_for_split(schemes_drift, yv, "DRIFT")

    # ------------------- Write CSV -----------------------
    csv_path = pathlib.Path(outdir) / f"{args.save_prefix}calibration_metrics_uce_h2.csv"
    fieldnames = [
        "split","scheme","n","ece_bins","uce_h2_bins",
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
