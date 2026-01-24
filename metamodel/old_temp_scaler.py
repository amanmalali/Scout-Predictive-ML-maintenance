import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader


class linear_model(nn.Module):
    """
    Feature-conditioned temperature head with hard bounds.

    Maps features `x` -> per-member temperatures `T(x)` via a linear layer with a square link,
    then clamps to [`T_min`, `T_max`].

    Args:
        in_features: Feature dimension D.
        K: Number of temperatures produced per sample (typically ensemble size).
        T_min, T_max: Clamp range for temperatures.
    Returns:
        temps: Tensor of shape (B, K).
    """
    def __init__(self, in_features: int, K: int = 5,
                 T_min: float = 0.5, T_max: float = 7.0):
        super().__init__()
        assert 0. < T_min < T_max
        self.T_min, self.T_max = T_min, T_max
        self.lin = nn.Linear(in_features, K)
        # nn.init.zeros_(self.lin.weight)
        # nn.init.constant_(self.lin.bias, 0.0)
        self.lin.bias.data.fill_(1.5)

    def forward(self, x):
        t_sq  = self.lin(x).pow(2)                    # square-link
        temps = torch.clamp(t_sq, self.T_min, self.T_max)
        return temps


class scaler_model(nn.Module):
    """
    Static per-member temperature scaler for ensemble logits.

    Args:
        models: Ensemble size M (number of temperatures).
    """
    def __init__(self,models=5):
        super(scaler_model,self).__init__()
        self.temp=nn.Parameter(torch.ones(models)*1.5)

    def forward(self,logits):
        """
        Scale logits by learned per-member temperatures.

        Args:
            logits: Tensor (B, M, C).
        Returns:
            scaled_logits: Tensor (B, M, C).
        """
        temps=self.temp.unsqueeze(1).expand(logits.size(0),logits.size(1),logits.size(2))
        return logits/temps

    
    def forward_ext_temp(self,logits,temp):
        """
        Scale logits using externally provided per-sample temperatures.

        Args:
            logits: Tensor (B, M, C).
            temp: Tensor (B, M) temperatures.
        Returns:
            scaled_logits: Tensor (B, M, C).
        """
        # print("TEMPS :",logits)
        temps=temp.unsqueeze(2).expand(logits.size(0),logits.size(1),logits.size(2))
        # print(logits.shape)
        # print(temp.shape)
        return logits/temps



def _inv_sigmoid_T(T_eff: float, T_min: float, T_max: float) -> float:
    """
    Inverse of the bounded-sigmoid temperature parameterization.

    Returns raw z such that: T_eff = T_min + (T_max - T_min) * sigmoid(z).
    """
    # clip to open interval to avoid inf logits at the ends
    frac = (T_eff - T_min) / (T_max - T_min)
    frac = min(max(frac, 1e-6), 1 - 1e-6)
    return math.log(frac / (1.0 - frac))


class UCECollisionEntropyLoss(torch.nn.Module):
    """
    UCE-style calibration loss using collision entropy (Renyi-2) as uncertainty.

    Args:
        n_bins: Number of uniform bins over the observed H2 range.
    """

    def __init__(self, n_bins: int = 10):
        super().__init__()
        self.n_bins = n_bins

    @staticmethod
    def _risk_from_H2(u: torch.Tensor) -> torch.Tensor:
        inner = torch.clamp(2.0 * 2.0**(-u) - 1.0, min=0.0)
        return 0.5 * (1.0 - torch.sqrt(inner))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        logits:  (B, M, C) or (B, C)
        labels:  (B,)
        Returns:
          - uce:      scalar calibration loss
          - err_bins: (n_bins,) empirical error per bin
          - H2_bins:  (n_bins,) avg H2 per bin
        """
        device = logits.device

        # --- 1) compute per-sample collision entropy H2 and 0/1 error ---
        if logits.dim() == 3:
            probs = F.softmax(logits, dim=-1).mean(dim=1)
        else:
            probs = F.softmax(logits, dim=-1)

        H2   = -torch.log2((probs**2).sum(-1) + 1e-12)  # (B,)
        pred = probs.argmax(-1)                        # (B,)
        err  = (pred != labels).float()                # (B,)

        # --- 2) uniform bins over [min H2, max H2] ---
        # FIX: Convert tensor min/max to python scalars using .item()
        h2_min = H2.min().item()
        h2_max = H2.max().item()

        edges = torch.linspace(h2_min, h2_max, self.n_bins+1, device=device)
        edges[-1] += 1e-6  # include max

        uce_terms = []
        err_bins  = []
        H2_bins   = []

        # --- 3) per-bin gap or zero if empty ---
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (H2 > lo) & (H2 <= hi)
            prop = mask.float().mean()

            if prop.item() == 0.0:
                # empty bin → append zero so stack() never fails
                uce_terms.append(torch.tensor(0.0, device=device))
                err_bins.append(torch.tensor(0.0, device=device))
                H2_bins.append(torch.tensor(0.0, device=device))
            else:
                H2_bar  = H2[mask].mean()
                err_bar = err[mask].mean()
                err_ref = self._risk_from_H2(H2_bar)

                uce_terms.append(torch.abs(err_bar - err_ref) * prop)
                err_bins.append(err_bar)
                H2_bins.append(H2_bar)

        # --- 4) final loss & diagnostics ---
        uce = torch.stack(uce_terms).sum()
        return uce, torch.stack(err_bins), torch.stack(H2_bins)



# --------------------------------------------------------------------------- #
class GlobalScaler(nn.Module):
    """Single scalar temperature for all ensemble members, smooth & bounded."""
    def __init__(self,
                 init_temp: float = 1.5,
                 T_min: float = 0.7,
                 T_max: float = 7.0):
        super().__init__()
        assert 0 < T_min < T_max
        self.T_min, self.T_max = T_min, T_max

        # store *raw* parameter so optimiser works in ℝ
        raw_init = _inv_sigmoid_T(init_temp, T_min, T_max)
        self.raw_temp = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T_eff = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self.raw_temp)
        return logits / T_eff


class EnsembleScaler(nn.Module):
    """Static per-member temperatures (default 5) with smooth bounds."""
    def __init__(self,
                 n_models: int = 5,
                 init_temp: float = 1.5,
                 T_min: float = 0.7,
                 T_max: float = 7.0):
        super().__init__()
        assert 0 < T_min < T_max
        self.T_min, self.T_max = T_min, T_max

        raw_init = _inv_sigmoid_T(init_temp, T_min, T_max)
        self.raw_temp = nn.Parameter(
            torch.full((n_models,), raw_init, dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits shape: (B, n_models, n_classes)
        T_eff = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self.raw_temp)
        t = T_eff.view(1, -1, 1).expand_as(logits)  # broadcast over batch & classes
        return logits / t

def train_scaler(data, epochs: int = 200, feat: bool = True, lr: float = 1e-3):
    
    """
    Train temperature scaling parameters on a calibration dataset.

    Modes:
        - feat=True: trains `linear_model` (feature->temperature); keeps the scaler frozen and
          uses `scaler.forward_ext_temp(logits, temps)`.
        - feat=False: trains a static scaler (expects `scaler` to expose a trainable raw parameter,
          e.g., `raw_temp`), optimizing against `UCECollisionEntropyLoss`.

    Args:
        data: Dataset yielding (x, logits, y) per batch; also expected to expose `X`, `logits`, `y`.
        epochs: Training epochs.
        feat: Whether to train feature-based temperatures.
        lr: Adam learning rate.
    Returns:
        (lin_model, scaler): `lin_model` is None when `feat=False`.
    """

    dataloader = DataLoader(data, batch_size=len(data), shuffle=True)
    scaler     = scaler_model()                     # GlobalScaler / EnsembleScaler
    loss_f     = UCECollisionEntropyLoss()                       # your UCE wrapper

    # ------------------------------------------------------------------ #
    #  1)  FEATURE-BASED head  (FBTS)
    # ------------------------------------------------------------------ #
    if feat:
        lin_model = linear_model(data.X.shape[1])   # ← already bounded via sigmoid
        optimizer = optim.Adam(lin_model.parameters(), lr=lr)

        best_state, best_loss = None, float("inf")

        scaler.eval()                               # frozen during FBTS training
        for e in range(epochs):
            lin_model.train()
            epoch_loss = 0.0

            for x, logits, y in dataloader:
                optimizer.zero_grad()

                temps   = lin_model(x)                              # (B, K)
                preds   = scaler.forward_ext_temp(logits, temps)    # user API
                loss, *_ = loss_f(preds, y)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if 0 < epoch_loss < best_loss:
                best_loss  = epoch_loss
                best_state = copy.deepcopy(lin_model.state_dict())

            print(f"EPOCH {e:03d} – total loss {epoch_loss:.4f}")

        if best_state is None:        # training diverged or all-zero loss
            return None, None

        lin_model.load_state_dict(best_state)
        print("Best FBTS loss :", best_loss)
        return lin_model, scaler

    # ------------------------------------------------------------------ #
    #  2)  STATIC global / per-member scaler
    # ------------------------------------------------------------------ #
    else:
        # optimiser must see the *raw* unconstrained parameter
        optimizer = optim.Adam([scaler.raw_temp], lr=lr)

        best_state, best_loss = None, float("inf")

        for e in range(epochs):
            scaler.train()
            for _, logits, y in dataloader:
                optimizer.zero_grad()
                loss, *_ = loss_f(scaler(logits), y)
                loss.backward()
                optimizer.step()

            # full-set evaluation
            scaler.eval()
            with torch.no_grad():
                logits_all = torch.tensor(data.logits, dtype=torch.float32)
                y_all      = torch.tensor(data.y)
                full_loss, *_ = loss_f(scaler(logits_all), y_all)

            if full_loss.item() < best_loss:
                best_loss  = full_loss.item()
                best_state = copy.deepcopy(scaler.state_dict())

            print(f"EPOCH {e:03d} – loss {full_loss.item():.4f}")

        scaler.load_state_dict(best_state)

        # pretty-print effective temperature(s)
        with torch.no_grad():
            T_eff = scaler.T_min + (scaler.T_max - scaler.T_min) * \
                    torch.sigmoid(scaler.raw_temp.data)
        print("Optimal effective temperature(s):", T_eff)

        return None, scaler



def _train_static_scaler(
    data,                    # custom_torch_dataset
    scaler: nn.Module,       # new GlobalScaler or EnsembleScaler
    epochs: int = 200,
    lr: float = 1e-3,
):
    """
    Train a bounded static scaler (global or per-member) using UCECollisionEntropyLoss.

    Args:
        data: Dataset with `logits` and `y` for full-set evaluation.
        scaler: GlobalScaler or EnsembleScaler (must expose `raw_temp`, `T_min`, `T_max`).
        epochs: Training epochs.
        lr: Adam learning rate.
    Returns:
        scaler: The best-performing scaler (by full-set loss).
    """
    loader   = DataLoader(data, batch_size=len(data), shuffle=True)
    # ---------------- optimiser on the *raw* parameter --------------------
    opt      = optim.Adam([scaler.raw_temp], lr=lr)
    loss_f   = UCECollisionEntropyLoss()                        # your UCE wrapper
    best_state, best_loss = None, float("inf")

    for epoch in range(epochs):
        scaler.train()
        for _, logits, labels in loader:
            opt.zero_grad()
            loss, *_ = loss_f(scaler(logits), labels)
            loss.backward()
            opt.step()                # <- no clamp needed

        # ---------- full-set UCE on hold-out / full dataset --------------
        scaler.eval()
        with torch.no_grad():
            full_loss, *_ = loss_f(
                scaler(torch.tensor(data.logits, dtype=torch.float32)),
                torch.tensor(data.y),
            )
        if full_loss.item() < best_loss:
            best_loss  = full_loss.item()
            best_state = copy.deepcopy(scaler.state_dict())

    scaler.load_state_dict(best_state)

    # ------------ pretty print effective temperatures --------------------
    with torch.no_grad():
        T_eff = scaler.T_min + (scaler.T_max - scaler.T_min) * \
                torch.sigmoid(scaler.raw_temp.data)
    print("Optimal effective temperature(s):", T_eff)

    return scaler



def train_temp_scaler(data, eval_data=None, mode: str = "feat", lr=1e-4, epochs=200):
    """Entry point for training temperature scaling.

    Args:
        data: Training/calibration dataset.
        eval_data: Optional evaluation dataset passed to `eval_temp_scaler`.
        mode: One of {"feat", "global", "ensemble"}.
        lr: Adam learning rate.
        epochs: Training epochs.
    Returns:
        (lin_model, scaler): `lin_model` is only returned for mode="feat".
    """
    print("Training temperature scaler")
    if mode == "feat":
        lin_model, scaler = train_scaler(data,epochs=epochs,lr=lr)              # original path
    elif mode == "global":
        lin_model = None
        scaler    = _train_static_scaler(data, GlobalScaler(),epochs=epochs,lr=lr)
    elif mode == "ensemble":
        lin_model = None
        scaler    = _train_static_scaler(data, EnsembleScaler(),epochs=epochs,lr=lr)
    else:
        raise ValueError("mode must be 'feat', 'global', or 'ensemble'")

    if eval_data is not None and lin_model is not None and scaler is not None:
        print("Eval temp scaler")
    return lin_model, scaler



"""
LEGACY uncer_loss

class uncer_loss(nn.Module):
    # Utilities for entropy-based uncertainty and an Uncertainty Calibration Error (UCE)-style loss.

    # Notes:
    #     - Expects `logits` shaped like (B, M, C) when averaging over M ensemble/MC samples.

    def __init__(self, beta=1):
        super(uncer_loss, self).__init__()
        self.diff_loss=nn.MSELoss()
        self.eps=1e-10

    def entropy(self,prob):
        return -1 * torch.log2(torch.sum(prob**2, dim=-1))

    # def entropy(self, prob):
    #     return -1 * torch.sum(prob * torch.log2(prob + self.eps), dim=-1)

    def expected_entropy(self,mc_preds):
        return torch.mean(self.entropy(mc_preds), dim=1)

    def model_uncertainty(self,mc_preds):
        return self.entropy(torch.mean(mc_preds, dim=1)) - self.expected_entropy(mc_preds)
    
    def forward(self,logits):
        probs=F.softmax(logits,dim=-1)
        en=self.entropy(torch.mean(probs, dim=1))
        # scaled_en=F.tanh(en)
        # print(scaled_en,labels)
        return en
    
    def uceloss(self,logits, labels, n_bins=10):

        # Compute UCE by binning uncertainty and comparing bin error vs bin uncertainty.

        # Args:
        #     logits: Tensor (B, M, C).
        #     labels: Tensor (B,).
        #     n_bins: Number of bins over uncertainty in [0, 1] (as implemented).
        # Returns:
        #     uce: Scalar tensor.
        #     err_in_bin: Tensor (#nonempty_bins,).
        #     avg_uncert_in_bin: Tensor (#nonempty_bins,).

        probs=F.softmax(logits,dim=-1)
        mean_probs=torch.mean(probs,dim=1)
        labels=labels.to(torch.long)
        confidences, predictions = torch.max(mean_probs, 1)
        
        uncertainties=self.entropy(mean_probs)
        # uncertainties=self.model_uncertainty(probs)

        d = logits.device
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=d)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        _, predictions = torch.max(mean_probs, 1)
        errors = predictions.ne(labels)
        # uncertainties = nentr(softmaxes, base=softmaxes.size(1))
        errors_in_bin_list = []
        avg_entropy_in_bin_list = []

        uce = torch.zeros(1, device=d)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculate |uncert - err| in each bin
            in_bin = uncertainties.gt(bin_lower.item()) * uncertainties.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()  # |Bm| / n
            if prop_in_bin.item() > 0.0:
                errors_in_bin = errors[in_bin].float().mean()  # err()
                avg_entropy_in_bin = uncertainties[in_bin].mean()  # uncert()
                uce += torch.abs(avg_entropy_in_bin - errors_in_bin) * prop_in_bin

                errors_in_bin_list.append(errors_in_bin)
                avg_entropy_in_bin_list.append(avg_entropy_in_bin)

        err_in_bin = torch.tensor(errors_in_bin_list, device=d)
        avg_entropy_in_bin = torch.tensor(avg_entropy_in_bin_list, device=d)

        return uce, err_in_bin, avg_entropy_in_bin
"""