"""ece_temp_scaler.py — Temperature scaling with probabilistic Soft-UCE
======================================================================
This version trains with **ProbUCELoss** (collision-entropy vs. *expected*
error = 1 − p_true), which is threshold-free.  UCEWrapper for evaluation
is unchanged.
"""
from __future__ import annotations
import copy, math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
# ---------------------------------------------------------------------
# 1.  Probabilistic Soft-UCE surrogate
# ---------------------------------------------------------------------
class ProbUCELossEF(nn.Module):
    """
    Prob-UCE with
      • equal-frequency (quantile) hard bins, recomputed every forward pass
      • equal bin weights  (simple mean over bins)

    Works with (B,C) or (B,K,C) logits.  Not differentiable w.r.t. the
    quantile edges, but fine for evaluation or “projection” fine-tuning.
    """

    def __init__(self, n_bins: int = 15):
        super().__init__()
        self.n_bins = n_bins

    @staticmethod
    def _entropy_collision(p: torch.Tensor) -> torch.Tensor:     # (B,)
        return -torch.log2((p ** 2).sum(-1) + 1e-12)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # ---- probs (B,C)
        if logits.dim() == 3:                                    # (B,K,C)
            probs = torch.softmax(logits, -1).mean(1)
        else:                                                    # (B,C)
            probs = torch.softmax(logits, -1)

        # ---- uncertainty  &  expected error
        u = self._entropy_collision(probs)                       # (B,)
        idx = torch.arange(len(labels), device=probs.device)
        p_true = probs[idx, labels.long()]
        e = 1.0 - p_true                                         # (B,)

        # ---- equal-frequency bin edges  (quantiles)
        edges = torch.quantile(u.detach(),                      # detach for stability
                               torch.linspace(0, 1, self.n_bins + 1,
                                              device=u.device))

        gaps, count = [], 0
        for lo, hi in zip(edges[:-1], edges[1:]):
            # include right edge only on the last bin
            mask = (u > lo) & (u <= hi) if count < self.n_bins - 1 else (u >= lo) & (u <= hi)
            count += 1
            if mask.any():
                gaps.append(torch.abs(u[mask].mean() - e[mask].mean()))
            else:                      # empty bin → gap = 0
                gaps.append(torch.tensor(0.0, device=u.device))

        # ---- equal-weight mean over bins
        return torch.stack(gaps).mean()

class ProbUCELoss(nn.Module):
    """Smooth entropy–error calibration loss; (B,C) or (B,K,C) logits."""

    def __init__(self, n_bins: int = 15, tau: float = 0.1):
        super().__init__()
        self.n_bins, self.tau = n_bins, tau
        centres = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins)
        self.register_buffer("centres", centres)
        self.delta = 1.0 / n_bins

    # soft weights  (B, n_bins)
    def _w(self, u: torch.Tensor) -> torch.Tensor:
        l = (u.unsqueeze(1) - (self.centres - self.delta / 2)) / self.tau
        r = (u.unsqueeze(1) - (self.centres + self.delta / 2)) / self.tau
        return torch.sigmoid(l) - torch.sigmoid(r)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:  # type: ignore
        # ---- probs (B,C)
        if logits.dim() == 3:            # (B,K,C) → ensemble mean
            probs = F.softmax(logits, -1).mean(1)
        else:                            # (B,C)
            probs = F.softmax(logits, -1)

        # ---- per-sample scalars
        u = -torch.log2((probs ** 2).sum(-1) + 1e-12)          # collision H2
        labels = labels.long()                                 # ensure int
        idx = torch.arange(labels.size(0), device=labels.device)
        p_true = probs[idx, labels]                            # (B,)
        e = 1.0 - p_true                                       # expected error

        # ---- soft histogram
        w  = self._w(u)                                        # (B,n_bins)
        pi = w.mean(0) + 1e-12
        u_b = (w * u.unsqueeze(1)).sum(0) / (pi * len(u))
        e_b = (w * e.unsqueeze(1)).sum(0) / (pi * len(u))
        return torch.sum(torch.abs(u_b - e_b) * pi)

def collision_entropy(p: torch.Tensor) -> torch.Tensor:
    """Scalar H₂ for a batch of probability vectors (…, C)."""
    return -torch.log2((p ** 2).sum(dim=-1) + 1e-12)


def err_from_collision_entropy(u: torch.Tensor) -> torch.Tensor:
    """The *theoretical* expected error 1−p  given collision entropy u.

    Uses the **confidence branch**  p ∈ [0.5, 1];
    valid for binary classification where u ∈ [0, 1] bits.
    """
    # invert H₂(p)  for  p ≥ 0.5
    inner = torch.clamp(2 * 2 ** (-u) - 1.0, min=0.0)
    return 0.5 * (1.0 - torch.sqrt(inner))  # = 1 - p_confidence


class ProbUCELossEFSoft_CE(nn.Module):
    """
    Collision‐entropy calibration loss with
      • equal‐frequency bins (quantiles of H2)
      • soft bin membership via sigmoid width τ
      • uniform bin weights (1 / n_bins)
    Objective per bin: | mean_err_in_soft_bin – f(H2_mean_in_soft_bin) |
    """
    def __init__(self, n_bins: int = 15, tau: float = 0.1):
        super().__init__()
        self.n_bins = n_bins
        self.tau    = tau

    @staticmethod
    def _risk_from_H2(u: torch.Tensor) -> torch.Tensor:
        inner = torch.clamp(2.0 * 2.0 ** (-u) - 1.0, min=0.0)
        return 0.5 * (1.0 - torch.sqrt(inner))      # f(H₂)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert logits.requires_grad, "Logits must require grad"

        # 1) get ensemble‐mean probs
        probs = (
            torch.softmax(logits, -1).mean(1)
            if logits.dim() == 3 else torch.softmax(logits, -1)
        )

        # 2) H2, error indicator
        H2       = collision_entropy(probs)                         # (B,)
        err_real = (probs.argmax(dim=-1) != labels).float()         # (B,)

        # 3) equal‐frequency edges
        edges = torch.quantile(
            H2.detach(),
            torch.linspace(0, 1, self.n_bins + 1, device=H2.device)
        )
        # ensure max H2 falls into last bin
        edges[-1] += 1e-6

        lo, hi = edges[:-1], edges[1:]   # each shape (M,)

        # 4) soft‐bin memberships w[b,m] = sigmoid((H2 - lo_m)/τ) - sigmoid((H2 - hi_m)/τ)
        #    → shape (B, M)
        H2_exp = H2.unsqueeze(1)                              # (B,1)
        lop    = (H2_exp - lo.unsqueeze(0)) / self.tau         # (B,M)
        hip    = (H2_exp - hi.unsqueeze(0)) / self.tau         # (B,M)
        w      = torch.sigmoid(lop) - torch.sigmoid(hip)       # (B,M)

        # 5) per‐bin aggregated H2 and error
        gaps = []
        for m in range(self.n_bins):
            wm    = w[:, m].unsqueeze(1)                       # (B,1)
            denom = wm.sum() + 1e-12
            u_bar = (wm * H2_exp).sum() / denom                # scalar
            e_bar = (wm * err_real.unsqueeze(1)).sum() / denom # scalar
            ref   = self._risk_from_H2(u_bar)                  # scalar
            gaps.append(torch.abs(e_bar - ref))

        # 6) uniform average over bins
        return torch.stack(gaps).mean()



# ---------------------------------------------------------------------------
# 1) Equal‑frequency hard‑bin Prob‑UCE for collision entropy
# ---------------------------------------------------------------------------

class ProbUCELossEF_CE(nn.Module):
    """
    Collision-entropy calibration loss with **equal-frequency hard bins**
    and **uniform bin weights** (1 / n_bins).

    Objective per bin:  |  empirical error  –  f(H₂) |.
    """

    def __init__(self, n_bins: int = 15):
        super().__init__()
        self.n_bins = n_bins

    @staticmethod
    def _risk_from_H2(u: torch.Tensor) -> torch.Tensor:
        inner = torch.clamp(2.0 * 2.0 ** (-u) - 1.0, min=0.0)
        return 0.5 * (1.0 - torch.sqrt(inner))      # f(H₂)

    # ------------------------------------------------------------------
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # --- make sure logits still have grad ---
        assert logits.requires_grad, "Pass non-detached logits into the loss"

        probs = (
            torch.softmax(logits, -1).mean(1)
            if logits.dim() == 3 else torch.softmax(logits, -1)
        )

        H2 = collision_entropy(probs)               # (B,)
        p_conf = probs.max(dim=-1).values
        r_pred = 1.0 - p_conf                       # (B,)
        err_real = (probs.argmax(-1) != labels).float()

        # equal-frequency edges
        edges = torch.quantile(
            H2.detach(), torch.linspace(0, 1, self.n_bins + 1, device=H2.device)
        )
        edges[-1] += 1e-6                           # include max exactly

        gaps = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (H2 > lo) & (H2 <= hi)
            if mask.any():
                u_bar   = H2[mask].mean()
                err_bar = err_real[mask].mean()
                ref     = self._risk_from_H2(u_bar)
                gaps.append(torch.abs(err_bar - ref))
            else:
                # keep graph alive even for empty bins
                gaps.append((H2[0] * 0.0).sum())

        return torch.stack(gaps).mean()    # weight = 1 / n_bins              # simple average


# ---------------------------------------------------------------------------
# 2) Soft / differentiable Prob‑UCE for collision entropy
# ---------------------------------------------------------------------------
class ProbUCELoss_CE(nn.Module):
    """
    Calibrates collision entropy so that   risk = f(H2).
    Uses confidence (max-prob) branch → monotone u ↔ error mapping.
    """

    def __init__(self, n_bins: int = 15, tau: float = 0.1):
        super().__init__()
        self.n_bins, self.tau = n_bins, tau
        centres = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins)
        self.register_buffer("centres", centres)
        self.delta = 1.0 / n_bins

    # soft bin weights
    def _w(self, u):
        l = (u.unsqueeze(1) - (self.centres - self.delta / 2)) / self.tau
        r = (u.unsqueeze(1) - (self.centres + self.delta / 2)) / self.tau
        return torch.sigmoid(l) - torch.sigmoid(r)

    @staticmethod
    def _risk_from_H2(u):
        inner = torch.clamp(2.0 * 2.0 ** (-u) - 1.0, min=0.0)
        return 0.5 * (1.0 - torch.sqrt(inner))          # f(H2)

    # ------------------------------------------------------------------
    def forward(self, logits, labels):
        # probabilities (B,C)
        probs = (
            torch.softmax(logits, -1).mean(1)
            if logits.dim() == 3 else torch.softmax(logits, -1)
        )

        # uncertainty & predicted risk
        u = collision_entropy(probs)                    # H2  (B,)
        p_conf = probs.max(dim=-1).values               # confidence ≥ 0.5
        r_pred = 1.0 - p_conf                           # risk 0…0.5 (B,)

        # realised 0/1 error
        err_real = (probs.argmax(-1) != labels).float() # (B,)

        # soft histogram
        w  = self._w(u)                                 # (B,n_bins)
        pi = w.mean(0) + 1e-12
        r_b   = (w * r_pred.unsqueeze(1)).sum(0) / (pi * len(u))
        err_b = (w * err_real.unsqueeze(1)).sum(0) / (pi * len(u))

        r_ref_b = self._risk_from_H2(u_b := (w * u.unsqueeze(1)).sum(0) / (pi * len(u)))

        return torch.sum(torch.abs(err_b - r_ref_b) * pi)


class ProbUCELossHard_CE(nn.Module):
    """
    Hard-binned collision-entropy calibration loss:
      • Bins centred at (0.5/n_bins, …, 1−0.5/n_bins)
      • Hard indicator membership
      • Population-weighted absolute gap |err_b – f(H2_b)|
    """
    def __init__(self, n_bins: int = 15):
        super().__init__()
        self.n_bins = n_bins
        centres = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins)
        self.register_buffer("centres", centres)
        self.delta = 1.0 / n_bins

    @staticmethod
    def _risk_from_H2(u: torch.Tensor) -> torch.Tensor:
        inner = torch.clamp(2.0 * 2.0 ** (-u) - 1.0, min=0.0)
        return 0.5 * (1.0 - torch.sqrt(inner))  # f(H2)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 1) Ensemble-mean probabilities
        probs = (
            torch.softmax(logits, -1).mean(1)
            if logits.dim() == 3 else torch.softmax(logits, -1)
        )
        # 2) Collision entropy H2, predicted risk r_pred, and true error
        u        = collision_entropy(probs)                       # (B,)
        p_conf   = probs.max(dim=-1).values                       # (B,)
        r_pred   = 1.0 - p_conf                                    # (B,)
        err_real = (probs.argmax(dim=-1) != labels).float()       # (B,)

        gaps = []
        pis  = []
        # 3) Loop over bins
        for c in self.centres:
            lo = c - self.delta/2
            hi = c + self.delta/2
            mask = (u >= lo) & (u < hi)  # (B,)
            pi   = mask.float().mean()   # population fraction
            if pi.item() > 0:
                # mean predicted risk and mean true error in this bin
                r_b   = r_pred[mask].mean()
                err_b = err_real[mask].mean()
                # reference risk from mean entropy
                u_b   = u[mask].mean()
                r_ref = self._risk_from_H2(u_b)
                gaps.append(torch.abs(err_b - r_ref) * pi)
            else:
                # no contribution for empty bins
                gaps.append(torch.tensor(0.0, device=u.device))
        # 4) sum over bins
        return torch.stack(gaps).sum()

# ---------------------------------------------------------------------
# 2.  Temperature layers (unchanged)
# ---------------------------------------------------------------------
def _inv_sigmoid(t_eff: float, t_min: float, t_max: float) -> float:
    r = (t_eff - t_min) / (t_max - t_min)
    r = min(max(r, 1e-6), 1 - 1e-6)
    return math.log(r / (1 - r))

class GlobalScaler(nn.Module):
    def __init__(self, init=1.5, t_min=0.5, t_max=20.0):
        super().__init__()
        self.t_min, self.t_max = t_min, t_max
        self.raw = nn.Parameter(torch.tensor(_inv_sigmoid(init, t_min, t_max)))

    def T(self):  # scalar
        return self.t_min + (self.t_max - self.t_min) * torch.sigmoid(self.raw)

    def forward(self, logits):          # (B,C) or (B,K,C)
        return logits / self.T()

    def forward_ext_temp(self, logits, temps):  # FBTS helper
        return logits / temps.unsqueeze(2)

class EnsembleScaler(nn.Module):
    def __init__(self, n_models=5, init=1.5, t_min=0.5, t_max=20.0):
        super().__init__()
        self.t_min, self.t_max = t_min, t_max
        raw = _inv_sigmoid(init, t_min, t_max)
        self.raw = nn.Parameter(torch.full((n_models,), raw))

    def T(self):     # (K,)
        return self.t_min + (self.t_max - self.t_min) * torch.sigmoid(self.raw)

    def forward(self, logits):          # (B,K,C)
        return logits / self.T().view(1, -1, 1)

    def forward_ext_temp(self, logits, temps):
        return logits / temps.unsqueeze(2)


class LinearFBTS(nn.Module):
    def __init__(self, in_dim, K=5, t_min=0.5, t_max=20.0):
        super().__init__()
        self.t_min, self.t_max = t_min, t_max
        self.lin = nn.Linear(in_dim, K)

        # 1  neutral bias
        # nn.init.zeros_(self.lin.bias)        # T ≈ 1

        # small weight init so T varies ±10 %
        nn.init.normal_(self.lin.weight, 0.0, 0.1)
        self.lin.bias.data.fill_(math.log(2.0))

    def forward(self, x):
        # 2  smooth non-linearity instead of square
        T = F.softplus(self.lin(x)) + 1e-4
        return torch.clamp(T, self.t_min, self.t_max)


class MLPFBTS(nn.Module):
    """
    Small MLP head for feature-based temperature scaling:
      x ∈ ℝ^D → hidden layer (H units) → K temperatures T_k ∈ [t_min, t_max]

    T_k(x) = clamp( softplus( W2·ReLU(W1·x + b1) + b2 ) + ε, t_min, t_max )
    """
    def __init__(self, in_dim: int, K: int = 5, H: int = 32,
                 t_min: float = 0.5, t_max: float = 20.0):
        super().__init__()
        self.t_min, self.t_max = t_min, t_max

        # hidden layer
        self.fc1 = nn.Linear(in_dim, H)
        # output layer
        self.fc2 = nn.Linear(H, K)

        # init: small weights on output so T varies ±10%
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        # bias so softplus(b2) ≈ 1  ⇒ b2 ≈ log(e¹ − 1) ≈ log(1.718) ≈ 0.54,
        # but we keep your original log(2) for a ~1.1 start
        nn.init.constant_(self.fc2.bias, math.log(2.0))

        # you can customise fc1 init if needed; default is fine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (B, in_dim) input features
        Returns:
          temps: (B, K) positive, clamped temperatures
        """
        h = F.relu(self.fc1(x))                   # (B, H)
        raw = self.fc2(h)                         # (B, K)
        T   = F.softplus(raw) + 1e-4              # ensure >0
        return torch.clamp(T, self.t_min, self.t_max)

# class LinearFBTS(nn.Module):
#     def __init__(self, in_dim: int, K=5, t_min=0.5, t_max=20.0):
#         super().__init__()
#         self.t_min, self.t_max = t_min, t_max
#         self.lin = nn.Linear(in_dim, K)
#         self.lin.bias.data.fill_(2.5)

#     def forward(self, x):               # (B,K)
#         return torch.clamp(self.lin(x).pow(2), self.t_min, self.t_max)

# ---------------------------------------------------------------------
# 3.  Training helpers (now use ProbUCELoss)
# ---------------------------------------------------------------------
def _train_static(data, scaler: nn.Module, epochs=200, lr=1e-3):
    loader = DataLoader(data, batch_size=256, shuffle=True)
    opt = optim.Adam([p for p in scaler.parameters() if p.requires_grad], lr=lr)
    crit = ProbUCELossEFSoft_CE()
    best_state, best_loss = None, float("inf")
    for _ in range(epochs):
        for _, logits, y in loader:
            opt.zero_grad()
            loss = crit(scaler(logits), y)
            loss.backward()
            opt.step()
        if loss.item() < best_loss:
            best_loss, best_state = loss.item(), copy.deepcopy(scaler.state_dict())
    scaler.load_state_dict(best_state)
    return scaler

def _train_fbts(data, in_dim: int, base: EnsembleScaler, epochs=200, lr=1e-3):
    loader = DataLoader(data, batch_size=256, shuffle=True)
    head = LinearFBTS(in_dim)
    opt  = optim.Adam(head.parameters(), lr=lr)
    crit = ProbUCELoss_CE()
    best_state, best_loss = None, float("inf")
    base.eval()
    for _ in range(epochs):
        for x, logits, y in loader:
            opt.zero_grad()
            loss = crit(base.forward_ext_temp(logits, head(x)), y)
            loss.backward()
            opt.step()
        if loss.item() < best_loss:
            best_loss, best_state = loss.item(), copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    return head, base

# ---------------------------------------------------------------------
# 4.  Public entry point
# ---------------------------------------------------------------------
def train_temp_scaler(data, mode="feat", epochs=400, lr=1e-2):
    if mode == "global":
        head, scaler = None, _train_static(data, GlobalScaler(), epochs, lr)
    elif mode == "ensemble":
        head, scaler = None, _train_static(data, EnsembleScaler(), epochs, lr)
    elif mode == "feat":
        base = EnsembleScaler()
        head, scaler = _train_fbts(data, data.X.shape[1], base, epochs, lr)  # type: ignore
    else:
        raise ValueError("mode must be 'feat', 'global', or 'ensemble'")
    return head, scaler

# ---------------------------------------------------------------------
# 5.  Hard UCE metric (unchanged)
# ---------------------------------------------------------------------
class UCEWrapper(nn.Module):
    @staticmethod
    def entropy(p): return -torch.log2((p ** 2).sum(-1) + 1e-12)

    def uceloss(self, logits, labels, n_bins=10):
        p = F.softmax(logits, -1).mean(1) if logits.dim() == 3 else F.softmax(logits, -1)
        u, e = self.entropy(p), (p.argmax(-1) != labels).float()
        edges = torch.linspace(0, 1, n_bins + 1, device=logits.device)
        uce = torch.zeros(1, device=logits.device)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (u > lo) & (u <= hi)
            if m.any():
                gap = torch.abs(u[m].mean() - e[m].mean())
                uce += gap * m.float().mean()
        return uce

__all__ = ["ProbUCELoss_CE","ProbUCELossEF_CE","ProbUCELossEF", "ProbUCELoss", "GlobalScaler", "EnsembleScaler",
           "LinearFBTS", "train_temp_scaler", "UCEWrapper"]
