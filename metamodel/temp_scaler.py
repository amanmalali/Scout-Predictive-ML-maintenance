import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #

def _inv_sigmoid_T(T_eff: float, T_min: float, T_max: float) -> float:
    """
    Inverse of the bounded-sigmoid temperature parameterization.
    Returns raw z such that: T_eff = T_min + (T_max - T_min) * sigmoid(z).
    """
    # clip to open interval to avoid inf logits at the ends
    frac = (T_eff - T_min) / (T_max - T_min)
    frac = min(max(frac, 1e-6), 1 - 1e-6)
    return math.log(frac / (1.0 - frac))

# --------------------------------------------------------------------------- #
# Model Definitions
# --------------------------------------------------------------------------- #

class linear_model(nn.Module):
    """
    Feature-conditioned temperature head (Linear or Non-Linear MLP).
    
    Args:
        in_features: Input feature dimension.
        K: Number of output temperatures (typically ensemble size).
        T_min, T_max: Temperature bounds.
        hidden_dim: Size of hidden layer. If 0, model is Linear. If > 0, model is MLP.
    """
    def __init__(self, in_features: int, K: int = 5,
                 T_min: float = 0.5, T_max: float = 7.0,
                 hidden_dim: int = 0):
        super().__init__()
        assert 0. < T_min < T_max
        self.T_min, self.T_max = T_min, T_max
        
        # Stability: BatchNorm helps significantly with unscaled features
        self.bn = nn.BatchNorm1d(in_features)
        
        if hidden_dim > 0:
            # --- Non-Linear MLP Version ---
            self.model = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1), # Slight regularization
                nn.Linear(hidden_dim, K)
            )
            # Initialize final layer to Zero so it starts flat
            final_layer = self.model[-1]
            nn.init.zeros_(final_layer.weight)
        else:
            # --- Linear Version ---
            self.model = nn.Linear(in_features, K)
            # Initialize to Zero
            nn.init.zeros_(self.model.weight)
            final_layer = self.model

        # Initialize bias to exactly T=1.5 using inverse sigmoid
        init_raw = _inv_sigmoid_T(1.5, T_min, T_max)
        final_layer.bias.data.fill_(init_raw)

    def forward(self, x):
        x = self.bn(x)
        t_raw = self.model(x)
        # Sigmoid parameterization for smooth gradients everywhere
        temps = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(t_raw)
        return temps


class scaler_model(nn.Module):
    """Static per-member temperature scaler for ensemble logits (Wrapper)."""
    def __init__(self, models=5):
        super(scaler_model, self).__init__()
        self.temp = nn.Parameter(torch.ones(models) * 1.5)

    def forward(self, logits):
        temps = self.temp.unsqueeze(1).expand(logits.size(0), logits.size(1), logits.size(2))
        return logits / temps
    
    def forward_ext_temp(self, logits, temp):
        temps = temp.unsqueeze(2).expand(logits.size(0), logits.size(1), logits.size(2))
        return logits / temps


class GlobalScaler(nn.Module):
    """Single scalar temperature for all ensemble members."""
    def __init__(self, init_temp: float = 1.5, T_min: float = 0.5, T_max: float = 7.0):
        super().__init__()
        assert 0 < T_min < T_max
        self.T_min, self.T_max = T_min, T_max
        raw_init = _inv_sigmoid_T(init_temp, T_min, T_max)
        self.raw_temp = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T_eff = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self.raw_temp)
        return logits / T_eff


class EnsembleScaler(nn.Module):
    """Static per-member temperatures."""
    def __init__(self, n_models: int = 5, init_temp: float = 1.5, T_min: float = 0.5, T_max: float = 7.0):
        super().__init__()
        assert 0 < T_min < T_max
        self.T_min, self.T_max = T_min, T_max
        raw_init = _inv_sigmoid_T(init_temp, T_min, T_max)
        self.raw_temp = nn.Parameter(
            torch.full((n_models,), raw_init, dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T_eff = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self.raw_temp)
        t = T_eff.view(1, -1, 1).expand_as(logits)
        return logits / t


# --------------------------------------------------------------------------- #
# Loss Function (Fixed TypeError)
# --------------------------------------------------------------------------- #

class UCECollisionEntropyLoss(torch.nn.Module):
    """UCE-style calibration loss."""
    def __init__(self, n_bins: int = 10):
        super().__init__()
        self.n_bins = n_bins

    @staticmethod
    def _risk_from_H2(u: torch.Tensor) -> torch.Tensor:
        inner = torch.clamp(2.0 * 2.0**(-u) - 1.0, min=0.0)
        return 0.5 * (1.0 - torch.sqrt(inner))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        device = logits.device

        # 1) compute per-sample collision entropy H2 and 0/1 error
        if logits.dim() == 3:
            probs = F.softmax(logits, dim=-1).mean(dim=1)
        else:
            probs = F.softmax(logits, dim=-1)

        H2   = -torch.log2((probs**2).sum(-1) + 1e-12)
        pred = probs.argmax(-1)
        err  = (pred != labels).float()

        # 2) uniform bins over [min H2, max H2]
        # [FIX] Use .item() to fix TypeError
        h2_min = H2.min().item()
        h2_max = H2.max().item()

        # [FIX] Prevent degenerate bins
        if h2_max <= h2_min + 1e-6:
            h2_max = h2_min + 1e-3

        edges = torch.linspace(h2_min, h2_max, self.n_bins+1, device=device)
        edges[-1] += 1e-6 

        uce_terms = []
        err_bins  = []
        H2_bins   = []

        # 3) per-bin gap
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (H2 > lo) & (H2 <= hi)
            prop = mask.float().mean()

            if prop.item() == 0.0:
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

        uce = torch.stack(uce_terms).sum()
        return uce, torch.stack(err_bins), torch.stack(H2_bins)


# --------------------------------------------------------------------------- #
# Training Loops (With Validation & Model Selection)
# --------------------------------------------------------------------------- #

def train_scaler(data, epochs: int = 200, feat: bool = True, lr: float = 1e-3, hidden_dim: int = 0):
    """
    Train feature-based temperature scaling (FBTS).
    """
    # Detect Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 80/20 Validation Split
    n_samples = len(data)
    n_val = int(0.2 * n_samples)
    n_train = n_samples - n_val
    
    indices = torch.randperm(n_samples).tolist()
    train_subset = Subset(data, indices[:n_train])
    val_subset = Subset(data, indices[n_train:])
    
    train_loader = DataLoader(train_subset, batch_size=len(train_subset), shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=len(val_subset), shuffle=False)
    
    scaler = scaler_model().to(device)
    loss_f = UCECollisionEntropyLoss()

    if feat:
        # Pass hidden_dim to choose Linear (0) or MLP (>0)
        lin_model = linear_model(data.X.shape[1], hidden_dim=hidden_dim).to(device)
        optimizer = optim.Adam(lin_model.parameters(), lr=lr)
        best_state, best_val_loss = None, float("inf")

        scaler.eval() 
        for e in range(epochs):
            # --- Train ---
            lin_model.train()
            for x, logits, y in train_loader:
                x, logits, y = x.to(device), logits.to(device), y.to(device)
                optimizer.zero_grad()
                temps = lin_model(x)
                preds = scaler.forward_ext_temp(logits, temps)
                loss, *_ = loss_f(preds, y)
                loss.backward()
                optimizer.step()

            # --- Validate ---
            lin_model.eval()
            with torch.no_grad():
                # Loop allows for larger val sets, though batch is usually full set here
                val_losses = []
                for x_val, logits_val, y_val in val_loader:
                    x_val, logits_val, y_val = x_val.to(device), logits_val.to(device), y_val.to(device)
                    temps_val = lin_model(x_val)
                    preds_val = scaler.forward_ext_temp(logits_val, temps_val)
                    val_l, *_ = loss_f(preds_val, y_val)
                    val_losses.append(val_l.item())
                
                # Average loss if multiple batches (usually 1)
                avg_val_loss = sum(val_losses) / len(val_losses)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = copy.deepcopy(lin_model.state_dict())

            if e % 20 == 0:
                print(f"EPOCH {e:03d} – Val UCE: {avg_val_loss:.4f}")

        if best_state is not None:
            lin_model.load_state_dict(best_state)
            print(f"Best FBTS Val UCE ({'MLP' if hidden_dim>0 else 'Linear'}):", best_val_loss)
            return lin_model, scaler
        return None, None
    else:
        return None, None


def _train_static_scaler(data, scaler: nn.Module, epochs: int = 200, lr: float = 1e-3):
    """
    Train a bounded static scaler (global or per-member).
    """
    # Detect Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = scaler.to(device)

    # 80/20 Validation Split
    n_samples = len(data)
    n_val = int(0.2 * n_samples)
    n_train = n_samples - n_val
    
    indices = torch.randperm(n_samples).tolist()
    train_subset = Subset(data, indices[:n_train])
    val_subset = Subset(data, indices[n_train:])
    
    train_loader = DataLoader(train_subset, batch_size=len(train_subset), shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=len(val_subset), shuffle=False)

    opt = optim.Adam([scaler.raw_temp], lr=lr)
    loss_f = UCECollisionEntropyLoss()
    
    best_state, best_val_loss = None, float("inf")

    for epoch in range(epochs):
        # --- Train ---
        scaler.train()
        for _, logits, labels in train_loader:
            logits, labels = logits.to(device), labels.to(device)
            opt.zero_grad()
            loss, *_ = loss_f(scaler(logits), labels)
            loss.backward()
            opt.step()

        # --- Validate ---
        scaler.eval()
        with torch.no_grad():
            val_losses = []
            for _, logits_val, labels_val in val_loader:
                logits_val, labels_val = logits_val.to(device), labels_val.to(device)
                val_l, *_ = loss_f(scaler(logits_val), labels_val)
                val_losses.append(val_l.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(scaler.state_dict())
            
        if epoch % 20 == 0:
            print(f"EPOCH {epoch:03d} – Val UCE: {avg_val_loss:.4f}")

    if best_state is not None:
        scaler.load_state_dict(best_state)

    with torch.no_grad():
        # Ensure we are on the same device for calculations
        T_eff = scaler.T_min + (scaler.T_max - scaler.T_min) * \
                torch.sigmoid(scaler.raw_temp.data)
    print("Optimal effective temperature(s):", T_eff)

    return scaler


def train_temp_scaler(data, eval_data=None, mode: str = "feat", lr=1e-3, epochs=200, 
                      non_linear: bool = False, hidden_dim: int = 64):
    """
    Entry point for training temperature scaling.
    
    Args:
        non_linear (bool): If True, uses an MLP for 'feat' mode.
        hidden_dim (int): Hidden dimension size if non_linear is True.
    """
    print(f"Training temperature scaler [{mode}]")
    
    # Configure hidden dimension based on flag
    actual_hidden_dim = hidden_dim if non_linear else 0
    
    if mode == "feat":
        lin_model, scaler = train_scaler(data, epochs=epochs, lr=lr, feat=True, hidden_dim=actual_hidden_dim)
    elif mode == "global":
        lin_model = None
        scaler    = _train_static_scaler(data, GlobalScaler(), epochs=epochs, lr=lr)
    elif mode == "ensemble":
        lin_model = None
        scaler    = _train_static_scaler(data, EnsembleScaler(), epochs=epochs, lr=lr)
    else:
        raise ValueError("mode must be 'feat', 'global', or 'ensemble'")

    return lin_model, scaler