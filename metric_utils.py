
import argparse, copy, pathlib, pickle, warnings
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import LBFGS
from sensitivity.new_temp_scaler import train_temp_scaler
from utils import custom_torch_dataset

def to_tensor(x): return torch.as_tensor(x,dtype=torch.float32 )

def softmax_probs(log):
    log=to_tensor(log); log=log.mean(1) if log.dim()==3 else log
    return F.softmax(log,dim=-1).detach()

def collision_entropy(p): return -torch.log2((p**2).sum(-1)+1e-12)

def epi_uncertainty(log):
    log=to_tensor(log)
    if log.dim()!=3: return torch.zeros(len(log))
    p_k=F.softmax(log,dim=-1)
    return (collision_entropy(p_k.mean(1))-collision_entropy(p_k).mean(1)).clamp(min=0.).detach()

def risk_from_H2(H): return 0.5-torch.sqrt(torch.clamp((2**(-H)-0.5)/2,min=0.))

def compute_uce_from_logits(logits: torch.Tensor,
                            labels: torch.Tensor,
                            n_bins: int = 15) -> float:
    """
    UCE in H₂‑entropy space (binary).  `logits` are already temperature‑scaled.
    """
    probs = softmax_probs(logits)
    H2    = collision_entropy(probs)
    err   = (probs.argmax(-1) != labels).float()

    edges = torch.linspace(0, 1, n_bins + 1)
    uce   = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (H2 >= lo) & (H2 < hi)
        if m.any():
            gap = torch.abs(err[m].mean() - risk_from_H2(H2[m].mean()))
            uce += m.float().mean() * gap
    return float(uce)