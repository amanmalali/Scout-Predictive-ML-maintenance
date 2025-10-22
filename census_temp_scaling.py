#!/usr/bin/env python3
# census_prob_vs_uncert_calib.py  – 2025‑07 fix
# ------------------------------------------------------------------
# Probability‑ vs uncertainty‑calibrated temperatures on Census,
# including entropy spreads and FBTS T distributions.
# ------------------------------------------------------------------
import argparse, copy, pathlib, pickle, warnings
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import LBFGS
from sensitivity.new_temp_scaler import train_temp_scaler
from utils import custom_torch_dataset

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["figure.dpi"] = 120
DEVICE = "cpu"                           # set to "cuda" if you like

COL_PROB = {"none":"tab:orange", "global":"tab:blue", "ensemble":"tab:green"}
COL_UNC  = {"none":"tab:orange", "global":"tab:blue",
            "ensemble":"tab:green","feat":"tab:purple"}


# --- add just below risk_from_H2 ------------------------------------------
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


# ──────────────── helpers ─────────────────────────────────────────
def load_split(tag):
    base="./census/data"
    if tag=="calib_over":
        X=np.load(f"{base}/calib_over_x.npy"); y=np.load(f"{base}/calib_over_y.npy")
    else:
        X=np.load(f"{base}/sense_census_{tag}_x.npy"); y=np.load(f"{base}/sense_census_{tag}_y.npy")
    return X, torch.as_tensor(y ).long()

def to_tensor(x): return torch.as_tensor(x,dtype=torch.float32 )

def softmax_probs(log):
    """
    For a 3-D tensor (B, K, C) return the *mean probability* across K
    members, **after** applying softmax to each member.
    For 2-D logits simply return softmax(log).
    """
    log = to_tensor(log)
    if log.dim() == 3:                 # (B, K, C)
        p_k = F.softmax(log, dim=-1)   # softmax per member
        return p_k.mean(1).detach()    # mean over K
    else:                              # (B, C)
        return F.softmax(log, dim=-1).detach()

def collision_entropy(p): return -torch.log2((p**2).sum(-1)+1e-12)

def epi_uncertainty(log):
    log=to_tensor(log)
    if log.dim()!=3: return torch.zeros(len(log))
    p_k=F.softmax(log,dim=-1)
    return (collision_entropy(p_k.mean(1))-collision_entropy(p_k).mean(1)).clamp(min=0.).detach()

def risk_from_H2(H): return 0.5-torch.sqrt(torch.clamp((2**(-H)-0.5)/2,min=0.))

# ───────────── probability TS ────────────────────────────────────
def nll_loss(l,y): return F.cross_entropy(l,y,reduction="mean")
def learn_global_T_prob(log,y):
    w=torch.nn.Parameter(torch.tensor(0. ))
    opt=LBFGS([w],lr=.1,max_iter=50,line_search_fn="strong_wolfe"); log=log.detach()
    def c(): opt.zero_grad(); loss=nll_loss(log/(0.5+19.5*torch.sigmoid(w)),y); loss.backward(); return loss
    opt.step(c); return float(0.5+19.5*torch.sigmoid(w).cpu())
def learn_per_T_prob(log3d,y):
    _,K,_=log3d.shape
    w=torch.nn.Parameter(torch.zeros(K ))
    opt=LBFGS([w],lr=.1,max_iter=50,line_search_fn="strong_wolfe"); l=log3d.detach()
    def c(): opt.zero_grad(); T=0.5+19.5*torch.sigmoid(w); loss=nll_loss((l/T.view(1,K,1)).mean(1),y); loss.backward(); return loss
    opt.step(c); return (0.5+19.5*torch.sigmoid(w.detach().cpu())).cpu()

def apply_prob_T(log, Tg=None, Tv=None):
    t=to_tensor(log)
    if Tg is not None:                         # scalar global T
        return (t/Tg).cpu().numpy()
    if Tv is not None:                         # per‑member vector
        if t.dim()==3 and t.shape[1]==len(Tv): # expected shape (B,K,C)
            return (t/Tv.view(1,-1,1).to(t)).cpu().numpy()
        # fallback: logits already collapsed → use mean(Tv)
        return (t/Tv.mean()).cpu().numpy()
    return log

# ───────────── plotting small helpers (diag/hist/RC) ─────────────
def rel_diag(ax,probs,y,label,color):
    edges=torch.linspace(0,1,16 )
    xs,ys,w=[],[],[]
    for i in range(15):
        m=(probs[:,1]>=edges[i])&(probs[:,1]<edges[i+1])
        if m.any(): xs.append(probs[m,1].mean()); ys.append((y[m]==1).float().mean()); w.append(m.float().mean())
    xs=torch.stack(xs); ys=torch.stack(ys); w=torch.stack(w)
    ece=(w*torch.abs(xs-ys)).sum().item()
    ax.plot(xs.cpu(),ys.cpu(),"o-",color=color,label=f"{label} (ECE {ece:.3f})")
    ax.plot([0,1],[0,1],"--",lw=.8,color="black"); ax.set(xlim=(0,1),ylim=(0,1))
def unc_diag(ax,u,err,label,color):
    u=(u-u.min())/(u.max()-u.min()+1e-12)
    edges=torch.linspace(0,1,16 )
    xs,ys,w=[],[],[]
    for i in range(15):
        m=(u>=edges[i])&(u<edges[i+1])
        if m.any(): xs.append(u[m].mean()); ys.append(err[m].mean()); w.append(m.float().mean())
    xs=torch.stack(xs); ys=torch.stack(ys); w=torch.stack(w)
    uce=(w*torch.abs(xs-ys)).sum().item()
    ax.plot(xs.cpu(),ys.cpu(),"o-",color=color,label=f"{label} (UCE {uce:.3f})")
    ax.plot([0,1],[0,1],"--",lw=.8,color="black"); ax.set(xlim=(0,1),ylim=(0,1))
def h2_diag(ax,H,err,label,color):
    bins=torch.linspace(H.min(),H.max(),16 )
    xs,ys,w=[],[],[]
    for i in range(15):
        m=(H>=bins[i])&(H<bins[i+1])
        if m.any(): xs.append(H[m].mean()); ys.append(err[m].mean()); w.append(m.float().mean())
    xs=torch.stack(xs); ys=torch.stack(ys); w=torch.stack(w)
    uce=(w*torch.abs(ys-risk_from_H2(xs))).sum().item()
    ax.plot(xs.cpu(),ys.cpu(),"o-",color=color,label=f"{label} (UCE {uce:.3f})")
    Href=torch.linspace(0,1,200)
    ax.plot(Href,risk_from_H2(Href),"--",lw=.8,color="black",alpha=.5)
    ax.set(xlim=(0,1),ylim=(0,0.5),xlabel=r"$H_2$",ylabel="emp. error")
def risk_cov(ax,u,err,label,color):
    idx=u.argsort(); risk=err[idx].cumsum(0)/torch.arange(1,len(err)+1 )
    cov=torch.arange(1,len(err)+1 )/len(err)
    ax.plot(cov.cpu(),risk.cpu(),lw=1.5,color=color,label=label)

def diagnostics_grid(pdf,method_dict,rows,labels,colors):
    fig,axes=plt.subplots(len(rows),4,figsize=(14,3*len(rows)))
    for c,t in enumerate(["Prob","Epi-U (norm)","Tot $H_2$","Epi $H_2$"]):
        axes[0][c].set_title(t)
    for r,tag in enumerate(rows):
        y=labels[tag]
        for m,col in colors.items():
            log=method_dict[tag][m]; p=softmax_probs(log); err=(p.argmax(-1)!=y).float()
            tot=collision_entropy(p); epi=epi_uncertainty(log)
            rel_diag(axes[r][0],p,y,m if r==0 else "",col)
            unc_diag(axes[r][1],epi,err,"",col)
            h2_diag(axes[r][2],tot,err,m if r==0 else "",col)
            h2_diag(axes[r][3],epi,err,"",col)
    axes[0][0].legend(frameon=False,fontsize=7); fig.tight_layout(); fig.savefig(pdf,dpi=300); plt.close(fig)

def rc_fig(pdf,method_dict,rows,labels,colors):
    fig,ax=plt.subplots(figsize=(4.8,3.5))
    for tag in rows:
        y=labels[tag]
        for m,col in colors.items():
            log=method_dict[tag][m]
            risk_cov(ax, epi_uncertainty(log),
                     (softmax_probs(log).argmax(-1)!=y).float(),
                     f"{tag}-{m}", col)
    ax.set(xlabel="coverage",ylabel="selective risk",title="Risk–coverage (epistemic)")
    ax.legend(frameon=False,fontsize=7); fig.tight_layout(); fig.savefig(pdf,dpi=300); plt.close(fig)

def epi_hist(pdf,methods_dict,colors):
    fig,axes=plt.subplots(1,len(methods_dict),figsize=(3.5*len(methods_dict),3))
    if len(methods_dict)==1: axes=[axes]
    for (m,log),ax in zip(methods_dict.items(),axes):
        ax.hist(epi_uncertainty(log).cpu(),bins=30,density=True,color=colors[m],alpha=.6)
        ax.set_title(m); ax.set_xlabel(r"$U_{\mathrm{epi}}$"); ax.set_ylabel("density")
    fig.tight_layout(); fig.savefig(pdf,dpi=300); plt.close(fig)

def fbts_hist(pdf,xinp,lin,K):
    xinp=to_tensor(xinp)
    T= lin.forward(xinp).detach()  # (B,K)

    fig,axes=plt.subplots(1,K,figsize=(3.2*K,3))
    for k in range(K):
        axes[k].hist(T[:,k],bins=30,color="tab:purple",alpha=.7)
        axes[k].set_title(f"member {k}"); axes[k].set_xlabel("T"); axes[k].set_ylabel("count")
    fig.tight_layout(); fig.savefig(pdf,dpi=300); plt.close(fig)

# ─────────────────── main ─────────────────────────────────────────
if __name__=="__main__":
    P=argparse.ArgumentParser()
    P.add_argument("--model",default="./census/saved_models/sense_model_v2.pkl")
    P.add_argument("--out",  default="./ts_eval_census")
    P.add_argument("--trials",type=int,default=3)
    args=P.parse_args(); out=pathlib.Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)

    with open(args.model,"rb") as f: sense=pickle.load(f)
    Xtr_np,ytr = load_split("train")
    Xva_np,yva = load_split("val")
    Xcal_np,ycal= load_split("calib_over")
    raw_tr=sense.gen_logits(Xtr_np); raw_va=sense.gen_logits(Xva_np); raw_cal=sense.gen_logits(Xcal_np)

    # ----- probability calibration -----
    Tg=learn_global_T_prob(to_tensor(raw_cal).mean(1),ycal)
    Tv=learn_per_T_prob(to_tensor(raw_cal),ycal)        # len K
    prob={"train":{"none":raw_tr,
                   "global":apply_prob_T(raw_tr,Tg),
                   "ensemble":apply_prob_T(raw_tr,None,Tv)},
          "val"  :{"none":raw_va,
                   "global":apply_prob_T(raw_va,Tg),
                   "ensemble":apply_prob_T(raw_va,None,Tv)}}
    labels={"train":ytr,"val":yva}
    diagnostics_grid(out/"census_diagnostics_prob_calib.pdf",prob,["train","val"],labels,COL_PROB)
    rc_fig(out/"risk_coverage_prob_calib.pdf",prob,["train","val"],labels,COL_PROB)
    epi_hist(out/"epi_spread_prob_calib.pdf",prob["val"],COL_PROB)

    # ----- uncertainty calibration (global / ensemble / feat) -----
    calib_ds=custom_torch_dataset(Xcal_np,raw_cal,ycal.cpu().numpy())
    # --- replace the old `fit` -------------------------------------------------
    def fit(mode: str, trials: int,
        X_train: np.ndarray, y_train: torch.Tensor):
        """
        Train `trials` initialisations of the temperature scaler for `mode`
        and keep the one with the *lowest UCE on the TRAIN split*.
        """
        best = None                                       # (uce, lin, sc)
        for _ in range(trials):
            lin, sc = train_temp_scaler(
                calib_ds, None,
                mode=mode,
                lr=1e-3,
                epochs=400,
            )
            tmp = copy.deepcopy(sense)
            tmp.lin_model, tmp.scaler = lin, sc

            logits_tr = tmp.inf(X_train, apply_temp=True, mode=mode)
            uce_tr    = compute_uce_from_logits(logits_tr, y_train)

            if best is None or uce_tr < best[0]:
                best = (uce_tr, lin, sc)

        return best[1], best[2]          # lin_module, scaler

    lin_g,sc_g=fit("global",2,Xtr_np, ytr) 
    sense.lin_model=lin_g
    sense.scaler=sc_g
    print("Trained global scaler")
    with open("./census/saved_models/sense_model_global.pkl",'wb') as output:
        pickle.dump(sense,output)

    lin_e,sc_e=fit("ensemble",2,Xtr_np, ytr)

    sense.lin_model=lin_e
    sense.scaler=sc_e

    print("Trained ensemble scaler")
    with open("./census/saved_models/sense_model_ensemble.pkl",'wb') as output:
        pickle.dump(sense,output)

    lin_f,sc_f=fit("feat",args.trials,Xtr_np, ytr)

    sense.lin_model=lin_f
    sense.scaler=sc_f

    print("Trained fbts scaler")
    with open("./census/saved_models/sense_model_fbts.pkl",'wb') as output:
        pickle.dump(sense,output)

    def build_unc(X):
        d={"none":sense.gen_logits(X)}
        for mode,(lin,sc) in {"global":(lin_g,sc_g),"ensemble":(lin_e,sc_e),"feat":(lin_f,sc_f)}.items():
            sense.lin_model,sense.scaler=lin,sc
            d[mode]=sense.inf(X,True,mode)
        return d
    unc={"train":build_unc(Xtr_np),"val":build_unc(Xva_np)}
    diagnostics_grid(out/"census_diagnostics_uncertainty_calib.pdf",unc,["train","val"],labels,COL_UNC)
    rc_fig(out/"risk_coverage_uncertainty_calib.pdf",unc,["train","val"],labels,COL_UNC)
    epi_hist(out/"epi_spread_uncertainty_calib.pdf",unc["val"],COL_UNC)
    fbts_hist(out/"fbts_temp_distribution.pdf",Xva_np,lin_f,len(Tv))

    print("✓ done – PDFs saved to",out)
