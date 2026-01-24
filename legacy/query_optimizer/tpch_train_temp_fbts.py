#!/usr/bin/env python3
# tpch_prob_vs_uncert_calib.py
# ------------------------------------------------------------------
# Re‑trains probability and uncertainty temperature scalers on TPCH
# and produces the same plots/metrics as for CIFAR & Census.
# ------------------------------------------------------------------
import argparse, copy, math, pathlib, pickle, warnings
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import LBFGS
from sensitivity.new_temp_scaler import train_temp_scaler
from utils import custom_torch_dataset
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["figure.dpi"] = 120


# colour maps
COL_PROB = {"none":"tab:orange","global":"tab:blue","ensemble":"tab:green"}
COL_UNC  = {"none":"tab:orange","global":"tab:blue",
            "ensemble":"tab:green","feat":"tab:purple"}

# ───────────────────── data helpers ──────────────────────────────
def load_split(tag):
    base="./data"
    if tag=="calib_over":
        x=np.load(f"{base}/calib_over_x.npy"); y=np.load(f"{base}/calib_over_y.npy")
    else:
        x=np.load(f"{base}/sense_tpch_{tag}_x.npy"); y=np.load(f"{base}/sense_tpch_{tag}_y.npy")
    return x, torch.as_tensor(y ).long()

def to_tensor(x): return torch.as_tensor(x,dtype=torch.float32 )

# ───────────────────── core uncertainty utils ───────────────────
def softmax_probs(log):
    log=to_tensor(log); log=log.mean(1) if log.dim()==3 else log
    return F.softmax(log,dim=-1).detach()
def collision_entropy(p): return -torch.log2((p**2).sum(-1)+1e-12)
def epi_uncert(log):
    log=to_tensor(log)
    if log.dim()!=3: return torch.zeros(len(log))
    p_k=F.softmax(log,dim=-1)
    return (collision_entropy(p_k.mean(1))-collision_entropy(p_k).mean(1)).clamp(min=0.).detach()

# analytic mis‑classification risk for H₂
def risk_from_H2(H): return 0.5 - torch.sqrt(torch.clamp((2**(-H) - 0.5)/2, min=0.))

# ───────────────────── probability scalers (NLL) ────────────────
def nll(l,y): return F.cross_entropy(l,y,reduction="mean")
def learn_Tg(log,y):
    w=torch.nn.Parameter(torch.tensor(0. ))
    opt=LBFGS([w],lr=.1,max_iter=50,line_search_fn="strong_wolfe")
    log=log.detach()
    def c(): opt.zero_grad(); loss=nll(log/(0.5+6.5*torch.sigmoid(w)),y); loss.backward(); return loss
    opt.step(c); return float((0.5+6.5*torch.sigmoid(w)).cpu())
def learn_Tv(log3d,y):
    _,K,_=log3d.shape; w=torch.nn.Parameter(torch.zeros(K ))
    opt=LBFGS([w],lr=.1,max_iter=50,line_search_fn="strong_wolfe"); l=log3d.detach()
    def c(): opt.zero_grad(); T=0.5+6.5*torch.sigmoid(w); loss=nll((l/T.view(1,K,1)).mean(1),y); loss.backward(); return loss
    opt.step(c); return (0.5+6.5*torch.sigmoid(w.detach().cpu())).cpu()
def apply_prob_T(log,Tg=None,Tv=None):
    t=to_tensor(log)
    if Tg is not None: return (t/Tg).cpu().numpy()
    if Tv is not None:
        if t.dim()==3 and t.shape[1]==len(Tv): return (t/Tv.view(1,-1,1).to(t)).cpu().numpy()
        return (t/Tv.mean()).cpu().numpy()
    return log

# ───────────────────── extended metrics block ───────────────────
def uce_equal_width_H2(log,y,n_bins=15):
    p=softmax_probs(log); H=collision_entropy(p); err=(p.argmax(-1)!=y).float()
    edges=torch.linspace(0,1,n_bins+1 )
    uce=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(H>lo)&(H<=hi); cnt=m.sum()
        if cnt:
            gap=torch.abs(err[m].mean()-risk_from_H2(H[m].mean()))
            uce+=cnt.float()/len(y)*gap
    return float(uce)
def selective_nll(log,y,keep_frac=.80):
    p=softmax_probs(log); H=collision_entropy(p)
    keep=H<=torch.quantile(H,keep_frac)
    return float(-torch.log(p[keep,y[keep]]+1e-12).mean())
def all_metrics(log,y):
    p=softmax_probs(log); err=(p.argmax(-1)!=y).float()
    def aurc(score):
        idx=score.argsort(); risk=err[idx].cumsum(0)/torch.arange(1,len(err)+1 )
        cov=torch.arange(1,len(err)+1 )/len(err)
        return float(torch.trapz(risk,cov))
    tot=collision_entropy(p); epi=epi_uncert(log)
    return dict(
        AURC_p=aurc(p[:,1]), AURC_tot=aurc(tot), AURC_epi=aurc(epi),
        UCE=uce_equal_width_H2(log,y), sNLL80=selective_nll(log,y)
    )

# ───────────────────── plotting primitives ───────────────────────
def rel(ax,p,y,label,col):
    bins=torch.linspace(0,1,16 ); xs=ys=w=[]
    xs=[];ys=[];w=[]
    for i in range(15):
        m=(p[:,1]>=bins[i])&(p[:,1]<bins[i+1])
        if m.any(): xs.append(p[m,1].mean()); ys.append((y[m]==1).float().mean()); w.append(m.float().mean())
    xs,ys,w=torch.stack(xs),torch.stack(ys),torch.stack(w)
    ax.plot(xs.cpu(),ys.cpu(),"o-",color=col,label=f"{label} (ECE {(w*abs(xs-ys)).sum():.3f})")
    ax.plot([0,1],[0,1],"--",lw=.8,color="black"); ax.set(xlim=(0,1),ylim=(0,1))
def unc_func(ax,u,err,label,col):
    u=(u-u.min())/(u.max()-u.min()+1e-12); bins=torch.linspace(0,1,16 )
    xs=ys=w=[]
    xs=[];ys=[];w=[]
    for i in range(15):
        m=(u>=bins[i])&(u<bins[i+1])
        if m.any(): xs.append(u[m].mean()); ys.append(err[m].mean()); w.append(m.float().mean())
    xs,ys,w=torch.stack(xs),torch.stack(ys),torch.stack(w)
    ax.plot(xs.cpu(),ys.cpu(),"o-",color=col,label=f"{label} (UCE {(w*abs(xs-ys)).sum():.3f})")
    ax.plot([0,1],[0,1],"--",lw=.8,color="black"); ax.set(xlim=(0,1),ylim=(0,1))
def h2(ax,H,err,label,col):
    bins=torch.linspace(H.min(),H.max(),16 ); xs=ys=w=[]
    xs=[];ys=[];w=[]
    for i in range(15):
        m=(H>=bins[i])&(H<bins[i+1])
        if m.any(): xs.append(H[m].mean()); ys.append(err[m].mean()); w.append(m.float().mean())
    xs,ys,w=torch.stack(xs),torch.stack(ys),torch.stack(w)
    ax.plot(xs.cpu(),ys.cpu(),"o-",color=col,label=f"{label} (UCE {(w*abs(ys-risk_from_H2(xs))).sum():.3f})")
    Href=torch.linspace(0,1,200)
    ax.plot(Href,risk_from_H2(Href),"--",lw=.8,color="black",alpha=.5)
    ax.set(xlim=(0,1),ylim=(0,0.5),xlabel=r"$H_2$",ylabel="emp. error")
def risk_cov(ax,u,err,label,col):
    idx=u.argsort(); risk=err[idx].cumsum(0)/torch.arange(1,len(err)+1 )
    cov=torch.arange(1,len(err)+1 )/len(err)
    ax.plot(cov.cpu(),risk.cpu(),lw=1.5,color=col,label=label)
def diag_grid(pdf,methods,rows,labels,colors):
    fig,axes=plt.subplots(len(rows),4,figsize=(14,3*len(rows)))
    for c,t in enumerate(["Prob","Epi‑U (norm)","Tot $H_2$","Epi $H_2$"]): axes[0][c].set_title(t)
    for r,tag in enumerate(rows):
        y=labels[tag]
        for m,col in colors.items():
            log=methods[tag][m]; p=softmax_probs(log); err=(p.argmax(-1)!=y).float()
            rel(axes[r][0],p,y,m if r==0 else "",col)
            unc_func(axes[r][1],epi_uncert(log),err,"",col)
            h2(axes[r][2],collision_entropy(p),err,m if r==0 else "",col)
            h2(axes[r][3],epi_uncert(log),err,"",col)
    axes[0][0].legend(frameon=False,fontsize=7); fig.tight_layout(); fig.savefig(pdf,dpi=300); plt.close(fig)
def rc_fig(pdf,methods,rows,labels,colors):
    fig,ax=plt.subplots(figsize=(5,3.5))
    for tag in rows:
        y=labels[tag]
        for m,col in colors.items():
            log=methods[tag][m]
            risk_cov(ax,epi_uncert(log),(softmax_probs(log).argmax(-1)!=y).float(),f"{tag}-{m}",col)
    ax.set(xlabel="coverage",ylabel="selective risk",title="Risk–coverage (epistemic)")
    ax.legend(frameon=False,fontsize=7); fig.tight_layout(); fig.savefig(pdf,dpi=300); plt.close(fig)

def epi_hist(pdf,method_dict,colors):
    fig,axes=plt.subplots(1,len(method_dict),figsize=(3.5*len(method_dict),3))
    if len(method_dict)==1: axes=[axes]
    for (m,log),ax in zip(method_dict.items(),axes):
        ax.hist(epi_uncert(log).cpu(),bins=30,density=True,color=colors[m],alpha=.6)
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

# ───────────────────── fbts trials util ─────────────────────────
def fbts_trials(sense, calib_ds, Xt, yt, Xv, yv, trials):
    best=None
    for t in range(trials):
        lin,sc=train_temp_scaler(calib_ds,None,mode="feat",lr=1e-3,epochs=300)
        sense.lin_model,sense.scaler=lin,sc
        m_tr=all_metrics(sense.inf(Xt,True,"feat"),yt)
        key=(m_tr["AURC_p"],m_tr["AURC_tot"])
        if best is None or key<best[0]:
            m_va=all_metrics(sense.inf(Xv,True,"feat"),yv)
            best=(key,copy.deepcopy(lin),copy.deepcopy(sc),m_tr,m_va)
    return best[1],best[2],best[3],best[4]

# ───────────────────── main ─────────────────────────────────────
if __name__=="__main__":
    P=argparse.ArgumentParser()
    P.add_argument("--model-raw",default="./saved_models/sense_model.pkl")
    P.add_argument("--out",default="./ts_eval_tpch")
    P.add_argument("--trials",type=int,default=5)
    args=P.parse_args(); out=pathlib.Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)

    with open(args.model_raw,"rb") as f: sense=pickle.load(f)

    Xt_np,yt=load_split("train"); Xv_np,yv=load_split("val"); Xc_np,yc=load_split("calib_over")
    raw_t=sense.gen_logits(Xt_np); raw_v=sense.gen_logits(Xv_np); raw_c=sense.gen_logits(Xc_np)

    # ------- probability calibration -------
    Tg=learn_Tg(to_tensor(raw_c).mean(1),yc); Tv=learn_Tv(to_tensor(raw_c),yc)
    prob={"iid":{"none":raw_t,"global":apply_prob_T(raw_t,Tg),"ensemble":apply_prob_T(raw_t,None,Tv)},
          "drift":{"none":raw_v,"global":apply_prob_T(raw_v,Tg),"ensemble":apply_prob_T(raw_v,None,Tv)}}
    labels={"iid":yt,"drift":yv}
    diag_grid(out/"tpch_diagnostics_prob_calib.pdf",prob,["iid","drift"],labels,COL_PROB)
    rc_fig(out/"risk_coverage_prob_calib.pdf",prob,["iid","drift"],labels,COL_PROB)
    epi_hist(out/"epi_spread_prob_calib.pdf",prob["drift"],COL_PROB)

    # write metrics
    rows=[]
    for sp in ["iid","drift"]:
        for m in prob[sp]:
            rows.append({"split":sp,"method":m}|all_metrics(prob[sp][m],labels[sp]))
    pd.DataFrame(rows).set_index(["split","method"]).to_csv(out/"metrics_prob_calib.csv")

    # ------- uncertainty calibration -------
    calib_ds=custom_torch_dataset(Xc_np,raw_c,yc.cpu().numpy())
    def fit(mode,trials):
        best=None
        for _ in range(trials):
            lin,sc=train_temp_scaler(calib_ds,None,mode=mode,lr=1e-3,epochs=300)
            sense.lin_model,sense.scaler=lin,sc
            key= all_metrics(sense.inf(Xc_np,True,mode),yc)["AURC_p"]
            if best is None or key<best[0]:
                best=(key,copy.deepcopy(lin),copy.deepcopy(sc))
        return best[1],best[2]

    lin_g,sc_g=fit("global",2)
    sense.lin_model=lin_g
    sense.scaler=sc_g
    print("Trained global scaler")
    with open("./saved_models/sense_model_global.pkl",'wb') as output:
        pickle.dump(sense,output)

    lin_e,sc_e=fit("ensemble",2)
    sense.lin_model=lin_e
    sense.scaler=sc_e
    print("Trained ensemble scaler")
    with open("./saved_models/sense_model_ens.pkl",'wb') as output:
        pickle.dump(sense,output)
    

    lin_f,sc_f, m_tr_fb, m_va_fb = fbts_trials(sense,calib_ds,Xt_np,yt,Xv_np,yv,args.trials)

    sense.lin_model=lin_f
    sense.scaler=sc_f

    print("Trained fbts scaler")
    with open("./saved_models/sense_model_fbts.pkl",'wb') as output:
        pickle.dump(sense,output)
    

    def build_unc(X):
        d={"none":sense.gen_logits(X)}        # raw logits (sense unchanged)
        for mode,(lin,sc) in {"global":(lin_g,sc_g),
                              "ensemble":(lin_e,sc_e),
                              "feat":(lin_f,sc_f)}.items():
            sense.lin_model,sense.scaler=lin,sc
            d[mode]=sense.inf(X,True,mode)
        return d
    unc={"iid":build_unc(Xt_np),"drift":build_unc(Xv_np)}
    diag_grid(out/"tpch_diagnostics_uncertainty_calib.pdf",unc,["iid","drift"],labels,COL_UNC)
    rc_fig(out/"risk_coverage_uncertainty_calib.pdf",unc,["iid","drift"],labels,COL_UNC)
    epi_hist(out/"epi_spread_uncertainty_calib.pdf",unc["drift"],COL_UNC)
    fbts_hist(out/"fbts_temp_distribution.pdf",Xv_np,lin_f,len(Tv))

    rows=[]
    for sp in ["iid","drift"]:
        for m in unc[sp]:
            rows.append({"split":sp,"method":m}|all_metrics(unc[sp][m],labels[sp]))
    pd.DataFrame(rows).set_index(["split","method"]).to_csv(out/"metrics_uncertainty_calib.csv")

    print("✓ all TPCH outputs saved to",out)
