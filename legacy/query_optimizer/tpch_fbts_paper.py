import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn.functional as F
from metamodel.uncertainty import renyi_entropy,model_uncertainty,entropy
from gen_arrival_sim.gen_time import find_nearest
from sklearn.linear_model import LinearRegression
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import mean_pinball_loss
from metamodel.sense_model import sensitivity_model

from sensitivity.new_temp_scaler import train_temp_scaler, uncer_loss
from utils import custom_torch_dataset
from torchmetrics.regression import MeanAbsolutePercentageError


import numpy as np
from utils import torch_dataset,run_inference_dataset
import torch
import pandas as pd
from metamodel.gen_train_data import build_sensitivity_training_set
from gen_arrival_sim.gen_time import add_timestamps_simple

from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from prophet.serialize import model_to_json, model_from_json
import sys
from scipy.stats import rankdata, norm
from sklearn.metrics import mean_absolute_error
import math
from scipy.stats import entropy as entropy2
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from test_tpch_temp_prob import fbts_trials


import time
from udao.data.extractors import PredicateEmbeddingExtractor, QueryStructureExtractor
from udao.data.extractors.tabular_extractor import TabularFeatureExtractor
from udao.data.handler.data_handler import DataHandler
from udao.data.handler.data_processor import FeaturePipeline, create_data_processor
from udao.data.iterators.query_plan_iterator import QueryPlanIterator
from udao.data.predicate_embedders import Word2VecEmbedder, Word2VecParams
from udao.data.preprocessors.normalize_preprocessor import NormalizePreprocessor
from udao.model.embedders.graph_averager import GraphAverager
from udao.model.model import FixedEmbeddingUdaoModel, UdaoModel
from udao.model.module import LearningParams, UdaoModule
from udao.model.regressors.mlp import MLP
from udao.model.utils.losses import WMAPELoss
from udao.model.utils.schedulers import UdaoLRScheduler, setup_cosine_annealing_lr
from udao.optimization import concepts
from udao.optimization.moo.progressive_frontier import SequentialProgressiveFrontier
from udao.optimization.soo.mogd import MOGD
from udao.utils.interfaces import UdaoEmbedInput
from udao.utils.logging import logger
from train_tpch import retrain_model
from load_model import generate_dataframes
from gen_loss_tpch import gen_loss_vals
from metric_utils import compute_uce_from_logits
import copy

#Temporary imports 

def fit(sense, calib_ds, mode: str, trials: int, X_train: np.ndarray, y_train: torch.Tensor):
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
            epochs=300,
        )
        tmp = copy.deepcopy(sense)
        tmp.lin_model, tmp.scaler = lin, sc
        y_train=torch.tensor(y_train)
        logits_tr = tmp.inf(X_train, apply_temp=True, mode=mode)
        uce_tr    = compute_uce_from_logits(logits_tr, y_train)

        if best is None or uce_tr < best[0]:
            best = (uce_tr, lin, sc)

    return best[1], best[2]

def js_distance_norm(ref_eu: np.ndarray,
                     run_eu: np.ndarray,
                     *,
                     sigma_train: float,
                     w_ref=None) -> float:

    raw = jensen_shannon_divergence(ref_eu, run_eu,
                                    bins=50, w_ref=w_ref)
    sigma_run = np.std(run_eu, ddof=1)
    var_ratio = max(sigma_run, 1e-9) / sigma_train
    return raw / var_ratio

def get_iterator(df_train,df_test):
    tensor_dtypes = torch.float32
    processor_getter = create_data_processor(QueryPlanIterator, "op_enc")
    data_processor = processor_getter(
        tensor_dtypes=tensor_dtypes,
        tabular_features=FeaturePipeline(
            extractor=TabularFeatureExtractor(
                columns=["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
                + ["s1", "s2", "s3", "s4"]
                + ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"],
            ),
            preprocessors=[NormalizePreprocessor(MinMaxScaler())],
        ),
        objectives=FeaturePipeline(
            extractor=TabularFeatureExtractor(["latency", "cost"]),
        ),
        query_structure=FeaturePipeline(
            extractor=QueryStructureExtractor(positional_encoding_size=10),
            preprocessors=[
                NormalizePreprocessor(MinMaxScaler(), "graph_features"),
                NormalizePreprocessor(MinMaxScaler(), "graph_meta_features"),
            ],
        ),
        op_enc=FeaturePipeline(
            extractor=PredicateEmbeddingExtractor(
                Word2VecEmbedder(Word2VecParams(vec_size=8))
            ),
        ),
    )

    # df_train=pd.read_csv('./data/train.csv')
    # df_test=pd.read_csv('./data/test.csv')
    # df_val=pd.read_csv('./data/val.csv')
    # df_val.to_csv('./val.csv')
    print("Training length :",len(df_train))
    df=pd.read_csv('./data/full_data.csv')

    data_handler = DataHandler(
        df,
        DataHandler.Params(
            index_column="id",
            stratify_on=None,#"tid",
            dryrun=False,#True,
            data_processor=data_processor,
        ), 
    ) 


    split_iterators=data_handler.get_iterators_file(df_train,df_test,df) #CHANGE BEFORE RUNNING FULLY
    return split_iterators


class scaled_label_gen:
    def __init__(self, losses) -> None:
        self.loss_avg = losses.mean()
        self.loss_std = losses.std()

    def get_loss_label(self, loss):
        """
        Linearly scale `loss` into [0, 1] using:
          - loss <= loss_avg               → 0.0
          - loss >= loss_avg + 2*loss_std  → 1.0
          - in between: (loss - loss_avg) / (2*loss_std)
        Values are clamped to [0.0, 1.0].
        """
        lower = self.loss_avg
        upper = self.loss_avg + 2 * self.loss_std

        # Guard against zero division
        if upper == lower:
            return 0.0

        scaled = (loss - lower) / (upper - lower)
        return min(1.0, max(0.0, scaled))

class label_gen:
    def __init__(self,losses) -> None:
        self.loss_avg=losses.mean()
        self.loss_std=losses.std()

    # def get_loss_label(self,pred,label):
    #     if pred.round()!=label:
    #         return 1
    #     else:
    #         return 0
    def get_loss_label(self,loss):
        if loss>self.loss_avg+2*self.loss_std:
            loss_label=1
        else:
            loss_label=0
        return loss_label
        
        # loss_label=int((loss-self.loss_avg)/self.loss_std)
        # if loss_label<0:
        #     loss_label=0
        # if loss<self.loss_avg+1*self.loss_std:
        #     loss_label=0
        # elif self.loss_avg+1*self.loss_std<=loss<self.loss_avg+2*self.loss_std:
        #     loss_label=1
        # elif self.loss_avg+2*self.loss_std<=loss<self.loss_avg+3*self.loss_std:
        #     loss_label=2
        # elif self.loss_avg+3*self.loss_std<=loss:
        #     loss_label=3
        # return loss_label



def get_baseline(data_x,data_y,data_loss,sense_model):
    pred_logits=sense_model.inf(data_x,apply_temp=True, mode="feat")
    probs=F.softmax(pred_logits,dim=-1)
    model_uncer=model_uncertainty(probs).detach().numpy()
    mean_probs=torch.mean(probs,dim=1)
    confidences, predictions = torch.max(mean_probs, 1)
    confidences=confidences.detach().numpy()
    predictions=predictions.detach().numpy()
    mean_probs=mean_probs.detach().numpy()
    uncer=np.apply_along_axis(renyi_entropy,axis=1,arr=mean_probs,alpha=2)

    correct_uncer=model_uncer[predictions==data_y]
    incorrect_uncer=model_uncer[predictions!=data_y]

    correct_conf=confidences[predictions==data_y]
    incorrect_conf=confidences[predictions!=data_y]

    print("Avg correct confidence :",correct_conf.mean())
    print("Avg incorrect confidence :",incorrect_conf.mean())

    
    return correct_uncer.mean(),incorrect_uncer.mean()


    # pred_loss=np.argmax(mean_probs)    


def dummy_baseline_retrain(predictions,model_uncer,calib_y):

    train_x=np.hstack([np.expand_dims(predictions,1),np.expand_dims(model_uncer,1)])
    # clf = LinearRegression()
    # mapie_reg = MapieRegressor(estimator=clf,cv=3)
    # mapie_reg = mapie_reg.fit(train_x, calib_y)

    qrf = RandomForestQuantileRegressor(n_estimators=1000)
    qrf.fit(train_x,calib_y)
    print(qrf)
    return qrf

def dummy_baseline_new(sense_model,scaled_labeler,calib_x,calib_y,calib_loss):

    pred_logits=sense_model.inf(calib_x,apply_temp=True, mode="feat")
    probs=F.softmax(pred_logits,dim=-1)
    model_uncer=model_uncertainty(probs).detach().numpy()
    model_uncer[model_uncer<0] = 0
    mean_probs=torch.mean(probs,dim=1)
    uncer=entropy(mean_probs).detach().numpy()
    confidences, predictions = torch.max(mean_probs, 1)
    qr_prob = mean_probs[:, 1].detach().cpu().numpy()


    confidences=confidences.detach().numpy()
    predictions=predictions.detach().numpy()
    # model_uncer[model_uncer<0]=0
    scaled_labels=[]
    for i in range(len(calib_loss)):
        scaled_labels.append(scaled_labeler.get_loss_label(calib_loss[i]))

    scaled_labels=np.array(scaled_labels)

    train_x=np.hstack([np.expand_dims(qr_prob,1),np.expand_dims(model_uncer,1),np.expand_dims(uncer,1)])

    qrf = RandomForestQuantileRegressor(n_estimators=500,random_state=42)
    qrf.fit(train_x,scaled_labels)
    # qrf.fit(X_balanced,y_balanced)
    print(qrf)

    return qrf

def dummy_baseline(sense_model,calib_x,calib_y):

    pred_logits=sense_model.inf(calib_x,apply_temp=True, mode="feat")
    probs=F.softmax(pred_logits,dim=-1)
    model_uncer=model_uncertainty(probs).detach().numpy()
    model_uncer[model_uncer<0] = 0
    mean_probs=torch.mean(probs,dim=1)
    uncer=entropy(mean_probs).detach().numpy()
    confidences, predictions = torch.max(mean_probs, 1)

    confidences=confidences.detach().numpy()
    predictions=predictions.detach().numpy()
    # model_uncer[model_uncer<0]=0
    # np.save('./cifar10/data/qrf_epis.npy',model_uncer)
    # np.save('./cifar10/data/qrf_uncer.npy',uncer)
    # np.save('./cifar10/data/qrf_predictions.npy',predictions)
    # x=input()
    
    train_x=np.hstack([np.expand_dims(predictions,1),np.expand_dims(model_uncer,1),np.expand_dims(uncer,1)])
    # print(train_x.shape)

    # train_x_majority = train_x[calib_y == 0]
    # train_x_minority = train_x[calib_y == 1]
    # train_x_minority_oversampled, y_minority_oversampled = resample(
    #     train_x_minority, calib_y[calib_y == 1], replace=True, n_samples=len(train_x_majority), random_state=42
    # )
    # X_balanced = np.concatenate([train_x_majority, train_x_minority_oversampled])
    # y_balanced = np.hstack([calib_y[calib_y == 0], y_minority_oversampled])

    # train_x=np.hstack([np.expand_dims(predictions,1),np.expand_dims(model_uncer,1),np.expand_dims(uncer,1)])
    # clf = LinearRegression()
    # mapie_reg = MapieRegressor(estimator=clf,cv=3)
    # mapie_reg = mapie_reg.fit(train_x, calib_y)

    qrf = RandomForestQuantileRegressor(n_estimators=500,random_state=42)
    qrf.fit(train_x,calib_y)
    print(qrf)

    return qrf


def expected_loss(sense_model,train_x,train_y,qrf):
    pred_logits=sense_model.inf(train_x,apply_temp=True, mode="feat")
    probs=F.softmax(pred_logits,dim=-1)
    model_uncer=model_uncertainty(probs).detach().numpy()

    mean_probs=torch.mean(probs,dim=1)
    uncer=entropy(mean_probs).detach().numpy()
    confidences, predictions = torch.max(mean_probs, 1)

    confidences=confidences.detach().numpy()
    predictions=predictions.detach().numpy()

    feat=np.hstack([np.expand_dims(predictions,1),np.expand_dims(model_uncer,1),np.expand_dims(uncer,1)])

    y_pred = qrf.predict(feat, quantiles=[0.3, 0.5, 0.85])
    
    low_y=sum(y_pred[:,0])
    high_y=sum(y_pred[:,2])

    avg_loss=(high_y-low_y)/2
    avg_loss+=low_y

    print(avg_loss/len(train_x))

    return low_y/len(train_x),avg_loss/len(train_x),high_y/len(train_x),sum(train_y)/len(train_x)


def gen_dist_labels(labels):
    labels=np.round(labels)
    base_labels=[0.0,1.0,2.0,3.0]
    unique_vals,count=np.unique(labels,return_counts=True)
    missing=sorted(set(base_labels).difference(set(unique_vals)))
    if len(missing)>0:
        for i in missing:
            count=np.insert(count, int(i), 0)
    density=count/sum(count)
    return density

class sub_distribution:
    def __init__(self,stat,dist):
        self.stat=stat
        self.dist=dist
        self.avg=dist.mean()
        self.std=dist.std()


class distribution:
    def __init__(self,label,epis,uncer):
        self.label=label
        self.epis=epis
        self.uncer=uncer

def gen_dist_cont(X):
    kde = gaussian_kde(X)
    return kde



# def jensen_shannon_divergence(array1, array2, bins=50):
#     """
#     Compute the Jensen-Shannon divergence between two distributions.
    
#     Parameters:
#         array1 (np.ndarray): The first array of continuous values.
#         array2 (np.ndarray): The second array of continuous values.
#         bins (int): Number of bins to discretize the distributions.

#     Returns:
#         float: Jensen-Shannon divergence.
#     """
#     # Compute histograms for both arrays
#     hist1, bin_edges = np.histogram(array1, bins=bins, density=True)
#     hist2, _ = np.histogram(array2, bins=bin_edges, density=True)
    
#     # Add a small value to avoid log(0) issues
#     hist1 += 1e-10
#     hist2 += 1e-10

#     # Compute the midpoint distribution
#     midpoint = 0.5 * (hist1 + hist2)

#     # Calculate Jensen-Shannon divergence
#     js_divergence = 0.5 * (entropy2(hist1, midpoint) + entropy2(hist2, midpoint))
#     return js_divergence

def jensen_shannon_divergence(a_ref: np.ndarray,
                  a_win: np.ndarray,
                  *,
                  bins = 50,
                  w_ref = None,
                  eps: float = 1e-9):

    # reference histogram — weighted if w_ref given
    h_ref, edges = np.histogram(a_ref, bins=bins,
                                weights=w_ref, density=False)
    h_win, _     = np.histogram(a_win, bins=edges, density=False)

    h_ref = h_ref.astype(float) + eps
    h_win = h_win.astype(float) + eps
    h_ref /= h_ref.sum()
    h_win /= h_win.sum()

    m = 0.5 * (h_ref + h_win)
    return 0.5 * (entropy2(h_ref, m) + entropy2(h_win, m))


def get_base_distribution(sense_model,train_x,train_y,reg_x,reg_y,reg,window_size=100,train_len=None,retrain=False):
    tests=100
    pred_logits=sense_model.inf(train_x,apply_temp=True, mode="feat")
    probs=F.softmax(pred_logits,dim=-1)
    model_uncer=model_uncertainty(probs).detach().numpy()
    model_uncer[model_uncer<0] = 0
    mean_probs=torch.mean(probs,dim=1)
    uncer=entropy(mean_probs).detach().numpy()
    confidences, predictions = torch.max(mean_probs, 1)
    qr_prob = mean_probs[:, 1]           # shape (N,)
    qr_prob = qr_prob.detach().cpu().numpy() 
    confidences=confidences.detach().numpy()
    predictions=predictions.detach().numpy()
    # np.save('./cifar10/data/test_epis.npy',model_uncer)
    # x=input()

    reg_input=np.hstack([np.expand_dims(qr_prob,1),np.expand_dims(model_uncer,1),np.expand_dims(uncer,1)])

    # reg_input=np.hstack([np.expand_dims(predictions,1),np.expand_dims(model_uncer,1),np.expand_dims(uncer,1)])
    y_pred = reg.predict(reg_input, quantiles=[0.3, 0.5, 0.85])
    y_pred=y_pred[:,2]
    label_density=gen_dist_labels(train_y)
    uncer_density=gen_dist_cont(uncer)
    epis_density=gen_dist_cont(model_uncer)
    dist_label=[]
    dist_uncer=[]
    dist_epis=[]
    for t in range(tests):
        idx=np.random.choice(np.arange(len(train_x)),window_size,replace=False)
        pred_idx=y_pred[idx]#predictions[idx]
        uncer_idx=uncer[idx]
        epis_idx=model_uncer[idx]
        pred_idx_dist=gen_dist_labels(pred_idx)
        uncer_kde=gen_dist_cont(uncer_idx)
        epis_kde=gen_dist_cont(epis_idx)

        x_vals = np.linspace(min(model_uncer.min(), epis_idx.min()), max(model_uncer.max(), epis_idx.max()), 500)

        pdf1 = epis_density(x_vals)
        pdf2 = epis_kde(x_vals)
        js_dist_epis = jensenshannon(pdf1, pdf2)
        dist_epis.append(js_dist_epis)

        x_vals = np.linspace(min(uncer.min(), uncer_idx.min()), max(uncer.max(), uncer_idx.max()), 500)

        pdf1 = uncer_density(x_vals)
        pdf2 = uncer_kde(x_vals)
        js_dist_uncer = jensenshannon(pdf1, pdf2)
        dist_uncer.append(js_dist_uncer)

        js_dist_label=jensenshannon(label_density,pred_idx_dist)
        dist_label.append(js_dist_label)
    
    label_stat=sub_distribution(label_density,np.array(dist_label))
    uncer_stat=sub_distribution(uncer_density,np.array(dist_uncer))
    epis_stat=sub_distribution(epis_density,np.array(dist_epis))
    pred_logits=sense_model.inf(reg_x,apply_temp=True, mode="feat")
    probs=F.softmax(pred_logits,dim=-1)
    reg_model_uncer=model_uncertainty(probs).detach().numpy()
    reg_model_uncer[reg_model_uncer<0] = 0
    mean_probs=torch.mean(probs,dim=1)
    reg_uncer=entropy(mean_probs).detach().numpy()

    full_dist=distribution(label_stat,epis_stat,uncer_stat)
    full_dist.quantile=CenteredQuantileTransformer(reg_model_uncer)

    full_dist.quantile2=CenteredQuantileTransformer(reg_uncer)
    
    test_quants=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
    y_pred_test = reg.predict(reg_input, quantiles=test_quants)
    y_pred_test=np.array(y_pred_test)

    train_quant=[]
    for e in model_uncer:
        train_quant.append(full_dist.quantile.find_quantile(e))
    train_quant=np.array(train_quant)

    tests=100
    train_avg=[]
    for t in range(tests):
        idx=np.random.choice(np.arange(len(model_uncer)),100)
        quant_sel=train_quant[idx]
        avg_quant=quant_sel.mean()
        train_avg.append(quant_sel.mean())
    train_avg=np.array(train_avg)

    full_dist.train_avg_quant=train_avg.mean()

    quantile_df=pd.DataFrame(y_pred_test,columns=test_quants)
    quantile_df['target']=train_y
    quantile_sum=quantile_df.cumsum()
    min_error=1000000000
    best_quant=0
    for i in range(len(test_quants)):
        mae_error=mean_absolute_error(quantile_sum['target'].values,quantile_sum[test_quants[i]].values)
        print(test_quants[i],mae_error)
        if mae_error<=min_error:
            min_error=mae_error
            best_quant=test_quants[i]
    
    full_dist.best_quant=best_quant
    df=pd.DataFrame(y_pred_test,columns=test_quants)
    df['y']=train_y
    df['epis']=model_uncer
    q=[]
    for i in range(len(df)):
        q.append(full_dist.quantile.find_quantile(df.loc[i,['epis']].values[0]))
    df['epis_quant']=q
    df['vote'] = df[test_quants].sum(axis=1)
    full_dist.quant_thresh=df.loc[(df['y']==1)&(df['epis_quant']>=0.1)]['epis_quant'].mean()
    full_dist.min_votes=df.loc[(df['y']==0)&(df['epis_quant']>full_dist.quant_thresh)]['vote'].mean()

    min_error=1000000000
    for i in range(len(test_quants)):
        mae_error=mean_absolute_error(df.loc[(df['y']==1)&(df['epis_quant']>full_dist.quant_thresh)]['y'].values,df.loc[(df['y']==1)&(df['epis_quant']>full_dist.quant_thresh)][test_quants[i]].values)
        mae_error2=mean_absolute_error(df.loc[(df['y']==0)&(df['epis_quant']>full_dist.quant_thresh)]['y'].values,df.loc[(df['y']==0)&(df['epis_quant']>full_dist.quant_thresh)][test_quants[i]].values)
        print(test_quants[i],mae_error,mae_error2,2*mae_error+mae_error2)
        if (2*mae_error+mae_error2)<=min_error:
            min_error=2*mae_error+mae_error2
            best_quant=test_quants[i]
    if best_quant>1:
        best_quant=0.99
    full_dist.best_upper_quant=best_quant
    print("BEST UPPER QUANT",full_dist.best_upper_quant)

    min_error=1000000000
    for i in range(len(test_quants)):
        mae_error=mean_absolute_error(df.loc[(df['vote']<full_dist.min_votes)&(df['epis_quant']>full_dist.quant_thresh)]['y'].values,df.loc[(df['vote']<full_dist.min_votes)&(df['epis_quant']>full_dist.quant_thresh)][test_quants[i]].values)
        print(test_quants[i],mae_error)
        if mae_error<=min_error:
            min_error=mae_error
            best_quant=test_quants[i]
    
    full_dist.best_mid_quant=(best_quant+full_dist.best_quant)/2
    quant_diff=round(float((full_dist.best_upper_quant-full_dist.best_quant)/3),3)
    full_dist.mid_quant_1=full_dist.best_quant+quant_diff
    full_dist.mid_quant_2=full_dist.best_quant+2*quant_diff
    print("BEST UPPER QUANT",full_dist.best_upper_quant)
    print("BEST MID QUANT",full_dist.mid_quant_1)
    print("BEST MID 2 QUANT",full_dist.mid_quant_2)
    print("BEST MIDDLE QUANT",full_dist.best_mid_quant)

    window   = 100     # sliding-window length used online
    tests    = 1000    # Monte-Carlo samples for μ, σ


    # ----------- OPTIONAL balancing -------------------------------------------
    # if train_len is not None:
    #     balance = True
    #     legacy_eu = model_uncer[:train_len]          # old training EU
    #     new_eu = model_uncer[train_len:]          # “deployment” EU seen in retrain
    # else:
    #     balance = False      # flip to False if you deliberately want raw counts
    # print("BALANCE :",balance)
    # if balance and len(new_eu) > 0:
    #     # weight each *slice* equally in the reference KDE / histogram
    #     w_legacy = np.ones(len(legacy_eu))
    #     w_new    = np.full(len(new_eu), max(1,len(legacy_eu) / len(new_eu)))
    #     ref_eu   = np.concatenate([legacy_eu, new_eu])
    #     ref_w    = np.concatenate([w_legacy, w_new])
    # else:
    #     ref_eu   = model_uncer
    #     ref_w    = None                     # uniform weight

    # --------------------------------------------------------------------------
    # Monte-Carlo estimate of JS baseline
    # --------------------------------------------------------------------------
    
    ref_eu   = model_uncer
    ref_w    = None  
    
    dist_avg = []
    N = len(model_uncer)

    # for _ in range(tests):
    #     start = np.random.randint(0, N - window)
    #     win   = model_uncer[start : start + window]      # contiguous slice
    #     dist_avg.append(jensen_shannon_divergence(ref_eu, win, bins=50, w_ref=ref_w))

    if retrain:
        for _ in range(tests):
            idx = np.random.choice(N, size=window, replace=True)    # random sample (with replacement)
            win = model_uncer[idx]
            dist_avg.append(jensen_shannon_divergence(ref_eu, win, bins=50, w_ref=ref_w))
    else:
        for _ in range(tests):
            start = np.random.randint(0, N - window)
            win   = model_uncer[start : start + window]      # contiguous slice
            dist_avg.append(jensen_shannon_divergence(ref_eu, win, bins=50, w_ref=ref_w))


    dist_avg = np.asarray(dist_avg)

    # Store artefacts for deployment
    full_dist.test_epis_arr = ref_eu           # reference EU vector
    full_dist.test_dist_arr = dist_avg
    full_dist.ref_w = ref_w 

    print("Avg distance :",dist_avg.mean())

    np.save('./data/reg_epis.npy',reg_model_uncer)
    np.save('./data/base_epis.npy',model_uncer)
    np.save('./data/quantile_test.npy',y_pred_test)
    np.save('./data/dist_avg.npy',dist_avg)
    return full_dist


def get_training_dist(sense_model,train_x,train_y,qrf):

    pred_logits=sense_model.inf(train_x,apply_temp=True, mode="feat")
    probs=F.softmax(pred_logits,dim=-1)
    
    model_uncer=model_uncertainty(probs).detach().numpy()


    mean_probs=torch.mean(probs,dim=1)

    confidences, predictions = torch.max(mean_probs, 1)

    confidences=confidences.detach().numpy()
    predictions=predictions.detach().numpy()

    feat=np.hstack([np.expand_dims(predictions,1),np.expand_dims(model_uncer,1)])


def scale_dist(dist, u_min, u_max):
    # Normalize the uncertainty to be in the same scale as prediction (0 to 3)
    scaled_dist = (dist - u_min) / (u_max - u_min) * 0.35
    return scaled_dist

# def diff_dist(base_dist,dist1):
#     diff=[]
#     for i in range(len(base_dist)):
#         diff.append()

# def expected_js(sense_model,train_x,train_y,qrf):
#     pred_logits=sense_model.inf(train_x,apply_temp=True,mode="ensemble")
#     probs=F.softmax(pred_logits,dim=-1)
#     model_uncer=model_uncertainty(probs).detach().numpy()

#     mean_probs=torch.mean(probs,dim=1)

#     confidences, predictions = torch.max(mean_probs, 1)

#     confidences=confidences.detach().numpy()
#     predictions=predictions.detach().numpy()

#     feat=np.hstack([np.expand_dims(predictions,1),np.expand_dims(model_uncer,1)])

#     y_pred = qrf.predict(feat, quantiles=[0.3, 0.5, 0.9])
    
#     low_y=y_pred[:,0]
#     median_y=y_pred[:,1]
#     high_y=y_pred[:,2]
    
#     _,low_counts=np.unique(low_y,)

#     distance.jensenshannon([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0)
    
def generate_time_intervals(start_time_seconds, num_intervals):
    # Convert start_time to datetime object
    start_datetime = datetime.utcfromtimestamp(start_time_seconds)

    # Generate 1-minute intervals
    intervals = [start_datetime + timedelta(minutes=i) for i in range(num_intervals)]

    # Create a DataFrame with the intervals
    df = pd.DataFrame({'ds': intervals})

    return df

def rolling_window_logistic_regression(X, y, num_preds):
    num_points = len(X)
    predictions = np.zeros(num_points)
    confidences = np.zeros(num_points)
    
    X_window = X.reshape(-1, 1)
    y_window = y#.astype('int')

    model = LinearRegression()
    model.fit(X_window, y_window)
    
    future_values = []
    last_observation = X[-1]+1
    for _ in range(num_preds):
        next_X = np.array(last_observation).reshape(1, -1)
        prediction = model.predict(next_X)[0]
        future_values.append(prediction)
        last_observation=last_observation+1

    return future_values

class CenteredQuantileTransformer:
    def __init__(self, arr):
        """
        Initializes the transformer to center the most common values
        around the 0.5 quantile in the transformed distribution.

        Parameters:
        arr (np.ndarray): The input array with arbitrary distribution.
        """
        self.original_array = arr
        self.flat_arr = arr.ravel()
        
        # Transform to quantiles in range [0, 1]
        self.quantiles = self._to_quantiles(self.flat_arr)
        
        # Optional: Map quantiles to standard normal
        self.normalized_values = norm.ppf(self.quantiles)

    def _to_quantiles(self, arr):
        """
        Converts the array values to quantiles in the range [0, 1].

        Parameters:
        arr (np.ndarray): The input array with arbitrary distribution.

        Returns:
        np.ndarray: Quantiles corresponding to each element in arr.
        """
        # Get the ranks of the data, adjusted to be in range [1, N]
        ranks = rankdata(arr, method="average")
        
        # Scale ranks to quantiles between 0 and 1
        quantiles = ranks / (len(arr) + 1)
        
        return quantiles

    def get_transformed_distribution(self, normal_output=True):
        """
        Returns the transformed distribution centered around the 0.5 quantile.

        Parameters:
        normal_output (bool): If True, returns the data mapped to a standard normal distribution.
                              If False, returns the quantiles directly.

        Returns:
        np.ndarray: The transformed array with common values centered around the 0.5 quantile.
        """
        if normal_output:
            return self.normalized_values.reshape(self.original_array.shape)
        else:
            return self.quantiles.reshape(self.original_array.shape)

    def find_quantile(self, value):
        """
        Finds the quantile of a given input value in the transformed distribution.

        Parameters:
        value (float): The input value from the original distribution.

        Returns:
        float: The quantile of the input value in the transformed distribution.
        """
        # Calculate rank of the input value within the original array
        rank = np.sum(self.flat_arr < value) + 1
        quantile = rank / (len(self.flat_arr) + 1)
        
        return quantile


def run_sim(data,alpha_model,sense_model,labeler,delta=1*60*60,reg=None,distro=None,avg_train_loss=-1,sc=None,split_iterators=None,kappa=5000,beta=0.5,reactive=False):
    uncer_counter=0
    count=0
    predicted_loss=0
    predicted_lower=0
    predicted_upper=0
    true_loss=0

    new_data=0
    last_match=-1

    pinball_loss_low=0
    pinball_loss_high=0

    ts=[]
    np_data=[]
    old_near_ts=-1

    store_columns=future_data.columns.values

    columns_to_drop=['y','timestamp','loss','loss_label','sense_pred_low','sense_pred','sense_pred_high','uncer','future_loss','prophet_count','sum_pred_loss']

    store_columns=np.hstack([['feat_0','feat_1','timestamp'],['model_pred','y','loss','loss_label','sense_pred_low','sense_pred','sense_pred_high','uncer','future_loss','prophet_count','sum_pred_loss']])

    storage_data=pd.DataFrame(columns=store_columns)
    data_df=pd.DataFrame(columns=['time','total'])

    np_data=[]

    new_data_idx=0

    sense_data_idx=0

    retrain_loss=0
    np_features=[]

    retrain_counter=0
    retraining_timestamps=[]

    sampling=None#'tbs'
    old_train_x=None
    old_train_y=None

    old_sense_train_x=None

    window_size=100

    js_loss_dist=-1
    js_dist_epis=-1
    js_dist_uncer=-1

    loss_window=np.array([])
    uncer_window=np.array([])
    epis_window=np.array([])


    data_needed=[]
    data_avail=[]

    random_start=0
    if int(sc)==1:
        random_start=30000
    loss_switch=0
    if distro is not None:
        quant_wanted=distro.best_quant#0.5
    else:
        quant_wanted=0.5
    # quant_wanted=0.5
    quant_new=0.5
    quant_window=[]
    quant_window2=[]
    max_loss=0
    total_arrivals=-1
    train_loss_sum=0
    future_train_loss=0

    reg_data=[]
    last_fixed=0
    correct_quants=[]
    epis_quants=[]
    quant_correction=0
    pred_fix=0
    quant2_window=[]
    quant_test=0.5
    excess=[]
    retrain_idx=0

    dyn_quant=[]
    time_spent_retraining=0
    last_retrain=-1
    new_model=False
    data_threshold=5000

    with open('./saved_models/prophet_tpch_long.json', 'r') as fin:
        prophet_model = model_from_json(fin.read())  # Load model


    if distro is not None:
        x_grid = np.linspace(0, 1, 10000)
        pdf_values = distro.epis.stat(x_grid)
        cdf_values = np.cumsum(pdf_values)
        cdf_values /= cdf_values[-1]  # Normalize to get a proper CDF

        # Create an interpolator to find the quantile for any value n
        cdf_interp = interp1d(x_grid, cdf_values)

    mean_abs_percentage_error = MeanAbsolutePercentageError()
    alpha_model.eval()
    for index,row in data[random_start:].iterrows():
        if old_near_ts==-1:
            old_near_ts=row['timestamp']
        if last_retrain==-1:
            last_retrain=row['timestamp']

        print(len(storage_data))
        ts.append(row['timestamp'])
        arr=row[['feat_0','feat_1','timestamp']].values
        in_y=row['y']
        in_ts=row['timestamp']

        if new_model and (in_ts-last_retrain)>7200:
            print("NEW MODEL DEPLOYED")
            new_model=False
            alpha_model=alpha_model_new
            alpha_model.eval()
            sense_model=sense_model_new
            labeler=labeler_new
            scaled_labeler=scaled_labeler_new
            avg_train_loss=avg_train_loss_new
            distro=distro_new
            reg=reg_new
            split_iterators=split_iterators_new
            last_retrain=in_ts

            dyn_quant=[]
            quant_window=[]
            excess=[]
            loss_window=[]
            predicted_loss=0
            predicted_lower=0
            predicted_upper=0
            retrain_loss=0
            true_loss=0
            train_loss_sum=0
            quant_correction=0
            retrain_loss=0
            max_loss=0
            future_train_loss=0
            data_needed=[]
            data_avail=[]
            sense_data_idx=len(storage_data)
            last_fixed = len(storage_data)

        in_x_id=row['feat_0']

        embedding=alpha_model.embedder(split_iterators['val'][in_x_id][0].embedding_input)

        input=torch.unsqueeze(split_iterators['val'][in_x_id][0].features,dim=0)
        
        alpha_model_pred=alpha_model.regressor(embedding,input)
        alpha_loss=mean_abs_percentage_error(alpha_model_pred[0][0],split_iterators['val'][in_x_id][1][0]).item()

        loss_label=labeler.get_loss_label(alpha_loss)

        print("Alpha loss :",alpha_loss,loss_label)
        # print("Prediction :",alpha_model_pred)
        # print("Ground truth :",split_iterators['val'][in_x_id][1])
        embedding=embedding.detach().numpy()[0]
        input=input.numpy()[0]
        sense_x=np.concatenate([input,embedding])
        sense_x=np.concatenate([sense_x,np.array([alpha_model_pred.detach().numpy()[0][0]])])
        
        pred_logits=sense_model.inf(np.array([sense_x]),apply_temp=True, mode="feat")
        probs=F.softmax(pred_logits,dim=-1)
        


        mean_probs=torch.mean(probs,dim=1)#.detach().numpy()[0]
        
        total_uncer=entropy(mean_probs).detach().numpy()
        mean_probs=torch.mean(probs,dim=1).detach().numpy()[0]
        qr_prob=mean_probs[1]

        sense_pred=np.argmax(mean_probs)
        conf=mean_probs[sense_pred]
        uncer=renyi_entropy(mean_probs,2)

        model_uncer=model_uncertainty(probs).detach().numpy()
        if model_uncer<0:
            model_uncer[0]=0.0

        # pred_loss=np.argmax(mean_probs)


        
        # quant_wanted=distro.quantile.find_quantile(model_uncer[0])
        # if quant_wanted<0.5:
        #     quant_wanted=0.5
        # y_pred = reg.predict([[sense_pred,model_uncer[0],total_uncer[0]]], quantiles=[0.40, 0.5, 0.9, quant_wanted])
        


        # q=distro.quantile.find_quantile(model_uncer[0])
        # test_quants=[0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
        # votes=reg.predict([[sense_pred,model_uncer[0],total_uncer[0]]], quantiles=test_quants).sum(axis=1)
        
        # print("VOTE :",votes)
        # if q>(distro.quant_thresh):
        #     if votes<distro.min_votes:
        #         quant_wanted=distro.best_mid_quant
        #     else:
        #         quant_wanted=distro.best_upper_quant
        # else:
        #     quant_wanted=distro.best_quant
        
        # if q>(distro.quant_thresh):
        #     excess.append(q)
        # else:
        #     excess.append(distro.best_quant)
        
        # print("NEW QUANT :",quant_wanted)

        dyn_quant.append(model_uncer[0])

        # if len(dyn_quant)>100:
        #     print("Dist : ",jensen_shannon_divergence(distro.test_epis_arr,dyn_quant[-100:]))
        #     if jensen_shannon_divergence(distro.test_epis_arr,dyn_quant[-100:])>(distro.test_dist_arr.mean()+2*distro.test_dist_arr.std()) and distro.test_epis_arr.mean() <= np.array(dyn_quant[-100:]).mean():
        #         quant_wanted=distro.best_upper_quant
        #     elif jensen_shannon_divergence(distro.test_epis_arr,dyn_quant[-100:])>(distro.test_dist_arr.mean()+1*distro.test_dist_arr.std()) and distro.test_epis_arr.mean() <= np.array(dyn_quant[-100:]).mean():
        #         quant_wanted=distro.mid_quant_2
        #     elif jensen_shannon_divergence(distro.test_epis_arr,dyn_quant[-100:])>(distro.test_dist_arr.mean()+0.5*distro.test_dist_arr.std()) and distro.test_epis_arr.mean() <= np.array(dyn_quant[-100:]).mean():
        #         quant_wanted=distro.mid_quant_1
        #     else:
        #         quant_wanted=distro.best_quant
        # else:
        #     quant_wanted=distro.best_quant

        if len(dyn_quant)>100:
            print("Dist : ",jensen_shannon_divergence(distro.test_epis_arr,dyn_quant[-100:],w_ref=distro.ref_w))
            if jensen_shannon_divergence(distro.test_epis_arr,dyn_quant[-100:],w_ref=distro.ref_w)>(distro.test_dist_arr.mean()+2*distro.test_dist_arr.std()) and distro.test_epis_arr.mean() <= np.array(dyn_quant[-100:]).mean():
                quant_wanted=distro.best_upper_quant
            elif jensen_shannon_divergence(distro.test_epis_arr,dyn_quant[-100:],w_ref=distro.ref_w)>(distro.test_dist_arr.mean()+1*distro.test_dist_arr.std()) and distro.test_epis_arr.mean() <= np.array(dyn_quant[-100:]).mean():
                quant_wanted=distro.mid_quant_2
            elif jensen_shannon_divergence(distro.test_epis_arr,dyn_quant[-100:],w_ref=distro.ref_w)>(distro.test_dist_arr.mean()+0.5*distro.test_dist_arr.std()) and distro.test_epis_arr.mean() <= np.array(dyn_quant[-100:]).mean():
                quant_wanted=distro.mid_quant_1
            else:
                quant_wanted=distro.best_quant
        else:
            quant_wanted=distro.best_quant

        
        print("NEW QUANT :",quant_wanted)
        y_pred = reg.predict([[qr_prob,model_uncer[0],total_uncer[0]]], quantiles=[0.40, 0.5, 0.99, quant_wanted])
        if y_pred[0][3]==1:# or model_uncer[0]>0.5:
            data_needed.append(index)
        quant_window.append(distro.quantile.find_quantile(model_uncer[0]))
        quant_window2.append(distro.quantile2.find_quantile(total_uncer[0]))
        if len(loss_window)<window_size:
            loss_window=np.append(loss_window,[y_pred[0][3]])
            # epis_window=np.append(epis_window,model_uncer)
            # uncer_window=np.append(uncer_window,total_uncer)

        else:
            loss_window=np.delete(loss_window,0)
            # epis_window=np.delete(epis_window,0)
            # uncer_window=np.delete(uncer_window,0)
            loss_window=np.append(loss_window,[y_pred[0][2]])
            # epis_window=np.append(epis_window,model_uncer)
            # uncer_window=np.append(uncer_window,total_uncer)

        if len(loss_window)==window_size and index>0:
            loss_dist=gen_dist_labels(loss_window)
            js_loss_dist=jensenshannon(loss_dist,distro.label.stat)
            quant_new=np.array(quant_window)[-100:].mean()#+quant_correction
            
            if quant_new<=distro.train_avg_quant:
                quant_new=distro.best_quant

            elif quant_new > distro.train_avg_quant:
                quant_new = distro.best_quant + (quant_new -distro.train_avg_quant) * 4

            if quant_new>1:
                quant_new=0.99


        if len(loss_window)==window_size:
            print("UNCER quant :",np.array(quant_window2)[-100:].mean())
            print("Quant :",quant_new)#np.array(quant_window)[-10:].mean())
       

        conform_low=y_pred[0][0]
        conform_up=y_pred[0][2]

        pinball_loss_low+=mean_pinball_loss([loss_label],[y_pred[0][0]],alpha=0.40)
        pinball_loss_high+=mean_pinball_loss([loss_label],[y_pred[0][2]],alpha=0.85)

        total_pinball_loss=pinball_loss_high+pinball_loss_low

        print(total_pinball_loss)

        # y_pred,y_pis=reg.predict([[sense_pred,model_uncer[0]]],alpha=0.5)
        # conform_low=y_pis[:, 0, :][0][0]
        # conform_up=y_pis[:, 1, :][0][0]

        # if conform_low<0:
        #     conform_low=0
        train_loss_sum+=avg_train_loss
        quant_wanted=quant_new
        true_loss+=loss_label

        predicted_loss+=y_pred[0][3]
        predicted_lower+=conform_low
        predicted_upper+=conform_up

        retrain_loss+=conform_up
        #['model_pred','loss','loss_label','sense_pred_low','sense_pred_median','sense_pred_high']
        new_entry=np.hstack([arr,[alpha_model_pred.cpu().detach().numpy()[0][0],in_y,alpha_loss,loss_label,conform_low,y_pred[0][3],conform_up,model_uncer[0],max_loss,total_arrivals,predicted_loss]])
        np_data.append(new_entry)

        near_ts=find_nearest(np.array(ts),row['timestamp']-delta)
        print(len(storage_data))
        if ts[near_ts]-old_near_ts>600:
            old_near_ts=ts[near_ts]
            print("ADDING NEW DATA")
            # print(len(np_data[:near_ts]))
            storage_data=pd.DataFrame(np_data[:near_ts], columns=store_columns)
            # storage_data = pd.concat([storage_data,pd.DataFrame(np_data[:near_ts], columns=store_columns)])
            # np_data=np_data[near_ts:]
            if len(data_needed)>0:
                if data_needed[0]<random_start+len(storage_data):
                    near_data=find_nearest(np.array(data_needed),random_start+len(storage_data))
                    data_avail.extend(data_needed[:near_data])
                    data_needed=data_needed[near_data:]

        print("Data Needed : {}  Data Available : {}".format(len(data_needed),len(data_avail)))
        # if len(storage_data)>10000:
        #     if len(storage_data[new_data_idx:])>500:
        #         new_data_idx=len(storage_data)-1
        #         predictions=storage_data['sense_pred'].values
        #         uncertainty=storage_data['uncer'].values

        #         calib_y=storage_data['loss_label'].values


        #         reg=dummy_baseline_retrain(predictions,uncertainty,calib_y)
        #         print("Trained new quantile regressor")
    
        if (js_loss_dist>distro.label.avg+3*distro.label.std ) and index%100==0 and index>0:
            loss_switch=1
            # data_needed.extend([t for t in range(index-100,index)])
            retrain_loss+=100


        if (js_loss_dist<=distro.label.avg+3*distro.label.std ) and index%100==0 and index>0:
            loss_switch=0


        new_row = {'time': in_ts, 'total': predicted_loss}
        data_df.loc[len(data_df)] = new_row


        if len(data_df) > window_size and index%100==0 and index>0:
            future_df=generate_time_intervals(in_ts,120)
            fcst = prophet_model.predict(future_df)
            
            total_arrivals = int(fcst['yhat_upper'].sum())
            
            X=np.arange(len(data_df)).reshape(-1, 1)
            pred=rolling_window_logistic_regression(X[-window_size:],data_df['total'].to_numpy()[-window_size:],total_arrivals)
            max_loss=max(pred)
            future_train_loss=total_arrivals*avg_train_loss+train_loss_sum
            
        if len(retraining_timestamps)>0:
            if storage_data['timestamp'].iloc[-1]<retraining_timestamps[-1][0]:
                last_fixed=len(storage_data)
                print("Moving last fixed:",last_fixed)

            if storage_data['timestamp'].iloc[-1]<last_retrain:
                last_fixed=len(storage_data)
                print("Moving last fixed for retrain:",last_fixed)
            

        if len(storage_data)>0 and (len(storage_data)-last_fixed)>100:
            pred_fix=storage_data['loss_label'][last_fixed:].sum()-storage_data['sense_pred'][last_fixed:].sum()
            last_fixed=len(storage_data)
            predicted_loss=predicted_loss+pred_fix
        print("Pred fix :",pred_fix)

        data_threshold=int(beta*kappa)

        #ONLY FOR REACTIVE SCOUT
        if reactive:
            max_loss=predicted_loss
            future_train_loss=train_loss_sum
        

        if len(storage_data)>retrain_idx:
            if (max_loss-future_train_loss)>kappa and storage_data['loss_label'][sense_data_idx:].sum()>data_threshold:
                retrain_idx=index
                new_model=True
                last_retrain=in_ts
                sense_data_idx=len(storage_data)

                last_fixed=len(storage_data)
                print("RETRAINED WOHOOOOO")
                # storage_data = pd.DataFrame(np_data, columns=store_columns)
                # storage_data.to_csv("./data/results_pred_long_v9_full_results_with_temp_with_fix_predictive_"+str(delta)+"_"+str(sc)+"_"+str(kappa)+"_"+str(beta)+".csv")

                dyn_quant=[]
                quant_window=[]
                excess=[]
                loss_window=[]
                quant_wanted=0.5
                quant_new=0.5
                predicted_loss=0
                predicted_lower=0
                predicted_upper=0
                retrain_loss=0
                true_loss=0
                train_loss_sum=0
                quant_correction=0
                retrain_loss=0
                max_loss=0
                future_train_loss=0

                new_train_idx=storage_data['feat_0'].values
                full_data=pd.read_csv('./data/full_data.csv')
                old_train_df=pd.read_csv('./data/train.csv')
                old_train_idx=old_train_df['Unnamed: 0'].values
                eu_train_len=len(old_train_idx)
                # train_idx=new_train_idx
                train_idx=np.concatenate([old_train_idx,new_train_idx])
                
                train_idx=np.unique(train_idx)

                new_train_df=full_data.iloc[train_idx].copy(deep=True)
                # new_train_df=new_train_df.drop(['Unnamed: 0'],axis=1)
                # new_train_df=pd.concat([old_train_df,new_train_df])
                print("Retraining model!")
                start=time.time()
                new_retrain_df,new_retest_df,new_model_path,model=retrain_model(new_train_df,dir_name="./checkpoints_retrain/pred_test_v16_full_results_with_temp_with_fix_predictive_"+str(int(delta))+"_"+str(sc)+"_"+str(kappa)+"_"+str(beta))
                time_spent_retraining=time.time()-start
                # df_train=pd.read_csv('./data/train.csv')
                # df_test=pd.read_csv('./data/test.csv')

                # model=torch.load('./base_model.pt',weights_only=False)

                # print(split_iterators["train"].shape)
                print("NEW MODEL PATH :",new_model_path)
                # new_retrain_df.to_csv('./dummy_data/pred_retrain_v24_'+str(delta)+'_'+str(sc)+'.csv')

                module = UdaoModule.load_from_checkpoint(
                        new_model_path,
                        model=model,
                        objectives=["latency", "cost"],
                )

                alpha_model_new=module.model
                split_iterators_new=get_iterator(new_retrain_df,new_retest_df)

                alpha_model_new.eval()

                train_x,train_model_pred,train_y=generate_dataframes(alpha_model_new,split_iterators_new["train"])
                test_x,test_model_pred,test_y=generate_dataframes(alpha_model_new,split_iterators_new["test"])


                train_loss=gen_loss_vals(train_model_pred['latency_pred'].values,train_y['latency'].values)

                test_loss=gen_loss_vals(test_model_pred['latency_pred'].values,test_y['latency'].values)

                labeler_new=label_gen(train_loss)
                scaled_labeler_new=scaled_label_gen(train_loss)

                train_latency_preds=train_model_pred['latency_pred'].values
                train_latency_preds=np.expand_dims(train_latency_preds,axis=1)
                sense_train_x,sense_train_y=build_sensitivity_training_set(train_x,train_latency_preds,train_loss,train_y['latency'].values,problem_type='reg')
                sense_train_x=np.array(sense_train_x)
                sense_train_y=np.array(sense_train_y)

                # test_latency_preds=test_model_pred['latency_pred'].values
                # test_latency_preds=np.expand_dims(test_latency_preds,axis=1)
                # sense_test_x,sense_test_y=build_sensitivity_training_set(test_x,test_latency_preds,test_loss,test_y['latency'].values,problem_type='reg',val=True,train_losses=train_loss)
                # sense_test_x=np.array(sense_test_x)
                # sense_test_y=np.array(sense_test_y)


                sense_model_new=sensitivity_model()

                train_x,train_y,train_loss,calib_over_x,calib_over_y,calib_loss,calib_un_x,calib_un_y,calib_un_loss=sense_model_new.gen_calib_data(sense_train_x,sense_train_y,train_loss,balance=True,test_size=500)

                sense_model_new.train_sense_model(train_x,train_y)
                # # xgb_sense_model=create_sense_model()
                # # xgb_sense_model_np=train_sense_model(xgb_sense_model,train_x,train_y)

                # dep_train_df=full_data.iloc[new_train_idx].copy(deep=True)
                # dep_split_iterators_new=get_iterator(dep_train_df,dep_train_df)
                # dep_train_x,dep_train_model_pred,dep_train_y=generate_dataframes(alpha_model_new,dep_split_iterators_new["train"])
                # dep_train_loss=gen_loss_vals(dep_train_model_pred['latency_pred'].values,dep_train_y['latency'].values)
                # dep_train_latency_preds=dep_train_model_pred['latency_pred'].values
                # dep_train_latency_preds=np.expand_dims(dep_train_latency_preds,axis=1)
                # dep_sense_train_x,dep_sense_train_y=build_sensitivity_training_set(dep_train_x,dep_train_latency_preds,dep_train_loss,dep_train_y['latency'].values,problem_type='reg')



                train_logits=sense_model_new.gen_logits(train_x)
                calib_logits=sense_model_new.gen_logits(calib_over_x)


                train_data=custom_torch_dataset(train_x,train_logits,train_y)
                calib_data=custom_torch_dataset(calib_over_x,calib_logits,calib_over_y)
                temp_sucess=False

                best_lin=None
                best_scaler=None
                while not temp_sucess:
                    sense_model_new.lin_model,sense_model_new.scaler=fit(sense_model_new, calib_data,"feat",5,train_x,train_y)
                    reg_new=dummy_baseline_new(sense_model_new,scaled_labeler_new,calib_un_x,calib_un_y,calib_un_loss)
                    try:
                        distro_new=get_base_distribution(sense_model_new,sense_train_x,sense_train_y,calib_un_x,calib_un_y,reg_new,train_len=eu_train_len)
                    except Exception as e:
                        for er in range(10):
                            print("DISTRO RAN INTO AN ERROR")
                            print(e)
                        continue
                    if distro_new.best_quant>=distro_new.best_upper_quant:
                        print("DIST QUANT FAILED")
                        continue
                    temp_sucess=True
                    
                quant_wanted=distro_new.best_quant
                print("Trained scaler")
                with open("./retrained_models/sense_model_retrain_v16_full_results_with_temp_with_fix_predictive_"+str(delta)+"_"+str(sc)+"_"+str(kappa)+"_"+str(beta)+".pkl",'wb') as output:
                    pickle.dump(sense_model_new,output)

                data_needed=[]
                data_avail=[]
                retraining_timestamps.append([in_ts,storage_data[-1:]['timestamp'].values[0],sense_train_y.mean(),time_spent_retraining])
                avg_train_loss_new=sense_train_y.mean()
                np.save('./data/retraining_ts_pred_long_v16_full_results_with_temp_with_fix_predictive_'+str(delta)+'_'+str(sc)+'_'+str(kappa)+'_'+str(beta)+'.npy',retraining_timestamps)



        print("Index : {}  True : {}   Predicted : {}  Lower Bound : {}  Upper Bound : {} ".format(index,true_loss,predicted_loss,predicted_lower,predicted_upper))

    storage_data = pd.DataFrame(np_data, columns=store_columns)

    storage_data.to_csv("./data/results_pred_long_v16_full_results_with_temp_with_fix_predictive_"+str(delta)+"_"+str(sc)+"_"+str(kappa)+"_"+str(beta)+".csv")



delta=float(sys.argv[1])
sc=sys.argv[2]
kappa=int(sys.argv[3])
beta=float(sys.argv[4])
reactive=bool(sys.argv[5])


future_data=pd.read_csv("./data/future_long_"+str(sc)+".csv",index_col=0)
# alpha_model=torch.load("./census/saved_models/census_classifier_v1.pt")

model=torch.load('./base_model.pt',weights_only=False)

    # print(split_iterators["train"].shape)

module = UdaoModule.load_from_checkpoint(
        "./checkpoints/96-val_WMAPE=0.21.ckpt",
        model=model,
        objectives=["latency", "cost"],
)

alpha_model=module.model
alpha_model.eval()

train_x=np.load("./data/sense_tpch_train_x.npy")
train_y=np.load("./data/sense_tpch_train_y.npy")
train_loss=np.load("./data/train_loss.npy")

labeler=label_gen(train_loss)
scaled_labeler=scaled_label_gen(train_loss)


with open("./saved_models/sense_model_fbts.pkl",'rb') as inp:
    sense_model=pickle.load(inp)

calib_x=np.load("./data/calib_un_x_temp.npy")
calib_y=np.load("./data/calib_un_y_temp.npy")
calib_loss=np.load("./data/calib_un_loss_temp.npy")

reg=dummy_baseline_new(sense_model,scaled_labeler,calib_x,calib_y,calib_loss)

# corr,uncorr=get_baseline(train_x,train_y,train_loss,sense_model)
full_dist=get_base_distribution(sense_model,train_x,train_y,calib_x,calib_y,reg)

print(full_dist.label.avg)
print(full_dist.train_avg_quant)
print(full_dist.best_quant)

df_train=pd.read_csv('./data/train.csv')
df_test=pd.read_csv('./data/test.csv')

initial_iterator=get_iterator(df_train,df_test)

run_sim(future_data,alpha_model,sense_model,labeler,delta=delta*60*60,reg=reg,distro=full_dist,avg_train_loss=train_y.mean(),sc=sc,split_iterators=initial_iterator,kappa=kappa,beta=beta,reactive=reactive)

