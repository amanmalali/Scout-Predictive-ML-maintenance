import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn.functional as F
from sensitivity.uncertainty import renyi_entropy,model_uncertainty,entropy
from sensitivity.uncertainty import entropy_shannon, model_uncertainty_shannon, expected_entropy_shannon
from arrival.gen_time import find_nearest
from sklearn.linear_model import LinearRegression
from mapie.regression import MapieRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import mean_pinball_loss
from sensitivity.sense_model import sensitivity_model
from metric_utils import compute_uce_from_logits

from sensitivity.new_temp_scaler import train_temp_scaler
from utils import custom_torch_dataset

from census.train_basic_classifier import train_model,test_model
import numpy as np
from utils import torch_dataset,run_inference_dataset
import torch
from census.gen_data import generate_data_drift
from census.train_basic_classifier import model_inf,calc_loss
import pandas as pd
from sensitivity.gen_train_data import build_sensitivity_training_set
from arrival.gen_time import add_timestamps_simple

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
import time
import copy
#Temporary imports 

#GLOBAL Def
ent_mode="collision"



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
            epochs=400,
        )
        tmp = copy.deepcopy(sense)
        tmp.lin_model, tmp.scaler = lin, sc
        y_train=torch.tensor(y_train)
        logits_tr = tmp.inf(X_train, apply_temp=True, mode=mode)
        uce_tr    = compute_uce_from_logits(logits_tr, y_train)

        if best is None or uce_tr < best[0]:
            best = (uce_tr, lin, sc)

    return best[1], best[2]  

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







def dummy_baseline_new(sense_model,scaled_labeler,calib_x,calib_y,calib_loss):

    pred_logits=sense_model.inf(calib_x,apply_temp=True)
    probs=F.softmax(pred_logits,dim=-1)
    


    if ent_mode=="shannon":
        model_uncer=model_uncertainty_shannon(probs).detach().numpy()
        model_uncer[model_uncer<0] = 0
        mean_probs=torch.mean(probs,dim=1)
        uncer=entropy_shannon(mean_probs).detach().numpy()

    if ent_mode=="collision":
        model_uncer=model_uncertainty(probs).detach().numpy()
        model_uncer[model_uncer<0] = 0
        mean_probs=torch.mean(probs,dim=1)
        uncer=entropy(mean_probs).detach().numpy()

    qr_prob = mean_probs[:, 1].detach().cpu().numpy()
    confidences, predictions = torch.max(mean_probs, 1)
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


def shuffle_dataframe(df, random_state=None):
    """
    Returns a shuffled copy of the input DataFrame without modifying the original.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - random_state (int, optional): Seed for reproducibility.

    Returns:
    - pd.DataFrame: A new, shuffled DataFrame.
    """
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def get_base_distribution(sense_model,train_x,train_y,reg_x,reg_y,reg,window_size=100,train_len=None):
    tests=100
    pred_logits=sense_model.inf(train_x,apply_temp=True)
    probs=F.softmax(pred_logits,dim=-1)



    if ent_mode=="shannon":
        model_uncer=model_uncertainty_shannon(probs).detach().numpy()
        model_uncer[model_uncer<0] = 0
        mean_probs=torch.mean(probs,dim=1)
        uncer=entropy_shannon(mean_probs).detach().numpy()

    if ent_mode=="collision":
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
    pred_logits=sense_model.inf(reg_x,apply_temp=True)
    probs=F.softmax(pred_logits,dim=-1)
    reg_model_uncer=model_uncertainty(probs).detach().numpy()
    reg_model_uncer[reg_model_uncer<0] = 0
    mean_probs=torch.mean(probs,dim=1)
    reg_uncer=entropy(mean_probs).detach().numpy()

    full_dist=distribution(label_stat,epis_stat,uncer_stat)
    full_dist.quantile=CenteredQuantileTransformer(reg_model_uncer)

    full_dist.quantile2=CenteredQuantileTransformer(reg_uncer)
    
    test_quants=[0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
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

    num_trials=100

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
    full_dist.min_votes=df.loc[(df['y']<1)&(df['epis_quant']>full_dist.quant_thresh)]['vote'].mean()

    min_error=1000000000
    for i in range(len(test_quants)):
        mae_error=mean_absolute_error(df.loc[(df['y']==1)]['y'].values,df.loc[(df['y']==1)][test_quants[i]].values)
        mae_error2=mean_absolute_error(df.loc[(df['y']<1)]['y'].values,df.loc[(df['y']<1)][test_quants[i]].values)
        print(test_quants[i],mae_error,mae_error2,2*mae_error+mae_error2)
        if (2*mae_error+mae_error2)<=min_error:
            min_error=2*mae_error+mae_error2
            best_quant=test_quants[i]
    if best_quant>1:
        best_quant=0.99
    full_dist.best_upper_quant=best_quant
    quant_diff=round(float((full_dist.best_upper_quant-full_dist.best_quant)/3),3)
    full_dist.mid_quant_1=full_dist.best_quant+quant_diff
    full_dist.mid_quant_2=full_dist.best_quant+2*quant_diff
    print("BEST UPPER QUANT",full_dist.best_upper_quant)
    print("BEST MID QUANT",full_dist.mid_quant_1)
    print("BEST MID 2 QUANT",full_dist.mid_quant_2)


    min_error=1000000000
    for i in range(len(test_quants)):
        mae_error=mean_absolute_error(df.loc[(df['vote']<full_dist.min_votes)]['y'].values,df.loc[(df['vote']<full_dist.min_votes)][test_quants[i]].values)
        print(test_quants[i],mae_error)
        if mae_error<=min_error:
            min_error=mae_error
            best_quant=test_quants[i]
    
    full_dist.best_mid_quant=(best_quant+full_dist.best_quant)/2
    print("BEST MIDDLE QUANT",full_dist.best_mid_quant)

    window   = 100     # sliding-window length used online
    tests    = 1000    # Monte-Carlo samples for μ, σ

    # legacy_eu = model_uncer[:train_len]          # old training EU
    # new_eu    = model_uncer[train_len:]          # “deployment” EU seen in retrain

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
    #     ref_w    = None                    # uniform weight

    # --------------------------------------------------------------------------
    # Monte-Carlo estimate of JS baseline
    # --------------------------------------------------------------------------
    ref_eu   = model_uncer
    ref_w    = None
    
    dist_avg = []
    N = len(model_uncer)

    for _ in range(tests):
        start = np.random.randint(0, N - window)
        win   = model_uncer[start : start + window]      # contiguous slice
        dist_avg.append(jensen_shannon_divergence(ref_eu, win, bins=50, w_ref=ref_w))

    dist_avg = np.asarray(dist_avg)

    # Store artefacts for deployment
    full_dist.test_epis_arr = ref_eu           # reference EU vector
    full_dist.test_dist_arr = dist_avg
    full_dist.ref_w = ref_w  

    print("AVG EPIS DIST :",full_dist.test_dist_arr.mean())
    print("STD EPIS DIST :",full_dist.test_dist_arr.std())

    # np.save('./census/data/reg_epis.npy',reg_model_uncer)
    # np.save('./census/data/base_epis.npy',model_uncer)
    # np.save('./census/data/quantile_test.npy',y_pred_test)
    # np.save('./census/data/quantile_target.npy',train_y)
    return full_dist



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

def run_sim(data,alpha_model,alpha_model_inf,sense_model,labeler,delta=1*60*60,reg=None,distro=None,avg_train_loss=-1,sc=None,kappa=5000,beta=0.75,reactive=False):
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

    columns_to_drop=['y','timestamp','loss','loss_label','sense_pred_low','sense_pred','sense_pred_high','uncer','total_uncer','future_loss','prophet_count','sum_pred_loss']

    store_columns=np.hstack([future_data.columns.values,['model_pred','loss','loss_label','sense_pred_low','sense_pred','sense_pred_high','uncer','total_uncer','future_loss','prophet_count','sum_pred_loss']])

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
    loss_switch=0
    if distro is not None:
        quant_wanted=distro.best_quant#0.5
    else:
        quant_wanted=0.5
    # quant_wanted=0.5
    quant_new=0.5
    quant_window=[]
    quant_window2=[]
    max_loss=-1
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

    dyn_quant=[]
    time_spent_retraining=0

    with open('./census/saved_models/prophet.json', 'r') as fin:
        prophet_model = model_from_json(fin.read())  # Load model


    if distro is not None:
        x_grid = np.linspace(0, 1, 10000)
        pdf_values = distro.epis.stat(x_grid)
        cdf_values = np.cumsum(pdf_values)
        cdf_values /= cdf_values[-1]  # Normalize to get a proper CDF

        # Create an interpolator to find the quantile for any value n
        cdf_interp = interp1d(x_grid, cdf_values)


    for index,row in data[random_start:].iterrows():
        if old_near_ts==-1:
            old_near_ts=row['timestamp']
        print(len(storage_data))
        ts.append(row['timestamp'])
        arr=row.values
        in_y=row['y']
        in_ts=row['timestamp']
        in_x=row.drop(labels=['y','timestamp']).values
        
        alpha_model_pred=alpha_model_inf(alpha_model,in_x)
        

        sense_x=np.append(in_x,alpha_model_pred.detach().numpy())
        pred_logits=sense_model.inf(np.array([sense_x]),apply_temp=True)
        probs=F.softmax(pred_logits,dim=-1)




        if ent_mode=="shannon":
            mean_probs=torch.mean(probs,dim=1)#.detach().numpy()[0]
            total_uncer=entropy_shannon(mean_probs).detach().numpy()
            mean_probs=torch.mean(probs,dim=1).detach().numpy()[0]

            sense_pred=np.argmax(mean_probs)

            model_uncer=model_uncertainty_shannon(probs).detach().numpy()
            if model_uncer<0:
                model_uncer[0]=0.0


        elif ent_mode=="collision":
            mean_probs=torch.mean(probs,dim=1)#.detach().numpy()[0]
            total_uncer=entropy(mean_probs).detach().numpy()
            mean_probs=torch.mean(probs,dim=1).detach().numpy()[0]

            sense_pred=np.argmax(mean_probs)
            conf=mean_probs[sense_pred]
            uncer=renyi_entropy(mean_probs,2)

            model_uncer=model_uncertainty(probs).detach().numpy()
            if model_uncer<0:
                model_uncer[0]=0.0

        pred_loss=np.argmax(mean_probs)
        qr_prob=mean_probs[1]

        alpha_loss=calc_loss(alpha_model_pred,in_y).item()

        loss_label=labeler.get_loss_label(alpha_loss)
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

        if len(dyn_quant)>100:
            print("Dist : ",jensen_shannon_divergence(distro.test_epis_arr,dyn_quant[-100:]))
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
        new_entry=np.hstack([arr,[alpha_model_pred.cpu().detach().numpy()[0],alpha_loss,loss_label,conform_low,y_pred[0][3],qr_prob,model_uncer[0],total_uncer[0],max_loss,total_arrivals,predicted_loss]])
        np_data.append(new_entry)

        near_ts=find_nearest(ts,row['timestamp']-delta)
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
            future_df=generate_time_intervals(in_ts,10)
            fcst = prophet_model.predict(future_df)
            
            total_arrivals = int(fcst['yhat_upper'].sum())
            
            X=np.arange(len(data_df)).reshape(-1, 1)
            pred=rolling_window_logistic_regression(X[-window_size:],data_df['total'].to_numpy()[-window_size:],total_arrivals)
            max_loss=max(pred)
            future_train_loss=total_arrivals*avg_train_loss+train_loss_sum
            print("Predicted future loss :",max_loss)
            
        if len(retraining_timestamps)>0:
            if storage_data['timestamp'].iloc[-1]<retraining_timestamps[-1][0]:
                last_fixed=len(storage_data)

        if len(storage_data)>0 and (len(storage_data)-last_fixed)>100:
            pred_fix=storage_data['loss_label'][last_fixed:len(storage_data)].sum()-storage_data['sense_pred'][last_fixed:len(storage_data)].sum()
            last_fixed=len(storage_data)
            predicted_loss=predicted_loss+pred_fix
        print("Pred fix :",pred_fix)

        #ONLY FOR REACTIVE SCOUT
        if reactive:
            max_loss=predicted_loss
            future_train_loss=train_loss_sum

        data_threshold=int(beta*kappa)

        if (max_loss-future_train_loss)>kappa and sum(storage_data['loss_label'][sense_data_idx:])>data_threshold:
        # if retrain_loss>4000 and len(storage_data)>10000:
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
            # old_train_x=np.load("./census/data/census_train_x.npy",allow_pickle=True)
            # old_train_y=np.load("./census/data/census_train_y.npy",allow_pickle=True)
            sense_data_idx=len(storage_data)
            last_fixed=len(storage_data)


            if old_train_x is None:
                old_train_x=np.load("./census/data/census_train_x.npy")
                old_train_y=np.load("./census/data/census_train_y.npy")

            new_train_x=storage_data.drop(columns_to_drop,axis=1)
            new_train_x=new_train_x.drop(['model_pred'],axis=1)

            new_train_x=new_train_x.values

            
            new_train_y=storage_data['y'].values
            full_train_x=np.concatenate([old_train_x,new_train_x],axis=0)
            full_train_y=np.concatenate([old_train_y,new_train_y],axis=0)
            eu_train_len=len(old_train_y)
            
            # train_y=np.expand_dims(train_y, axis=1)
            print(len(old_train_y))
            print(len(new_train_y))
            print(len(full_train_y))
            np.save('./census/data/debug.npy',full_train_y)
            
            train_x, test_x, train_y, test_y = train_test_split(full_train_x, full_train_y, test_size=0.2, random_state=42)

            new_train_data=torch_dataset(train_x,train_y)
            new_test_data=torch_dataset(test_x,test_y)
            full_train_data=torch_dataset(full_train_x,full_train_y)
            deployment_data=torch_dataset(new_train_x,new_train_y)

            model_path="./census/saved_models/census_classifier_v1_paper_with_temp_with_fix_shannon_reactive_"+str(int(delta))+"_"+str(sc)+"_"+str(kappa)+"_"+str(beta)+".pt"
            start_time=time.time()
            train_model(new_train_data,new_test_data,model_path,epochs=100)
            time_taken=time.time()-start_time
            time_spent_retraining+=time_taken
            
            alpha_model=torch.load(model_path)

            hidden_x=np.load("./census/data/census_hidden_x.npy")
            hidden_y=np.load("./census/data/census_hidden_y.npy")
            hidden_data=torch_dataset(hidden_x,hidden_y)
            hidden_acc=test_model(hidden_data,alpha_model)

            train_pred,train_loss,_=run_inference_dataset(alpha_model,model_inf,full_train_data,calc_loss)

            labeler=label_gen(train_loss)
            scaled_labeler=scaled_label_gen(train_loss)

            dep_train_pred,dep_train_loss,_=run_inference_dataset(alpha_model,model_inf,deployment_data,calc_loss)
            sense_train_x,sense_train_y=build_sensitivity_training_set(full_train_x,train_pred,train_loss,full_train_y,problem_type="reg")
            dep_sense_train_x,dep_sense_train_y=build_sensitivity_training_set(new_train_x,dep_train_pred,dep_train_loss,new_train_y,problem_type="reg")
            
            dep_sense_train_x=np.array(dep_sense_train_x)
            dep_sense_train_y=np.array(dep_sense_train_y)
        #     # print(len(storage_data['model_pred'].values)==len(train_pred))

        #     # print(len(storage_data['loss_label'].values)==len(sense_train_x))

            # storage_data['model_pred']=train_pred
            # storage_data['loss_label']=sense_train_x
            # storage_data['loss']=train_loss

            
            sense_train_x=np.array(sense_train_x)
            sense_train_y=np.array(sense_train_y)
            

            sense_model=sensitivity_model()
            train_x,train_y,train_loss,calib_over_x,calib_over_y,calib_loss,calib_un_x,calib_un_y,calib_un_loss=sense_model.gen_calib_data(sense_train_x,sense_train_y,train_loss,balance=True)


            sense_model.train_sense_model(train_x,train_y)
            # xgb_sense_model=create_sense_model()
            # xgb_sense_model_np=train_sense_model(xgb_sense_model,train_x,train_y)



            train_logits=sense_model.gen_logits(train_x)
            calib_logits=sense_model.gen_logits(calib_over_x)


            train_data=custom_torch_dataset(train_x,train_logits,train_y)
            calib_data=custom_torch_dataset(calib_over_x,calib_logits,calib_over_y)

            temp_sucess=False

            while not temp_sucess:
                # sense_model.lin_model,sense_model.scaler=train_temp_scaler(calib_data,train_data)
                sense_model.lin_model,sense_model.scaler=fit(sense_model, calib_data,"feat",5,train_x,train_y)
                reg=dummy_baseline_new(sense_model,scaled_labeler,calib_un_x,calib_un_y,calib_un_loss)
                distro=get_base_distribution(sense_model,sense_train_x,sense_train_y,calib_un_x,calib_un_y,reg,train_len=eu_train_len)
                temp_sucess=True
            
            quant_wanted=distro.best_quant
            print("Trained scaler")
            with open("./census/saved_models/sense_model_retrain_v1_"+str(delta)+"_"+str(sc)+"_"+str(kappa)+"_"+str(beta)+".pkl",'wb') as output:
                pickle.dump(sense_model,output)

            data_needed=[]
            data_avail=[]
            retraining_timestamps.append([in_ts,storage_data[-1:]['timestamp'].values[0],sense_train_y.mean(),time_spent_retraining])
            avg_train_loss=train_y.mean()
            np.save('./census/data/retraining_ts_scout_v1_'+str(delta)+'_'+str(sc)+'_'+str(kappa)+'_'+str(beta)+'.npy',retraining_timestamps)
            # x=input()

        

        # if model_uncer[0]<0.1:
        #     predicted_loss+=y_pred
        #     predicted_lower+=conform_low
        #     predicted_upper+=conform_up
        # else:
        #     predicted_loss+=sense_pred
        #     predicted_lower+=sense_pred
        #     predicted_upper+=sense_pred

        # print("Input prediction {}, Input uncertainty {}".format(sense_pred,model_uncer))
        # if model_uncer>0.8:
        #     predicted_loss+=sense_pred
        #     predicted_lower+=conform_low
        #     predicted_upper+=conform_up
        #     # print("Prediction :",y_pred)
        #     # print("Lower Bound :",y_pis[:, 0, :][0][0])
        #     # print("Upper Bound :",y_pis[:, 1, :][0][0])
        # else:
        #     predicted_loss+=sense_pred
        #     predicted_lower+=sense_pred

        #     predicted_upper+=sense_pred

        print("Index : {}  True : {}   Predicted : {}  Lower Bound : {}  Upper Bound : {} ".format(index,true_loss,predicted_loss,predicted_lower,predicted_upper))
        
        # print("Correct : {} Sense pred : {}  Uncertainty : {}".format(loss_label, sense_pred, model_uncer[0]))

        # print("Correct : {} Sense pred : {}  Uncertainty : {}".format(loss_label, sense_pred, model_uncer[0]))

        # print("Lower Bound : {} Upper Bound : {}".format(y_pred[0][0],y_pred[0][2]))
        # print(y_pred[0][1])
        
        # print("True pred {} Predicted lower {} Predicted upper {}".format(true_loss, predicted_lower,predicted_upper))
    storage_data = pd.DataFrame(np_data, columns=store_columns)

    storage_data.to_csv("./census/data/scout_results_v1_"+str(delta)+"_"+str(sc)+"_"+str(kappa)+"_"+str(beta)+".csv")



delta=float(sys.argv[1])
sc=sys.argv[2]
kappa=int(sys.argv[3])
beta=float(sys.argv[4])
reactive=bool(sys.argv[5])


future_data=pd.read_csv("./census/data/future_"+str(sc)+".csv",index_col=0)
alpha_model=torch.load("./census/saved_models/census_classifier_v1.pt")

train_x=np.load("./census/data/sense_census_train_x.npy")
train_y=np.load("./census/data/sense_census_train_y.npy")
train_loss=np.load("./census/data/census_train_loss.npy")



labeler=label_gen(train_loss)
scaled_labeler=scaled_label_gen(train_loss)


scaled_train_y=[]
for i in range(len(train_loss)):
    scaled_train_y.append(scaled_labeler.get_loss_label(train_loss[i]))

scaled_train_y=np.array(scaled_train_y)


with open("./census/saved_models/sense_model_fbts.pkl",'rb') as inp:
    sense_model=pickle.load(inp)

calib_x=np.load("./census/data/calib_un_x.npy")
calib_y=np.load("./census/data/calib_un_y.npy")
calib_loss=np.load("./census/data/calib_un_loss.npy")
reg=dummy_baseline_new(sense_model,scaled_labeler,calib_x,calib_y,calib_loss)
# corr,uncorr=get_baseline(train_x,train_y,train_loss,sense_model)
full_dist=get_base_distribution(sense_model,train_x,train_y,calib_x,calib_y,reg)

print(full_dist.label.avg)
print(full_dist.train_avg_quant)
print(full_dist.best_quant)

run_sim(future_data,alpha_model,model_inf,sense_model,labeler,delta=delta*60*60,reg=reg,distro=full_dist,avg_train_loss=train_y.mean(),sc=sc,kappa=kappa,beta=beta,reactive=reactive)

