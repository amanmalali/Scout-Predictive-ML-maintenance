import copy
import pickle
import sys
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from imblearn.over_sampling import RandomOverSampler
from mapie.regression import MapieRegressor
from prophet.serialize import model_from_json, model_to_json
from quantile_forest import RandomForestQuantileRegressor
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde, norm, rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_pinball_loss, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

from arrival.gen_time import add_timestamps_simple, find_nearest
from census.gen_data import generate_data_drift
from census.train_basic_classifier import (calc_loss, model_inf, test_model,
                                           train_model)
from sensitivity.gen_train_data import (build_sensitivity_training_ddla,
                                        build_sensitivity_training_set)
from sensitivity.risk_advisor import riskadvisor_model
from sensitivity.temp_scaler import train_temp_scaler
from sensitivity.uncertainty import entropy_shannon, model_uncertainty_shannon, expected_entropy_shannon
from utils import (add_chunks_periodically_separate,
                   combine_xy_with_transition, custom_torch_dataset,
                   generate_exponential_decay_array, run_inference_dataset,
                   torch_dataset, unison_shuffled_copies)

import math
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_recall_curve

def best_fbeta_threshold(y_true, scores, beta: float = 1.0):
    """
    Pick the score cut-off that maximises Fβ on a validation set.

    Parameters
    ----------
    y_true : 1-D array-like of shape (n_samples,)
        Ground-truth binary labels (0/1).
    scores : 1-D array-like of shape (n_samples,)
        Continuous prediction scores or probabilities (larger ⇒ more positive).
    beta   : float, default 1.0   (β > 1 favours recall, β < 1 favours precision)

    Returns
    -------
    t_best : float
        Threshold that achieves the highest Fβ.
    f_best : float
        Value of Fβ at `t_best`.
    """
    # Precision–recall (PR) curve: len(thresholds) == len(p) - 1
    p, r, thr = precision_recall_curve(y_true, scores)

    # Fβ for every PR point (avoid divide-by-zero with tiny eps)
    fbeta = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-12)

    # Index of the maximum
    i_best = np.nanargmax(fbeta)          # nan-safe arg-max

    # precision_recall_curve returns an *extra* point; handle that edge case
    if i_best == len(thr):                # PR point with no corresponding threshold
        t_best = scores.min() - 1e-12     # any value below min(score) gives same labels
    else:
        t_best = thr[i_best]

    return float(t_best), float(fbeta[i_best])

def find_best_thresh(sense_model,calib_x,calib_y):
    pred_logits=sense_model.inf(calib_x,apply_temp=False)
    probs=F.softmax(pred_logits,dim=-1)

    mean_probs=torch.mean(probs,dim=1)#.detach().numpy()
    total_uncer=entropy_shannon(mean_probs).detach().numpy()
    mean_probs=mean_probs.detach().numpy()


    sense_pred=np.argmax(mean_probs)

    mean_probs = mean_probs[:, 1]


    risk_score=total_uncer+mean_probs

    return best_fbeta_threshold(calib_y,risk_score)



class label_gen:
    def __init__(self,losses) -> None:
        self.loss_avg=losses.mean()
        self.loss_std=losses.std()

    # def get_loss_label(self,pred,label):
    #     if pred.round()!=label:
    #         return 1
    #     else:
    #         return 0
    def get_loss_label(self,true,pred):
        if np.round(pred)==true:
            loss_label=0
        else:
            loss_label=1
        return loss_label
        







def run_sim(data,alpha_model,alpha_model_inf,sense_model,labeler,delta=1*60*60,reg=None,distro=None,avg_train_loss=-1,sc=None,kappa=1000,beta=0.75,risk_thresh=None):
    
    #Only for DDLA
    # ddla_train_x=np.load("./census/data/ddla_train_x.npy")
    # ddla_train_y=np.load("./census/data/ddla_train_y.npy")

    # decision_paths=ddla_build(ddla_train_x,ddla_train_y)
    # cols=[]
    # for f in range(ddla_train_x.shape[1]):
    #     cols.append('feat_'+str(f))
    # # cols.append("y")
    # # all_data=np.concatenate([train_data,val_data],axis=0)
    # df_train=pd.DataFrame(data=ddla_train_x,columns=cols)
    
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

    store_columns=np.hstack([future_data.columns.values,['model_pred','loss','loss_label','sense_pred_low','sense_pred','sense_pred_high','uncer','future_loss','prophet_count','sum_pred_loss']])

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
    quant_wanted=0.5
    quant_new=0.5
    quant_window=[]
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
    retrain_ts=0
    train_proportion=0
    batch_proportion=0
    baseline_ddla=0
    high_loss_ddla=0
    sense_data_idx=0

    high_loss_risk=0

    with open('./census/saved_models/prophet.json', 'r') as fin:
        prophet_model = model_from_json(fin.read())  # Load model


    if distro is not None:
        x_grid = np.linspace(0, 1, 10000)
        pdf_values = distro.epis.stat(x_grid)
        cdf_values = np.cumsum(pdf_values)
        cdf_values /= cdf_values[-1]  # Normalize to get a proper CDF

        # Create an interpolator to find the quantile for any value n
        cdf_interp = interp1d(x_grid, cdf_values)

    ddla_input=[]
    for index,row in data.iterrows():
        if old_near_ts==-1:
            old_near_ts=row['timestamp']
            retrain_ts=row['timestamp']
        print(len(storage_data))
        ts.append(row['timestamp'])
        arr=row.values
        in_y=row['y']
        in_ts=row['timestamp']
        in_x=row.drop(labels=['y','timestamp']).values
        
        ddla_input.append(in_x)

        alpha_model_pred=alpha_model_inf(alpha_model,in_x)
        
        alpha_loss=calc_loss(alpha_model_pred,in_y).item()

        loss_label=labeler.get_loss_label(in_y,alpha_model_pred.detach().numpy()[0])

        sense_x=np.append(in_x,alpha_model_pred.detach().numpy())
        pred_logits=sense_model.inf(np.array([sense_x]),apply_temp=False)
        probs=F.softmax(pred_logits,dim=-1)

        


        mean_probs=torch.mean(probs,dim=1)#.detach().numpy()[0]
        total_uncer=entropy_shannon(mean_probs).detach().numpy()
        mean_probs=torch.mean(probs,dim=1).detach().numpy()[0]

        sense_pred=np.argmax(mean_probs)

        high_loss_risk=high_loss_risk+sense_pred

        model_uncer=model_uncertainty_shannon(probs).detach().numpy()
        if model_uncer<0:
            model_uncer[0]=0.0

        
        baseline_ddla=baseline_ddla+avg_train_loss
        risk_score=total_uncer[0]+mean_probs[1]
        if risk_score>=risk_thresh:
            risk_prediction=1
        else:
            risk_prediction=0

        high_loss_ddla=high_loss_ddla+risk_prediction

        if (total_uncer[0]+mean_probs[1])>1.0:
            data_needed.append(index)
        # y_pred,y_pis=reg.predict([[sense_pred,model_uncer[0]]],alpha=0.5)
        # conform_low=y_pis[:, 0, :][0][0]
        # conform_up=y_pis[:, 1, :][0][0]

        # if conform_low<0:
        #     conform_low=0
        train_loss_sum+=avg_train_loss
        
        true_loss+=loss_label

        #['model_pred','loss','loss_label','sense_pred_low','sense_pred_median','sense_pred_high']
        new_entry=np.hstack([arr,[alpha_model_pred.cpu().detach().numpy()[0],alpha_loss,loss_label,high_loss_ddla,risk_prediction,risk_score,model_uncer[0],-1,-1,high_loss_ddla]])
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
        
        if len(retraining_timestamps)>0:
            if storage_data['timestamp'].iloc[-1]<retraining_timestamps[-1][0]:
                last_fixed=len(storage_data)

        if len(storage_data)>0 and (len(storage_data)-last_fixed)>100:
            pred_fix=storage_data['loss_label'][last_fixed:len(storage_data)].sum()-storage_data['sense_pred'][last_fixed:len(storage_data)].sum()
            last_fixed=len(storage_data)
            high_loss_ddla=high_loss_ddla+pred_fix
        print("Pred fix :",pred_fix)

        # if len(storage_data)>0:
        #     if storage_data.iloc[-1]['timestamp']<=(retrain_ts+300)                 :
        #         print("Moving last fixed",last_fixed)
        #         last_fixed=len(storage_data)

        # if len(storage_data)>0 and (len(storage_data)-last_fixed)>100:
        #     pred_fix=storage_data['loss_label'][last_fixed:].sum()-(storage_data.iloc[-1]['sense_pred_low']-storage_data.iloc[last_fixed]['sense_pred_low'])
            
        #     print("FIX VAL :",pred_fix)
        #     print("Last fixed :",last_fixed)
        #     last_fixed=len(storage_data)
        #     high_loss_ddla=high_loss_ddla+pred_fix

        print("Pred fix :",pred_fix)

        print("Data Needed : {}  Data Available : {}".format(len(data_needed),len(data_avail)))
        print("Baseline : {}  Current : {}".format(baseline_ddla,high_loss_ddla))

        data_threshold=int(beta*kappa)

        if len(storage_data)>0:
            if storage_data.iloc[[-1]]['timestamp'].values[0]>retrain_ts:
                new_close_ts=find_nearest(ts,retrain_ts)
                if (high_loss_ddla-baseline_ddla)>kappa and sum(storage_data['loss_label'][sense_data_idx:])>data_threshold:
        # if retrain_loss>4000 and len(storage_data)>10000:
                    baseline_ddla=0
                    high_loss_risk=0
                    high_loss_ddla=0
                    true_loss=0
                    data_needed=[]
                    data_avail=[]
                    train_loss_sum=0
                    retrain_ts=in_ts
                    sense_data_idx=len(storage_data)
                    last_retraining=len(storage_data)
                    last_fixed=len(storage_data)
                    # old_train_x=np.load("./census/data/census_train_x.npy",allow_pickle=True)
                    # old_train_y=np.load("./census/data/census_train_y.npy",allow_pickle=True)
                    if old_train_x is None:
                        old_train_x=np.load("./census/data/census_train_x.npy")
                        old_train_y=np.load("./census/data/census_train_y.npy")

                    new_train_x=storage_data.drop(columns_to_drop,axis=1)
                    new_train_x=new_train_x.drop(['model_pred'],axis=1)

                    new_train_x=new_train_x.values

                    
                    new_train_y=storage_data['y'].values
                    train_x=np.concatenate([old_train_x,new_train_x],axis=0)
                    train_y=np.concatenate([old_train_y,new_train_y],axis=0)
                    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
                    # train_y=np.expand_dims(train_y, axis=1)
                    print(old_train_y.shape)
                    print(new_train_y.shape)
                    print(train_y.shape)
                    new_train_data=torch_dataset(train_x,train_y)
                    new_test_data=torch_dataset(test_x,test_y)
                    model_path="./census/saved_models/census_classifier_v6_paper_with_fix_reactive_"+str(int(delta))+"_"+str(sc)+"_"+str(kappa)+"_"+str(beta)+".pt"
                    train_model(new_train_data,new_test_data,model_path,epochs=100)
                    alpha_model=torch.load(model_path)
                    
                    hidden_x=np.load("./census/data/census_hidden_x.npy")
                    hidden_y=np.load("./census/data/census_hidden_y.npy")
                    hidden_data=torch_dataset(hidden_x,hidden_y)
                    hidden_acc=test_model(hidden_data,alpha_model)
                    
                    train_pred,train_loss,_=run_inference_dataset(alpha_model,model_inf,new_train_data,calc_loss)

                    labeler=label_gen(train_loss)
                    temp_sense_train_y=[]
                    for l in range(len(train_loss)):
                        temp_sense_train_y.append(labeler.get_loss_label(train_y[l],train_pred[l]))
                    
                    temp_sense_train_y=np.array(temp_sense_train_y)
                    ddla_train_x,ddla_train_y=build_sensitivity_training_ddla(train_x,train_pred,train_loss,train_y,problem_type='class')
                #     # print(len(storage_data['model_pred'].values)==len(train_pred))

                #     # print(len(storage_data['loss_label'].values)==len(sense_train_x))

                    # storage_data['model_pred']=train_pred
                    # storage_data['loss_label']=sense_train_x
                    # storage_data['loss']=train_loss
                    ddla_train_x=np.array(ddla_train_x)
                    ddla_train_y=np.array(ddla_train_y)
                    
                    risk_train_x=np.concatenate([ddla_train_x,train_pred],axis=1)
                    risk_train_y=ddla_train_y

                    sense_model=riskadvisor_model()

                    # train_x,train_y,train_loss,calib_over_x,calib_over_y,calib_loss,calib_un_x,calib_un_y=sense_model.gen_calib_data(train_x,train_y,train_loss,balance=True,test_size=500)
                    train_x,train_y,train_loss,calib_un_x,calib_un_y,calib_un_loss=sense_model.gen_calib_data(risk_train_x,risk_train_y,train_loss,balance=True,test_size=0.1)


                    # np.save("./census/data/calib_un_x.npy",calib_un_x)
                    # np.save("./census/data/calib_un_y.npy",calib_un_y)

                    sense_model.train_sense_model(risk_train_x,risk_train_y)
                    with open("./census/saved_models/risk_model_retrain_v6_paper_with_fix_reactive_"+str(delta)+"_"+str(sc)+"_"+str(kappa)+"_"+str(beta)+".pkl",'wb') as output:
                        pickle.dump(sense_model,output)

                    risk_thresh,risk_f1_score=find_best_thresh(sense_model,calib_un_x,calib_un_y)


                    avg_train_loss=ddla_train_y.mean()


                    retraining_timestamps.append([in_ts,storage_data[-1:]['timestamp'].values[0],ddla_train_y.mean()])
                    np.save('./census/data/retraining_ts_risk_v6_paper_with_fix_reactive_'+str(delta)+'_'+str(sc)+'_'+str(kappa)+'_'+str(beta)+'.npy',retraining_timestamps)
                    # x=input()

        


        print("Index : {}  True : {}   Predicted : {}  Lower Bound : {}  Upper Bound : {} ".format(index,true_loss,high_loss_ddla,predicted_lower,predicted_upper))
        
        # print("Correct : {} Sense pred : {}  Uncertainty : {}".format(loss_label, sense_pred, model_uncer[0]))

        # print("Correct : {} Sense pred : {}  Uncertainty : {}".format(loss_label, sense_pred, model_uncer[0]))

        # print("Lower Bound : {} Upper Bound : {}".format(y_pred[0][0],y_pred[0][2]))
        # print(y_pred[0][1])
        
        # print("True pred {} Predicted lower {} Predicted upper {}".format(true_loss, predicted_lower,predicted_upper))
    storage_data = pd.DataFrame(np_data, columns=store_columns)

    storage_data.to_csv("./census/data/risk_results_v6_paper_with_fix_reactive_"+str(delta)+"_"+str(sc)+"_"+str(kappa)+"_"+str(beta)+".csv")


delta=float(sys.argv[1])
sc=sys.argv[2]
kappa=int(sys.argv[3])
beta=float(sys.argv[4])


future_data=pd.read_csv("./census/data/future_"+str(sc)+".csv",index_col=0)
alpha_model=torch.load("./census/saved_models/census_classifier_v1.pt")

train_x=np.load("./census/data/ddla_train_x.npy")
train_y=np.load("./census/data/ddla_train_y.npy")
train_loss=np.load("./census/data/census_train_loss.npy")

labeler=label_gen(train_loss)

with open("./census/saved_models/risk_advisor_model_v2.pkl",'rb') as inp:
    sense_model=pickle.load(inp)


calib_x=np.load("./census/data/calib_un_risk_x.npy")
calib_y=np.load("./census/data/calib_un_risk_y.npy")

reg=None

# corr,uncorr=get_baseline(train_x,train_y,train_loss,sense_model)
full_dist=None

risk_thresh,risk_f1_score=find_best_thresh(sense_model,calib_x,calib_y)

run_sim(future_data,alpha_model,model_inf,sense_model,labeler,delta=delta*60*60,reg=reg,distro=full_dist,avg_train_loss=train_y.mean(),sc=sc,kappa=kappa,beta=beta,risk_thresh=risk_thresh)
