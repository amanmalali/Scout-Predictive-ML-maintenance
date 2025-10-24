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
from sensitivity.sense_model import sensitivity_model
from sensitivity.temp_scaler import train_temp_scaler
from sensitivity.uncertainty import entropy, model_uncertainty, renyi_entropy
from utils import (add_chunks_periodically_separate,
                   combine_xy_with_transition, custom_torch_dataset,
                   generate_exponential_decay_array, run_inference_dataset,
                   torch_dataset, unison_shuffled_copies)

import math
from sklearn.model_selection import train_test_split
import time
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
        



def get_baseline(data_x,data_y,data_loss,sense_model):
    pred_logits=sense_model.inf(data_x)
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


def dummy_baseline(sense_model,calib_x,calib_y):

    pred_logits=sense_model.inf(calib_x,apply_temp=True)
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
    # clf = LinearRegression()
    # mapie_reg = MapieRegressor(estimator=clf,cv=3)
    # mapie_reg = mapie_reg.fit(train_x, calib_y)

    qrf = RandomForestQuantileRegressor(n_estimators=100)
    qrf.fit(train_x,calib_y)
    print(qrf)

    return qrf


def extract_decision_paths(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i >= 0 else "undefined!"
        for i in tree_.feature
    ]
    paths = []

    def recurse(node, path_conditions):
        if tree_.feature[node] >= 0:  # Not a leaf node
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], path_conditions + [(name, "<=", threshold)])
            recurse(tree_.children_right[node], path_conditions + [(name, ">", threshold)])
        else:
            if tree_.value[node][0][1] > tree_.value[node][0][0]:  # Node predicts 1
                paths.append(path_conditions)

    recurse(0, [])
    return paths

def sample_matches_path(sample, path):
    return all(sample[feature] <= threshold if condition == "<=" else sample[feature] > threshold
               for feature, condition, threshold in path)

# Function: Calculate proportions of samples matching decision paths
def calculate_path_proportions(data, paths):
    proportions = []
    for path in paths:
        matches = data.apply(lambda x: sample_matches_path(x, path), axis=1)
        proportion = np.mean(matches)
        proportions.append(proportion)
    return proportions




def ddla_build(ddla_train_x,ddla_train_y):
    cols=[]
    for f in range(ddla_train_x.shape[1]):
        cols.append('feat_'+str(f))
    # cols.append("y")
    # all_data=np.concatenate([train_data,val_data],axis=0)
    df_train=pd.DataFrame(data=ddla_train_x,columns=cols)

    proportion_label_0 = (ddla_train_y == 0).mean()
    proportion_label_1 = (ddla_train_y == 1).mean()

    print(f"Proportion of label 0: {proportion_label_0:.2f}, label 1: {proportion_label_1:.2f}")


    decision_tree_default = DecisionTreeClassifier()#, min_samples_split= 10, random_state= 42, splitter= 'random')
    decision_tree_default.fit(df_train, ddla_train_y)


    # print("Best Parameters:", grid_search.best_params_)
    # print("Best Cross-Validation Accuracy:", grid_search.best_score_)
    # x=input()

    # y_pred_dt_default = decision_tree_default.predict(df_train)
    # accuracy_dt_default = accuracy_score(ddla_train_y, y_pred_dt_default)
    # accuracy_balanced = balanced_accuracy_score(ddla_train_y, y_pred_dt_default)
    # print(f"Accuracy with Default Parameters: {accuracy_dt_default}")
    # print(f"Balanced Accuracy with Default Parameters: {accuracy_balanced}")

    # df=pd.concat([df_train,df_val])
    # y=np.concatenate([ddla_train_y,ddla_val_y])
    # y_pred_dt_default = decision_tree_default.predict(df)
    # accuracy_dt_default = accuracy_score(y, y_pred_dt_default)
    # accuracy_balanced = balanced_accuracy_score(y, y_pred_dt_default)
    # print(f"Accuracy with Default Parameters: {accuracy_dt_default}")
    # print(f"Balanced Accuracy with Default Parameters: {accuracy_balanced}")

    decision_paths = extract_decision_paths(decision_tree_default, cols)

    return decision_paths



# df_train_sac=df_train.copy()
# for i in range(0, df.shape[0], 500):
#     batch = df.iloc[i:i+500]
#     X_test = batch[cols]
#     y_test = y[i:i+500]
#     test_data=torch_dataset(X_test.values,y_test)
#     y_pred_batch,loss,_=run_inference_dataset(model,model_inf,test_data,calc_loss)
#     accuracy_batch = accuracy_score(y_test, (np.round(y_pred_batch)))
#     print(f"Batch {i//500} Accuracy before retrain: {accuracy_batch:.2f}")
    
#     # Calculate the distribution proportions of the current training set and test batch
#     # train_proportions = calculate_path_proportions(df_train, decision_paths)
#     # batch_proportions = calculate_path_proportions(X_test, decision_paths)
#     # print("Train propss :",np.array(train_proportions).sum())
#     # print("batch propss :",np.array(batch_proportions).sum())
#     # Calculate the overall distribution proportion of the current training set and test batch
#     train_matches = df_train_sac.apply(lambda x: any(sample_matches_path(x, path) for path in decision_paths), axis=1)
#     batch_matches = X_test.apply(lambda x: any(sample_matches_path(x, path) for path in decision_paths), axis=1)
    
#     train_proportion = train_matches.mean()
#     batch_proportion = batch_matches.mean()
#     print("Train prop :",train_proportion)
#     print("batch prop :",batch_proportion)
#     print("magic :",(batch_proportion-train_proportion)/train_proportion )
#     v,c=np.unique(y_test,return_counts=True)
#     print("True prop :", c[1]/500)





def run_sim(data,alpha_model,alpha_model_inf,sense_model,labeler,delta=1*60*60,reg=None,distro=None,avg_train_loss=-1,sc=None,kappa=1000,beta=0.75):
    
    #Only for DDLA
    ddla_train_x=np.load("./census/data/ddla_train_x.npy")
    ddla_train_y=np.load("./census/data/ddla_train_y.npy")

    decision_paths=ddla_build(ddla_train_x,ddla_train_y)
    cols=[]
    for f in range(ddla_train_x.shape[1]):
        cols.append('feat_'+str(f))
    # cols.append("y")
    # all_data=np.concatenate([train_data,val_data],axis=0)
    df_train=pd.DataFrame(data=ddla_train_x,columns=cols)
    train_matches = df_train.apply(lambda x: any(sample_matches_path(x, path) for path in decision_paths), axis=1)
    
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

    window_size=1000

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
    time_spent_retraining=0
    last_fixed=0
    sense_data_idx=0

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
        
        # y_pred,y_pis=reg.predict([[sense_pred,model_uncer[0]]],alpha=0.5)
        # conform_low=y_pis[:, 0, :][0][0]
        # conform_up=y_pis[:, 1, :][0][0]

        # if conform_low<0:
        #     conform_low=0
        train_loss_sum+=avg_train_loss
        
        true_loss+=loss_label

        #['model_pred','loss','loss_label','sense_pred_low','sense_pred_median','sense_pred_high']

        #if index%500==0 and index>0:
        ddla_df=pd.DataFrame(np.array(ddla_input[-1:]),columns=cols)
        print("DDLA SIZE :",len(ddla_df))
        batch_matches = ddla_df.apply(lambda x: any(sample_matches_path(x, path) for path in decision_paths), axis=1)
        train_proportion = train_matches.mean()
        batch_proportion = batch_matches.mean()
        baseline_ddla=baseline_ddla+train_proportion*1
        # high_loss_ddla=high_loss_ddla+batch_proportion*500
        score=(batch_proportion-train_proportion)/train_proportion

        # if score>0.5:
        #     high_loss_ddla=high_loss_ddla+batch_proportion*500
        # else:
        high_loss_ddla=high_loss_ddla+batch_proportion*1

        # for dn in range(index-500,int(index-500+batch_proportion*500)):
        #     data_needed.append(dn)
        
        new_entry=np.hstack([arr,[alpha_model_pred.cpu().detach().numpy()[0],alpha_loss,loss_label,high_loss_ddla,batch_proportion,-1,-1,-1,-1,high_loss_ddla]])
        np_data.append(new_entry)
        
        print("Train prop :",train_proportion)
        print("batch prop :",batch_proportion)


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

            
            if old_train_x is None:
                old_train_x=np.load("./census/data/census_train_x.npy")
                old_train_y=np.load("./census/data/census_train_y.npy")

            new_train_x=storage_data.drop(columns_to_drop,axis=1)
            new_train_x=new_train_x.drop(['model_pred'],axis=1)

            new_train_x=new_train_x.values

            
            new_train_y=storage_data['y'].values
            train_x=np.concatenate([old_train_x,new_train_x],axis=0)
            train_y=np.concatenate([old_train_y,new_train_y],axis=0)
            new_train_data=torch_dataset(train_x,train_y)

            train_pred,train_loss,_=run_inference_dataset(alpha_model,model_inf,new_train_data,calc_loss)
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
            decision_paths=ddla_build(ddla_train_x,ddla_train_y)
            cols=[]
            for f in range(ddla_train_x.shape[1]):
                cols.append('feat_'+str(f))
            # cols.append("y")
            # all_data=np.concatenate([train_data,val_data],axis=0)
            df_train=pd.DataFrame(data=ddla_train_x,columns=cols)


        if len(retraining_timestamps)>0:
            if storage_data['timestamp'].iloc[-1]<retraining_timestamps[-1][0]:
                last_fixed=len(storage_data)

        if len(storage_data)>0 and (len(storage_data)-last_fixed)>100:
            pred_fix=storage_data['loss_label'][last_fixed:len(storage_data)].sum()-storage_data['sense_pred'][last_fixed:len(storage_data)].sum()
            last_fixed=len(storage_data)
            high_loss_ddla=high_loss_ddla+pred_fix

        print("PRED FIX :",pred_fix)


        print("Data Needed : {}  Data Available : {}".format(len(data_needed),len(data_avail)))
        print("Baseline : {}  Current : {}".format(baseline_ddla,high_loss_ddla))
        data_threshold=int(beta*kappa)
        if len(storage_data)>0:
            if storage_data.iloc[[-1]]['timestamp'].values[0]>retrain_ts:
                new_close_ts=find_nearest(ts,retrain_ts)
                if (high_loss_ddla-baseline_ddla)>kappa and sum(storage_data['loss_label'][sense_data_idx:])>data_threshold:
        # if retrain_loss>4000 and len(storage_data)>10000:
                    baseline_ddla=0
                    high_loss_ddla=0
                    true_loss=0
                    data_needed=[]
                    data_avail=[]
                    train_loss_sum=0
                    retrain_ts=in_ts
                    last_retraining=len(storage_data)
                    last_fixed=len(storage_data)
                    sense_data_idx=len(storage_data)
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
                    model_path="./census/saved_models/census_classifier_v1_retrained_ddla_"+str(int(delta))+"_"+str(sc)+"_"+str(kappa)+"_"+str(beta)+".pt"
                    start_time=time.time()
                    train_model(new_train_data,new_test_data,model_path,epochs=100)
                    time_taken=time.time()-start_time
                    time_spent_retraining+=time_taken
                    
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
                    decision_paths=ddla_build(ddla_train_x,ddla_train_y)
                    cols=[]
                    for f in range(ddla_train_x.shape[1]):
                        cols.append('feat_'+str(f))
                    # cols.append("y")
                    # all_data=np.concatenate([train_data,val_data],axis=0)
                    df_train=pd.DataFrame(data=ddla_train_x,columns=cols)
                    train_matches = df_train.apply(lambda x: any(sample_matches_path(x, path) for path in decision_paths), axis=1)
                    
                    avg_train_loss=ddla_train_y.mean()
                    retraining_timestamps.append([in_ts,storage_data[-1:]['timestamp'].values[0],temp_sense_train_y.mean(),time_spent_retraining,train_loss.mean(),train_loss.std()])
                    np.save('./census/data/retraining_ts_ddla_v1_'+str(delta)+'_'+str(sc)+'_'+str(kappa)+'_'+str(beta)+'.npy',retraining_timestamps)
                    # x=input()

        


        print("Index : {}  True : {}   Predicted : {}  Lower Bound : {}  Upper Bound : {} ".format(index,true_loss,high_loss_ddla,predicted_lower,predicted_upper))
        
        # print("Correct : {} Sense pred : {}  Uncertainty : {}".format(loss_label, sense_pred, model_uncer[0]))

        # print("Correct : {} Sense pred : {}  Uncertainty : {}".format(loss_label, sense_pred, model_uncer[0]))

        # print("Lower Bound : {} Upper Bound : {}".format(y_pred[0][0],y_pred[0][2]))
        # print(y_pred[0][1])
        
        # print("True pred {} Predicted lower {} Predicted upper {}".format(true_loss, predicted_lower,predicted_upper))
    storage_data = pd.DataFrame(np_data, columns=store_columns)

    storage_data.to_csv("./census/data/ddla_results_v1_"+str(delta)+"_"+str(sc)+"_"+str(kappa)+"_"+str(beta)+".csv")


delta=float(sys.argv[1])
sc=sys.argv[2]
kappa=int(sys.argv[3])
beta=float(sys.argv[4])
print("Started running sim....",delta,sc,kappa,beta)

future_data=pd.read_csv("./census/data/future_"+str(sc)+".csv",index_col=0)
alpha_model=torch.load("./census/saved_models/census_classifier_v1.pt")

train_x=np.load("./census/data/ddla_train_x.npy")
train_y=np.load("./census/data/ddla_train_y.npy")
train_loss=np.load("./census/data/census_train_loss.npy")

labeler=label_gen(train_loss)

with open("./census/saved_models/sense_model_v2.pkl",'rb') as inp:
    sense_model=pickle.load(inp)

calib_x=np.load("./census/data/calib_un_x.npy")
calib_y=np.load("./census/data/calib_un_y.npy")

reg=None

# corr,uncorr=get_baseline(train_x,train_y,train_loss,sense_model)
full_dist=None

run_sim(future_data,alpha_model,model_inf,sense_model,labeler,delta=delta*60*60,reg=reg,distro=full_dist,avg_train_loss=train_y.mean(),sc=sc,kappa=kappa,beta=beta)
