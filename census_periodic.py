import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn.functional as F
from sensitivity.uncertainty import renyi_entropy,model_uncertainty,entropy
from arrival.gen_time import find_nearest
from sklearn.linear_model import LinearRegression
from mapie.regression import MapieRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import mean_pinball_loss
from sensitivity.sense_model import sensitivity_model

from sensitivity.temp_scaler import train_temp_scaler
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
from sklearn.model_selection import train_test_split
import time



#Temporary imports 


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

def get_base_distribution(sense_model,train_x,train_y,reg,window_size=1000):
    tests=100
    pred_logits=sense_model.inf(train_x,apply_temp=True)
    probs=F.softmax(pred_logits,dim=-1)
    model_uncer=model_uncertainty(probs).detach().numpy()
    model_uncer[model_uncer<0] = 0
    mean_probs=torch.mean(probs,dim=1)
    uncer=entropy(mean_probs).detach().numpy()
    confidences, predictions = torch.max(mean_probs, 1)

    confidences=confidences.detach().numpy()
    predictions=predictions.detach().numpy()
    # np.save('./cifar10/data/test_epis.npy',model_uncer)
    # x=input()
    reg_input=np.hstack([np.expand_dims(predictions,1),np.expand_dims(model_uncer,1),np.expand_dims(uncer,1)])
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

    full_dist=distribution(label_stat,epis_stat,uncer_stat)
    full_dist.quantile=CenteredQuantileTransformer(model_uncer)

    return full_dist

    
def get_training_dist(sense_model,train_x,train_y,qrf):

    pred_logits=sense_model.inf(train_x,apply_temp=True)
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
#     pred_logits=sense_model.inf(train_x,apply_temp=True)
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

def run_sim(data,alpha_model,alpha_model_inf,sense_model,labeler,delta=1*60*60,reg=None,distro=None,avg_train_loss=-1,sc=None,retraining_period=None):
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
        
        alpha_model_pred=alpha_model_inf(alpha_model,in_x)
        
        alpha_loss=calc_loss(alpha_model_pred,in_y).item()

        loss_label=labeler.get_loss_label(alpha_loss)
        
        # y_pred,y_pis=reg.predict([[sense_pred,model_uncer[0]]],alpha=0.5)
        # conform_low=y_pis[:, 0, :][0][0]
        # conform_up=y_pis[:, 1, :][0][0]

        # if conform_low<0:
        #     conform_low=0
        train_loss_sum+=avg_train_loss
        
        true_loss+=loss_label

        #['model_pred','loss','loss_label','sense_pred_low','sense_pred_median','sense_pred_high']
        new_entry=np.hstack([arr,[alpha_model_pred.cpu().detach().numpy()[0],alpha_loss,loss_label,-1,-1,-1,-1,-1,-1,-1]])
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


        if len(storage_data)>0:
            if (in_ts-retrain_ts)>retraining_period and len(storage_data)>1000:  # if retrain_loss>4000 and len(storage_data)>10000:
                train_loss_sum=0
                retrain_ts=in_ts
                last_retraining=len(storage_data)
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

                model_path="./census/saved_models/census_classifier_v1_retrained_periodic_"+str(retraining_period)+"_"+str(sc)+"_"+str(delta)+".pt"
                start_time=time.time()
                train_model(new_train_data,new_test_data,model_path,epochs=100)
                time_taken=time.time()-start_time
                time_spent_retraining+=time_taken
                alpha_model=torch.load(model_path)
                
                    
                train_pred,train_loss,_=run_inference_dataset(alpha_model,model_inf,new_train_data,calc_loss)

                labeler=label_gen(train_loss)
                temp_sense_train_y=[]
                for l in range(len(train_loss)):
                    temp_sense_train_y.append(labeler.get_loss_label(train_loss[l]))
                
                temp_sense_train_y=np.array(temp_sense_train_y)

            #     # print(len(storage_data['model_pred'].values)==len(train_pred))

            #     # print(len(storage_data['loss_label'].values)==len(sense_train_x))

                # storage_data['model_pred']=train_pred
                # storage_data['loss_label']=sense_train_x
                # storage_data['loss']=train_loss

                

                retraining_timestamps.append([in_ts,storage_data[-1:]['timestamp'].values[0],temp_sense_train_y.mean(),time_spent_retraining])
                np.save('./census/data/retraining_ts_periodic_v1_'+str(retraining_period)+'_'+str(sc)+'_'+str(delta)+'.npy',retraining_timestamps)
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

    storage_data.to_csv("./census/data/periodic_results_v1_"+str(retraining_period)+"_"+str(sc)+"_"+str(delta)+".csv")


delta=float(sys.argv[1])
sc=sys.argv[2]
retrain_period=float(sys.argv[3])

future_data=pd.read_csv("./census/data/future_"+str(sc)+".csv",index_col=0)
alpha_model=torch.load("./census/saved_models/census_classifier_v1.pt")

train_x=np.load("./census/data/sense_census_train_x.npy")
train_y=np.load("./census/data/sense_census_train_y.npy")
train_loss=np.load("./census/data/census_train_loss.npy")

labeler=label_gen(train_loss)

with open("./census/saved_models/sense_model_fbts.pkl",'rb') as inp:
    sense_model=pickle.load(inp)

calib_x=np.load("./census/data/calib_un_x.npy")
calib_y=np.load("./census/data/calib_un_y.npy")

reg=None

# corr,uncorr=get_baseline(train_x,train_y,train_loss,sense_model)
full_dist=None

run_sim(future_data,alpha_model,model_inf,sense_model,labeler,delta=delta*60*60,reg=reg,distro=full_dist,avg_train_loss=train_y.mean(),sc=sc,retraining_period=retrain_period*60*60)

