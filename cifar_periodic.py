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

import numpy as np
from utils import torch_dataset,run_inference_dataset
import torch
from cifar10.train_image_classifier import model_inf,calc_loss
import pandas as pd
from sensitivity.gen_train_data import build_sensitivity_training_set_v2
from arrival.gen_time import add_timestamps_simple
import torchvision.transforms as transforms
import numpy as np
from scipy.stats import rankdata, norm

from cifar10.models.resnet import ResNet18
import torch.backends.cudnn as cudnn
from utils import image_dataset
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
from cifar10.train_image_classifier import train_img_model
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from prophet.serialize import model_to_json, model_from_json
import sys
from sklearn.metrics import mean_absolute_error
import math
from scipy.stats import entropy as entropy2
from sklearn.utils import resample
from sklearn.metrics import f1_score,precision_score,recall_score
import time
# n = len(sys.argv)
# print("Total arguments passed:", n)

# # Arguments passed
# print("\nName of Python script:", sys.argv[0])

# print("ARGUMENT 1 :",sys.argv[1])
# #Temporary imports 


class label_gen:
    def __init__(self,losses) -> None:
        self.loss_avg=losses.mean()
        self.loss_std=losses.std()

    def get_loss_label(self,loss):
        
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
        if loss>self.loss_avg+2*self.loss_std:
            loss_label=1
        else:
            loss_label=0
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
    # train_x_majority = train_x[calib_y == 0]
    # train_x_minority = train_x[calib_y == 1]
    # train_x_minority_oversampled, y_minority_oversampled = resample(
    #     train_x_minority, calib_y[calib_y == 1], replace=True, n_samples=len(train_x_majority), random_state=42
    # )
    # X_balanced = np.concatenate([train_x_majority, train_x_minority_oversampled])
    # y_balanced = np.hstack([calib_y[calib_y == 0], y_minority_oversampled])

    qrf = RandomForestQuantileRegressor(n_estimators=500)
    qrf.fit(train_x,calib_y)
    # qrf.fit(X_balanced,y_balanced)
    print(qrf)

    return qrf


def expected_loss(sense_model,train_x,train_y,qrf):
    pred_logits=sense_model.inf(train_x,apply_temp=True)
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


def jensen_shannon_divergence(array1, array2, bins=50):
    """
    Compute the Jensen-Shannon divergence between two distributions.
    
    Parameters:
        array1 (np.ndarray): The first array of continuous values.
        array2 (np.ndarray): The second array of continuous values.
        bins (int): Number of bins to discretize the distributions.

    Returns:
        float: Jensen-Shannon divergence.
    """
    # Compute histograms for both arrays
    hist1, bin_edges = np.histogram(array1, bins=bins, density=True)
    hist2, _ = np.histogram(array2, bins=bin_edges, density=True)
    
    # Add a small value to avoid log(0) issues
    hist1 += 1e-10
    hist2 += 1e-10

    # Compute the midpoint distribution
    midpoint = 0.5 * (hist1 + hist2)

    # Calculate Jensen-Shannon divergence
    js_divergence = 0.5 * (entropy2(hist1, midpoint) + entropy2(hist2, midpoint))
    return js_divergence

def get_base_distribution(sense_model,train_x,train_y,reg_x,reg_y,reg,window_size=1000):
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

    pred_logits=sense_model.inf(reg_x,apply_temp=True)
    probs=F.softmax(pred_logits,dim=-1)
    reg_model_uncer=model_uncertainty(probs).detach().numpy()
    reg_model_uncer[reg_model_uncer<0] = 0
    mean_probs=torch.mean(probs,dim=1)
    reg_uncer=entropy(mean_probs).detach().numpy()

    full_dist=distribution(label_stat,epis_stat,uncer_stat)
    full_dist.quantile=CenteredQuantileTransformer(reg_model_uncer)
    test_quants=[0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
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

    print("BEST QUANT",full_dist.best_quant)
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
    print("BEST MIDDLE QUANT",full_dist.best_mid_quant)

    tests=1000

    dist_avg=[]
    for t in range(tests):
        idx=np.random.choice(np.arange(len(df)),100)
        quant_sel=df.loc[idx]['epis'].values
        dist_avg.append(jensen_shannon_divergence(df['epis'].values,quant_sel))
        avg_quant=quant_sel.mean()
    dist_avg=np.array(dist_avg)
    full_dist.test_epis_arr=df['epis'].values
    full_dist.test_dist_arr=dist_avg

    # np.save('./census/data/base_epis.npy',model_uncer)
    # np.save('./census/data/reg_cifar_epis.npy',reg_model_uncer)
    # np.save('./census/data/base_cifar_epis.npy',model_uncer)
    # np.save('./census/data/quantile_cifar_test.npy',y_pred_test)
    # np.save('./census/data/quantile_cifar_target.npy',train_y)
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

def run_sim(imgs,data,alpha_model,alpha_model_inf,sense_model,labeler,delta=8*60*60,reg=None,distro=None,avg_train_loss=-1,sc=None,retraining_period=None):
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    vit.to(device)
    vit.eval()
    embedding_transform = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
    
    uncer_counter=0
    count=0
    predicted_loss=0
    predicted_lower=0
    predicted_upper=0
    true_loss=0
    time_retraining=0

    new_data=0
    last_match=-1

    pinball_loss_low=0
    pinball_loss_high=0

    ts=[]
    np_data=[]
    old_near_ts=-1

    # store_columns=future_data.columns.values

    columns_to_drop=['y','timestamp','loss','loss_label','sense_pred_low','sense_pred','sense_pred_high','uncer','future_loss','prophet_count']

    store_columns=np.hstack([future_data.columns.values,['model_pred','loss','loss_label','sense_pred_low','sense_pred','sense_pred_high','uncer','future_loss','prophet_count']])

    dist_columns=np.hstack([future_data.columns.values,['model_pred','loss','loss_label','sense_pred','quant_pred','label_dist','epis_dist','uncer_dist']])

    dist_data=pd.DataFrame(columns=dist_columns)
    storage_data=pd.DataFrame(columns=store_columns)
    data_df=pd.DataFrame(columns=['time','total'])

    np_data=[]

    dist_df_array=[]
    new_data_idx=0

    sense_data_idx=0

    retrain_data_idx=0

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

    if int(sc)==1:
        random_start=40000
    loss_switch=0
    quant_wanted=0.5
    quant_new=0.5
    quant_window=[]
    max_loss=-1
    total_arrivals=-1
    train_loss_sum=0
    last_retraining=0
    retrain_time=-1
    retrain_ts=-1
    new_model=False
    time_spent_retraining=0
    retrain_path='./cifar10/checkpoint/retrain_ckpt_periodic_long_v1_'+str(int(delta))+'_'+str(sc)+'.pth'

    # with open('./cifar10/saved_models/prophet.json', 'r') as fin:
    #     prophet_model = model_from_json(fin.read())  # Load model

    # retraining_period=12*60*60
    if distro is not None:
        x_grid = np.linspace(0, 2, 10000)
        pdf_values = distro.epis.stat(x_grid)
        cdf_values = np.cumsum(pdf_values)
        cdf_values /= cdf_values[-1]  # Normalize to get a proper CDF

        # Create an interpolator to find the quantile for any value n
        cdf_interp = interp1d(x_grid, cdf_values)


    for index,row in data[random_start:].iterrows():
        if old_near_ts==-1:
            old_near_ts=row['timestamp']
        if retrain_time==-1:
            retrain_time=row['timestamp']
            retrain_ts=row['timestamp']
        ts.append(row['timestamp'])
        arr=row.values
        in_y=row['y']
        in_ts=row['timestamp']

        if new_model and (in_ts-retrain_ts)>3600:
            new_model=False
            labeler=new_labeler
            avg_train_loss=new_avg_train_loss
            alpha_model.load_state_dict(state['net'])
            alpha_model.eval()

        in_x=transform_test(imgs[index][0])

        alpha_model_logits=alpha_model_inf(alpha_model,in_x)
        _,alpha_model_pred=torch.max(alpha_model_logits,1)

        
        alpha_loss=calc_loss(alpha_model_logits,in_y).item()

        loss_label=labeler.get_loss_label(alpha_loss)


        
        true_loss+=loss_label


        
        #['model_pred','loss','loss_label','sense_pred_low','sense_pred_median','sense_pred_high']
        new_entry=np.hstack([arr,[alpha_model_pred.cpu().detach().numpy()[0],alpha_loss,loss_label,-1,-1,-1,-1,-1,-1]])
        np_data.append(new_entry)


        near_ts=find_nearest(ts,row['timestamp']-delta)
        print(len(storage_data))
        if len(data_needed)>0:
            print("DEBUG :",data_needed[0],random_start+len(dist_data))

        if ts[near_ts]-old_near_ts>600:
            old_near_ts=ts[near_ts]
            print("ADDING NEW DATA")
            # print(len(np_data[:near_ts]))
            storage_data=pd.DataFrame(np_data[:near_ts], columns=store_columns)
            


        if len(storage_data)>0:
            print("Retraining Threshold :",storage_data['loss_label'].sum()-avg_train_loss*len(storage_data['loss_label'][last_retraining:]))
            if (in_ts-retrain_time)>retraining_period:
                retrain_time=in_ts

                retrain_ts=in_ts
                new_model=True
                train_loss_sum=0
                last_retraining=len(storage_data)
                transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

                transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                if old_train_x is None:
                    old_train_x=np.load("./cifar10/data/train_x.npy")
                    old_train_y=np.load("./cifar10/data/train_y.npy")
                

                
                new_train_x=imgs.data[random_start:random_start+len(storage_data)]

                new_train_y=storage_data['y'].values

                # filler_data_size=len(old_train_x)-len(new_train_x)

                # filler_idx=np.random.choice(np.arange(len(old_train_x)),filler_data_size,replace=False)

                # old_data_x_slice=old_train_x[filler_idx]
                # old_data_y_slice=old_train_y[filler_idx]

                train_x=np.concatenate([old_train_x,new_train_x],axis=0)
                train_y=np.concatenate([old_train_y,new_train_y],axis=0)

                # all_data_x=np.concatenate([old_train_x,imgs.data[:sense_data_idx]],axis=0)
                # all_data_y=np.concatenate([old_train_y,dist_data[:sense_data_idx]['y'].values],axis=0)

                retrain_x,retest_x,retrain_y,retest_y=train_test_split(train_x, train_y, test_size=0.2, random_state=42, shuffle=True)

                trainset = image_dataset(retrain_x,retrain_y,transform=transform_train)
                testset = image_dataset(retest_x,retest_y,transform=transform_test)

                evalset=image_dataset(train_x,train_y,transform=transform_test)
                start=time.time()
                net=train_img_model(trainset,testset,epochs=100,retrain=True,retrain_path=retrain_path)
                net.eval()
                time_spent_retraining=time.time()-start
                state=torch.load(retrain_path)
                # alpha_model.load_state_dict(state['net'])
                # alpha_model.eval()
                net.load_state_dict(state['net'])
                net.eval()

                train_pred,train_loss,_=run_inference_dataset(net,model_inf,evalset,calc_loss,neural=True)
                temp_label=[]
                new_labeler=label_gen(train_loss)
                for p in train_loss:
                    lab=new_labeler.get_loss_label(p)
                    temp_label.append(lab)
                temp_label=np.array(temp_label)
                new_avg_train_loss=temp_label.mean()
                
                retraining_timestamps.append([in_ts,storage_data[-1:]['timestamp'].values[0],avg_train_loss,time_spent_retraining])
                np.save('./cifar10/data/retraining_ts_periodic_long_v1_'+str(retraining_period)+'_'+str(sc)+'_'+str(delta)+'.npy',retraining_timestamps)




        print("Index : {}  True : {}   Predicted : {}  Lower Bound : {}  Upper Bound : {}  Baseline :{}".format(index,true_loss,predicted_loss,predicted_lower,predicted_upper,train_loss_sum))
        
    storage_data = pd.DataFrame(np_data, columns=store_columns)
    # np.save('./cifar10/data/cifar_periodic_v12_'+str(retraining_period)+'_'+str(sc)+'_'+str(delta)+'.npy',np_features)
    storage_data.to_csv('./cifar10/data/results_test_cifar_periodic_long_v1_'+str(retraining_period)+'_'+str(sc)+'_'+str(delta)+'.csv')



delta=float(sys.argv[1])
sc=sys.argv[2]
retrain_period=float(sys.argv[3])

future_data=pd.read_csv("./cifar10/data/future_long_scenario_"+str(sc)+".csv",index_col=0)


device='cuda'

net=ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

state=torch.load("./cifar10/checkpoint/cifar_image_classifier.pth")

net.load_state_dict(state['net'])

train_x=np.load("./cifar10/data/sense_cifar_train_x.npy")
train_y=np.load("./cifar10/data/sense_cifar_train_y.npy")
train_loss=np.load("./cifar10/data/cifar_train_loss.npy")

labeler=label_gen(train_loss)

with open("./cifar10/saved_models/sense_model_fbts.pkl",'rb') as inp:
    sense_model=pickle.load(inp)

calib_x=np.load("./cifar10/data/calib_un_x.npy")
calib_y=np.load("./cifar10/data/calib_un_y.npy")

reg=dummy_baseline(sense_model,calib_x,calib_y)
imgs=np.load("./cifar10/data/future_long_scenario_"+str(sc)+"_imgs"+".npy")
img_ds=image_dataset(imgs,np.array([1]*len(imgs)))

if len(imgs)==len(future_data):
    print("Images and future data lengths match")
# corr,uncorr=get_baseline(train_x,train_y,train_loss,sense_model)


full_dist=get_base_distribution(sense_model,train_x,train_y,calib_x,calib_y,reg)

print(full_dist.label.avg)

run_sim(img_ds,future_data,net,model_inf,sense_model,labeler,delta=delta*60*60,reg=reg,distro=full_dist,avg_train_loss=train_y.mean(),sc=sc,retraining_period=retrain_period*60*60)

