import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn.functional as F
from sensitivity.uncertainty import entropy_shannon, model_uncertainty_shannon, expected_entropy_shannon
from arrival.gen_time import find_nearest
from sklearn.linear_model import LinearRegression
from mapie.regression import MapieRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import mean_pinball_loss
from sensitivity.sense_model import sensitivity_model
from sensitivity.risk_advisor import riskadvisor_model
import os


from sensitivity.temp_scaler import train_temp_scaler
from utils import custom_torch_dataset

import numpy as np
from utils import torch_dataset,run_inference_dataset
import torch
from cifar10.train_image_classifier import model_inf,calc_loss
import pandas as pd
from sensitivity.gen_train_data import build_sensitivity_training_set_v2,build_sensitivity_training_ddla
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

# n = len(sys.argv)
# print("Total arguments passed:", n)

# # Arguments passed
# print("\nName of Python script:", sys.argv[0])

# print("ARGUMENT 1 :",sys.argv[1])
# #Temporary imports 

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

def run_sim(imgs,data,alpha_model,alpha_model_inf,sense_model,labeler,delta=4*60*60,reg=None,distro=None,avg_train_loss=-1,sc=None,kappa=5000,beta=0.75,risk_thresh=None):
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    vit.to(device)
    vit.eval()
    embedding_transform = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
    
    
    predicted_lower=0
    predicted_upper=0
    
    ts=[]
    np_data=[]
    old_near_ts=-1

    # store_columns=future_data.columns.values

    columns_to_drop=['y','timestamp','loss','loss_label','sense_pred_low','sense_pred','sense_pred_high','uncer','future_loss','prophet_count']

    store_columns=np.hstack([future_data.columns.values,['model_pred','loss','loss_label','sense_pred_low','sense_pred','sense_pred_high','uncer','future_loss','prophet_count','sum_pred_loss']])

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
    random_start=0
    if int(sc)==1:
        random_start=40000
    
    print("RANDOM START :",random_start)
    predicted_loss=0
    train_loss_sum=0
    true_loss=0
    loss_switch=0
    quant_wanted=0.5
    quant_new=0.5
    quant_window=[]
    max_loss=-1
    total_arrivals=-1
    
    future_train_loss=0

    reg_data=[]
    last_fixed=0
    correct_quants=[]
    epis_quants=[]
    quant_correction=0
    pred_fix=0
    excess=[]
    dyn_quant=[]
    baseline_ddla=0
    high_loss_ddla=0

    high_loss_risk=0
    new_model=False
    retrain_ts=-1

    with open('./cifar10/saved_models/prophet_long.json', 'r') as fin:
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
            retrain_ts=row['timestamp']


        ts.append(row['timestamp'])
        arr=row.values
        in_y=row['y']
        in_ts=row['timestamp']

        if new_model and (in_ts-retrain_ts)>3600:
            new_model=False
            alpha_model.load_state_dict(state['net'])
            alpha_model.eval()
            sense_model=sense_model_new
            labeler=labeler_new
            avg_train_loss=avg_train_loss_new
            
            baseline_ddla=0
            high_loss_risk=0
            high_loss_ddla=0
            max_loss=0
            future_train_loss=0
            excess=[]
            quant_window=[]
            loss_window=[]
            dyn_quant=[]
            quant_wanted=0.5
            quant_new=0.5
            predicted_loss=0
            predicted_lower=0
            predicted_upper=0
            retrain_loss=0
            true_loss=0
            train_loss_sum=0
            data_needed=[]
            data_avail=[]
            last_fixed=len(storage_data)
            retrain_ts=in_ts
            risk_thresh=risk_thresh_new


        in_x=transform_test(imgs[index][0])

        alpha_model_logits=alpha_model_inf(alpha_model,in_x)
        _,alpha_model_pred=torch.max(alpha_model_logits,1)

        # alpha_model_pred=alpha_model_pred.cpu().detach().numpy()[0]

        vit_x=embedding_transform(imgs[index][0]).unsqueeze(0)
        vit_x=vit_x.to(device)
        feats = vit._process_input(vit_x)

        batch_class_token = vit.class_token.expand(vit_x.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)

        feats = vit.encoder(feats)

        # We're only interested in the representation of the CLS token that we appended at position 0
        sense_x = feats[:, 0]
        sense_x=sense_x.cpu().detach().numpy()[0]
        np_features.append(sense_x)
        # print("SENSE SHAPE :",sense_x.shape)
        sense_x=np.append(sense_x,alpha_model_pred.cpu().detach().numpy()[0])
        pred_logits=sense_model.inf(np.array([sense_x]),apply_temp=False)
        probs=F.softmax(pred_logits,dim=-1)



        mean_probs=torch.mean(probs,dim=1)#.detach().numpy()[0]
        total_uncer=entropy_shannon(mean_probs).detach().numpy()
        mean_probs=torch.mean(probs,dim=1).detach().numpy()[0]
        sense_pred=np.argmax(mean_probs)

        model_uncer=model_uncertainty_shannon(probs).detach().numpy()
        if model_uncer<0:
            model_uncer[0]=0.0


        train_loss_sum+=avg_train_loss
        pred_loss=np.argmax(mean_probs)
        alpha_loss=calc_loss(alpha_model_logits,in_y).item()

        loss_label=labeler.get_loss_label(in_y,alpha_model_pred.cpu().detach().numpy()[0])
        baseline_ddla=baseline_ddla+avg_train_loss

        risk_score=total_uncer[0]+mean_probs[1]
        if risk_score>=risk_thresh:
            risk_prediction=1
        else:
            risk_prediction=0

        high_loss_ddla=high_loss_ddla+risk_prediction
        
        if (total_uncer[0]+mean_probs[1])>1.0:
            data_needed.append(index)

        true_loss+=loss_label

        predicted_loss+=sense_pred

        # true_loss=true_loss-base[3]

        # predicted_loss=abs(predicted_loss-base[1])
        

        # predicted_lower=abs(predicted_lower-base[0])
        # predicted_upper=abs(predicted_upper-base[3])

        # if predicted_upper<predicted_lower:
        #     predicted_upper=predicted_lower

        # retrain_loss+=1# base[3]


        #['model_pred','loss','loss_label','sense_pred_low','sense_pred_median','sense_pred_high']
        new_entry=np.hstack([arr,[alpha_model_pred.cpu().detach().numpy()[0],alpha_loss,loss_label,high_loss_ddla,risk_prediction,risk_score,model_uncer[0],max_loss,total_arrivals,high_loss_ddla]])
        np_data.append(new_entry)


        dist_entry=np.hstack([arr,[alpha_model_pred.cpu().detach().numpy()[0],alpha_loss,loss_label,sense_pred,-1,js_loss_dist,js_dist_epis,js_dist_uncer]])
        dist_df_array.append(dist_entry)
        near_ts=find_nearest(ts,row['timestamp']-delta)
        print(len(storage_data))
        # if len(data_needed)>0:
        #     print("DEBUG :",data_needed[0],random_start+len(dist_data))
        if ts[near_ts]-old_near_ts>600:
            old_near_ts=ts[near_ts]
            # print(len(np_data[:near_ts]))
            storage_data=pd.DataFrame(np_data[:near_ts], columns=store_columns)
            dist_data=pd.DataFrame(dist_df_array[:near_ts], columns=dist_columns)
            # dist_data.to_csv("./cifar10/data/retraining_dist_test_new.csv")
            if len(data_needed)>0:
                if data_needed[0]<random_start+len(dist_data):
                    near_data=find_nearest(np.array(data_needed),random_start+len(dist_data))
                    data_avail.extend(data_needed[:near_data])
                    data_needed=data_needed[near_data:]

        print("Data Needed : {}  Data Available : {}".format(len(data_needed),len(data_avail)))
            # storage_data.to_csv("./cifar10/data/storage_test.csv")
            # storage_data = pd.concat([storage_data,pd.DataFrame(np_data[:near_ts], columns=store_columns)])
            # np_data=np_data[near_ts:]



        new_row = {'time': in_ts, 'total': predicted_loss}
        data_df.loc[len(data_df)] = new_row


        # if index%100==0 and index>0 and js_loss_dist>0:
        #     excess_dist=js_loss_dist-distro.label.avg
        #     if excess_dist<0:
        #         excess_dist=0
        #     scaled_dist=scale_dist(excess_dist,0,0.13)
        #     if excess_dist>0:
        #         quant_wanted=0.5+2*excess_dist
        #     else:
        #         quant_wanted=0.5
        
        # print("Quant Wanted :",quant_wanted)
        # print("Quant New :",quant_new)

        if len(retraining_timestamps)>0:
            if storage_data['timestamp'].iloc[-1]<retraining_timestamps[-1][0]:
                last_fixed=len(storage_data)

        if len(storage_data)>0 and (len(storage_data)-last_fixed)>100:
            pred_fix=storage_data['loss_label'][last_fixed:len(storage_data)].sum()-storage_data['sense_pred'][last_fixed:len(storage_data)].sum()
            last_fixed=len(storage_data)
            high_loss_ddla=high_loss_ddla+pred_fix

        # if len(storage_data)>0:
        #     if storage_data.iloc[-1]['timestamp']<=retrain_ts:
        #         last_fixed=len(storage_data)

        # if len(storage_data)>0 and (len(storage_data)-last_fixed)>100:
        #     pred_fix=storage_data['loss_label'][last_fixed:len(storage_data)].sum()-(storage_data.iloc[-1]['sense_pred_low']-storage_data.iloc[last_fixed]['sense_pred_low'])
            
        #     print("FIX VAL :",pred_fix)
        #     last_fixed=len(storage_data)
        #     high_loss_ddla=high_loss_ddla+pred_fix
            
        print("PRED FIX :",pred_fix)

        data_threshold=int(beta*kappa)

        if len(storage_data)>0:
            if storage_data.iloc[[-1]]['timestamp'].values[0]>retrain_ts:
                new_close_ts=find_nearest(ts,retrain_ts)
                if (high_loss_ddla-baseline_ddla)>kappa and sum(storage_data['loss_label'][sense_data_idx:])>data_threshold:
                    retrain_ts=in_ts
                    new_model=True
                    baseline_ddla=0
                    high_loss_risk=0
                    high_loss_ddla=0
                    max_loss=0
                    future_train_loss=0
                    excess=[]
                    quant_window=[]
                    loss_window=[]
                    dyn_quant=[]
                    quant_wanted=0.5
                    quant_new=0.5
                    predicted_loss=0
                    predicted_lower=0
                    predicted_upper=0
                    retrain_loss=0
                    true_loss=0
                    train_loss_sum=0
                    quant_correction=0
                    print("Starting retraining")
                    sense_data_idx=len(storage_data)
                    last_fixed=len(storage_data)
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
                        old_train_x=np.load("./cifar10/data/train_x_ablation.npy")
                        old_train_y=np.load("./cifar10/data/train_y_ablation.npy")
                    

                    
                    # new_train_x=imgs.data[data_avail]

                    # data_avail=list(np.array(data_avail)-random_start)
                    # new_train_y=dist_data['y'].values[data_avail]

                    # filler_data_size=len(old_train_x)-len(new_train_x)

                    # filler_idx=np.random.choice(np.arange(len(old_train_x)),filler_data_size,replace=False)

                    # old_data_x_slice=old_train_x[filler_idx]
                    # old_data_y_slice=old_train_y[filler_idx]

                    # train_x=np.concatenate([old_data_x_slice,new_train_x],axis=0)
                    # train_y=np.concatenate([old_data_y_slice,new_train_y],axis=0)

                    new_train_x=imgs.data[random_start:random_start+len(storage_data)]

                    new_train_y=storage_data['y'].values


                    train_x=np.concatenate([old_train_x,new_train_x],axis=0)
                    train_y=np.concatenate([old_train_y,new_train_y],axis=0)

                    retrain_x,retest_x,retrain_y,retest_y=train_test_split(train_x, train_y, test_size=0.2, random_state=42, shuffle=True)

                    trainset = image_dataset(retrain_x,retrain_y,transform=transform_train)
                    testset = image_dataset(retest_x,retest_y,transform=transform_test)

                    evalset=image_dataset(train_x,train_y,transform=transform_test)

                    if sc is not None:
                        retrain_path='./cifar10/checkpoint/retrain_ckpt_long_risk_with_threshold_v1_'+str(kappa)+'_'+str(int(delta))+'_'+str(sc)+'_'+str(beta)+'.pth'
                    net=train_img_model(trainset,testset,epochs=100,retrain=True,retrain_path=retrain_path)
                    net.eval()
                    if sc is not None:
                        state=torch.load(retrain_path)
                    else:
                        state=torch.load("./cifar10/checkpoint/retrain_ckpt4232.pth")

                    # alpha_model.load_state_dict(state['net'])
                    # alpha_model.eval()

                    net.load_state_dict(state['net'])
                    net.eval()

                    train_pred,train_loss,_=run_inference_dataset(net,model_inf,evalset,calc_loss,neural=True)
                    labeler_new=label_gen(train_loss)


                    # all_data_x=np.concatenate([old_train_x,imgs.data[:sense_data_idx]],axis=0)
                    # all_data_y=np.concatenate([old_train_y,dist_data[:sense_data_idx]['y'].values],axis=0)

                    # retrain_x,retest_x,retrain_y,retest_y=train_test_split(all_data_x, all_data_y, test_size=0.2, random_state=42, shuffle=True)

                    # trainset = image_dataset(retrain_x,retrain_y,transform=transform_train)
                    # testset = image_dataset(retest_x,retest_y,transform=transform_test)

                    # evalset=image_dataset(train_x,train_y,transform=transform_test)
                    # if sc is not None:
                    #     retrain_path='./cifar10/checkpoint/retrain_ckpt_'+str(int(delta))+'_'+str(sc)+'.pth'
                    # net=train_img_model(trainset,testset,epochs=100,retrain=True,retrain_path=retrain_path)
                    # net.eval()
                    # if sc is None:
                    #     state=torch.load("./cifar10/checkpoint/retrain_ckpt.pth")
                    # else:
                    #     state=torch.load(retrain_path)
                    # alpha_model.load_state_dict(state['net'])
                    # alpha_model.eval()
                    # train_pred,train_loss,_=run_inference_dataset(alpha_model,model_inf,evalset,calc_loss,neural=True)
                    # labeler=label_gen(train_loss)
                    
                    # allset=image_dataset(all_data_x,all_data_y,transform=transform_test)

                    sense_train_pred,sense_train_loss,_=run_inference_dataset(net,model_inf,evalset,calc_loss,neural=True)
                    sense_train_pred=sense_train_pred.squeeze(1)
                    sense_train_pred=np.argmax(sense_train_pred,axis=1)
                    sense_train_pred=np.expand_dims(sense_train_pred, axis=1)
                    
                    if old_sense_train_x is None:
                        old_sense_train_x=np.load("./cifar10/data/ddla_cifar_train_x.npy")

                    # sense_distro_x=np.concatenate([old_sense_train_x[filler_idx],np.array(np_features)[data_avail]])

                    # sense_distro_x,sense_distro_y=build_sensitivity_training_set_v2(sense_distro_x,train_loss,train_y,problem_type="reg",val=True,train_losses=train_loss)

                    sense_train_features=np.array(np_features)[:sense_data_idx]

                    sense_train_features=np.concatenate([old_sense_train_x,sense_train_features],axis=0)

                    risk_train_x,risk_train_y=build_sensitivity_training_ddla(sense_train_features,sense_train_pred,sense_train_loss,train_y,problem_type="class")
                    
                    risk_train_x=np.array(risk_train_x)
                    risk_train_y=np.array(risk_train_y)
                    print(sense_train_pred)
                    print("SHAPE :",sense_train_pred.shape)
                    risk_train_x=np.concatenate([risk_train_x,sense_train_pred],axis=1)

                    risk_train_x,risk_train_y,risk_train_loss,calib_un_x,calib_un_y,calib_un_loss=sense_model.gen_calib_data(risk_train_x,risk_train_y,train_loss,balance=True,test_size=0.1)

                    avg_train_loss_new=risk_train_y.mean()
                    
                    # sense_distro_x=np.array(sense_distro_x)
                    # sense_distro_y=np.array(sense_distro_y)

                    sense_model_new=riskadvisor_model()


                    sense_model_new.train_sense_model(risk_train_x,risk_train_y)

                    risk_thresh_new,risk_f1_score_new=find_best_thresh(sense_model_new,calib_un_x,calib_un_y)

                    # train_y_np=np.array(train_y)
                    # avg_train_loss=train_y_np.mean()

                    data_needed=[]
                    data_avail=[]
                    retraining_timestamps.append([in_ts,storage_data[-1:]['timestamp'].values[0],avg_train_loss])
                    np.save('./cifar10/data/retraining_ts_long_risk_with_threshold_v1_'+str(kappa)+'_'+str(delta)+'_'+str(sc)+'_'+str(beta)+'.npy',retraining_timestamps)   
        print("Index : {}  True : {}   RiskScore : {}  Lower Bound : {}  Upper Bound : {}  Baseline :{}".format(index,true_loss,high_loss_ddla,predicted_lower,predicted_upper,train_loss_sum))

    storage_data = pd.DataFrame(np_data, columns=store_columns)
    # # np.save('./cifar10/data/resnet_simple_correction_ablation_v12_'+str(delta)+'_'+str(sc)+'.npy',np_features)
    storage_data.to_csv('./cifar10/data/results_long_risk_with_threshold_v1_'+str(kappa)+'_'+str(delta)+'_'+str(sc)+'_'+str(beta)+'.csv')
    # dist_data=pd.DataFrame(dist_df_array, columns=dist_columns)
    # dist_data.to_csv('./cifar10/data/retraining_ts_long_v1_'+str(delta)+'_'+str(sc)+'.csv')


def file_in_directory(filename, directory):
    """Check if a file exists in the given directory."""
    return os.path.isfile(os.path.join(directory, filename))


delta=float(sys.argv[1])
sc=sys.argv[2]
kappa=int(sys.argv[3])
beta=float(sys.argv[4])

future_data=pd.read_csv("./cifar10/data/future_long_scenario_"+str(sc)+".csv",index_col=0)

filename='results_long_risk_with_threshold_v1_'+str(kappa)+'_'+str(delta*60*60)+'_'+str(sc)+'_'+str(beta)+'.csv'
directory = "./cifar10/data/"



if file_in_directory(filename, directory):
    print("Simulation already exisits")

else:
    device='cuda'
    net=ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    state=torch.load("./cifar10/checkpoint/cifar_image_classifier.pth")

    net.load_state_dict(state['net'])

    train_x=np.load("./cifar10/data/ddla_cifar_train_x.npy")
    train_y=np.load("./cifar10/data/ddla_cifar_train_y.npy")

    train_loss=np.load("./cifar10/data/cifar_train_loss.npy")

    labeler=label_gen(train_loss)

    with open("./cifar10/saved_models/risk_advisor_model.pkl",'rb') as inp:
        sense_model=pickle.load(inp)
    
    calib_x=np.load("./cifar10/data/calib_un_risk_x.npy")
    calib_y=np.load("./cifar10/data/calib_un_risk_y.npy")


    reg=None 
    imgs=np.load("./cifar10/data/future_long_scenario_"+str(sc)+"_imgs"+".npy")
    img_ds=image_dataset(imgs,np.array([1]*len(imgs)))

    if len(imgs)==len(future_data):
        print("Images and future data lengths match")
    # corr,uncorr=get_baseline(train_x,train_y,train_loss,sense_model)

    risk_thresh,risk_f1_score=find_best_thresh(sense_model,calib_x,calib_y)

    full_dist=None


    run_sim(img_ds,future_data,net,model_inf,sense_model,labeler,delta=delta*60*60,reg=reg,distro=full_dist,avg_train_loss=train_y.mean(),sc=sc,kappa=kappa,beta=beta,risk_thresh=risk_thresh)

