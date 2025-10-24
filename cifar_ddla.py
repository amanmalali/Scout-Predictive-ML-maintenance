import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn.functional as F
import os
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# n = len(sys.argv)
# print("Total arguments passed:", n)

# # Arguments passed
# print("\nName of Python script:", sys.argv[0])

# print("ARGUMENT 1 :",sys.argv[1])
# #Temporary imports 
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

def ddla_build_grid_search(ddla_train_x, ddla_train_y):
    # Create column names
    cols = ['feat_' + str(f) for f in range(ddla_train_x.shape[1])]
    df_train = pd.DataFrame(data=ddla_train_x, columns=cols)

    # Print label distribution
    proportion_label_0 = (ddla_train_y == 0).mean()
    proportion_label_1 = (ddla_train_y == 1).mean()
    print(f"Proportion of label 0: {proportion_label_0:.2f}, label 1: {proportion_label_1:.2f}")

    # Compute total number of samples
    n_samples = ddla_train_x.shape[0]

    # Define parameter grid
    param_grid = {
        'max_depth': list(range(3, 11)),  # 3 to 10
        'min_samples_leaf': [max(1, int(n_samples * p)) for p in np.linspace(0.00, 0.05, 6)]  # 0% to 5%
    }

    # Set up GridSearchCV with 10-fold cross-validation
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid=param_grid,
        scoring='accuracy',
        cv=10,
        n_jobs=-1
    )

    # Fit grid search
    grid_search.fit(df_train, ddla_train_y)

    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    # Use best estimator
    best_tree = grid_search.best_estimator_

    # Extract decision paths from the best tree
    decision_paths = extract_decision_paths(best_tree, cols)

    return decision_paths

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


    decision_paths = extract_decision_paths(decision_tree_default, cols)

    return decision_paths

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

def run_sim(imgs,data,alpha_model,alpha_model_inf,sense_model,labeler,delta=4*60*60,reg=None,distro=None,avg_train_loss=-1,sc=None,kappa=5000,beta=0.75):
    
    ddla_train_x=np.load("./cifar10/data/ddla_cifar_train_x.npy")
    ddla_train_y=np.load("./cifar10/data/ddla_cifar_train_y.npy")
    
    decision_paths=ddla_build(ddla_train_x,ddla_train_y)
    cols=[]
    for f in range(ddla_train_x.shape[1]):
        cols.append('feat_'+str(f))
    # cols.append("y")
    # all_data=np.concatenate([train_data,val_data],axis=0)
    df_train=pd.DataFrame(data=ddla_train_x,columns=cols)
    train_matches = df_train.apply(lambda x: any(sample_matches_path(x, path) for path in decision_paths), axis=1)
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

    store_columns=np.hstack([future_data.columns.values,['model_pred','loss','loss_label','sense_pred_low','sense_pred','sense_pred_high','uncer','future_loss','prophet_count','sum_pred_loss','og_loss','og_pred']])

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
    ddla_input=[]

    train_proportion=0
    batch_proportion=0
    baseline_ddla=0
    high_loss_ddla=0
    old_train_loss=None
    major_retrain=0
    ddla_rebuild=0
    data_threshold=4000
    new_model=False
    retrain_ts=-1
    avg_train_loss_new=0

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
            labeler=labeler_new
            avg_train_loss=avg_train_loss_new

            train_matches=train_matches_new
            df_train=df_train_new
            decision_paths=decision_paths_new

            baseline_ddla=0
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
        # sense_x=np.append(sense_x,alpha_model_pred.cpu().detach().numpy()[0])

        ddla_input.append(sense_x)

        train_loss_sum+=avg_train_loss
        alpha_loss=calc_loss(alpha_model_logits,in_y).item()

        loss_label=labeler.get_loss_label(in_y,alpha_model_pred.cpu().detach().numpy()[0])
        
        true_loss+=loss_label

    
       

       

        #if index%1==0 and index>0:

        ddla_df=pd.DataFrame(np.array(ddla_input[-1:]),columns=cols)
        print("DDLA SIZE :",len(ddla_df))
        # train_matches = df_train.apply(lambda x: any(sample_matches_path(x, path) for path in decision_paths), axis=1)
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

        # for dn in range(index-500,int(index-500+batch_proportion*500+10)):
        #     data_needed.append(dn)
        
        print("Train prop :",train_proportion)
        print("batch prop :",batch_proportion)

        #['model_pred','loss','loss_label','sense_pred_low','sense_pred_median','sense_pred_high']
        new_entry=np.hstack([arr,[alpha_model_pred.cpu().detach().numpy()[0],alpha_loss,loss_label,high_loss_ddla,batch_proportion,-1,-1,max_loss,total_arrivals,high_loss_ddla,alpha_loss,alpha_model_pred.cpu().detach().numpy()[0]]])
        np_data.append(new_entry)

        dist_entry=np.hstack([arr,[alpha_model_pred.cpu().detach().numpy()[0],alpha_loss,loss_label,-1,-1,js_loss_dist,js_dist_epis,js_dist_uncer]])
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



            # if (len(storage_data)-ddla_rebuild)>1000:
            #     ddla_rebuild=len(storage_data)
            #     if old_train_x is None:
            #         old_train_x=np.load("./cifar10/data/train_x_ablation.npy")
            #         old_train_y=np.load("./cifar10/data/train_y_ablation.npy")
            #         old_ddla_train_x=np.load("./cifar10/data/ddla_cifar_train_x2_simple_ablation.npy")
            #         old_ddla_train_y=np.load("./cifar10/data/ddla_cifar_train_y2_simple_ablation.npy")
    

                
            #     if old_train_loss is None:
            #         old_train_loss=np.load('./cifar10/data/cifar_train_loss_ablation.npy')
            #         old_train_pred=np.squeeze(np.load('./cifar10/data/cifar_train_pred_ablation.npy'))
                

            #     new_train_x=ddla_input[random_start:random_start+len(storage_data)]

            #     new_train_y=storage_data['y'].values

            #     train_x=np.concatenate([old_ddla_train_x,new_train_x],axis=0)
            #     train_y=np.concatenate([old_train_y,new_train_y],axis=0)


                
            #     new_train_loss=storage_data[major_retrain:]['loss'].values
            #     train_loss=np.concatenate([old_train_loss,new_train_loss])

                
            #     new_train_pred=storage_data[major_retrain:]['model_pred'].values
            #     train_pred=np.concatenate([old_train_pred,new_train_pred])

            #     ddla_train_x,ddla_train_y=build_sensitivity_training_ddla(train_x,train_pred,train_loss,train_y,problem_type='class')
            #     ddla_train_x=np.array(ddla_train_x)
            #     ddla_train_y=np.array(ddla_train_y)
            #     decision_paths=ddla_build(ddla_train_x,ddla_train_y)
            #     cols=[]
            #     for f in range(ddla_train_x.shape[1]):
            #         cols.append('feat_'+str(f))
            #     # cols.append("y")
            #     # all_data=np.concatenate([train_data,val_data],axis=0)
            #     df_train=pd.DataFrame(data=ddla_train_x,columns=cols)
            #     train_matches = df_train.apply(lambda x: any(sample_matches_path(x, path) for path in decision_paths), axis=1)

        print("Data Needed : {}  Data Available : {}".format(len(data_needed),len(data_avail)))
            # storage_data.to_csv("./cifar10/data/storage_test.csv")
            # storage_data = pd.concat([storage_data,pd.DataFrame(np_data[:near_ts], columns=store_columns)])
            # np_data=np_data[near_ts:]


        # if len(storage_data)>20000:
        #     if len(storage_data[new_data_idx:])>5000:
        #         new_data_idx=len(storage_data)-1
        #         predictions=storage_data['sense_pred'].values
        #         uncertainty=storage_data['uncer'].values

        #         calib_y=storage_data['loss_label'].values


        #         reg=dummy_baseline_retrain(predictions,uncertainty,calib_y)
        #         print("Trained new quantile regressor")




        # print("Retrain Counter :",retrain_counter)
        # print("Retrain if only training:",retrain_loss)

        # print("Epis dist :",js_dist_epis)
        # print("Epis dist :",js_dist_uncer)

        # or js_dist_epis>distro.epis.avg+3*distro.epis.std
        # if (js_loss_dist>distro.label.avg+3*distro.label.std ) and index%100==0 and index>0:
        #     loss_switch=1
        #     # data_needed.extend([t for t in range(index-100,index)])
        #     retrain_loss+=100


        # if (js_loss_dist<=distro.label.avg+3*distro.label.std ) and index%100==0 and index>0:
        #     loss_switch=0


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

        

                
                
        # if len(storage_data)>0:
        #     if storage_data.iloc[-1]['timestamp']<=retrain_ts:
        #         last_fixed=len(storage_data)

        # if len(retraining_timestamps)>0:
        #     if storage_data['timestamp'].iloc[-1]<retraining_timestamps[-1][0]:
        #         last_fixed=len(storage_data)

        #     if storage_data['timestamp'].iloc[-1]<retrain_ts:
        #         last_fixed=len(storage_data)



        # if len(storage_data)>0 and (len(storage_data)-last_fixed)>100:
        #     pred_fix=storage_data['loss_label'][last_fixed:len(storage_data)].sum()-(storage_data.iloc[-1]['sense_pred_low']-storage_data.iloc[last_fixed]['sense_pred_low'])
            
        #     print("FIX VAL :",pred_fix)
        #     last_fixed=len(storage_data)
        #     high_loss_ddla=high_loss_ddla+pred_fix

        
        if len(retraining_timestamps)>0:
            if storage_data['timestamp'].iloc[-1]<retraining_timestamps[-1][0]:
                last_fixed=len(storage_data)
            
            if storage_data['timestamp'].iloc[-1]<retrain_ts:
                last_fixed=len(storage_data)

        if len(storage_data)>0 and (len(storage_data)-last_fixed)>100:
            pred_fix=storage_data['loss_label'][last_fixed:len(storage_data)].sum()-storage_data['sense_pred'][last_fixed:len(storage_data)].sum()
            last_fixed=len(storage_data)
            high_loss_ddla=high_loss_ddla+pred_fix
        
        print("PRED FIX :",pred_fix)

        print("Trying to fix with :",pred_fix)
        print("Data Needed : {}  Data Available : {}".format(len(data_needed),len(data_avail)))
        print("Baseline : {}  Current : {}".format(baseline_ddla,high_loss_ddla))


        # if kappa==10000:
        #     data_threshold=4000
        # else:
        #     data_threshold=int((3/4)*kappa)

        data_threshold=int(beta*kappa)

        if (high_loss_ddla-baseline_ddla)>kappa and sum(storage_data['loss_label'][sense_data_idx:])>data_threshold:
            major_retrain=len(storage_data)
            retrain_ts=in_ts
            new_model=True
            baseline_ddla=0
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
                old_train_x=np.load("./cifar10/data/train_x.npy")
                old_train_y=np.load("./cifar10/data/train_y.npy")
            

            
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
                retrain_path='./cifar10/checkpoint/retrain_ckpt_ddla_kappa_v1_'+str(kappa)+'_'+str(int(delta))+'_'+str(sc)+'_'+str(beta)+'.pth'
            net=train_img_model(trainset,testset,epochs=100,retrain=True,retrain_path=retrain_path)
            net.eval()
            if sc is not None:
                state=torch.load(retrain_path)
            else:
                state=torch.load("./cifar10/checkpoint/retrain_ckpt4232.pth")

            net.load_state_dict(state['net'])
            net.eval()

            # train_pred,train_loss,_=run_inference_dataset(net,model_inf,evalset,calc_loss,neural=True)
            

            sense_train_pred,sense_train_loss,_=run_inference_dataset(net,model_inf,evalset,calc_loss,neural=True)
            
            labeler_new=label_gen(sense_train_loss)
            
            sense_train_pred=sense_train_pred.squeeze(1)
            sense_train_pred=np.argmax(sense_train_pred,axis=1)
            sense_train_pred=np.expand_dims(sense_train_pred, axis=1)
            
            if old_sense_train_x is None:
                old_sense_train_x=np.load("./cifar10/data/ddla_cifar_train_x.npy")

            # sense_distro_x=np.concatenate([old_sense_train_x[filler_idx],np.array(np_features)[data_avail]])

            # sense_distro_x,sense_distro_y=build_sensitivity_training_set_v2(sense_distro_x,train_loss,train_y,problem_type="reg",val=True,train_losses=train_loss)

            sense_train_features=np.array(np_features)[:sense_data_idx]

            sense_train_features=np.concatenate([old_sense_train_x,sense_train_features],axis=0)

            ddla_train_x,ddla_train_y=build_sensitivity_training_ddla(sense_train_features,sense_train_pred,sense_train_loss,train_y,problem_type="class")
            
            ddla_train_x=np.array(ddla_train_x)
            ddla_train_y=np.array(ddla_train_y)

            avg_train_loss_new=ddla_train_y.mean()

            decision_paths_new=ddla_build(ddla_train_x,ddla_train_y)
            cols=[]
            for f in range(ddla_train_x.shape[1]):
                cols.append('feat_'+str(f))
            # cols.append("y")
            # all_data=np.concatenate([train_data,val_data],axis=0)
            df_train_new=pd.DataFrame(data=ddla_train_x,columns=cols)
            train_matches_new = df_train.apply(lambda x: any(sample_matches_path(x, path) for path in decision_paths), axis=1)
            data_needed=[]
            data_avail=[]
            
            retraining_timestamps.append([in_ts,storage_data[-1:]['timestamp'].values[0],avg_train_loss])
            np.save('./cifar10/data/retraining_ts_ddla_kappa_v1_'+str(kappa)+'_'+str(delta)+'_'+str(sc)+'_'+str(beta)+'.npy',retraining_timestamps)
        

        #######
        


        print("Index : {}  True : {}   Predicted : {}  Lower Bound : {}  Upper Bound : {}  Baseline :{}".format(index,true_loss,high_loss_ddla,predicted_lower,predicted_upper,train_loss_sum))
        
    storage_data = pd.DataFrame(np_data, columns=store_columns)
    # np.save('./cifar10/data/resnet_simple_correction_ablation_v11_'+str(delta)+'_'+str(sc)+'.npy',np_features)
    storage_data.to_csv('./cifar10/data/ddla_results_kappa_v1_'+str(kappa)+'_'+str(delta)+'_'+str(sc)+'_'+str(beta)+'.csv')
    dist_data=pd.DataFrame(dist_df_array, columns=dist_columns)
    # dist_data.to_csv('./cifar10/data/retraining_simple_correction_ablation_v11_'+str(delta)+'_'+str(sc)+'.csv')


def file_in_directory(filename, directory):
    """Check if a file exists in the given directory."""
    return os.path.isfile(os.path.join(directory, filename))



delta=float(sys.argv[1])
sc=sys.argv[2]
kappa=int(sys.argv[3])
beta=float(sys.argv[4])



future_data=pd.read_csv("./cifar10/data/future_long_scenario_"+str(sc)+".csv",index_col=0)


filename='ddla_results_kappa_v1_'+str(kappa)+'_'+str(delta*60*60)+'_'+str(sc)+'_'+str(beta)+'.csv'
directory = "./cifar10/data/"



device='cuda'

if file_in_directory(filename, directory):
    print("Simulation already exisits")

else:
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

    with open("./cifar10/saved_models/sense_model_vit_cifar.pkl",'rb') as inp:
        sense_model=pickle.load(inp)

    calib_x=np.load("./cifar10/data/calib_un_x.npy")
    calib_y=np.load("./cifar10/data/calib_un_y.npy")

    reg=None
    imgs=np.load("./cifar10/data/future_long_scenario_"+str(sc)+"_imgs"+".npy")
    img_ds=image_dataset(imgs,np.array([1]*len(imgs)))

    if len(imgs)==len(future_data):
        print("Images and future data lengths match")
    # corr,uncorr=get_baseline(train_x,train_y,train_loss,sense_model)


    full_dist=None

    run_sim(img_ds,future_data,net,model_inf,sense_model,labeler,delta=delta*60*60,reg=reg,distro=full_dist,avg_train_loss=train_y.mean(),sc=sc,kappa=kappa,beta=beta)

