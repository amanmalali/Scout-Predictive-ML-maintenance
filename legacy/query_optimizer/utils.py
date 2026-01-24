from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np


def scale_data(features,scale_func=None):
    if scale_func is not None:
        scaled_df=scale_func.transform(features)
    else:
        scale_func=MinMaxScaler()
        scaled_df=scale_func.fit_transform(features)
    
    return scaled_df,scale_func



class torch_dataset(Dataset):
    def __init__(self,X,y,scale=False):
        super().__init__()
        self.X=X.astype(np.float32)
        self.y=y.astype(np.float32)
        if scale:
            self.X,self.scale_func=scale_data(self.X)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index,:],self.y[index]


class custom_torch_dataset(Dataset):
    def __init__(self,X,logits,y,scale=False):
        super().__init__()
        self.X=X.astype(np.float32)
        self.logits=logits.astype(np.float32)
        self.y=y.astype(np.float32)
        if scale:
            self.X,self.scale_func=scale_data(self.X)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index,:],self.logits[index,:],self.y[index]
    
def run_inference_dataset(model,inf_func,data,loss_func,return_embeddings=False,neural=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictions=[]
    losses=[]
    if return_embeddings:
        embeddings=[]
    else:
        embeddings=None
    for X,y in data:
        y_pred=inf_func(model,X)
        loss=loss_func(y_pred,y)
        if return_embeddings:
            y_pred,emb=y_pred
        if device=='cuda':
            predictions.append(y_pred.cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy())
        else:
            predictions.append(y_pred.detach().numpy())
            losses.append(loss.detach().numpy())
        if return_embeddings:
            embeddings.append(embeddings.detach().numpy())

    predictions=np.array(predictions)
    losses=np.array(losses)

    if neural:
        print(predictions.shape)

    if return_embeddings:
        embeddings=np.array(embeddings)
    return predictions,losses,embeddings



class image_dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        x = transforms.ToPILImage()(x)
        if self.transform:
            x = self.transform(x)
    
        return x, y
    
    def __len__(self):
        return len(self.data)

def generate_exponential_decay_array(length, decay_factor):
    """
    Generate an array with values that start at 1 and decay exponentially.

    Parameters:
    length (int): The length of the array.
    decay_factor (float): The decay factor. Higher values lead to faster decay.

    Returns:
    numpy.ndarray: Array of exponentially decaying values.
    """
    # Generate an array of indices from 0 to length-1
    indices = np.arange(length)
    
    # Compute the exponential decay
    decay_array = np.exp(-decay_factor * indices)
    
    return decay_array


def combine_xy_with_transition(x1, y1, x2, y2, transition_window):
    
    window_probs = np.linspace(1, 0, transition_window)

    transition_start=int(len(y1)-transition_window//2)

    probs=np.ones(transition_start)
    probs=np.concatenate([probs,window_probs],axis=0)
    drift_probs=np.zeros(len(y1)+len(y2)-len(probs))
    probs=np.concatenate([probs,drift_probs],axis=0)
    sim_x=[]
    sim_y=[]
    train_i=0
    test_i=0
    drifted_i=0
    print("LENGTHS :",len(probs))
    for i in range(len(x1)+len(x2)):
        pick=np.random.choice([0,1],1,p=[probs[i],1-probs[i]])
        if pick==0:
            if train_i<len(x1):
                sim_x.append(x1[train_i])
                sim_y.append(y1[train_i])
                train_i+=1
            elif drifted_i<len(x2):
                sim_x.append(x2[drifted_i])
                sim_y.append(y2[drifted_i])
                drifted_i+=1
        elif pick==1:
            if drifted_i<len(x2):
                sim_x.append(x2[drifted_i])
                sim_y.append(y2[drifted_i])
                drifted_i+=1
            elif train_i<len(x1):
                sim_x.append(x1[train_i])
                sim_y.append(y1[train_i])
                train_i+=1
    
    return np.array(sim_x),np.array(sim_y)

def add_chunks_periodically_separate(a_x, a_y, b_x, b_y, chunk_size, period):
    a_start=0
    b_start=0
    a_end=chunk_size
    b_end=period
    chunks=len(a_x)//chunk_size
    sim_x=a_x[a_start:a_end]
    sim_y=a_y[a_start:a_end]
    for c in range(chunks):
        if c==0:
            sim_x=a_x[a_start:a_end]
            sim_y=a_y[a_start:a_end]
        else:
            sim_x=np.concatenate([sim_x,a_x[a_start:a_end]])
            sim_y=np.concatenate([sim_y,a_y[a_start:a_end]])
        sim_x=np.concatenate([sim_x,b_x[b_start:b_end]])
        sim_y=np.concatenate([sim_y,b_y[b_start:b_end]])

        a_start=a_end
        a_end=a_end+chunk_size

        b_start=b_end
        b_end=b_end+period

        if a_end>len(a_x):
            a_end=len(a_x)
        
        if b_end>len(b_x):
            b_end=len(b_x)

    if b_end<len(b_x):
        b_end=len(b_x)
    sim_x=np.concatenate([sim_x,b_x[b_start:b_end]])
    sim_y=np.concatenate([sim_y,b_y[b_start:b_end]])
    return sim_x,sim_y


def find_linear_equation(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the slope (m)
    slope = (y2 - y1) / (x2 - x1)

    # Use one of the points to find the y-intercept (b)
    intercept = y1 - slope * x1

    return slope, intercept


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
