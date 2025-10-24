import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from arrival.gen_time import find_nearest
from sensitivity.risk_advisor import riskadvisor_model
from sensitivity.sense_model import sensitivity_model
from sensitivity.new_temp_scaler import train_temp_scaler
from sensitivity.uncertainty import model_uncertainty, renyi_entropy
from utils import custom_torch_dataset

# from sensitivity.sense_model import create_sense_model,train_sense_model

# from sensitivity.sense_model import create_sense_model,train_sense_model



start=time.time()

train_x=np.load("./cifar10/data/sense_cifar_train_x.npy")
train_y=np.load("./cifar10/data/sense_cifar_train_y.npy")
train_loss=np.load("./cifar10/data/cifar_train_loss.npy")

test_x=np.load("./cifar10/data/sense_cifar_test_x.npy")
test_y=np.load("./cifar10/data/sense_cifar_test_y.npy")
test_loss=np.load("./cifar10/data/cifar_test_loss.npy")

print(train_x.shape)


val_x=np.load("./cifar10/data/sense_cifar_val_x.npy")
val_y=np.load("./cifar10/data/sense_cifar_val_y.npy")
val_loss=np.load("./cifar10/data/cifar_val_loss.npy")

sense_model=sensitivity_model()

train_x,train_y,train_loss,calib_over_x,calib_over_y,calib_loss,calib_un_x,calib_un_y,calib_un_loss=sense_model.gen_calib_data(train_x,train_y,train_loss,balance=True)

np.save("./cifar10/data/calib_un_x.npy",calib_un_x)
np.save("./cifar10/data/calib_un_y.npy",calib_un_y)
np.save("./cifar10/data/calib_un_loss.npy",calib_un_loss)

np.save("./cifar10/data/calib_over_x.npy",calib_over_x)
np.save("./cifar10/data/calib_over_y",calib_over_y)
np.save("./cifar10/data/calib_over_loss.npy",calib_loss)


sense_model.train_sense_model(train_x,train_y)

train_logits=sense_model.gen_logits(train_x)
val_logits=sense_model.gen_logits(val_x)
calib_logits=sense_model.gen_logits(calib_over_x)

train_data=custom_torch_dataset(train_x,train_logits,train_y)
val_data=custom_torch_dataset(val_x,val_logits,val_y)
calib_data=custom_torch_dataset(calib_over_x,calib_logits,calib_over_y)

sense_model.lin_model,sense_model.scaler=train_temp_scaler(calib_data,val_data)

pred_logits=sense_model.inf(val_x,apply_temp=True)
probs=F.softmax(pred_logits,dim=-1)
model_uncer=model_uncertainty(probs).detach().numpy()
mean_probs=torch.mean(probs,dim=1)
confidences, predictions = torch.max(mean_probs, 1)
confidences=confidences.detach().numpy()
predictions=predictions.detach().numpy()
mean_probs=mean_probs.detach().numpy()
uncer=np.apply_along_axis(renyi_entropy,axis=1,arr=mean_probs,alpha=2)

print(np.unique(predictions,return_counts=True))
print(np.unique(val_y,return_counts=True))

correct_uncer=model_uncer[predictions==val_y]
incorrect_uncer=model_uncer[predictions!=val_y]

print(correct_uncer.mean())
print(incorrect_uncer.mean())



pred_logits=sense_model.inf(test_x,apply_temp=True)
probs=F.softmax(pred_logits,dim=-1)
model_uncer=model_uncertainty(probs).detach().numpy()
mean_probs=torch.mean(probs,dim=1)
confidences, predictions = torch.max(mean_probs, 1)
confidences=confidences.detach().numpy()
predictions=predictions.detach().numpy()
mean_probs=mean_probs.detach().numpy()
uncer=np.apply_along_axis(renyi_entropy,axis=1,arr=mean_probs,alpha=2)

print(np.unique(predictions,return_counts=True))
print(np.unique(test_y,return_counts=True))

correct_uncer=model_uncer[predictions==test_y]
incorrect_uncer=model_uncer[predictions!=test_y]

print(correct_uncer.mean())
print(incorrect_uncer.mean())

print("Trained scaler")
with open("./cifar10/saved_models/sense_model_vit_cifar.pkl",'wb') as output:
    pickle.dump(sense_model,output)




start=time.time()

ddla_train_x=np.load("./cifar10/data/ddla_cifar_train_x.npy")
ddla_train_y=np.load("./cifar10/data/ddla_cifar_train_y.npy")
ddla_train_pred=np.load('./cifar10/data/cifar_train_pred.npy')
ddla_train_loss=np.load("./cifar10/data/cifar_train_loss.npy")

ddla_train_x=np.concatenate([ddla_train_x,ddla_train_pred],axis=1)

sense_model=riskadvisor_model()

train_x,train_y,train_loss,calib_un_x,calib_un_y,calib_un_loss=sense_model.gen_calib_data(ddla_train_x,ddla_train_y,ddla_train_loss,balance=True,test_size=0.1)


np.save("./cifar10/data/calib_un_risk_x.npy",calib_un_x)
np.save("./cifar10/data/calib_un_risk_y.npy",calib_un_y)


sense_model.train_sense_model(ddla_train_x,ddla_train_y)

with open("./cifar10/saved_models/risk_advisor_model.pkl",'wb') as output:
    pickle.dump(sense_model,output)


