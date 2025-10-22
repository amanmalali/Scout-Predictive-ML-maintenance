import copy
import pickle
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from arrival.gen_time import add_timestamps_simple
from census.gen_data import generate_data_drift
from census.train_basic_classifier import (calc_loss, model_inf, test_model,train_model)
from sensitivity.gen_train_data import (build_sensitivity_training_ddla,build_sensitivity_training_set)
from sensitivity.risk_advisor import riskadvisor_model
from sensitivity.sense_model import sensitivity_model
from sensitivity.temp_scaler import train_temp_scaler, uncer_loss
from utils import (add_chunks_periodically_separate,
                   combine_xy_with_transition, custom_torch_dataset,
                   generate_exponential_decay_array, run_inference_dataset,
                   torch_dataset, unison_shuffled_copies)

# from sensitivity.sense_model import create_sense_model,train_sense_model

# from sensitivity.sense_model import create_sense_model,train_sense_model


def train_risk_model():
    start=time.time()

    train_x=np.load("./census/data/ddla_train_x.npy")
    train_y=np.load("./census/data/ddla_train_y.npy")
    train_loss=np.load("./census/data/ddla_train_y.npy")


    val_x=np.load("./census/data/ddla_val_x.npy")
    val_y=np.load("./census/data/ddla_val_y.npy")
    model_path="./census/saved_models/census_classifier_v1.pt"

    train_data=torch_dataset(train_x,train_y)


    model=torch.load(model_path)

    pred,loss,_=run_inference_dataset(model,model_inf,train_data,calc_loss)

    sense_model=riskadvisor_model()
    train_x=np.concatenate([train_x,pred],axis=1)

    np.save("./census/data/ddla_train_x_pred.npy",train_x)
    # train_x,train_y,train_loss,calib_over_x,calib_over_y,calib_loss,calib_un_x,calib_un_y=sense_model.gen_calib_data(train_x,train_y,train_loss,balance=True,test_size=500)
    train_x,train_y,train_loss,calib_un_x,calib_un_y,calib_un_loss=sense_model.gen_calib_data(train_x,train_y,train_loss,balance=True,test_size=500)


    np.save("./census/data/calib_un_risk_x.npy",calib_un_x)
    np.save("./census/data/calib_un_risk_y.npy",calib_un_y)

    # np.save("./census/data/calib_over_risk_x.npy",calib_over_x)
    # np.save("./census/data/calib_over_risk_y.npy",calib_over_y)

    sense_model.train_sense_model(train_x,train_y)
    # xgb_sense_model=create_sense_model()
    # xgb_sense_model_np=train_sense_model(xgb_sense_model,train_x,train_y)


    with open("./census/saved_models/risk_advisor_model_v2.pkl",'wb') as output:
        pickle.dump(sense_model,output)



def train_sense_model():
    start=time.time()

    train_x=np.load("./census/data/sense_census_train_x.npy")
    train_y=np.load("./census/data/sense_census_train_y.npy")
    train_loss=np.load("./census/data/census_train_loss.npy")

    val_x=np.load("./census/data/sense_census_val_x.npy")
    val_y=np.load("./census/data/sense_census_val_y.npy")
    val_loss=np.load("./census/data/census_val_loss.npy")

    sense_model=sensitivity_model()


    train_x,train_y,train_loss,calib_over_x,calib_over_y,calib_loss,calib_un_x,calib_un_y,calib_un_loss=sense_model.gen_calib_data(train_x,train_y,train_loss,balance=True,test_size=500)


    np.save("./census/data/calib_un_x.npy",calib_un_x)
    np.save("./census/data/calib_un_y.npy",calib_un_y)
    np.save("./census/data/calib_un_loss.npy",calib_un_loss)

    np.save("./census/data/calib_over_x.npy",calib_over_x)
    np.save("./census/data/calib_over_y.npy",calib_over_y)
    np.save("./census/data/calib_over_loss.npy",calib_loss)

    sense_model.train_sense_model(train_x,train_y)
    # xgb_sense_model=create_sense_model()
    # xgb_sense_model_np=train_sense_model(xgb_sense_model,train_x,train_y)



    train_logits=sense_model.gen_logits(train_x)
    val_logits=sense_model.gen_logits(val_x)
    calib_logits=sense_model.gen_logits(calib_over_x)


    train_data=custom_torch_dataset(train_x,train_logits,train_y)
    val_data=custom_torch_dataset(val_x,val_logits,val_y)
    calib_data=custom_torch_dataset(calib_over_x,calib_logits,calib_over_y)



    # sense_model.lin_model,sense_model.scaler=train_temp_scaler(calib_data,train_data)


    best_lin=None
    best_scaler=None
    min_uce_loss=1000000
    uloss   = uncer_loss()
    for tr in range(5):
        sense_model.lin_model,sense_model.scaler=train_temp_scaler(calib_data,train_data)
        if sense_model.lin_model is None or sense_model.scaler is None:
            print("Bad temp scaling training")
            continue
        if best_lin is None:
            best_lin=sense_model.lin_model
            best_scaler=sense_model.scaler
        logits_temp = sense_model.inf(train_x, apply_temp=True)
        loss_tensor, err_bin,ent_bin = uloss.uceloss(
        torch.as_tensor(logits_temp, dtype=torch.float32),
        torch.as_tensor(train_y, dtype=torch.float32),n_bins=10
        )
        print("Trial UCE LOSS :",loss_tensor.item())
        if loss_tensor.item()<=min_uce_loss:
            min_uce_loss=loss_tensor.item()
            best_lin=sense_model.lin_model
            best_scaler=sense_model.scaler
        print("Best UCE so far.... :",loss_tensor.item())

    sense_model.lin_model=best_lin
    sense_model.scaler=best_scaler

    print("Trained scaler")
    with open("./census/saved_models/sense_model_v2.pkl",'wb') as output:
        pickle.dump(sense_model,output)


