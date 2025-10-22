from census.train_basic_classifier import train_model,test_model
import numpy as np
from utils import torch_dataset,run_inference_dataset, combine_xy_with_transition,generate_exponential_decay_array, add_chunks_periodically_separate,unison_shuffled_copies
import torch
from census.gen_data import generate_data_drift
from census.train_basic_classifier import model_inf,calc_loss
import pandas as pd
from sensitivity.gen_train_data import build_sensitivity_training_set,build_sensitivity_training_ddla
from arrival.gen_time import add_timestamps_simple
import copy
from sklearn.model_selection import train_test_split
from census_sense import train_risk_model,train_sense_model

def gen_eval_labels(model):
    train_x=np.load("./census/data/census_train_x.npy",allow_pickle=True)
    train_y=np.load("./census/data/census_train_y.npy",allow_pickle=True)
    

    val_x=np.load("./census/data/census_val_x.npy",allow_pickle=True)
    val_y=np.load("./census/data/census_val_y.npy",allow_pickle=True)

    train_data=torch_dataset(train_x,train_y)
    pred,loss,_=run_inference_dataset(model,model_inf,train_data,calc_loss)
    sense_train_x,sense_train_y=build_sensitivity_training_ddla(train_x,pred,loss,train_y,problem_type='class')
    
    val_data=torch_dataset(val_x,val_y)
    pred_val,loss_val,_=run_inference_dataset(model,model_inf,val_data,calc_loss)


    sense_val_x,sense_val_y=build_sensitivity_training_ddla(val_x,pred_val,loss_val,val_y,problem_type='class',val=True,train_losses=loss)

    
    return sense_train_x,sense_train_y,sense_val_x,sense_val_y

if __name__=='__main__':

    df=pd.read_csv("./census/data/adult.csv")
    generate_data_drift(df)

    train_x=np.load("./census/data/census_train_x.npy",allow_pickle=True)
    train_y=np.load("./census/data/census_train_y.npy",allow_pickle=True)

    drifted_x=np.load("./census/data/census_val_x.npy",allow_pickle=True)
    drifted_y=np.load("./census/data/census_val_y.npy",allow_pickle=True)

    hidden_train_idx=np.random.choice(np.arange(len(train_x)),250)
    hidden_drift_idx=np.random.choice(np.arange(len(drifted_x)),1000)
    
    hidden_x=np.concatenate([train_x[hidden_train_idx],drifted_x[hidden_drift_idx]],axis=0)
    hidden_y=np.concatenate([train_y[hidden_train_idx],drifted_y[hidden_drift_idx]],axis=0)

    if len(hidden_x)!=len(hidden_y):
        raise Exception("Hidden data X and Y lengths don't match")

    np.save("./census/data/census_hidden_x.npy",hidden_x)
    np.save("./census/data/census_hidden_y.npy",hidden_y)


    # Add timing information from another dataset

    # ts=np.load('./arrival/data/timestamps.npy')
    # counts=np.load('./arrival/data/counts.npy')

    # val_data=np.concatenate([val_x,np.expand_dims(val_y,axis=1)],axis=1)
    # train_data=np.concatenate([train_x,np.expand_dims(train_y,axis=1)],axis=1)

    
    # df,events=add_timestamps_simple(train_data,val_data,ts,counts)

    # df.to_csv("./census/data/future.csv")

    # event_ts,event_count=np.unique(events,return_counts=True)
    # historical_data=np.hstack([np.expand_dims(event_ts,axis=1),np.expand_dims(event_count,axis=1)])
    
    # np.save("./census/data/historical.npy",historical_data)


    #Scenario 1
    print("Scenario 1")
    # sim_train_x=copy.deepcopy(train_x)
    # sim_train_y=copy.deepcopy(train_y)
    
    sim_train_x=np.load("./census/data/census_train_x.npy",allow_pickle=True)
    sim_train_y=np.load("./census/data/census_train_y.npy",allow_pickle=True)

    drifted_x=np.load("./census/data/census_val_x.npy",allow_pickle=True)
    drifted_y=np.load("./census/data/census_val_y.npy",allow_pickle=True)


    # sim_train_x=np.delete(sim_train_x,hidden_train_idx,axis=0)
    # sim_train_y=np.delete(sim_train_y,hidden_train_idx,axis=0)

    # drifted_x=np.delete(drifted_x,hidden_drift_idx,axis=0)
    # drifted_y=np.delete(drifted_y,hidden_drift_idx,axis=0)

    drifted_x,drifted_y=unison_shuffled_copies(drifted_x,drifted_y)
    sim_train_x,sim_train_y=unison_shuffled_copies(sim_train_x,sim_train_y)
    sim_length=48
    total_data_len=(len(train_x)+len(drifted_x))
    trans_window=2
    trans_data=int((total_data_len/sim_length)*trans_window)
    sim_x,sim_y=combine_xy_with_transition(sim_train_x,sim_train_y,drifted_x,drifted_y,trans_data)
    # sim_x=sim_data[:,0]
    # sim_y=sim_data[:,1]

    # sim_x=np.concatenate([train_x,test_x,drifted_x],axis=0)
    # sim_y=np.concatenate([train_y,test_y,drifted_y])

    ts=np.load('./arrival/data/timestamps.npy')
    counts=np.load('./arrival/data/counts.npy')
    
    df,events=add_timestamps_simple(sim_x,sim_y,ts,counts)

    df.to_csv("./census/data/future_1.csv")

    train_x=np.load("./census/data/census_train_x.npy",allow_pickle=True)
    train_y=np.load("./census/data/census_train_y.npy",allow_pickle=True)

    drifted_x=np.load("./census/data/census_val_x.npy",allow_pickle=True)
    drifted_y=np.load("./census/data/census_val_y.npy",allow_pickle=True)

    # train_x=np.delete(train_x,hidden_train_idx,axis=0)
    # train_y=np.delete(train_y,hidden_train_idx,axis=0)

    # drifted_x=np.delete(drifted_x,hidden_drift_idx,axis=0)
    # drifted_y=np.delete(drifted_y,hidden_drift_idx,axis=0)
    
    drifted_x,drifted_y=unison_shuffled_copies(drifted_x,drifted_y)
    train_x,train_y=unison_shuffled_copies(train_x,train_y)
    #Scenario 2
    print("Scenario 2")
    # probs = np.linspace(1, 0, len(train_x)+len(test_x)+len(drifted_x))
    probs=generate_exponential_decay_array(len(train_x)+len(drifted_x),0.0001)
    sim_x=[]
    sim_y=[]
    train_i=0
    test_i=0
    drifted_i=0
    for i in range(len(train_x)+len(drifted_x)):
        pick=np.random.choice([0,1],1,p=[probs[i],1-probs[i]])
        if pick==0:
            if train_i<len(train_x):
                sim_x.append(train_x[train_i])
                sim_y.append(train_y[train_i])
                train_i+=1
            elif drifted_i<len(drifted_x):
                sim_x.append(drifted_x[drifted_i])
                sim_y.append(drifted_y[drifted_i])
                drifted_i+=1
        elif pick==1:
            if drifted_i<len(drifted_x):
                sim_x.append(drifted_x[drifted_i])
                sim_y.append(drifted_y[drifted_i])
                drifted_i+=1
            elif train_i<len(train_x):
                sim_x.append(train_x[train_i])
                sim_y.append(train_y[train_i])
                train_i+=1
    
    sim_x=np.array(sim_x)
    sim_y=np.array(sim_y)

    ts=np.load('./arrival/data/timestamps.npy')
    counts=np.load('./arrival/data/counts.npy')
    
    df,events=add_timestamps_simple(sim_x,sim_y,ts,counts)

    df.to_csv("./census/data/future_2.csv")
    # np.save("./census/data/future_new_imgs_2.npy",sim_x)



    #Scenario 3
    print("Scenario 3")
    sim_train_x=np.load("./census/data/census_train_x.npy",allow_pickle=True)
    sim_train_y=np.load("./census/data/census_train_y.npy",allow_pickle=True)

    drifted_x=np.load("./census/data/census_val_x.npy",allow_pickle=True)
    drifted_y=np.load("./census/data/census_val_y.npy",allow_pickle=True)

    # sim_train_x=np.delete(sim_train_x,hidden_train_idx,axis=0)
    # sim_train_y=np.delete(sim_train_y,hidden_train_idx,axis=0)

    # drifted_x=np.delete(drifted_x,hidden_drift_idx,axis=0)
    # drifted_y=np.delete(drifted_y,hidden_drift_idx,axis=0)
    
    drifted_x,drifted_y=unison_shuffled_copies(drifted_x,drifted_y)
    sim_train_x,sim_train_y=unison_shuffled_copies(sim_train_x,sim_train_y)

    sim_x,sim_y=add_chunks_periodically_separate(sim_train_x,sim_train_y,drifted_x,drifted_y,5000,10000)


    ts=np.load('./arrival/data/timestamps.npy')
    counts=np.load('./arrival/data/counts.npy')
    
    df,events=add_timestamps_simple(sim_x,sim_y,ts,counts)

    df.to_csv("./census/data/future_3.csv")
    # np.save("./census/data/future_new_imgs_3.npy",sim_x)


    #Scenario 4
    sim_train_x=np.load("./census/data/census_train_x.npy",allow_pickle=True)
    sim_train_y=np.load("./census/data/census_train_y.npy",allow_pickle=True)
    
    drifted_x=np.load("./census/data/census_val_x.npy",allow_pickle=True)
    drifted_y=np.load("./census/data/census_val_y.npy",allow_pickle=True)

    # sim_train_x=np.delete(sim_train_x,hidden_train_idx,axis=0)
    # sim_train_y=np.delete(sim_train_y,hidden_train_idx,axis=0)

    # drifted_x=np.delete(drifted_x,hidden_drift_idx,axis=0)
    # drifted_y=np.delete(drifted_y,hidden_drift_idx,axis=0)

    new_len=len(train_x)+len(drifted_x)
    train_len=len(sim_train_x)
    reps=(new_len//len(sim_train_x))

    sim_x=np.load("./census/data/census_train_x.npy",allow_pickle=True)
    sim_y=np.load("./census/data/census_train_y.npy",allow_pickle=True)
    for r in range(reps):
        shuffled_idx=np.random.choice(np.arange(train_len),train_len,replace=False)
        new_sim_x=sim_train_x[shuffled_idx]
        new_sim_y=sim_train_y[shuffled_idx]
        sim_x=np.concatenate([sim_x,new_sim_x],axis=0)
        sim_y=np.concatenate([sim_y,new_sim_y])
    
    sim_x=sim_x[:new_len]
    sim_y=sim_y[:new_len]

    ts=np.load('./arrival/data/timestamps.npy')
    counts=np.load('./arrival/data/counts.npy')

    df,events=add_timestamps_simple(sim_x,sim_y,ts,counts)
    print(len(df))
    df.to_csv("./census/data/future_4.csv")
    # np.save("./census/data/future_imgs_4.npy",sim_x)

    

    event_ts,event_count=np.unique(events,return_counts=True)
    historical_data=np.hstack([np.expand_dims(event_ts,axis=1),np.expand_dims(event_count,axis=1)])
    
    np.save("./census/data/historical.npy",historical_data)


    train_x=np.load("./census/data/census_train_x.npy",allow_pickle=True)
    train_y=np.load("./census/data/census_train_y.npy",allow_pickle=True)

    val_x=np.load("./census/data/census_val_x.npy",allow_pickle=True)
    val_y=np.load("./census/data/census_val_y.npy",allow_pickle=True)

    # train_x=np.delete(train_x,hidden_train_idx,axis=0)
    # train_y=np.delete(train_y,hidden_train_idx,axis=0)

    # val_x=np.delete(val_x,hidden_drift_idx,axis=0)
    # val_y=np.delete(val_y,hidden_drift_idx,axis=0)

    # np.save("./census/data/census_new_train_x.npy",train_x)
    # np.save("./census/data/census_new_train_y.npy",train_y)

    # np.save("./census/data/census_new_val_x.npy",val_x)
    # np.save("./census/data/census_new_val_y.npy",val_y)


    #Train classifier
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
    train_data=torch_dataset(train_x,train_y)
    test_data=torch_dataset(test_x,test_y)
    val_data=torch_dataset(val_x,val_y)

    eval_data=torch_dataset(hidden_x,hidden_y)
    model_path="./census/saved_models/census_classifier_v1.pt"


    train_model(train_data,test_data,model_path,epochs=100)
    model=torch.load(model_path)
    print(test_model(val_data,model))
    print("EVALUATION ON THE HIDDEN SET")
    print(test_model(eval_data,model))
    #Generate training set for sensitivity model

    pred,loss,_=run_inference_dataset(model,model_inf,train_data,calc_loss)

    sense_train_x,sense_train_y=build_sensitivity_training_set(train_x,pred,loss,train_y,problem_type='reg')

    np.save("./census/data/sense_census_train_x.npy",sense_train_x)
    np.save("./census/data/sense_census_train_y.npy",sense_train_y)
    np.save("./census/data/census_train_loss.npy",loss)



    pred_val,loss_val,_=run_inference_dataset(model,model_inf,val_data,calc_loss)


    sense_val_x,sense_val_y=build_sensitivity_training_set(val_x,pred_val,loss_val,val_y,problem_type='reg',val=True,train_losses=loss)

    np.save("./census/data/sense_census_val_x.npy",sense_val_x)
    np.save("./census/data/sense_census_val_y.npy",sense_val_y)
    np.save("./census/data/census_val_loss.npy",loss_val)

    ddla_train_x,ddla_train_y,ddla_val_x,ddla_val_y=gen_eval_labels(model)
    np.save("./census/data/ddla_train_x.npy",ddla_train_x)
    np.save("./census/data/ddla_train_y.npy",ddla_train_y)
    np.save("./census/data/ddla_val_x.npy",ddla_val_x)
    np.save("./census/data/ddla_val_y.npy",ddla_val_y)

    #Train Sensitivity and Risk Model
    train_sense_model()
    train_risk_model()