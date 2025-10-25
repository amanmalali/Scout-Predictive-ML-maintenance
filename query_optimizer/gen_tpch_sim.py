import numpy as np
from utils import torch_dataset,run_inference_dataset, combine_xy_with_transition,generate_exponential_decay_array, add_chunks_periodically_separate,unison_shuffled_copies
import torch
import pandas as pd
from sensitivity.gen_train_data import build_sensitivity_training_set
from arrival.gen_time import add_timestamps_simple
import copy
from sklearn.model_selection import train_test_split


if __name__=='__main__':


    df_train=pd.read_csv('./data/train.csv')
    df_test=pd.read_csv('./data/test.csv')
    df_val=pd.read_csv('./data/val.csv')
    df_full=pd.read_csv('./data/full_data.csv')
    

    #Scenario 1
    print("Scenario 1")
    # sim_train_x=copy.deepcopy(train_x)
    # sim_train_y=copy.deepcopy(train_y)
    
    sim_train_x=df_train.values
    sim_train_y=df_train['latency'].values

    sim_test_x=df_test.values
    sim_test_y=df_test['latency'].values

    sim_train_x=np.concatenate([sim_train_x,sim_test_x])
    sim_train_y=np.concatenate([sim_train_y,sim_test_y])

    drifted_x=df_val.values
    drifted_y=df_val['latency'].values



    drifted_x,drifted_y=unison_shuffled_copies(drifted_x,drifted_y)
    sim_train_x,sim_train_y=unison_shuffled_copies(sim_train_x,sim_train_y)
    sim_length=48
    total_data_len=(len(sim_train_x)+len(drifted_x))
    trans_window=2
    trans_data=int((total_data_len/sim_length)*trans_window)
    sim_x,sim_y=combine_xy_with_transition(sim_train_x,sim_train_y,drifted_x,drifted_y,trans_data)

    ts=np.load('./arrival/data/timestamps.npy')
    counts=np.load('./arrival/data/counts.npy')
    
    df,events=add_timestamps_simple(sim_x,sim_y,ts,counts,future_hours=160)

    df.to_csv("./data/future_long_1.csv")

    df_train=pd.read_csv('./data/train.csv')
    df_test=pd.read_csv('./data/test.csv')
    df_val=pd.read_csv('./data/val.csv')
    df_full=pd.read_csv('./data/full_data.csv')
    

    sim_train_x=df_train.values
    sim_train_y=df_train['latency'].values

    sim_test_x=df_test.values
    sim_test_y=df_test['latency'].values

    sim_train_x=np.concatenate([sim_train_x,sim_test_x])
    sim_train_y=np.concatenate([sim_train_y,sim_test_y])

    drifted_x=df_val.values
    drifted_y=df_val['latency'].values

    drifted_x,drifted_y=unison_shuffled_copies(drifted_x,drifted_y)
    sim_train_x,sim_train_y=unison_shuffled_copies(sim_train_x,sim_train_y)
    #Scenario 2
    print("Scenario 2")
    # probs = np.linspace(1, 0, len(train_x)+len(test_x)+len(drifted_x))
    # probs=generate_exponential_decay_array(len(sim_train_x)+len(drifted_x),0.0001)
    probs=np.linspace(1, 0, len(sim_train_x)+len(drifted_x))
    sim_x=[]
    sim_y=[]
    train_i=0
    test_i=0
    drifted_i=0
    for i in range(len(sim_train_x)+len(drifted_x)):
        pick=np.random.choice([0,1],1,p=[probs[i],1-probs[i]]) #Changed
        # print(pick)
        if pick==0:
            if train_i<len(sim_train_x):
                sim_x.append(sim_train_x[train_i])
                sim_y.append(sim_train_y[train_i])
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
            elif train_i<len(sim_train_x):
                sim_x.append(sim_train_x[train_i])
                sim_y.append(sim_train_y[train_i])
                train_i+=1
    
    sim_x=np.array(sim_x)
    sim_y=np.array(sim_y)

    ts=np.load('./arrival/data/timestamps.npy')
    counts=np.load('./arrival/data/counts.npy')
    
    df,events=add_timestamps_simple(sim_x,sim_y,ts,counts,future_hours=160)

    df.to_csv("./data/future_long_2.csv")
    # # np.save("./census/data/future_new_imgs_2.npy",sim_x)



    #Scenario 3
    print("Scenario 3")
    df_train=pd.read_csv('./data/train.csv')
    df_test=pd.read_csv('./data/test.csv')
    df_val=pd.read_csv('./data/val.csv')
    df_full=pd.read_csv('./data/full_data.csv')
    

    sim_train_x=df_train.values
    sim_train_y=df_train['latency'].values

    sim_test_x=df_test.values
    sim_test_y=df_test['latency'].values

    sim_train_x=np.concatenate([sim_train_x,sim_test_x])
    sim_train_y=np.concatenate([sim_train_y,sim_test_y])

    drifted_x=df_val.values
    drifted_y=df_val['latency'].values

    # sim_train_x=np.delete(sim_train_x,hidden_train_idx,axis=0)
    # sim_train_y=np.delete(sim_train_y,hidden_train_idx,axis=0)

    # drifted_x=np.delete(drifted_x,hidden_drift_idx,axis=0)
    # drifted_y=np.delete(drifted_y,hidden_drift_idx,axis=0)
    
    drifted_x,drifted_y=unison_shuffled_copies(drifted_x,drifted_y)
    sim_train_x,sim_train_y=unison_shuffled_copies(sim_train_x,sim_train_y)

    sim_x,sim_y=add_chunks_periodically_separate(sim_train_x,sim_train_y,drifted_x,drifted_y,10000,20000)


    ts=np.load('./arrival/data/timestamps.npy')
    counts=np.load('./arrival/data/counts.npy')
    
    df,events=add_timestamps_simple(sim_x,sim_y,ts,counts,future_hours=160)

    df.to_csv("./data/future_long_3.csv")
    # np.save("./census/data/future_new_imgs_3.npy",sim_x)


    #Scenario 4
    df_train=pd.read_csv('./data/train.csv')
    df_test=pd.read_csv('./data/test.csv')
    df_val=pd.read_csv('./data/val.csv')
    df_full=pd.read_csv('./data/full_data.csv')
    

    sim_train_x=df_train.values
    sim_train_y=df_train['latency'].values

    sim_test_x=df_test.values
    sim_test_y=df_test['latency'].values

    sim_train_x=np.concatenate([sim_train_x,sim_test_x])
    sim_train_y=np.concatenate([sim_train_y,sim_test_y])

    drifted_x=df_val.values
    drifted_y=df_val['latency'].values

    # sim_train_x=np.delete(sim_train_x,hidden_train_idx,axis=0)
    # sim_train_y=np.delete(sim_train_y,hidden_train_idx,axis=0)

    # drifted_x=np.delete(drifted_x,hidden_drift_idx,axis=0)
    # drifted_y=np.delete(drifted_y,hidden_drift_idx,axis=0)

    new_len=len(sim_train_x)+len(drifted_x)
    train_len=len(sim_train_x)
    reps=(new_len//len(sim_train_x))

    sim_x=sim_train_x
    sim_y=sim_train_y
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

    df,events=add_timestamps_simple(sim_x,sim_y,ts,counts,future_hours=160)
    print(len(df))
    # df.to_csv("./data/future_long_4.csv")
    # np.save("./census/data/future_imgs_4.npy",sim_x)

    

    event_ts,event_count=np.unique(events,return_counts=True)
    historical_data=np.hstack([np.expand_dims(event_ts,axis=1),np.expand_dims(event_count,axis=1)])
    
    np.save("./data/historical_long.npy",historical_data)

    pred=np.load('./data/train_latency_pred.npy')
    loss=np.load('./data/train_loss.npy')
    train_x=np.load('./data/train_x.npy')
    train_y=np.load('./data/train_latency_y.npy')
    pred=np.expand_dims(pred,axis=1)
    sense_train_x,sense_train_y=build_sensitivity_training_set(train_x,pred,loss,train_y,problem_type='reg')

    np.save("./data/sense_tpch_train_x.npy",sense_train_x)
    np.save("./data/sense_tpch_train_y.npy",sense_train_y)

    pred_test=np.load('./data/test_latency_pred.npy')
    pred_test=np.expand_dims(pred_test,axis=1)
    loss_test=np.load('./data/test_loss.npy')
    test_x=np.load('./data/test_x.npy')
    test_y=np.load('./data/test_latency_y.npy')
    sense_test_x,sense_test_y=build_sensitivity_training_set(test_x,pred_test,loss_test,test_y,problem_type='reg',val=True,train_losses=loss)

    np.save("./data/sense_tpch_test_x.npy",sense_test_x)
    np.save("./data/sense_tpch_test_y.npy",sense_test_y)



    # pred_val,loss_val,_=run_inference_dataset(model,model_inf,val_data,calc_loss)

    pred_val=np.load('./data/val_latency_pred.npy')
    pred_val=np.expand_dims(pred_val,axis=1)
    loss_val=np.load('./data/val_loss.npy')
    val_x=np.load('./data/val_x.npy')
    val_y=np.load('./data/val_latency_y.npy')
    sense_val_x,sense_val_y=build_sensitivity_training_set(val_x,pred_val,loss_val,val_y,problem_type='reg',val=True,train_losses=loss)

    np.save("./data/sense_tpch_val_x.npy",sense_val_x)
    np.save("./data/sense_tpch_val_y.npy",sense_val_y)
