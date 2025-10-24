'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from cifar10.train_image_classifier import train_img_model,model_inf,calc_loss
from sensitivity.gen_train_data import build_sensitivity_training_set
from cifar10.gen_data import gen_data
import os
import argparse
import numpy as np
from utils import image_dataset,run_inference_dataset,generate_exponential_decay_array,combine_xy_with_transition,add_chunks_periodically_separate
from cifar10.models.resnet import ResNet18
from sensitivity.gen_train_data import generate_image_embeddings_v2
from arrival.gen_time import add_timestamps_images



def setup_data():
    trainset = torchvision.datasets.CIFAR10(
        root='./cifar10/data', train=True, download=True)

    testset = torchvision.datasets.CIFAR10(
        root='./cifar10/data', train=False, download=True)
    
    train_x=trainset.data
    train_y=trainset.targets


    test_x=testset.data
    test_y=testset.targets

    gen_data(train_x,test_x,train_y,test_y)





def main():
    print('==> Preparing data..')

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

    train_x=np.load("./cifar10/data/train_x.npy")
    train_y=np.load("./cifar10/data/train_y.npy")

    test_x=np.load("./cifar10/data/test_x.npy")
    test_y=np.load("./cifar10/data/test_y.npy")


    drifted_x=np.load("./cifar10/data/drifted_x.npy")
    drifted_y=np.load("./cifar10/data/drifted_y.npy")

    #Scenario 1
    print("Scenario 1")
    sim_train_x=np.concatenate([train_x,test_x],axis=0)
    sim_train_y=np.concatenate([train_y,test_y],axis=0)
    sim_length=48
    total_data_len=(len(train_x)+len(test_x)+len(drifted_x))
    trans_window=2
    trans_data=int((total_data_len/sim_length)*trans_window)
    sim_x,sim_y=combine_xy_with_transition(sim_train_x,sim_train_y,drifted_x,drifted_y,trans_data)
    # sim_x=sim_data[:,0]
    # sim_y=sim_data[:,1]

    # sim_x=np.concatenate([train_x,test_x,drifted_x],axis=0)
    # sim_y=np.concatenate([train_y,test_y,drifted_y])

    ts=np.load('./arrival/data/timestamps.npy')
    counts=np.load('./arrival/data/counts.npy')
    
    df,events=add_timestamps_images(sim_x,sim_y,ts,counts,future_hours=270)

    df.to_csv("./cifar10/data/future_long_scenario_1.csv")
    np.save("./cifar10/data/future_long_scenario_1_imgs.npy",sim_x)

    #Scenario 2
    print("Scenario 2")
    # probs = np.linspace(1, 0, len(train_x)+len(test_x)+len(drifted_x))
    probs=generate_exponential_decay_array(len(train_x)+len(test_x)+len(drifted_x),0.00001)
    sim_x=[]
    sim_y=[]
    train_i=0
    test_i=0
    drifted_i=0
    for i in range(len(train_x)+len(test_x)+len(drifted_x)):
        pick=np.random.choice([0,1],1,p=[probs[i],1-probs[i]])
        if pick==0:
            if train_i<len(train_x):
                sim_x.append(train_x[train_i])
                sim_y.append(train_y[train_i])
                train_i+=1
            elif test_i<len(test_x):
                sim_x.append(test_x[test_i])
                sim_y.append(test_y[test_i])
                test_i+=1
            elif drifted_i<len(drifted_x):
                sim_x.append(drifted_x[drifted_i])
                sim_y.append(drifted_y[drifted_i])
                drifted_i+=1
        elif pick==1:
            sim_x.append(drifted_x[drifted_i])
            sim_y.append(drifted_y[drifted_i])
            drifted_i+=1
    
    sim_x=np.array(sim_x)
    sim_y=np.array(sim_y)

    ts=np.load('./arrival/data/timestamps.npy')
    counts=np.load('./arrival/data/counts.npy')
    
    df,events=add_timestamps_images(sim_x,sim_y,ts,counts,future_hours=270)

    df.to_csv("./cifar10/data/future_long_scenario_2.csv")
    np.save("./cifar10/data/future_long_scenario_2_imgs.npy",sim_x)

    #Scenario 3
    print("Scenario 3")
    sim_train_x=np.concatenate([train_x,test_x],axis=0)
    sim_train_y=np.concatenate([train_y,test_y])

    sim_x,sim_y=add_chunks_periodically_separate(sim_train_x,sim_train_y,drifted_x,drifted_y,10000,20000)


    ts=np.load('./arrival/data/timestamps.npy')
    counts=np.load('./arrival/data/counts.npy')
    
    df,events=add_timestamps_images(sim_x,sim_y,ts,counts,future_hours=270)

    df.to_csv("./cifar10/data/future_long_scenario_3.csv")
    np.save("./cifar10/data/future_long_scenario_3_imgs.npy",sim_x)

    #Scenario 4
    sim_train_x=np.concatenate([train_x,test_x],axis=0)
    sim_train_y=np.concatenate([train_y,test_y])
    new_len=len(train_x)+len(test_x)+len(drifted_x)
    train_len=len(sim_train_x)
    reps=(new_len//len(sim_train_x))

    sim_x=np.concatenate([train_x,test_x],axis=0)
    sim_y=np.concatenate([train_y,test_y])
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

    df,events=add_timestamps_images(sim_x,sim_y,ts,counts,future_hours=270)
    print(len(df))
    df.to_csv("./cifar10/data/future_long_scenario_4.csv")
    np.save("./cifar10/data/future_long_scenario_4_imgs.npy",sim_x)

    

    event_ts,event_count=np.unique(events,return_counts=True)
    historical_data=np.hstack([np.expand_dims(event_ts,axis=1),np.expand_dims(event_count,axis=1)])
    
    np.save("./cifar10/data/historical_data.npy",historical_data)

    train_x=np.load("./cifar10/data/train_x.npy")
    train_y=np.load("./cifar10/data/train_y.npy")

    test_x=np.load("./cifar10/data/test_x.npy")
    test_y=np.load("./cifar10/data/test_y.npy")


    drifted_x=np.load("./cifar10/data/drifted_x.npy")
    drifted_y=np.load("./cifar10/data/drifted_y.npy")

    trainset = image_dataset(train_x,train_y,transform=transform_train)
    testset = image_dataset(test_x,test_y,transform=transform_test)

    driftset=image_dataset(drifted_x,drifted_y,transform=transform_test)

    # net=train_img_model(trainset,testset,driftset,epochs=100)
    # net.eval()

    device='cuda'

    net=ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # state=torch.load("./cifar10/checkpoint/ckpt_ablation.pth")
    net=train_img_model(trainset,testset,driftset,epochs=100)
    net.eval()
    state=torch.load("./cifar10/checkpoint/cifar_image_classifier.pth")
    net.load_state_dict(state['net'])
    net.eval()


    train_pred,train_loss,_=run_inference_dataset(net,model_inf,trainset,calc_loss,neural=True)
    train_pred=train_pred.squeeze(1)
    train_max_preds=np.argmax(train_pred,axis=1)
    train_max_preds=np.expand_dims(train_max_preds, axis=1)

    sense_trainset=image_dataset(train_x,train_y)
    sense_train_features=generate_image_embeddings_v2(sense_trainset)

    np.save("./cifar10/data/cifar_train_loss.npy",train_loss)
    np.save("./cifar10/data/sense_cifar_train_feat.npy",sense_train_features)
    np.save("./cifar10/data/cifar_train_pred.npy",train_max_preds)

    test_pred,test_loss,_=run_inference_dataset(net,model_inf,testset,calc_loss,neural=True)
    test_pred=test_pred.squeeze(1)
    test_max_preds=np.argmax(test_pred,axis=1)
    test_max_preds=np.expand_dims(test_max_preds, axis=1)

    sense_testset=image_dataset(test_x,test_y)
    sense_test_features=generate_image_embeddings_v2(sense_testset)

    np.save("./cifar10/data/cifar_test_loss.npy",test_loss)
    np.save("./cifar10/data/sense_cifar_test_feat.npy",sense_test_features)
    np.save("./cifar10/data/cifar_test_pred.npy",test_max_preds)

    val_pred,val_loss,_=run_inference_dataset(net,model_inf,driftset,calc_loss,neural=True)
    val_pred=val_pred.squeeze(1)
    val_max_preds=np.argmax(val_pred,axis=1)
    val_max_preds=np.expand_dims(val_max_preds, axis=1)

    sense_valset=image_dataset(drifted_x,drifted_y)
    sense_val_features=generate_image_embeddings_v2(sense_valset)

    np.save("./cifar10/data/cifar_val_loss.npy",val_loss)
    np.save("./cifar10/data/sense_cifar_val_feat.npy",sense_val_features)
    np.save("./cifar10/data/cifar_val_pred.npy",val_max_preds)


if __name__=="__main__":
    setup_data()
    main()