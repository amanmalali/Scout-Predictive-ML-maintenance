from .models.basic_classifier import basic_classifier
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import copy
# from inference import run_inference_dataset


# df=pd.read_csv("/Users/aragh/project/testing/dataverse_files/mixed_0101_abrupto.csv")
# df=df[:10000]



def train_model(data,test_data,model_path,lr=0.001,epochs=100,output=1):
    train_dataloader=DataLoader(data,batch_size=32,shuffle=True)
    class_model=basic_classifier(input_size=data.X.shape[1],output_size=output)
    optim=torch.optim.Adam(class_model.parameters(),lr=lr)
    loss_fn = nn.BCELoss()
    best_model=None
    best_accuracy=0
    best_loss=1000000000
    for num_epoch in range(epochs):
        loss_total=0
        acc_total=0
        for x,y in train_dataloader:
            class_model.train()
            optim.zero_grad()
            y_pred=class_model(x)
            # print(y_pred)
            y_pred=torch.squeeze(y_pred)
            # print(y_pred)
            try:
                loss=loss_fn(y_pred,y)
            except:
                print(y_pred)
                print(y)
                continue
            loss.backward()
            optim.step()
            loss_total+=loss.item()*x.shape[0]
            acc=(y_pred.round()==y).float().mean()
            acc_total+=acc*x.shape[0]
        
        accuracy=acc_total/len(train_dataloader.dataset)
        total_loss=loss_total/len(train_dataloader.dataset)
        test_accuracy,test_loss=test_model(test_data,class_model)
        
        if test_loss<best_loss:
            best_loss=test_loss
            best_model=copy.deepcopy(class_model.state_dict())

        # if test_accuracy>=best_accuracy:
        #     best_accuracy=test_accuracy
        #     best_model=copy.deepcopy(class_model.state_dict())

        print("Loss for epoch :{} is {} and accuracy {}".format(num_epoch,total_loss,accuracy))
    
    class_model.load_state_dict(best_model)
    torch.save(class_model,model_path)
    print("Model saved!")
    


def test_model(data,model):
    val_dataloader=DataLoader(data,batch_size=64,shuffle=True)
    acc_total=0
    loss_fn = nn.BCELoss()
    loss_total=0
    for x,y in val_dataloader:
        model.eval()
        y_pred=model(x)
        y_pred=torch.squeeze(y_pred)
        loss=loss_fn(y_pred,y)
        loss_total+=loss.item()*x.shape[0]
        acc=(y_pred.round()==y).float().mean()
        acc_total+=acc*x.shape[0]
        
    accuracy=acc_total/len(val_dataloader.dataset)
    l=loss_total/len(val_dataloader.dataset)
    print("Testing model : Accuracy {}  Loss {}".format(accuracy,l))
    return accuracy,l


def model_inf(model,X):
    x=torch.tensor(X,dtype=torch.float32)
    pred=model(x)
    return pred

def calc_loss(y_pred,y):
    y=torch.tensor([y],dtype=torch.float32)
    loss_fn = nn.BCELoss()
    loss=loss_fn(y_pred,y)
    return loss


# model.eval()
# def test_inf(X):
#     x=torch.tensor(X)
#     pred=model(x)
#     return pred

# def test_loss_func(y_pred,y):
#     y=torch.tensor([y])
#     l=loss_fn(y_pred,y)
#     return l


# pred,loss_inf,_=run_inference_dataset(test_inf,data,test_loss_func)

