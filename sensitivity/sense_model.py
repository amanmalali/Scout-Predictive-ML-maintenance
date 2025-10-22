import xgboost as xgb
import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import torch
import ray

n_estimators=[400,600,1000,600,800]
lr=[0.01,0.01,0.01,0.01,0.01] #Might want to change this
max_depth=[10,10,10,5,5]
subsample=[0.7,0.6,0.7,0.5,0.7]
tree_method=['hist']
grow_policy=['depthwise','lossguide','depthwise','lossguide','lossguide']
min_child_weight=[5,5,0.5,5,0.2]

@ray.remote(num_gpus=0.1)
def train_xg(model,train_x,train_y,classes_weights):
    model.fit(train_x,train_y,sample_weight=classes_weights)
    return True


class sensitivity_model:
    def __init__(self,objective=None) -> None:
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        if objective is None:
            self.xgb_models=self.create_sense_model()
        else:
            self.xgb_models=self.create_sense_model(objective=objective)
        self.sense_model=None
        self.lin_model=None
        self.scaler=None

    def create_sense_model(self,num_models=5,objective='multi:softmax'):
        xgb_models=[]
        for i in range(num_models):
            clf=xgb.XGBClassifier(device=self.device,n_estimators=n_estimators[i],learning_rate=lr[i],
                                max_depth=max_depth[i],subsample=subsample[i],tree_method='hist',
                                grow_policy=grow_policy[i],min_child_weight=min_child_weight[i],
                                n_jobs=-1,objective='multi:softmax',num_class=2)
            # calib_clf=CalibratedClassifierCV(estimator=clf,method='isotonic',n_jobs=-1)
            # calib_clf.fit(train_x,train_y)
            # conf_clf=MapieClassifier(estimator=clf,method="aps",n_jobs=-1)
            xgb_models.append(("clf"+str(i),clf))
        
        return xgb_models


    def gen_calib_data(self,train_x,train_y,train_loss,balance=False,test_size=0.02):

        train_y=train_y.reshape(-1,1)
        train_loss=train_loss.reshape(-1,1)
        train_labels=np.concatenate([train_y,train_loss],axis=1)
        train_x, calib_x, train_labels, calib_labels = train_test_split(train_x, train_labels, test_size=test_size, shuffle=True)
        train_x, calib_un_x, train_labels, calib_un_labels = train_test_split(train_x, train_labels, test_size=test_size, shuffle=True)
        train_y=train_labels[:,0]
        train_loss=train_labels[:,1]

        calib_y=calib_labels[:,0]
        calib_loss=calib_labels[:,1]

        calib_un_y=calib_un_labels[:,0]
        calib_un_loss=calib_un_labels[:,1]

        if balance:
            ros = RandomOverSampler()#random_state=42)
            calib_over_x , calib_over_y = ros.fit_resample(calib_x, calib_y)
        else:
            calib_over_x=calib_x
            calib_over_y=calib_y
        return train_x,train_y,train_loss,calib_over_x,calib_over_y,calib_loss,calib_un_x,calib_un_y,calib_un_loss



    def gen_logits(self,input_x):
        model_logits=[]
        num_models=len(self.sense_model)
        for n in range(num_models):
            logits=np.array(self.sense_model[n][1].predict(input_x,output_margin=True))
            model_logits.append(logits)
        model_logits=np.array(model_logits)
        logits_preds=[]
        for row in range(len(input_x)):
            logits=model_logits[:,row,:]
            logits_preds.append(logits)
        logits_preds=np.array(logits_preds)

        return logits_preds


    def train_sense_model(self,train_x,train_y):
        classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=train_y
        )
        for i in range(len(self.xgb_models)):
            print("Training model :",i)
            self.xgb_models[i][1].fit(train_x,train_y,sample_weight=classes_weights)
        xgb_models_np=np.array(self.xgb_models)
        self.sense_model=xgb_models_np


    def train_sense_model_unbalanced(self,train_x,train_y):
        for i in range(len(self.xgb_models)):
            print("Training model :",i)
            self.xgb_models[i][1].fit(train_x,train_y)
        xgb_models_np=np.array(self.xgb_models)
        self.sense_model=xgb_models_np

    def train_sense_model_ray(self,train_x,train_y):
        classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=train_y
        )

        futures= [train_xg.remote(self.xgb_models[i][1],train_x,train_y,classes_weights) for i in range(len(self.xgb_models))]
        # for i in range(len(self.xgb_models)):
        #     print("Training model :",i)
        #     self.xgb_models[i][1].fit(train_x,train_y,sample_weight=classes_weights)
        # xgb_models_np=np.array(self.xgb_models)
        # self.sense_model=xgb_models_np

        model_status=ray.get(futures)
        print(model_status)


    def inf(self,X,apply_temp=True,mode="feat"):
        logits=self.gen_logits(X)
        if mode=="feat":
            if apply_temp:
                new_temps=self.lin_model(torch.tensor(X,dtype=torch.float32))
                scaled_logits=self.scaler.forward_ext_temp(torch.tensor(logits,dtype=torch.float32),new_temps)
            else:
                scaled_logits=torch.tensor(logits)
        else:
            if apply_temp:
                scaled_logits=self.scaler.forward(torch.tensor(logits,dtype=torch.float32))
            else:
                scaled_logits=torch.tensor(logits)
        
        return scaled_logits