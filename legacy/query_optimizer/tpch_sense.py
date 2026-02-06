import numpy as np
from sensitivity.new_temp_scaler import train_temp_scaler,uncer_loss
from utils import custom_torch_dataset
import time
import pickle
import torch
# from sensitivity.sense_model import create_sense_model,train_sense_model

from metamodel.sense_model import sensitivity_model



start=time.time()

train_x=np.load("./data/sense_tpch_train_x.npy")
train_y=np.load("./data/sense_tpch_train_y.npy")
train_loss=np.load("./data/train_loss.npy")

val_x=np.load("./data/sense_tpch_val_x.npy")
val_y=np.load("./data/sense_tpch_val_y.npy")
val_loss=np.load("./data/val_loss.npy")

sense_model=sensitivity_model()

# with open("./saved_models/sense_model_long.pkl",'rb') as inp:
#     sense_model=pickle.load(inp)

train_x,train_y,train_loss,calib_over_x,calib_over_y,calib_loss,calib_un_x,calib_un_y,calib_un_loss=sense_model.gen_calib_data(train_x,train_y,train_loss,balance=True)


np.save("./data/calib_un_x_temp.npy",calib_un_x)
np.save("./data/calib_un_y_temp.npy",calib_un_y)
np.save("./data/calib_un_loss_temp.npy",calib_un_loss)

np.save("./data/calib_over_x_temp.npy",calib_over_x)
np.save("./data/calib_over_y_temp.npy",calib_over_y)
np.save("./data/calib_over_loss_temp.npy",calib_loss)

sense_model.train_sense_model(train_x,train_y)
# xgb_sense_model=create_sense_model()
# xgb_sense_model_np=train_sense_model(xgb_sense_model,train_x,train_y)



train_logits=sense_model.gen_logits(train_x)
val_logits=sense_model.gen_logits(val_x)
calib_logits=sense_model.gen_logits(calib_over_x)


train_data=custom_torch_dataset(train_x,train_logits,train_y)
val_data=custom_torch_dataset(val_x,val_logits,val_y)
calib_data=custom_torch_dataset(calib_over_x,calib_logits,calib_over_y)


best_lin=None
best_scaler=None
min_uce_loss=1000000
uloss   = uncer_loss()
for tr in range(10):
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
with open("./saved_models/sense_model.pkl",'wb') as output:
    pickle.dump(sense_model,output)


