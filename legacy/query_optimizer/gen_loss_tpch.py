import torch
from torchmetrics.regression import MeanAbsolutePercentageError
import pandas as pd
import numpy as np

def gen_loss_vals(preds,targets):
    all_errors=[]
    mean_abs_percentage_error = MeanAbsolutePercentageError()
    for i in range(len(targets)):
        errors=mean_abs_percentage_error(torch.tensor(preds[i]), torch.tensor(targets[i]))
        all_errors.append(errors.detach().numpy())

    return np.array(all_errors)


if __name__ == "__main__":
    preds_train=pd.read_csv('./data/train_model_pred.csv')
    targets_train=pd.read_csv('./data/train_y.csv')

    # print(preds['latency_pred'].values)
    # print(targets['latency'].values)

    train_loss=gen_loss_vals(preds_train['latency_pred'].values,targets_train['latency'].values)
    print("Train Loss : ",train_loss.mean())

    np.save('./data/train_latency_pred.npy',preds_train['latency_pred'].values)
    np.save('./data/train_latency_y.npy',targets_train['latency'].values)
    np.save('./data/train_loss.npy',train_loss)

    preds_test=pd.read_csv('./data/test_model_pred.csv')
    targets_test=pd.read_csv('./data/test_y.csv')

    # print(preds['latency_pred'].values)
    # print(targets['latency'].values)
    test_loss=gen_loss_vals(preds_test['latency_pred'].values,targets_test['latency'].values)
    np.save('./data/test_latency_pred.npy',preds_test['latency_pred'].values)
    np.save('./data/test_latency_y.npy',targets_test['latency'].values)
    np.save('./data/test_loss.npy',test_loss)
    print("Test Loss : ",test_loss.mean())


    preds_val=pd.read_csv('./data/val_model_pred.csv')
    targets_val=pd.read_csv('./data/val_y.csv')

    # print(preds['latency_pred'].values)
    # print(targets['latency'].values)
    val_loss=gen_loss_vals(preds_val['latency_pred'].values,targets_val['latency'].values)
    np.save('./data/val_latency_pred.npy',preds_val['latency_pred'].values)
    np.save('./data/val_latency_y.npy',targets_val['latency'].values)
    np.save('./data/val_loss.npy',val_loss)
    print("Val Loss : ",val_loss.mean())