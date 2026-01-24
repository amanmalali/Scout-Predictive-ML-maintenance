import torch
import numpy as np
from utils import run_inference_dataset,image_dataset
from sensitivity.gen_train_data import generate_image_embeddings_v2,build_sensitivity_training_set_v2,build_sensitivity_training_ddla


def gen_sense_data():
    train_x=np.load("./cifar10/data/train_x.npy")
    train_y=np.load("./cifar10/data/train_y.npy")
    train_loss=np.load("./cifar10/data/cifar_train_loss.npy")
    train_pred=np.load("./cifar10/data/cifar_train_pred.npy")

    test_x=np.load("./cifar10/data/test_x.npy")
    test_y=np.load("./cifar10/data/test_y.npy")
    test_loss=np.load("./cifar10/data/cifar_test_loss.npy")
    test_pred=np.load("./cifar10/data/cifar_test_pred.npy")

    drifted_x=np.load("./cifar10/data/drifted_x.npy")
    drifted_y=np.load("./cifar10/data/drifted_y.npy")
    val_loss=np.load("./cifar10/data/cifar_val_loss.npy")
    val_pred=np.load("./cifar10/data/cifar_val_pred.npy")


    sense_trainset=image_dataset(train_x,train_y)

    sense_train_features=np.load("./cifar10/data/sense_cifar_train_feat.npy")
    sense_train_x,sense_train_y=build_sensitivity_training_set_v2(sense_train_features,train_pred,train_loss,train_y,problem_type="reg")
    ddla_train_x,ddla_train_y=build_sensitivity_training_ddla(sense_train_features,train_pred,train_loss,train_y,problem_type="class")


    print("Training data x :",sense_train_x.shape)
    print("Training data y :",np.unique(sense_train_y,return_counts=True))

    np.save("./cifar10/data/sense_cifar_train_x.npy",sense_train_x)
    np.save("./cifar10/data/sense_cifar_train_y.npy",sense_train_y)
    np.save("./cifar10/data/ddla_cifar_train_x.npy",ddla_train_x)
    np.save("./cifar10/data/ddla_cifar_train_y.npy",ddla_train_y)


    sense_testset=image_dataset(test_x,test_y)

    sense_test_features=np.load("./cifar10/data/sense_cifar_test_feat.npy")

    sense_test_x,sense_test_y=build_sensitivity_training_set_v2(sense_test_features,test_pred,test_loss,test_y,problem_type="reg",val=True,train_losses=train_loss)
    ddla_test_x,ddla_test_y=build_sensitivity_training_ddla(sense_test_features,test_pred,test_loss,test_y,problem_type="class",val=True,train_losses=train_loss)

    print("Test data x :",sense_test_x.shape)
    print("Testing data y :",np.unique(sense_test_y,return_counts=True))

    np.save("./cifar10/data/sense_cifar_test_x.npy",sense_test_x)
    np.save("./cifar10/data/sense_cifar_test_y.npy",sense_test_y)
    np.save("./cifar10/data/ddla_cifar_test_x.npy",ddla_test_x)
    np.save("./cifar10/data/ddla_cifar_test_y.npy",ddla_test_y)


    sense_valset=image_dataset(drifted_x,drifted_y)

    sense_val_features=np.load("./cifar10/data/sense_cifar_val_feat.npy")
    sense_val_x,sense_val_y=build_sensitivity_training_set_v2(sense_val_features,val_pred,val_loss,drifted_y,problem_type="reg",val=True,train_losses=train_loss)
    ddla_val_x,ddla_val_y=build_sensitivity_training_ddla(sense_val_features,val_pred,val_loss,drifted_y,problem_type="class",val=True,train_losses=train_loss)


    print("Validation data x :",sense_val_x.shape)
    print("Validation data y :",np.unique(sense_val_y,return_counts=True))

    np.save("./cifar10/data/sense_cifar_val_x.npy",sense_val_x)
    np.save("./cifar10/data/sense_cifar_val_y.npy",sense_val_y)
    np.save("./cifar10/data/ddla_cifar_val_x.npy",ddla_val_x)
    np.save("./cifar10/data/ddla_cifar_val_y.npy",ddla_val_y)

if __name__=="__main__":
    gen_sense_data()