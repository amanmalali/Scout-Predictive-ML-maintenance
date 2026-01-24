import numpy as np
from sklearn.model_selection import train_test_split
import copy

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def gen_data(train_x,test_x,train_y,test_y,split=1000):


    x1=np.load("./cifar10/data/CIFAR-10-C/brightness.npy")
    x2=np.load("./cifar10/data/CIFAR-10-C/contrast.npy")
    x3=np.load("./cifar10/data/CIFAR-10-C/defocus_blur.npy")
    x4=np.load("./cifar10/data/CIFAR-10-C/jpeg_compression.npy")
    x5=np.load("./cifar10/data/CIFAR-10-C/fog.npy")
    x6=np.load("./cifar10/data/CIFAR-10-C/frost.npy")
    y=copy.deepcopy(test_y)
    

    sub_drifted_x=np.concatenate([x1[30000:],x2[30000:]])
    sub_drifted_y=np.tile(y,4)
    idx=np.random.choice(np.arange(len(sub_drifted_y)),split)

    drifted_x=np.concatenate([x1[30000:],x2[30000:],x3[30000:],x4[30000:],x5[30000:],x6[30000:]])
    drifted_y=np.tile(y,12)

    drifted_x,drifted_y=unison_shuffled_copies(drifted_x,drifted_y)


    # sim_x=np.concatenate([train_x,drifted_x],axis=0)
    # sim_y=np.concatenate([train_y,drifted_y],axis=0)

    # Remove to revert to original version
    # idx=np.random.choice(np.arange(len(drifted_y)),split)

    # train_x_drifted=drifted_x[idx]
    # train_y_drifted=drifted_y[idx]

    train_x_drifted=sub_drifted_x[idx]
    train_y_drifted=sub_drifted_y[idx]

    new_train_x=np.concatenate([train_x,train_x_drifted,test_x],axis=0)
    new_train_y=np.concatenate([train_y,train_y_drifted,test_y],axis=0)


    train_x, test_x, train_y, test_y = train_test_split(new_train_x, new_train_y, test_size=0.2, shuffle=True, random_state=42)

    np.save("./cifar10/data/train_x.npy",train_x)
    np.save("./cifar10/data/train_y.npy",train_y)

    np.save("./cifar10/data/test_x.npy",test_x)
    np.save("./cifar10/data/test_y.npy",test_y)

    np.save("./cifar10/data/drifted_x.npy",drifted_x)
    np.save("./cifar10/data/drifted_y.npy",drifted_y)