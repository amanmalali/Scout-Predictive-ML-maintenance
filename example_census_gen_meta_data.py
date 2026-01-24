import numpy as np
from metamodel.gen_train_data import build_metamodel_training_set

train_x=np.load("./census/data/census_train_x.npy",allow_pickle=True)
train_y=np.load("./census/data/census_train_y.npy",allow_pickle=True)
train_losses=np.load("./census/data/train_losses.npy",allow_pickle=True)
train_predictions=np.load("./census/data/train_preds.npy",allow_pickle=True)


metamodel_train_x,metamodel_train_y=build_metamodel_training_set(predictions=train_predictions ,features=train_x,losses=train_losses,labels=train_y,problem_type="reg")


val_x=np.load("./census/data/census_val_x.npy",allow_pickle=True)
val_y=np.load("./census/data/census_val_y.npy",allow_pickle=True)
val_losses=np.load("./census/data/val_losses.npy",allow_pickle=True)
val_predictions=np.load("./census/data/val_preds.npy",allow_pickle=True)



metamodel_val_x,metamodel_val_y=build_metamodel_training_set(predictions=val_predictions ,features=val_x,losses=val_losses,labels=val_y,problem_type="reg",val=True,train_losses=train_losses)

print("High loss labels in training:",metamodel_train_y.sum())

print("High loss labels in validation:",metamodel_val_y.sum())

np.save("./census/data/census_metamodel_train_x.npy",metamodel_train_x)
np.save("./census/data/census_metamodel_train_y.npy",metamodel_train_y)
np.save("./census/data/census_metamodel_val_x.npy",metamodel_val_x)
np.save("./census/data/census_metamodel_val_y.npy",metamodel_val_y)
