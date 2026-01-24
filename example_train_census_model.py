import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split

from census.train_basic_classifier import train_model, model_inf, calc_loss
from utils import torch_dataset, scale_data
import numpy as np
from metamodel.gen_train_data import build_metamodel_training_set

def generate_results(model, X_scaled, y_true):
    """
    Helper function to iterate over numpy arrays, generate predictions,
    calculate losses, and return them as numpy arrays.
    """
    preds_list = []
    losses_list = []
    
    # Switch model to eval mode just in case
    model.eval()
    
    # Iterate through each sample
    for i in range(len(X_scaled)):
        sample_x = X_scaled[i]
        sample_y = y_true[i]
        
        # 1. Prediction (model_inf handles tensor conversion)
        pred_tensor = model_inf(model, sample_x)
        
        # 2. Loss Calculation Fix:
        # calc_loss expects shape [1], but .squeeze() was making it shape [].
        # We use .reshape(1) to ensure it matches the target shape defined in calc_loss.
        loss_tensor = calc_loss(pred_tensor.reshape(1), sample_y)
        
        # 3. Store (detach from graph)
        preds_list.append(pred_tensor.detach().cpu().item())
        losses_list.append(loss_tensor.detach().cpu().item())
        
    return np.array(preds_list), np.array(losses_list)

def main():
    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    data_x_path = './census/data/census_train_x.npy'
    data_y_path = './census/data/census_train_y.npy'
    inf_x_path  = './census/data/census_val_x.npy'
    inf_y_path  = './census/data/census_val_y.npy' 

    print("Loading data...")
    try:
        X_all = np.load(data_x_path,allow_pickle=True)
        y_all = np.load(data_y_path,allow_pickle=True)
        X_inf = np.load(inf_x_path,allow_pickle=True)
        y_inf = np.load(inf_y_path,allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

    # ==========================================
    # 2. TRAIN / TEST SPLIT
    # ==========================================
    print("Splitting Training Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    # ==========================================
    # 3. SCALING
    # ==========================================
    print("Scaling data...")
    # Fit scaler ONLY on training data
    X_train_scaled, scaler = scale_data(X_train)
    
    # Transform Test and Inference data using the training scaler
    X_test_scaled, _ = scale_data(X_test, scale_func=scaler)
    X_inf_scaled, _  = scale_data(X_inf, scale_func=scaler)

    X_all_scaled,_=scale_data(X_all, scale_func=scaler)

    # ==========================================
    # 4. TRAIN MODEL
    # ==========================================
    train_data = torch_dataset(X_train_scaled, y_train, scale=False)
    test_data  = torch_dataset(X_test_scaled, y_test, scale=False)

    model_output_path = './census/saved_models/basic_classifier.pt'
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    print("Starting training...")
    train_model(
        data=train_data,
        test_data=test_data,
        model_path=model_output_path,
        lr=0.001,
        epochs=50,
        output=1
    )

    # ==========================================
    # 5. GENERATE PREDICTIONS & LOSSES
    # ==========================================
    print("\n--- Generating Results ---")
    
    # Load the best saved model
    loaded_model = torch.load(model_output_path)
    
    # A. Results for Training Data
    print(f"Processing Training Data ({len(X_train_scaled)} samples)...")
    train_preds, train_losses = generate_results(loaded_model, X_all_scaled, y_all)

    # B. Results for Inference Data
    print(f"Processing Inference Data ({len(X_inf_scaled)} samples)...")
    inf_preds, inf_losses = generate_results(loaded_model, X_inf_scaled, y_inf)

    # Summary Output
    print("\n--- Summary ---")
    print(f"Training Results - Preds: {train_preds.shape}, Losses: {train_losses.shape}")
    print(f"Inference Results - Preds: {inf_preds.shape}, Losses: {inf_losses.shape}")
    
    return train_preds, train_losses, inf_preds, inf_losses

if __name__ == "__main__":
    t_preds, t_losses, i_preds, i_losses = main()
    np.save("./census/data/train_preds.npy",t_preds)
    np.save("./census/data/val_preds.npy",i_preds)
    np.save("./census/data/train_losses.npy",t_losses)
    np.save("./census/data/val_losses.npy",i_losses)


    '''
    This code is used to generate the convert the labels into binary high/low loss labels
    it also creates the feature set used to train the metamodel.
    
    '''
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
