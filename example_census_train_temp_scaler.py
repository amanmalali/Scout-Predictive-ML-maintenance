import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import itertools
import pickle

# 1. Setup Path to import local modules
sys.path.append(os.path.join(os.getcwd(), 'metamodel'))

from utils import custom_torch_dataset
from metamodel.train_metamodel import metamodel
import metamodel.temp_scaler as ts 

def load_data_from_files():
    """
    Loads features, labels, and loss from numpy files.
    """
    features_path = "./census/data/census_metamodel_train_x.npy"
    labels_path = "./census/data/census_metamodel_train_y.npy"
    loss_path = "./census/data/train_losses.npy"

    if os.path.exists(features_path) and os.path.exists(labels_path):
        X = np.load(features_path, allow_pickle=True)
        y = np.load(labels_path, allow_pickle=True)
        X = np.array(X).astype(np.float32)
        y = np.array(y).astype(np.float32)
        
        if os.path.exists(loss_path):
            loss = np.load(loss_path, allow_pickle=True).astype(np.float32)
        else:
            loss = np.zeros(len(y), dtype=np.float32)
        return X, y, loss
    else:
        raise FileNotFoundError(f"Data files not found.")

def main():
    # --- 1. Load Data ---
    try:
        raw_X, raw_y, raw_loss = load_data_from_files()
    except Exception as e:
        print(e)
        return

    meta = metamodel()

    # --- 2. Data Splitting ---
    # train_*:      Trains the Sense Model (XGBoost)
    # calib_over_*: Trains the Temperature Scaler (Internal validation split happens inside temp_scaler.py)
    # calib_un_*:   Evaluates/Selects the final Temperature Scaler (Holdout Test Set)
    (train_x, train_y, train_loss, 
     calib_over_x, calib_over_y, calib_over_loss, 
     calib_un_x, calib_un_y, calib_un_loss) = meta.gen_calib_data(
         raw_X, raw_y, raw_loss, balance=True, test_size=0.2
    )

    # --- 3. Feature Scaling (CRITICAL) ---
    print("\n--- Scaling Features (StandardScaler) ---")
    scaler_x = StandardScaler()
    scaler_x.fit(train_x) 
    
    train_x = scaler_x.transform(train_x).astype(np.float32)
    calib_over_x = scaler_x.transform(calib_over_x).astype(np.float32)
    calib_un_x = scaler_x.transform(calib_un_x).astype(np.float32)

    # --- 4. Train Sense Model ---
    print("\n--- Training Sense Model ---")
    meta.data_scaler=scaler_x
    meta.train_sense_model(train_x, train_y)

    # --- 5. Generate Logits ---
    print("\n--- Generating Logits ---")
    logits_calib_train = meta.gen_logits(calib_over_x).astype(np.float32)
    logits_calib_eval  = meta.gen_logits(calib_un_x).astype(np.float32)

    # --- 6. Prepare Datasets ---
    dataset_calib_train = custom_torch_dataset(
        calib_over_x, logits_calib_train, calib_over_y
    )

    # Holdout Tensors (Test Set)
    t_x_eval = torch.tensor(calib_un_x)
    t_logits_eval = torch.tensor(logits_calib_eval)
    t_y_eval = torch.tensor(calib_un_y)

    # --- 7. Hyperparameter Tuning Definitions ---
    uce_loss_fn = ts.UCECollisionEntropyLoss()
    
    # Define Search Grids
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    hidden_dims    = [32, 64, 128] # Only for MLP
    epochs_fixed   = 300 # Fixed to save time, or add to grid if needed

    # Generate Configurations
    configs = []

    # A. Global & Ensemble (Static Scalers)
    for mode in ["global", "ensemble"]:
        for lr in learning_rates:
            configs.append({
                "mode": mode,
                "lr": lr,
                "epochs": epochs_fixed,
                "non_linear": False,
                "hidden_dim": 0,
                "desc": f"{mode.upper()} (lr={lr})"
            })

    # B. Feature-Based Linear (FBTS-Linear)
    for lr in learning_rates:
        configs.append({
            "mode": "feat",
            "lr": lr,
            "epochs": epochs_fixed,
            "non_linear": False, # Linear Mode
            "hidden_dim": 0,
            "desc": f"FBTS-Linear (lr={lr})"
        })

    # C. Feature-Based Non-Linear (FBTS-MLP)
    for lr, h_dim in itertools.product(learning_rates, hidden_dims):
        configs.append({
            "mode": "feat",
            "lr": lr,
            "epochs": epochs_fixed,
            "non_linear": True,  # MLP Mode
            "hidden_dim": h_dim,
            "desc": f"FBTS-MLP (lr={lr}, hidden={h_dim})"
        })

    # --- 8. Tuning Loop ---
    print(f"\n--- Starting Hyperparameter Tuning ({len(configs)} configurations) ---")
    
    best_overall_loss = float('inf')
    best_config_desc = None
    best_mode_name = None  # To store "feat", "global", or "ensemble"
    best_models = (None, None) # (lin_model, scaler)

    results_log = []

    for i, conf in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing: {conf['desc']}")
        
        # Train
        lin_model, scaler = ts.train_temp_scaler(
            dataset_calib_train, 
            mode=conf['mode'], 
            lr=conf['lr'], 
            epochs=conf['epochs'],
            non_linear=conf['non_linear'],
            hidden_dim=conf['hidden_dim']
        )
        
        # If training failed (returned None), skip
        if scaler is None:
            print("   -> Training diverged or failed.")
            continue

        # Evaluate on Holdout (Test Set)
        scaler.eval()
        if lin_model: lin_model.eval()

        with torch.no_grad():
            if conf['mode'] == "feat":
                temps = lin_model(t_x_eval)
                scaled_logits = scaler.forward_ext_temp(t_logits_eval, temps)
            else:
                scaled_logits = scaler(t_logits_eval)
            
            # Metric: UCE on Holdout
            loss_val, _, _ = uce_loss_fn(scaled_logits, t_y_eval)
            loss_val = loss_val.item()

        print(f"   -> Holdout UCE: {loss_val:.6f}")
        
        # Log Result
        results_log.append({
            "config": conf['desc'],
            "uce": loss_val
        })

        # Update Best
        if loss_val < best_overall_loss:
            best_overall_loss = loss_val
            best_config_desc = conf['desc']
            best_mode_name = conf['mode']
            best_models = (lin_model, scaler)
            print(f"   *** New Best Model found! ***")

    # --- 9. Final Results ---
    print("\n" + "="*60)
    print(f"HYPERPARAMETER TUNING COMPLETE")
    print(f"Best Configuration: {best_config_desc}")
    print(f"Best Holdout UCE:   {best_overall_loss:.6f}")
    print("="*60)
    
    # Print Top 5
    results_df = pd.DataFrame(results_log).sort_values(by="uce")
    print("\nTop 5 Configurations:")
    print(results_df.head(5))

    # Store best models in the metamodel wrapper
    meta.lin_model = best_models[0]
    meta.scaler = best_models[1]
    meta.temp_mode=best_mode_name

    #saving metamodel
    with open("./census/saved_models/meta_model_with_ts.pkl",'wb') as output:
        pickle.dump(meta,output)

if __name__ == "__main__":
    main()