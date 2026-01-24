import numpy as np
import os
import torch
import random
import pickle
from sklearn.preprocessing import StandardScaler


from metamodel.tune_metamodel import run_sensitivity_search
from metamodel.train_metamodel import metamodel
from metamodel.uncertainty import model_uncertainty, entropy, expected_entropy

def main():
    # =========================================================
    # 1. Setup & Data Loading
    # =========================================================
    print("--- 1. Loading Data ---")
    
    # Dummy paths as requested
    x_path = './census/data/census_metamodel_train_x.npy'
    y_path = './census/data/census_metamodel_train_y.npy'

    # Validation to ensure files exist before crashing
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Data files not found at {x_path} or {y_path}. "
            "Please ensure the 'data' folder exists and contains these .npy files."
        )

    X = np.load(x_path,allow_pickle=True)
    y = np.load(y_path,allow_pickle=True)

    # Ensure y is integer (required for multi:softmax)
    y = y.astype(int)
    
    print(f"Loaded X: {X.shape}")
    print(f"Loaded y: {y.shape}")

    scaler_x = StandardScaler()
    scaler_x.fit(X)
    
    X = scaler_x.transform(X).astype(np.float32)

    # =========================================================
    # 2. Hyperparameter Tuning (Ray Tune)
    # =========================================================
    print("\n--- 2. Running Hyperparameter Search ---")
    
    # Run the distributed search to find best params for the ensemble
    # Returns a dictionary of parameters for the 5 models
    best_params = run_sensitivity_search(
        X, 
        y, 
        num_trials=20,  # Adjust based on available time/compute
        test_size=0.2
    )

    print("\n--- Best Parameters Found ---")
    print(best_params)

    # =========================================================
    # 3. Training the Final Ensemble
    # =========================================================
    print("\n--- 3. Training Final Metamodel ---")

    # Initialize the class
    final_model = metamodel(objective='multi:softmax')
    
    # Overwrite the default XGBoost models with our TUNED parameters
    final_model.xgb_models = final_model.create_sense_model(
        num_models=5, 
        parameters=best_params
    )
    
    # Train on the full dataset
    final_model.data_scaler=scaler_x
    final_model.train_sense_model(X, y)
    
    print("Training complete.")

    #saving metamodel
    with open("./census/saved_models/meta_model_with_ts.pkl",'wb') as output:
        pickle.dump(final_model,output)

    # =========================================================
    # 4. Inference & Uncertainty Example
    # =========================================================
    print("\n--- 4. Inference on Random Train Point ---")
    
    # A. Select a random index
    rand_idx = random.randint(0, len(X) - 1)
    
    # Select the sample (keeping 2D shape: [1, n_features])
    sample_x = X[rand_idx : rand_idx + 1]
    true_label = y[rand_idx]
    
    print(f"Selected Sample Index: {rand_idx}")
    
    # B. Generate Logits
    # shape: (N_samples, N_models, N_classes) -> (1, 5, 2)
    logits_np = final_model.gen_logits(sample_x)
    probs_tensor = torch.softmax(torch.tensor(logits_np, dtype=torch.float32), dim=-1)
    
    # --- CALCULATE UNCERTAINTIES ---
    
    # 1. Total Uncertainty: H(Mean(Prob))
    # Entropy of the average prediction across the ensemble
    mean_probs = torch.mean(probs_tensor, dim=1)
    total_unc = entropy(mean_probs)
    
    # 2. Aleatoric Uncertainty: Mean(H(Prob))
    # The average entropy of individual ensemble members (Data uncertainty)
    aleatoric_unc = expected_entropy(probs_tensor)
    
    # 3. Epistemic Uncertainty: Total - Aleatoric
    # The mutual information / divergence (Model uncertainty)
    epistemic_unc = model_uncertainty(probs_tensor)
    
    # Get Prediction
    predicted_label = torch.argmax(mean_probs, dim=1).item()

    # --- DISPLAY RESULTS ---
    print("-" * 40)
    print(f"Sample Index:      {rand_idx}")
    print(f"True Label:        {true_label}")
    print(f"Predicted Label:   {predicted_label}")
    print("-" * 40)
    print(f"Total Uncertainty:      {total_unc.item():.6f}")
    print(f"Aleatoric Uncertainty:  {aleatoric_unc.item():.6f}")
    print(f"Epistemic Uncertainty:  {epistemic_unc.item():.6f}")
    print("-" * 40)
    
    # Verification of Additivity
    sum_check = aleatoric_unc.item() + epistemic_unc.item()
    print(f"Verification (Aleatoric + Epistemic): {sum_check:.6f}")
    print(f"Matches Total? {np.isclose(total_unc.item(), sum_check)}")

if __name__ == "__main__":
    main()