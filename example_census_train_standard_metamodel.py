import numpy as np
import os
import torch
import random
import pickle
from sklearn.preprocessing import StandardScaler


# --- Imports from the 'metamodel' package ---
from metamodel.train_metamodel import metamodel
# Import the specific uncertainty functions
from metamodel.uncertainty import model_uncertainty, entropy, expected_entropy

def main():
    # =========================================================
    # 1. Setup & Data Loading
    # =========================================================
    print("--- 1. Loading Data ---")
    
    x_path = './census/data/census_metamodel_train_x.npy'
    y_path = './census/data/census_metamodel_train_y.npy'

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Data files not found at {x_path} or {y_path}.")

    X = np.load(x_path,allow_pickle=True)
    y = np.load(y_path,allow_pickle=True).astype(int)
    
    print(f"Data Loaded. X: {X.shape}, y: {y.shape}")

    scaler_x = StandardScaler()
    scaler_x.fit(X)
    
    X = scaler_x.transform(X).astype(np.float32)

    # =========================================================
    # 2. Training (Default Parameters)
    # =========================================================
    print("\n--- 2. Training Metamodel ---")
    final_model = metamodel(objective='multi:softmax')
    final_model.data_scaler=scaler_x
    final_model.train_sense_model(X, y)
    print("Training complete.")

    with open("./census/saved_models/stardard_metamodel.pkl",'wb') as output:
        pickle.dump(final_model,output)

    # =========================================================
    # 3. Inference & Uncertainty Decomposition
    # =========================================================
    print("\n--- 3. Uncertainty Decomposition Example ---")
    
    # Select random sample
    rand_idx = random.randint(0, len(X) - 1)
    sample_x = X[rand_idx : rand_idx + 1]
    true_label = y[rand_idx]
    
    # Generate Probabilities (Shape: 1 sample, 5 models, 2 classes)
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