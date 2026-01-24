import numpy as np
import os
import torch
import pickle
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- Imports ---
from metamodel.train_metamodel import metamodel
from quantile_reg.quantile_model import QuantileErrorEstimator

# =========================================================
# Original Scaled Label Generator (Mean + 2*Std)
# =========================================================
class scaled_label_gen:
    def __init__(self, losses) -> None:
        self.loss_avg = losses.mean()
        self.loss_std = losses.std()

    def get_loss_label(self, loss):
        """
        Linearly scale `loss` into [0, 1] using:
          - loss <= loss_avg               → 0.0
          - loss >= loss_avg + 2*loss_std  → 1.0
          - in between: (loss - loss_avg) / (2*loss_std)
        Values are clamped to [0.0, 1.0].
        """
        lower = self.loss_avg
        upper = self.loss_avg + 2 * self.loss_std

        # Guard against zero division
        if upper == lower:
            return 0.0

        scaled = (loss - lower) / (upper - lower)
        return min(1.0, max(0.0, scaled))

    def transform_array(self, loss_array):
        """Helper to transform a numpy array of losses."""
        return np.array([self.get_loss_label(l) for l in loss_array])


# =========================================================
# Main Execution
# =========================================================
def main():
    print("--- 1. Loading Raw Data & Losses ---")
    
    # Paths
    x_path = './census/data/census_metamodel_train_x.npy'
    y_path = './census/data/census_metamodel_train_y.npy'
    loss_path = './census/data/train_losses.npy'

    if not os.path.exists(x_path):
        raise FileNotFoundError("Data files not found.")

    X_raw = np.load(x_path, allow_pickle=True)
    y_raw = np.load(y_path, allow_pickle=True).astype(int)
    loss_raw = np.load(loss_path, allow_pickle=True).astype(np.float32)
    
    print(f"Data Loaded. X: {X_raw.shape}, y: {y_raw.shape}, loss: {loss_raw.shape}")

    # =========================================================
    # 2. Splitting Data
    # =========================================================
    print("\n--- 2. Splitting Data (Train / Calibration) ---")
    
    temp_model = metamodel()
    # This splits all arrays consistently
    (train_x_raw, train_y, train_loss, 
     _, _, _, 
     calib_un_x_raw, calib_un_y, calib_un_loss) = temp_model.gen_calib_data(
         X_raw, y_raw, loss_raw, balance=False, test_size=0.2
    )
    
    print(f"Split Sizes -> Train: {train_x_raw.shape[0]}, Calibration: {calib_un_x_raw.shape[0]}")

    # =========================================================
    # 3. Feature Scaling
    # =========================================================
    print("\n--- 3. Applying Feature Scaler ---")
    scaler = StandardScaler()
    scaler.fit(train_x_raw) # Fit on Train
    
    # Transform BOTH sets
    train_x_scaled = scaler.transform(train_x_raw).astype(np.float32)
    calib_un_x_scaled = scaler.transform(calib_un_x_raw).astype(np.float32)

    # =========================================================
    # 4. Training the Metamodel
    # =========================================================
    print("\n--- 4. Training Metamodel (on Train Set) ---")
    final_model = metamodel(objective='multi:softmax')
    final_model.data_scaler = scaler
    final_model.train_sense_model(train_x_scaled, train_y)

    # =========================================================
    # 5. Preparing Quantile Regressor Targets
    # =========================================================
    print("\n--- 5. Preparing Continuous Scaled Targets ---")
    
    # Initialize Label Scaler on TRAIN losses (defining the "normal" range)
    label_scaler = scaled_label_gen(train_loss)
    print(f"Loss Scaler: Avg={label_scaler.loss_avg:.4f}, Std={label_scaler.loss_std:.4f}")
    
    # Generate Scaled Targets [0-1] for BOTH sets
    # We need both because we fit on one and calibrate on the other
    train_scaled_targets = label_scaler.transform_array(train_loss)
    calib_scaled_targets = label_scaler.transform_array(calib_un_loss)
    
    print(f"Train Targets Mean: {train_scaled_targets.mean():.4f}")
    print(f"Calib Targets Mean: {calib_scaled_targets.mean():.4f}")

    # =========================================================
    # 6. Training & Calibrating Quantile Regressor
    # =========================================================
    print("\n--- 6. Training Quantile Error Estimator ---")
    
    q_estimator = QuantileErrorEstimator(n_estimators=500, ent_mode='collision')
    
    # STEP A: FIT on CALIBRATION SET
    # The QRF learns to predict error magnitude using the Calibration set
    print("Fitting QRF on Calibration Set...")
    q_estimator.fit(final_model, calib_un_x_scaled, calib_scaled_targets)
    
    # STEP B: CALIBRATE on TRAINING SET
    # We use the Training set (which the QRF has NOT seen) to find the best thresholds.
    print("Calibrating Thresholds using Training Set...")
    q_estimator.calibrate(final_model, train_x_scaled, train_y, train_scaled_targets)

    # =========================================================
    # 7. Saving Models
    # =========================================================
    print("\n--- 7. Saving Models ---")
    os.makedirs("./census/saved_models", exist_ok=True)
    
    with open("./census/saved_models/standard_metamodel.pkl", 'wb') as f:
        pickle.dump(final_model, f)
    with open("./census/saved_models/quantile_estimator.pkl", 'wb') as f:
        pickle.dump(q_estimator, f)
    with open("./census/saved_models/loss_scaler.pkl", 'wb') as f:
        pickle.dump(label_scaler, f)
    
    print("All models saved.")

    # =========================================================
    # 8. Inference Demo (Using Calibration Set)
    # =========================================================
    print("\n--- 8. Inference Demo ---")
    
    # Pick top loss samples + random samples
    high_loss_idx = np.argsort(calib_un_loss)[-3:] 
    random_idx = np.random.choice(len(calib_un_x_scaled), 2)
    idx = np.concatenate([high_loss_idx, random_idx])
    
    sample_x = calib_un_x_scaled[idx]
    sample_true_loss = calib_un_loss[idx]
    sample_scaled = calib_scaled_targets[idx]
    
    tiered_preds = q_estimator.predict_tiered(final_model, sample_x)
    
    print(f"{'Idx':<4} | {'TrueLoss':<9} | {'Scaled':<6} || {'Base Pred':<10} | {'Upper Pred':<10}")
    print("-" * 65)
    
    for i in range(len(idx)):
        base = tiered_preds['base_pred'][i]
        upper = tiered_preds['upper_pred'][i]
        t_loss = sample_true_loss[i]
        s_loss = sample_scaled[i]
        
        print(f"{i:<4} | {t_loss:.5f}   | {s_loss:.4f} || {base:.4f}     | {upper:.4f}")

if __name__ == "__main__":
    main()