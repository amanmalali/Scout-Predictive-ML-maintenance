import os
import sys
import numpy as np
import pickle
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# Scout Imports
from metamodel.train_metamodel import metamodel
from metamodel.gen_train_data import build_metamodel_training_set, create_loss_labels, generate_image_embeddings_v2
from quantile_regressor.quantile_model import QuantileErrorEstimator
from utils import move_metamodel_to_device


# --- HELPER DATASET FOR ViT EMBEDDING EXTRACTION ---
class _MetaImageDataset(Dataset):
    """
    A minimal PyTorch Dataset wrapper that allows generate_image_embeddings_v2 
    to dynamically inject its ViT transform preprocessing.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.transform = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = torch.tensor(self.x[idx], dtype=torch.float32)
        if self.transform is not None:
            img = self.transform(img)
        label = self.y[idx]
        return img, label


class MetamodelPipeline:
    def __init__(self, workspace_dir, task):
        self.workspace = workspace_dir
        self.task = task
        self.data_dir = os.path.join(workspace_dir, "data")
        self.model_dir = os.path.join(workspace_dir, "models")
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize Core Components
        self.meta_model = metamodel()
        self.q_estimator = QuantileErrorEstimator()

    def _save_split(self, name, array):
        np.save(os.path.join(self.data_dir, f"{name}.npy"), array)

    def _load_split(self, name):
        return np.load(os.path.join(self.data_dir, f"{name}.npy"))

    def _exists(self, filename, subdir="models"):
        path = os.path.join(self.workspace, subdir, filename)
        return os.path.exists(path)

    def run(self, base_model, X_base, X_meta, y_true, force=False, use_embeddings=False):
        """
        Runs the full Metamodel Pipeline.
        """
        print(f"--- Starting Metamodel Pipeline [Workspace: {self.workspace}] ---")

        # --- THE FIX: FEATURE EXTRACTION VIA ViT ---
        if use_embeddings:
            print(">>> [0/7] Extracting Embeddings for Metamodel Features using ViT-B/16...")
            meta_dataset = _MetaImageDataset(X_meta, y_true)
            features_for_meta = generate_image_embeddings_v2(meta_dataset)
        else:
            features_for_meta = X_meta

        # 1. Inference & Loss Generation
        if not self._exists("meta_raw_loss.npy", "data") or force:
            print(">>> [1/7] Running Inference (using Base Scaler) & Calculating Losses...")
            preds, losses = self.task.predict_with_loss(base_model, X_base, y_true)
            
            self._save_split("meta_raw_preds", preds)
            self._save_split("meta_raw_loss", losses)
            self._save_split("meta_raw_x", features_for_meta) # Passes ViT embeddings or raw
            self._save_split("meta_raw_y", y_true)
        else:
            print(">>> [1/7] Loading Cached Losses...")
            preds = self._load_split("meta_raw_preds")
            losses = self._load_split("meta_raw_loss")

        # 2. Data Construction & Splitting
        if not self._exists("split_calib_bal_x.npy", "data") or force:
            print(">>> [2/7] Constructing Metamodel Set & Splitting...")
            
            # A. Build Metamodel Feature Set AND Binary Labels
            meta_x, meta_y = build_metamodel_training_set(
                features=features_for_meta, # Replaced X_meta with dynamic features
                predictions=preds,
                losses=losses,
                labels=y_true,
                problem_type="reg" # Uses Mean+2*Std threshold internally
            )
            self._save_split("meta_train_x", meta_x)
            self._save_split("meta_train_y", meta_y)
            
            # B. Generate Splits (Train / Balanced / Unbalanced)
            splits = self.meta_model.gen_calib_data(
                train_x=meta_x, 
                train_y=meta_y, 
                train_loss=losses,
                balance=False # CRITICAL: Balance the calibration set
            )
            
            # Unpack the 9 outputs
            (train_x, train_y, train_loss, 
             calib_bal_x, calib_bal_y, calib_bal_loss, 
             calib_un_x, calib_un_y, calib_un_loss) = splits

            print("    Saving splits to disk...")
            self._save_split("split_train_x", train_x)
            self._save_split("split_train_y", train_y) # Binary labels
            self._save_split("split_train_loss", train_loss)
            
            self._save_split("split_calib_bal_x", calib_bal_x)
            self._save_split("split_calib_bal_y", calib_bal_y) # Binary labels
            self._save_split("split_calib_bal_loss", calib_bal_loss)
            
            self._save_split("split_calib_un_x", calib_un_x)
            self._save_split("split_calib_un_y", calib_un_y) # Binary labels
            self._save_split("split_calib_un_loss", calib_un_loss)
        else:
            print(">>> [2/7] Loading Cached Splits...")
            train_x = self._load_split("split_train_x")
            train_y = self._load_split("split_train_y")
            train_loss = self._load_split("split_train_loss")
            
            calib_bal_x = self._load_split("split_calib_bal_x")
            calib_bal_y = self._load_split("split_calib_bal_y")
            calib_bal_loss = self._load_split("split_calib_bal_loss")
            
            calib_un_x = self._load_split("split_calib_un_x")
            calib_un_y = self._load_split("split_calib_un_y")
            calib_un_loss = self._load_split("split_calib_un_loss")

        # 4. Data Scaler (Metamodel Specific)
        scaler_path = os.path.join(self.model_dir, "meta_data_scaler.pkl")
        if not os.path.exists(scaler_path) or force:
            print(">>> [3/7] Fitting Metamodel Data Scaler (Internal)...")
            scaler = StandardScaler()
            train_x_scaled = scaler.fit_transform(train_x)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            self.meta_model.data_scaler = scaler
        else:
            print(">>> [3/7] Loading Metamodel Data Scaler (Internal)...")
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            train_x_scaled = scaler.transform(train_x)
            self.meta_model.data_scaler = scaler

        # Scale calibration splits
        calib_bal_x_scaled = scaler.transform(calib_bal_x)
        calib_un_x_scaled = scaler.transform(calib_un_x)

        # 5. Train Metamodel
        meta_model_path = os.path.join(self.model_dir, "scout_metamodel.pkl")
        if not os.path.exists(meta_model_path) or force:
            print(">>> [4/7] Training Metamodel (XGBoost)...")
            self.meta_model.train_sense_model(train_x_scaled, train_y)
            with open(meta_model_path, "wb") as f:
                pickle.dump(self.meta_model, f)
        else:
            print(">>> [4/7] Loading Metamodel...")
            with open(meta_model_path, "rb") as f:
                self.meta_model = pickle.load(f)

        # # 6. Tune & Train Temperature Scaler
        # if not hasattr(self.meta_model, 'scaler') or self.meta_model.scaler is None or force:
        #     print(">>> [5/7] Tuning Temperature Scaler (Distributed)...")
            
        #     logits_bal = self.meta_model.gen_logits(calib_bal_x_scaled)
        #     logits_un = self.meta_model.gen_logits(calib_un_x_scaled)
            
        #     # Note: For validation in search, we use the Unbalanced calibration set
        #     # We must generate binary labels for the Unbalanced set using our Label Generator
        #     # to ensure the threshold is consistent with the Training set.
        #     # (calib_un_y from split is also valid, but using label_gen is safer for OOD consistency checks)

        #     lin_model, scaler_model, best_config = search_best_temp_scaler(
        #         train_x=calib_bal_x_scaled,
        #         train_y=calib_bal_y,
        #         train_logits=logits_bal,
        #         val_x=calib_un_x_scaled,
        #         val_y=calib_un_y,
        #         val_logits=logits_un
        #     )
            
        #     self.meta_model.lin_model = lin_model
        #     self.meta_model.scaler = scaler_model
        #     self.meta_model.temp_mode = best_config['arch']['mode']
            
        #     with open(meta_model_path, "wb") as f:
        #         pickle.dump(self.meta_model, f)
        # else:
        #     print(">>> [5/7] Temperature Scaler already attached.")

        # 7. Quantile Regression & Calibration
        
        q_model_path = os.path.join(self.model_dir, "quantile_estimator.pkl")
        self.meta_model = move_metamodel_to_device(self.meta_model, device='cpu')
        if not os.path.exists(q_model_path) or force:
            print(">>> [6/7] Training & Calibrating Quantile Regressor...")
            self.q_estimator.fit(
                sense_model=self.meta_model,
                X=calib_un_x_scaled,
                y_loss=calib_un_loss,
                train_loss_distribution=train_loss 
            )
            
        else:
            print(">>> [6/7] Loading Quantile Estimator...")
            with open(q_model_path, "rb") as f:
                self.q_estimator = pickle.load(f)

        print(">>> [7/7] Calibrating Quantile Estimator...")
        self.q_estimator.calibrate(
                sense_model=self.meta_model,
                X_train=train_x_scaled,
                y_train_binary=train_y
            )
        
        with open(q_model_path, "wb") as f:
            pickle.dump(self.q_estimator, f)

        print(f"--- Pipeline Complete. Artifacts in {self.workspace} ---")
        return self.meta_model, self.q_estimator