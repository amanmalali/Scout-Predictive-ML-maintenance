"""
RiskAdvisorNewPipeline
======================
Variant of RiskAdvisorPipeline with two key differences:

  1. Feature construction uses only the argmax predicted label (a single integer
     column) instead of the full prediction probability distribution.  This means
     meta_x = [base_features | argmax_label]  rather than
     meta_x = [base_features | full_probs].

  2. No StandardScaler is fitted or applied.  Training features and calibration
     features are passed directly (unscaled) to the XGBoost ensemble and to the
     threshold-finding step.

Everything else — inference, label generation (classification-error labels),
90/10 train/calibration split, 10-member XGBoost ensemble, F1-optimal threshold,
base_rate saving — is identical to RiskAdvisorPipeline.
"""

import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from risk_advisor.risk_advisor_model import riskadvisor_model, find_best_thresh
from metamodel.gen_train_data import create_loss_labels, generate_image_embeddings_v2


class _MetaImageDataset(Dataset):
    """Minimal Dataset wrapper so generate_image_embeddings_v2 can inject its ViT transform."""

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
        return img, self.y[idx]


class RiskAdvisorNewPipeline:
    """
    Trains and calibrates a riskadvisor_model using argmax-only features and
    no data scaling.

    Usage
    -----
    pipeline = RiskAdvisorNewPipeline(workspace_dir, task)
    risk_model, threshold = pipeline.run(
        base_model, X_base, X_meta, y_true,
        force=True, use_embeddings=False
    )
    """

    def __init__(self, workspace_dir: str, task):
        self.workspace = workspace_dir
        self.task = task
        self.data_dir = os.path.join(workspace_dir, "data")
        self.model_dir = os.path.join(workspace_dir, "models")

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.risk_model = riskadvisor_model()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _save(self, name: str, array: np.ndarray):
        np.save(os.path.join(self.data_dir, f"{name}.npy"), array)

    def _load(self, name: str) -> np.ndarray:
        return np.load(os.path.join(self.data_dir, f"{name}.npy"))

    def _exists(self, filename: str, subdir: str = "models") -> bool:
        return os.path.exists(os.path.join(self.workspace, subdir, filename))

    # ── main entry point ─────────────────────────────────────────────────────

    def run(self, base_model, X_base, X_meta, y_true,
            force: bool = False, use_embeddings: bool = False):
        """
        Run the full Risk Advisor New Pipeline.

        Parameters
        ----------
        base_model     : trained base ML model (task-agnostic)
        X_base         : features passed to task.predict_with_loss  (scaled for base model)
        X_meta         : features used to build risk advisor input
                         (raw tabular OR pre-extracted embeddings when use_embeddings=False)
        y_true         : true labels (int for classification)
        force          : regenerate all artifacts even if cached
        use_embeddings : if True, extract ViT embeddings from X_meta before building features

        Returns
        -------
        risk_model : riskadvisor_model   Trained and calibrated risk advisor (data_scaler=None).
        threshold  : float               F-beta optimal decision threshold.
        """
        print(f"--- Starting Risk Advisor New Pipeline [Workspace: {self.workspace}] ---")

        # ── [0] Optional ViT embedding extraction ────────────────────────────
        if use_embeddings:
            print(">>> [0/5] Extracting ViT-B/16 Embeddings for Risk Advisor features...")
            meta_dataset = _MetaImageDataset(X_meta, y_true)
            features_for_meta = generate_image_embeddings_v2(meta_dataset)
        else:
            features_for_meta = X_meta

        # ── [1] Inference, Argmax Label Features & Classification-Error Labels ─
        if not self._exists("ra_new_raw_loss.npy", "data") or force:
            print(">>> [1/5] Running Inference & Building Argmax-Label Features...")
            preds, losses = self.task.predict_with_loss(base_model, X_base, y_true)

            # Use only the argmax predicted label as the prediction feature.
            # For multi-class (2-D preds): argmax over class axis.
            # For binary (1-D preds): round sigmoid to 0/1.
            if preds.ndim == 2:
                preds_argmax = np.argmax(preds, axis=1).astype(np.float32)
            else:
                preds_argmax = np.round(preds).astype(np.float32)

            # Classification-error labels: 1 if argmax(pred) != true_y, else 0.
            # create_loss_labels handles the argmax conversion internally for 2-D preds.
            meta_y_list = create_loss_labels(preds, losses, y_true, problem_type='class')
            meta_y = np.array(meta_y_list)

            # meta_x = [base_features | argmax_label]  (single column instead of full probs)
            meta_x = np.column_stack(
                (features_for_meta, preds_argmax.reshape(-1, 1))
            ).astype(np.float32)

            self._save("ra_new_raw_preds",  preds)
            self._save("ra_new_raw_loss",   losses)
            self._save("ra_new_raw_x",      features_for_meta)
            self._save("ra_new_raw_y",      y_true)
            self._save("ra_new_meta_x",     meta_x)
            self._save("ra_new_meta_y",     meta_y)
        else:
            print(">>> [1/5] Loading Cached Inference & Labels...")
            preds   = self._load("ra_new_raw_preds")
            losses  = self._load("ra_new_raw_loss")
            meta_x  = self._load("ra_new_meta_x")
            meta_y  = self._load("ra_new_meta_y")

        # ── [2] Data Construction & Splitting ────────────────────────────────
        if not self._exists("ra_new_split_calib_un_x.npy", "data") or force:
            print(">>> [2/5] Splitting Data (Train / Calibration)...")
            splits = self.risk_model.gen_calib_data(
                train_x=meta_x,
                train_y=np.array(meta_y),
                train_loss=losses,
                test_size=0.1,
            )
            (train_x, train_y, train_loss,
             calib_un_x, calib_un_y, calib_un_loss) = splits

            self._save("ra_new_split_train_x",       train_x)
            self._save("ra_new_split_train_y",       train_y)
            self._save("ra_new_split_train_loss",    train_loss)
            self._save("ra_new_split_calib_un_x",    calib_un_x)
            self._save("ra_new_split_calib_un_y",    calib_un_y)
            self._save("ra_new_split_calib_un_loss", calib_un_loss)
        else:
            print(">>> [2/5] Loading Cached Splits...")
            train_x      = self._load("ra_new_split_train_x")
            train_y      = self._load("ra_new_split_train_y")
            train_loss   = self._load("ra_new_split_train_loss")
            calib_un_x   = self._load("ra_new_split_calib_un_x")
            calib_un_y   = self._load("ra_new_split_calib_un_y")
            calib_un_loss = self._load("ra_new_split_calib_un_loss")

        # ── [3] No Data Scaler — features used as-is ─────────────────────────
        print(">>> [3/5] Skipping Data Scaler (risk_advisor_new uses raw features)...")
        self.risk_model.data_scaler = None   # explicitly None; no transform applied

        # ── [4] Train Risk Advisor ────────────────────────────────────────────
        model_path = os.path.join(self.model_dir, "risk_advisor_new_model.pkl")
        if not os.path.exists(model_path) or force:
            print(">>> [4/5] Training Risk Advisor New (10× XGBoost, no scaler, argmax features)...")
            self.risk_model.train_sense_model(train_x, train_y)
            with open(model_path, "wb") as f:
                pickle.dump(self.risk_model, f)
        else:
            print(">>> [4/5] Loading Risk Advisor New...")
            with open(model_path, "rb") as f:
                self.risk_model = pickle.load(f)
            # Ensure data_scaler is None (never used in this variant)
            self.risk_model.data_scaler = None

        # ── [5] Find Threshold on Calibration Set (no scaling) ───────────────
        print(">>> [5/5] Finding F1-optimal Risk Score Threshold (unscaled features)...")
        threshold, f_score = find_best_thresh(
            self.risk_model, calib_un_x, calib_un_y
        )
        print(f"    Threshold: {threshold:.6f}  (F1={f_score:.4f})")

        threshold_path = os.path.join(self.model_dir, "risk_advisor_new_threshold.json")
        with open(threshold_path, "w") as f:
            json.dump({"threshold": threshold, "f1_score": f_score}, f, indent=2)

        # Also store base_rate on training set (fraction of positive labels)
        base_rate = float(np.mean(train_y))
        base_rate_path = os.path.join(self.model_dir, "risk_advisor_new_base_rate.json")
        with open(base_rate_path, "w") as f:
            json.dump({"base_rate": base_rate}, f, indent=2)

        print(f"    Training base_rate (avg risk label): {base_rate:.4f}")
        print(f"--- Risk Advisor New Pipeline Complete. Artifacts in {self.workspace} ---")

        return self.risk_model, threshold
