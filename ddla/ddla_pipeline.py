"""
DDLAPipeline
============
End-to-end pipeline for training a DDLAModel.

Key differences vs RiskAdvisorPipeline
---------------------------------------
* Feature space: [features | predictions] via ``build_metamodel_training_set``
  with ``problem_type='class'`` (same as risk_advisor).
* No StandardScaler — decision trees are scale-invariant.
* No threshold search — the runtime threshold is a simulation parameter.
* Saves: ``ddla_model.pkl``, ``ddla_base_rate.json``
* Returns: ``(DDLAModel, train_proportion)``
"""

import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from ddla.ddla_model import DDLAModel
from metamodel.gen_train_data import build_metamodel_training_set, generate_image_embeddings_v2


class _MetaImageDataset(Dataset):
    """Minimal Dataset wrapper for generate_image_embeddings_v2."""

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


class DDLAPipeline:
    """
    Trains and stores a DDLAModel end-to-end.

    Usage
    -----
    pipeline = DDLAPipeline(workspace_dir, task)
    ddla_model, train_proportion = pipeline.run(
        base_model, X_base, X_meta, y_true,
        force=True, use_embeddings=False
    )

    Parameters for run()
    --------------------
    base_model     : trained base ML model
    X_base         : features for ``task.predict_with_loss`` (appropriately
                     scaled for the base model)
    X_meta         : features for the DDLA tree:
                       - tabular datasets: raw feature vectors
                       - image datasets (use_embeddings=False): pre-extracted
                         ViT embeddings passed directly
                       - image datasets (use_embeddings=True): raw images from
                         which ViT embeddings are extracted inside the pipeline
    y_true         : true labels (int class indices)
    force          : regenerate all artifacts even if cached
    use_embeddings : if True, extract ViT-B/16 embeddings from X_meta
    """

    def __init__(self, workspace_dir: str, task):
        self.workspace = workspace_dir
        self.task      = task
        self.data_dir  = os.path.join(workspace_dir, "data")
        self.model_dir = os.path.join(workspace_dir, "models")

        os.makedirs(self.data_dir,  exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _save(self, name: str, array: np.ndarray):
        np.save(os.path.join(self.data_dir, f"{name}.npy"), array)

    def _load(self, name: str) -> np.ndarray:
        return np.load(os.path.join(self.data_dir, f"{name}.npy"))

    def _exists(self, filename: str, subdir: str = "data") -> bool:
        return os.path.exists(os.path.join(self.workspace, subdir, filename))

    # ── main entry point ─────────────────────────────────────────────────────

    def run(self, base_model, X_base, X_meta, y_true,
            force: bool = False, use_embeddings: bool = False):
        """
        Run the full DDLA Pipeline.

        Returns
        -------
        ddla_model      : DDLAModel   Fitted model with decision paths.
        train_proportion: float       Baseline proportion (fraction of training
                                      data falling in high-error paths).
        """
        print(f"--- Starting DDLA Pipeline [Workspace: {self.workspace}] ---")

        # ── [0] Optional ViT embedding extraction ────────────────────────────
        if use_embeddings:
            print(">>> [0/3] Extracting ViT-B/16 Embeddings for DDLA features...")
            meta_dataset = _MetaImageDataset(X_meta, y_true)
            X_feat = generate_image_embeddings_v2(meta_dataset)
        else:
            # X_meta is already the feature array for the decision tree
            X_feat = X_meta

        # ── [1] Inference & Classification-Error Label Generation ─────────────
        if not self._exists("ddla_labels.npy") or force:
            print(">>> [1/3] Running Inference & Generating Classification-Error Labels...")
            preds, losses = self.task.predict_with_loss(base_model, X_base, y_true)

            # For multi-class models preds is (N, C); create_loss_labels with
            # problem_type='class' does predictions[i].round() != labels[i] which
            # requires a scalar.  Convert to argmax predicted-class indices.
            preds_for_labels = (
                np.argmax(preds, axis=1).astype(np.float32)
                if preds.ndim == 2 else preds
            )

            X_ddla, y_ddla = build_metamodel_training_set(
                features=X_feat,
                predictions=preds_for_labels,
                losses=losses,
                labels=y_true,
                problem_type='class',
            )
            self._save("ddla_raw_preds",  preds)
            self._save("ddla_raw_losses", losses)
            self._save("ddla_feat_x",     X_feat)
            self._save("ddla_x",          X_ddla)
            self._save("ddla_labels",     np.array(y_ddla))
        else:
            print(">>> [1/3] Loading Cached Inference & Labels...")
            preds   = self._load("ddla_raw_preds")
            losses  = self._load("ddla_raw_losses")
            X_ddla  = self._load("ddla_x")
            y_ddla  = self._load("ddla_labels")

        # ── [2] Fit DDLAModel ─────────────────────────────────────────────────
        model_path = os.path.join(self.model_dir, "ddla_model.pkl")
        if not os.path.exists(model_path) or force:
            print(">>> [2/3] Training DDLAModel (DecisionTreeClassifier)...")
            ddla_model = DDLAModel()
            ddla_model.fit(X_ddla, np.array(y_ddla))
            with open(model_path, "wb") as f:
                pickle.dump(ddla_model, f)
        else:
            print(">>> [2/3] Loading Cached DDLAModel...")
            with open(model_path, "rb") as f:
                ddla_model = pickle.load(f)

        # ── [3] Save Base Rate Artifacts ──────────────────────────────────────
        train_proportion = float(ddla_model.train_proportion)
        # base_rate in DDLA = train_proportion (fraction of training data in
        # high-error paths) — used as the per-sample baseline failure rate.
        base_rate_path = os.path.join(self.model_dir, "ddla_base_rate.json")
        with open(base_rate_path, "w") as f:
            json.dump({
                "train_proportion": train_proportion,
                "base_rate":        train_proportion,
                "n_paths":          len(ddla_model.decision_paths),
                "n_features":       int(ddla_model.n_features),
            }, f, indent=2)

        print(f"    train_proportion = {train_proportion:.4f}  "
              f"(#paths={len(ddla_model.decision_paths)})")
        print(f"--- DDLA Pipeline Complete. Artifacts in {self.workspace} ---")

        return ddla_model, train_proportion
