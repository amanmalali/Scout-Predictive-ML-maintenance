import os
import sys
import torch
import numpy as np
import ray
from ray import tune, train

# --- 1. Robust Path Setup (Driver Side) ---
# Goal: Find '.../Scout-Predictive-ML-maintenance' containing 'metamodel'
# File is at: .../Scout-Predictive-ML-maintenance/metamodel/tune_temp_scaler.py

current_file_path = os.path.abspath(__file__)
metamodel_dir = os.path.dirname(current_file_path) # .../metamodel
repo_root = os.path.dirname(metamodel_dir)         # .../Scout-Predictive-ML-maintenance

# Add to driver sys.path immediately so imports work here
if repo_root not in sys.path:
    sys.path.append(repo_root)

# --- 2. Imports ---
try:
    from metamodel.temp_scaler import train_temp_scaler
    from utils import custom_torch_dataset
    try:
        from metric_utils import compute_uce_from_logits
    except ImportError:
        pass 
except ImportError as e:
    print(f"[Driver] Import Error in tune_temp_scaler: {e}")
    print(f"[Driver] Calculated Repo Root: {repo_root}")

def train_scaler_wrapper(config, data_package):
    """
    Ray Trainable Function (Executes on Worker).
    """
    # --- Worker-Side Path Patch (Safety Net) ---
    # Ensure the worker can see the repo packages even if env_vars failed
    # We reconstruct the path assuming the file structure is preserved
    worker_file_path = os.path.abspath(__file__)
    worker_repo_root = os.path.dirname(os.path.dirname(worker_file_path))
    if worker_repo_root not in sys.path:
        sys.path.append(worker_repo_root)
        
    # Lazy imports inside worker to ensure sys.path is ready
    from metamodel.temp_scaler import train_temp_scaler
    from utils import custom_torch_dataset
    try:
        from metric_utils import compute_uce_from_logits
    except ImportError:
        compute_uce_from_logits = None

    # Unpack data
    (X_train, logits_train, y_train, X_eval, logits_eval, y_eval) = data_package
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create Dataset (Train on Balanced)
    ds_train = custom_torch_dataset(X_train, logits_train, y_train)
    
    # Tensors for Eval (Test on Unbalanced)
    t_logits_eval = torch.tensor(logits_eval, dtype=torch.float32).to(device)
    t_feats_eval = torch.tensor(X_eval, dtype=torch.float32).to(device)
    t_y_eval = torch.tensor(y_eval, dtype=torch.float32).to(device)
    
    arch = config["arch"]
    
    try:
        # Train
        lin_model, scaler_model = train_temp_scaler(
            ds_train,
            mode=arch['mode'],
            lr=config['lr'],
            epochs=config['epochs'], 
            non_linear=arch['non_linear'],
            hidden_dim=arch['hidden_dim']
        )
        
        # Evaluate UCE
        with torch.no_grad():
            if arch['mode'] == 'feat':
                temps = lin_model(t_feats_eval)
                calib_logits = scaler_model.forward_ext_temp(t_logits_eval, temps)
            else:
                calib_logits = scaler_model(t_logits_eval)
            
            if compute_uce_from_logits:
                uce_val = compute_uce_from_logits(calib_logits, t_y_eval)
            else:
                # Fallback proxy if metric_utils missing
                uce_val = float(torch.nn.functional.cross_entropy(calib_logits, t_y_eval.long()).item())
            
        tune.report({"uce": uce_val})
        
    except Exception as e:
        print(f"[Worker] Trial failed: {e}")
        tune.report({"uce": 999.0})

def search_best_temp_scaler(train_x, train_y, train_logits, val_x, val_y, val_logits, fast_epochs=100, final_epochs=200):
    
    # --- CRITICAL: Initialize Ray with PYTHONPATH ---
    if not ray.is_initialized():
        print(f">>> [Metamodel] Initializing Ray with PYTHONPATH={repo_root}")
        # Pass the repo_root to the PYTHONPATH of all workers
        ray.init(
            ignore_reinit_error=True, 
            runtime_env={"env_vars": {"PYTHONPATH": repo_root}}
        )

    # Pack data
    data_package = (train_x, train_logits, train_y, val_x, val_logits, val_y)

    # Search Space
    architectures = [
        {"mode": "global",   "non_linear": False, "hidden_dim": 0,  "name": "Global (Scalar)"},
        {"mode": "ensemble", "non_linear": False, "hidden_dim": 0,  "name": "Ensemble (Vector)"},
        {"mode": "feat",     "non_linear": False, "hidden_dim": 0,  "name": "FBTS (Linear)"},
        {"mode": "feat",     "non_linear": True,  "hidden_dim": 32, "name": "FBTS (MLP-32)"},
    ]
    
    search_space = {
        "arch": tune.grid_search(architectures),
        "lr": tune.grid_search([0.01, 0.005, 0.001]),
        "epochs": fast_epochs
    }

    print(f">>> [Metamodel] Starting Distributed Search...")
    
    analysis = tune.run(
        tune.with_parameters(train_scaler_wrapper, data_package=data_package),
        config=search_space,
        metric="uce",
        mode="min",
        num_samples=1,
        resources_per_trial={"cpu": 2, "gpu": 0.2 if torch.cuda.is_available() else 0},
        verbose=1
    )

    best_config = analysis.best_config
    best_arch = best_config["arch"]
    print(f">>> [Metamodel] Best Config: {best_arch['name']} (lr={best_config['lr']}, Score={analysis.best_result['uce']:.4f})")

    # Retrain on full data
    print(f">>> [Metamodel] Retraining Best Scaler ({final_epochs} epochs)...")
    ds_final = custom_torch_dataset(train_x, train_logits, train_y)
    
    lin_model, scaler_model = train_temp_scaler(
        ds_final,
        mode=best_arch['mode'],
        lr=best_config['lr'],
        epochs=final_epochs, 
        non_linear=best_arch['non_linear'],
        hidden_dim=best_arch['hidden_dim']
    )
    
    return lin_model, scaler_model, best_config