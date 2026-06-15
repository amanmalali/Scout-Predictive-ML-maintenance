import numpy as np
import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

try:
    from .train_metamodel import metamodel
    from .uncertainty import entropy
except ImportError:
    from train_metamodel import metamodel
    from uncertainty import entropy


search_space = {
        "n_estimators": tune.choice([400, 600, 800, 1000]),
        "lr": tune.loguniform(1e-3, 0.1),
        "max_depth": tune.choice([3, 5, 8, 10, 12]),
        "subsample": tune.uniform(0.5, 0.8),
        "grow_policy": tune.choice(['depthwise', 'lossguide']),
        "min_child_weight": tune.choice([0.2, 0.5, 1, 5]),
    }

def evaluate_uncertainty_score(model, val_x, val_y, uncertainty_weight=0.05):
    """
    Calculates score: Accuracy + weight * (Uncertainty_Wrong - Uncertainty_Correct)
    
    This penalizes the model if it is uncertain about correct predictions, 
    and rewards it if it is uncertain about wrong predictions.
    """
    # 1. Get Logits and Probabilities
    logits = model.gen_logits(val_x)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    probs_t = torch.nn.functional.softmax(logits_t, dim=-1)
    
    # 2. Ensemble Average (Total Uncertainty)
    mean_probs = torch.mean(probs_t, dim=1)
    unc = entropy(mean_probs)
    
    # 3. Predictions and Masks
    preds = torch.argmax(mean_probs, dim=1).numpy()
    accuracy = np.mean(preds == val_y)
    
    wrong_mask = preds != val_y
    correct_mask = preds == val_y
    
    # 4. Calculate Gap Components
    # Handle edge cases where accuracy is 0% or 100%
    if np.sum(wrong_mask) > 0:
        mean_unc_wrong = torch.mean(unc[torch.tensor(wrong_mask)]).item()
    else:
        # If perfectly accurate, we assume ideal behavior for "wrong" (high uncertainty)
        # to not penalize perfection, though this term essentially vanishes.
        mean_unc_wrong = 0.0 

    if np.sum(correct_mask) > 0:
        mean_unc_correct = torch.mean(unc[torch.tensor(correct_mask)]).item()
    else:
        mean_unc_correct = 0.0
        
    # 5. Calculate Score
    # We want (Wrong - Correct) to be POSITIVE and LARGE.
    uncertainty_gap = mean_unc_wrong - mean_unc_correct
    
    # Weight of 0.05 implies: Increasing the "Gap" by 0.20 (20%) is worth 1% Accuracy.
    score = accuracy + (uncertainty_weight * uncertainty_gap)
    
    return score, accuracy, mean_unc_wrong, mean_unc_correct

def objective(config, data):
    train_x, val_x, train_y, val_y = data
    
    temp_params = {
        "n_estimators": [config["n_estimators"]] * 5,
        "lr": [config["lr"]] * 5,
        "max_depth": [config["max_depth"]] * 5,
        "subsample": [config["subsample"]] * 5,
        "tree_method": ['hist'] * 5,
        "grow_policy": [config["grow_policy"]] * 5,
        "min_child_weight": [config["min_child_weight"]] * 5,
    }
    
    sm = metamodel(objective='multi:softmax')
    sm.xgb_models = sm.create_sense_model(num_models=5, parameters=temp_params)
    sm.train_sense_model(train_x, train_y)
    
    score, acc, unc_wrong, unc_correct = evaluate_uncertainty_score(sm, val_x, val_y)
    
    train.report({
        "score": score, 
        "accuracy": acc, 
        "unc_wrong": unc_wrong, 
        "unc_correct": unc_correct,
        "unc_gap": unc_wrong - unc_correct
    })

def run_sensitivity_search(X, Y, num_trials=20, test_size=0.2, search_space=search_space):
    train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size=test_size, random_state=42)

    algo = OptunaSearch()

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print(f"Starting Distributed Search with {num_trials} trials...")
    
    analysis = tune.run(
        tune.with_parameters(objective, data=(train_x, val_x, train_y, val_y)),
        config=search_space,
        metric="accuracy",
        mode="max",
        num_samples=num_trials,
        resources_per_trial={"cpu": 4},
        search_alg=algo,
        verbose=1
    )

    top_results = analysis.dataframe().sort_values("accuracy", ascending=False).head(5)
    
    final_params = {
        "n_estimators": [],
        "lr": [],
        "max_depth": [],
        "subsample": [],
        "tree_method": [],
        "grow_policy": [],
        "min_child_weight": []
    }

    print("\nTop 5 Configurations found:")
    for i, row in top_results.iterrows():
        # Display the GAP to verify logic is working
        gap = row['unc_wrong'] - row['unc_correct']
        print(f"Trial {i}: Acc: {row['accuracy']:.4f}, Gap: {gap:.4f} (W: {row['unc_wrong']:.2f}, C: {row['unc_correct']:.2f}), Score: {row['score']:.4f}")
        
        final_params["n_estimators"].append(int(row["config/n_estimators"]))
        final_params["lr"].append(float(row["config/lr"]))
        final_params["max_depth"].append(int(row["config/max_depth"]))
        final_params["subsample"].append(float(row["config/subsample"]))
        final_params["tree_method"].append('hist')
        final_params["grow_policy"].append(row["config/grow_policy"])
        final_params["min_child_weight"].append(float(row["config/min_child_weight"]))

    return final_params