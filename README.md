
# Metamodel & Uncertainty Quantification Framework

This repository provides a framework for auditing machine learning models. It trains a **Metamodel** (an ensemble of XGBoost classifiers) to predict when a base model is likely to fail (high loss). It includes tools for **Temperature Scaling** (calibration) and **Uncertainty Decomposition** (separating Aleatoric vs. Epistemic uncertainty).


## Quick Start: The Census Workflow

Follow these steps in order to reproduce the full pipeline.
### Step 0: Generate training and drifted data

Before running the metamodel or the census binary classification model, the training and drifed data must be generated first. From root run

```bash
python ./census/gen_data.py


This will generate training and validation features with it's labels under ./census/data/

### Step 1: Train the Base Model

First, we must train the primary task model (a classifier on Census data). This script trains a PyTorch model, generates predictions, and calculates per-sample losses.

```bash
python example_train_census_model.py

```

* **Action:** Splits data, trains a neural network, and performs inference on validation data.
* **Outputs:** Saves predictions (`train_preds.npy`) and losses (`train_losses.npy`) to `./census/data/`.

### Step 2: Generate Metamodel Data

We convert the regression/classification losses from Step 1 into binary "failure" labels (0 = Low Loss, 1 = High Loss). This creates the dataset  that the Metamodel will learn to predict.

```bash
python example_census_gen_meta_data.py

```

* **Logic:** Uses `build_metamodel_training_set`. By default, samples with loss >  are labeled as "High Loss" (1).
* **Outputs:** `census_metamodel_train_x.npy` and `census_metamodel_train_y.npy`.

### Step 3: Train the Metamodel

You have three options for training the auditor (metamodel):

#### Option A: Standard Metamodel (Fast)

Trains a default ensemble of 5 XGBoost classifiers.

```bash
python example_census_train_standard_metamodel.py

```

* **Result:** A serialized model at `./census/saved_models/stardard_metamodel.pkl`.
* **Includes:** Example code for decomposing uncertainty into Total, Aleatoric, and Epistemic components.

#### Option B: Tuned Metamodel (High Performance)

Uses **Ray Tune** to search for optimal XGBoost hyperparameters (learning rate, depth, subsample) before training the final ensemble.

```bash
python example_census_tune_metamodel.py

```

* **Optimization:** Maximizes a custom score that rewards high accuracy and appropriate uncertainty gaps between correct and incorrect predictions.

#### Option C: Metamodel with Temperature Scaling (Calibrated)

Trains the ensemble and then fits a **Temperature Scaler** (using `temp_scaler.py`) to ensure the predicted probabilities are calibrated. It compares "Global", "Ensemble", and "Feature-Based" (FBTS) scaling.

```bash
python example_census_train_temp_scaler.py

```

* **Result:** Saves a model wrapper containing both the XGBoost ensemble and the best PyTorch temperature scaler to `./census/saved_models/meta_model_with_ts.pkl`.

---

## Component Guide

### 1. The Metamodel

The `metamodel` class (in `metamodel/train_metamodel.py`) wraps an ensemble of XGBoost classifiers. It is used to estimate the probability that the base model has failed on a specific input.

**How to use:**

```python
from metamodel.train_metamodel import metamodel

# Load or Initialize
meta = metamodel(objective='multi:softmax')

# Train (expects balanced class weights handled internally)
meta.train_sense_model(train_x, train_y)

# Generate Raw Logits (Shape: [N_samples, N_models, N_classes])
logits = meta.gen_logits(test_x)

```

### 2. Temperature Scaler

Located in `metamodel/temp_scaler.py`, this module calibrates the logits output by the metamodel. It implements the **UCE (Uncertainty Calibration Error)** loss.

**Modes:**

* **Global:** Single scalar temperature for all inputs.
* **Ensemble:** One scalar temperature per ensemble member.
* **Feat (FBTS):** A neural network (Linear or MLP) predicts the temperature based on input features .

**How to use:**

```python
import torch
import metamodel.temp_scaler as ts

# Assuming you have uncalibrated logits and features
scaler_model = ts.linear_model(in_features=data_dim) # For Feature-based

# Apply scaling
# forward_ext_temp uses dynamic temperatures from the linear model
temps = scaler_model(features)
calibrated_logits = scaler_model.forward_ext_temp(logits, temps)

```

### 3. Uncertainty Quantification

Located in `metamodel/uncertainty.py`. This allows you to decompose the ensemble's predictions into specific types of uncertainty using entropy calculations.

* **Total Uncertainty:** Entropy of the average prediction.
* **Aleatoric Uncertainty (Data):** Average entropy of individual predictions.
* **Epistemic Uncertainty (Model):** Divergence (Total - Aleatoric).

**How to use:**

```python
from metamodel.uncertainty import model_uncertainty, entropy, expected_entropy

# probs shape: (N_samples, N_models, N_classes)
mean_probs = torch.mean(probs, dim=1)

total_unc = entropy(mean_probs)
aleatoric = expected_entropy(probs)
epistemic = model_uncertainty(probs) # or total_unc - aleatoric

```

### 4. Quantile Regressor

The Quantile Regressor is used when you need to predict specific error bounds (e.g., "The error is likely between 0.1 and 0.5") rather than a binary "Success/Fail" label.

**How to use:**

```bash
python example_census_quantile_regressor.py

```

This script relies on the `quantile-forest` package to train a quantile regression model

```

```