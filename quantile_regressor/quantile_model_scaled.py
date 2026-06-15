import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import mean_absolute_error
import copy
from sklearn.isotonic import IsotonicRegression
import numpy as np
import pandas as pd

# Internal imports assuming standard Scout structure
from metamodel.uncertainty import model_uncertainty, entropy, entropy_shannon, model_uncertainty_shannon

class ScaledLabelGen:
    """
    Helper class to scale regression losses into a [0, 1] probability-like range
    based on the mean and standard deviation of the training loss.
    """
    def __init__(self, losses):
        self.loss_avg = losses.mean()
        self.loss_std = losses.std()
        
    def get_loss_label(self, loss):
        # Baseline is the average loss
        lower = self.loss_avg
        # Upper bound is mean + 2 sigma (approx 95% CI upper bound)
        upper = self.loss_avg + 2 * self.loss_std
        
        if upper == lower: 
            return 0.0
            
        # Linear scaling
        scaled = (loss - lower) / (upper - lower)
        
        # Clip to [0, 1]
        return min(1.0, max(0.0, scaled))

    def transform_array(self, loss_array):
        return np.array([self.get_loss_label(l) for l in loss_array])


class QuantileErrorEstimator:
    def __init__(self, n_estimators=500, random_state=42, ent_mode="collision"):
        """
        Args:
            n_estimators (int): Number of trees in the Quantile Forest.
            random_state (int): Seed for reproducibility.
            ent_mode (str): 'collision' or 'shannon' for uncertainty calculation.
        """
        self.qrf = RandomForestQuantileRegressor(n_estimators=n_estimators, random_state=random_state)
        self.ent_mode = ent_mode
        self.scaler = None
        
        # Calibration Artefacts
        self.q_low = 0.5   # Default safe
        self.q_high = 0.9  # Default safe
        
        # Z-Score Runway Artefacts
        self.roll_mean = 0.0
        self.roll_std = 1.0
        self.z_start = 0.0
        self.z_end = 2.0

    def _extract_features(self, sense_model, X, apply_temp_scaling=True):
        """
        Extracts features: [Class 1 Prob, Epistemic Uncertainty, Total Entropy]
        
        Args:
            sense_model: The metamodel instance.
            X: Input features.
            apply_temp_scaling (bool): Whether to apply temperature scaling during prediction. 
                                       Defaults to True.
        """
        # Use the metamodel's predict method
        if hasattr(sense_model, 'predict'):
            # The metamodel returns a tensor of logits
            # Note: The caller is responsible for passing scaled X if the metamodel expects it
            pred_logits = sense_model.predict(X, apply_temp=apply_temp_scaling)
        else:
            # Fallback if a raw model is passed
            pred_logits = torch.tensor(sense_model.predict(X, output_margin=True))

        # Ensure output is a tensor on CPU for numpy conversion
        if not isinstance(pred_logits, torch.Tensor):
            pred_logits = torch.tensor(pred_logits)
        
        # Softmax over classes (dim=-1)
        probs = F.softmax(pred_logits, dim=-1)
        
        # Average probabilities across the ensemble (dim=1)
        mean_probs = torch.mean(probs, dim=1)
        
        # Determine which uncertainty functions to use
        if self.ent_mode == "shannon":
            try:
                model_uncer = model_uncertainty_shannon(probs).detach().cpu().numpy()
                uncer = entropy_shannon(mean_probs).detach().cpu().numpy()
            except NameError: 
                 model_uncer = model_uncertainty(probs).detach().cpu().numpy()
                 uncer = entropy(mean_probs).detach().cpu().numpy()
        else:
            model_uncer = model_uncertainty(probs).detach().cpu().numpy()
            uncer = entropy(mean_probs).detach().cpu().numpy()

        model_uncer[model_uncer < 0] = 0
        qr_prob = mean_probs[:, 1].detach().cpu().numpy()

        # Stack features: [Prediction, Epistemic, Aleatoric]
        features = np.hstack([
            np.expand_dims(qr_prob, 1),
            np.expand_dims(model_uncer, 1),
            np.expand_dims(uncer, 1)
        ])
        
        return features

    def fit(self, sense_model, X, y_loss, train_loss_distribution=None):
        """
        Trains the Quantile Random Forest.
        
        Args:
            sense_model: The trained metamodel (for feature extraction).
            X: Input features for the training set (e.g., Calibration Unbalanced X).
            y_loss: The loss values to be predicted.
            train_loss_distribution: Optional. If provided, a ScaledLabelGen will be fits
                                     on this distribution and used to scale `y_loss` 
                                     before training. If None, `y_loss` is assumed 
                                     to be already scaled.
        """
        print("Extracting features for QRF Training...")
        # Note: We apply temp scaling by default during training if available
        features = self._extract_features(sense_model, X, apply_temp_scaling=False)
        
        # Handle Scaling
        if train_loss_distribution is not None:
            print("Fitting Loss Scaler on training distribution...")
            self.scaler = ScaledLabelGen(train_loss_distribution)
            target = self.scaler.transform_array(y_loss)
        else:
            # Assume y_loss is already scaled [0,1] or is the raw target desired
            target = y_loss
            
        print(f"Training Quantile Regressor on {len(target)} samples...")
        self.qrf.fit(features, target)
        print("QRF Trained.")

    # def calibrate(self, sense_model, X_train, y_train_binary):
    #     """
    #     Calibrates Q_Low, Q_High and Z-Score Baselines using the 'Concrete Z-Score' method.
        
    #     Args:
    #         sense_model: Trained metamodel.
    #         X_train: Full training set features (balanced or representative).
    #         y_train_binary: Binary labels (0=Nominal, 1=Fail) for prevalence matching.
    #     """
    #     print("--- Starting Concrete Z-Score Calibration ---")
    #     features = self._extract_features(sense_model, X_train, apply_temp_scaling=True)
        
    #     # Generate predictions for a dense grid of quantiles
    #     test_quants = np.linspace(0.05, 0.99, 95)
    #     print(f"Predicting calibration grid ({len(test_quants)} quantiles)...")
    #     y_pred_matrix = self.qrf.predict(features, quantiles=test_quants.tolist())
        
    #     # Extract Epistemic Uncertainty (Index -2 based on _extract_features)
    #     global_eu = features[:, -2]

    #     # ---------------------------------------------------------
    #     # 1. Q_LOW: Global Prevalence Match
    #     # ---------------------------------------------------------
    #     global_binary_sum = y_train_binary.sum()
    #     expected_sums = y_pred_matrix.sum(axis=0)
        
    #     # Find quantile where Sum(Predictions) ~= Sum(True Failures)
    #     idx_low = np.abs(expected_sums - global_binary_sum).argmin()
    #     q_low = test_quants[idx_low]

    #     # ---------------------------------------------------------
    #     # 2. Q_HIGH: Synthetic Prior Match (2x Nominal)
    #     # ---------------------------------------------------------
    #     fail_idx = np.where(y_train_binary == 1)[0]
    #     nom_idx = np.where(y_train_binary == 0)[0]
        
    #     # Mathematically simulate a severe drift environment (2x failure rate vs nominal)
    #     n_severe_nominals = min(len(fail_idx) * 2, len(nom_idx))
        
    #     rng = np.random.RandomState(42)
    #     severe_nom_idx = rng.choice(nom_idx, n_severe_nominals, replace=False)
    #     severe_drift_idx = np.concatenate([fail_idx, severe_nom_idx])
        
    #     severe_binary_sum = y_train_binary[severe_drift_idx].sum()
    #     severe_expected_sums = y_pred_matrix[severe_drift_idx, :].sum(axis=0)
        
    #     idx_high = np.abs(severe_expected_sums - severe_binary_sum).argmin()
    #     q_high = test_quants[idx_high]

    #     # Failsafe: Ensure Q_high is strictly greater than Q_low
    #     if q_high <= q_low:
    #          # Push high quantile up by at least 10 steps or to max
    #         q_high = test_quants[min(len(test_quants)-1, idx_low + 10)]

    #     # ---------------------------------------------------------
    #     # 3. Z-Score Baselines (Rolling Window)
    #     # ---------------------------------------------------------
    #     rolling_eu_train = pd.Series(global_eu).rolling(window=1000, min_periods=1).mean().values
    #     valid_rolling_train = rolling_eu_train[~np.isnan(rolling_eu_train)]
        
    #     roll_mean = float(np.mean(valid_rolling_train))
    #     roll_std = float(np.std(valid_rolling_train))

    #     # Z_START: Pure immediate tracking (0.0 standard deviations from the mean)
    #     z_start = 0.0 

    #     # Z_END: Map the 2x Nominal Severe Subset directly into the rolling Z-space
    #     severe_eu_mean = float(np.mean(global_eu[severe_drift_idx]))
    #     z_end = (severe_eu_mean - roll_mean) / (roll_std + 1e-8)

    #     # Failsafe: Ensure a minimal physical runway exists
    #     if z_end <= z_start:
    #         z_end = z_start + 2.0

    #     # Store Artifacts
    #     self.q_low = float(q_low)
    #     self.q_high = float(q_high)
    #     self.roll_mean = roll_mean
    #     self.roll_std = roll_std
    #     self.z_start = z_start
    #     self.z_end = z_end

    #     print("\n" + "="*50)
    #     print(f"Calibration Complete")
    #     print(f"Q_low:  {self.q_low:.3f} (Global Match)")
    #     print(f"Q_high: {self.q_high:.3f} (Severe Prior Match)")
    #     print(f"Rolling Z-Runway: {z_start:.2f} to {z_end:.2f}")
    #     print("="*50)


    def calibrate(self, sense_model, X_train, y_train_binary):
        """
        Calibrates the model using Epistemic Uncertainty (EU) Vectoring.
        EU is the only mathematically proven signal that survives retraining cycles 
        without succumbing to Uncertainty Inversion on memorized anomalies.
        """
        import pandas as pd
        import numpy as np
        
        print("--- Starting Epistemic Uncertainty (EU) Calibration ---")
        
        # STRICT REQUIREMENT: apply_temp_scaling=False to preserve tree variance (EU)
        features = self._extract_features(sense_model, X_train, apply_temp_scaling=False)
        
        # Shift anchor exclusively to EU
        eu = features[:, -2]

        test_quants = np.linspace(0.05, 0.99, 95)
        print(f"Predicting calibration grid ({len(test_quants)} quantiles)...")
        y_pred_matrix = self.qrf.predict(features, quantiles=test_quants.tolist())
        expected_rates_global = np.maximum.accumulate(y_pred_matrix.mean(axis=0))

        window_size = 100

        # 1. EXTRACT CONTINUOUS ROLLING CURVES
        rolling_rate = pd.Series(y_train_binary).rolling(window=window_size, min_periods=window_size).mean().values
        rolling_eu = pd.Series(eu).rolling(window=window_size, min_periods=window_size).mean().values

        valid_mask = ~np.isnan(rolling_rate) & ~np.isnan(rolling_eu)
        valid_rates = rolling_rate[valid_mask]
        valid_eu = rolling_eu[valid_mask]

        # 2. STATISTICAL MOMENTS & DEADBAND
        mu_rate = float(np.mean(valid_rates))
        sigma_rate = float(np.std(valid_rates))
        if sigma_rate == 0: sigma_rate = 1e-6

        self.mu_eu = float(np.mean(valid_eu))
        self.sigma_eu = float(np.std(valid_eu))
        if self.sigma_eu == 0: self.sigma_eu = 1e-6

        self.q_baseline = float(np.interp(mu_rate, expected_rates_global, test_quants))

        # 3. THE 3-SIGMA CRISIS PROJECTION
        rate_3sigma = min(0.99, mu_rate + (3.0 * sigma_rate))
        eu_3sigma = self.mu_eu + (3.0 * self.sigma_eu)
        
        q_3sigma = float(np.interp(rate_3sigma, expected_rates_global, test_quants))
        
        # Geometric Failsafe: Ensure Q strictly scales up with risk
        if q_3sigma <= self.q_baseline:
            idx_base = np.abs(test_quants - self.q_baseline).argmin()
            q_3sigma = float(test_quants[min(len(test_quants)-1, idx_base + 10)])

        # 4. EXTRACT THE EPISTEMIC GRADIENT
        self.slope_q = float(max(0.0, (q_3sigma - self.q_baseline) / (eu_3sigma - self.mu_eu)))

        print("\n" + "="*50)
        print(f"Epistemic Calibration Complete")
        print(f"Physics Baseline: Rate={mu_rate:.4f} | 3-Sigma Rate={rate_3sigma:.4f}")
        print("-" * 50)
        print(f"Deadband Floor: EU <= {self.mu_eu + self.sigma_eu:.4f} (Locked at Q={self.q_baseline:.3f})")
        print(f"Crisis Vector:  +{self.slope_q:.4f} Q per EU unit above Deadband")
        print("="*50)


    def predict(self, sense_model, X):
        """
        Predicts calibrated low and high quantiles.
        
        Returns:
            dict: {
                'q_low': predictions at calibrated low quantile,
                'q_high': predictions at calibrated high quantile,
                'features': extracted features used for prediction
            }
        """
        features = self._extract_features(sense_model, X, apply_temp_scaling=True)
        
        # Predict at the calibrated levels
        preds = self.qrf.predict(features, quantiles=[self.q_low, self.q_high])
        
        return {
            "q_low_pred": preds[:, 0],
            "q_high_pred": preds[:, 1],
            # Helper for Z-score calculation downstream
            "epistemic_uncertainty": features[:, -2] 
        }

    def get_z_score_params(self):
        """Returns the rolling mean/std and runway bounds for external drift trackers."""
        return {
            "mean": self.roll_mean,
            "std": self.roll_std,
            "z_start": self.z_start,
            "z_end": self.z_end
        }