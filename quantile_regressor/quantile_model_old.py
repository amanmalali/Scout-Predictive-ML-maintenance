import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import mean_absolute_error


# Clean absolute import
from metamodel.uncertainty import model_uncertainty, entropy, entropy_shannon, model_uncertainty_shannon
from utils import jensen_shannon_divergence

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
        
        # Calibration Artefacts (Default initialization)
        self.best_quant = 0.5
        self.mid_quant_1 = 0.6
        self.mid_quant_2 = 0.7
        self.best_upper_quant = 0.9

    def _extract_features(self, sense_model, X,apply_temp_scaling):
        """
        Extracts features: [Class 1 Prob, Epistemic Uncertainty, Total Entropy]
        """
        # Check if the model has a scaler loaded to decide on temperature scaling
        # apply_temp_scaling = False
        # if hasattr(sense_model, 'scaler') and sense_model.scaler is not None:
        #     apply_temp_scaling = True
            
        # Use the metamodel's predict method
        if hasattr(sense_model, 'predict'):
            # The metamodel returns a tensor of logits
            pred_logits = sense_model.predict(X, apply_temp=apply_temp_scaling)
        # else:
        #     # Fallback if a raw model is passed (though not expected based on architecture)
        #     pred_logits = torch.tensor(sense_model.predict(X, output_margin=True))

        # Ensure output is a tensor
        if not isinstance(pred_logits, torch.Tensor):
            pred_logits = torch.tensor(pred_logits)

        # Softmax over classes (dim=-1)
        probs = F.softmax(pred_logits, dim=-1)
        
        # Average probabilities across the ensemble (dim=1)
        # Assuming shape is (Batch, Models, Classes) -> (Batch, Classes)
        mean_probs = torch.mean(probs, dim=1)
        
        # Determine which uncertainty functions to use
        if self.ent_mode == "shannon":
            try:
                model_uncer = model_uncertainty_shannon(probs).detach().cpu().numpy()
                uncer = entropy_shannon(mean_probs).detach().cpu().numpy()
            except NameError: 
                 # Fallback if specific functions are missing in the import
                 model_uncer = model_uncertainty(probs).detach().cpu().numpy()
                 uncer = entropy(mean_probs).detach().cpu().numpy()
        else:
            model_uncer = model_uncertainty(probs).detach().cpu().numpy()
            uncer = entropy(mean_probs).detach().cpu().numpy()

        model_uncer[model_uncer < 0] = 0
        qr_prob = mean_probs[:, 1].detach().cpu().numpy()

        # Stack features
        features = np.hstack([
            np.expand_dims(qr_prob, 1),
            np.expand_dims(model_uncer, 1),
            np.expand_dims(uncer, 1)
        ])
        
        return features

    # def fit(self, sense_model, train_x, train_loss):
    #     """
    #     Trains the Quantile Random Forest on features extracted from the sense_model.
    #     """
    #     print("Extracting features and training Quantile Regressor...")
    #     features = self._extract_features(sense_model, train_x)
    #     self.qrf.fit(features, train_loss)
    #     print("QRF Trained.")

    def fit(self, sense_model, X, y_loss, train_loss_distribution=None, apply_temp_scaling=False):
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
        features = self._extract_features(sense_model, X, apply_temp_scaling)
        
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

    def calibrate_coverage(self, sense_model, val_x, val_y, apply_temp_scaling=False):
        print("Calibrating Quantiles (Prevalence & Coverage Logic)...")
        features = self._extract_features(sense_model, val_x, apply_temp_scaling)
        
        test_quants = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        y_pred_test = self.qrf.predict(features, quantiles=test_quants)
        df = pd.DataFrame(y_pred_test, columns=test_quants)
        df['y'] = val_y

        # --- Logic A: Base Quantile (Prevalence Match) ---
        target_prevalence = df['y'].mean()
        self.best_quant = min(test_quants, key=lambda q: abs(df[q].mean() - target_prevalence))
        print(f"Base Quantile (Prevalence Match {target_prevalence:.4f}): {self.best_quant}")

        # --- Logic B: Upper Quantile (95% Class 1 Coverage) ---
        # We find the first quantile where 95% of Class 1 samples have a prediction >= 0.5
        # (Assuming 0.5 is the binary decision threshold for the error hedge)
        self.best_upper_quant = 0.95 # Default fallback
        for q in test_quants:
            class1_preds = df.loc[df['y'] == 1, q]
            coverage = (class1_preds >= 0.5).mean()
            if coverage >= 0.95:
                self.best_upper_quant = q
                break
        
        # Safety: Ensure Upper is strictly greater than Base
        if self.best_upper_quant <= self.best_quant:
            idx = test_quants.index(self.best_quant)
            self.best_upper_quant = test_quants[min(len(test_quants)-1, idx + 4)]

        print(f"Upper Quantile (95% Class 1 Coverage): {self.best_upper_quant}")

        # --- Logic C: Interpolation ---
        quant_diff = round(float((self.best_upper_quant - self.best_quant) / 3), 3)
        self.mid_quant_1 = round(self.best_quant + quant_diff, 3)
        self.mid_quant_2 = round(self.best_quant + 2 * quant_diff, 3)
        print(f"Interpolated Mids: {self.mid_quant_1}, {self.mid_quant_2}")

    def calibrate(self, sense_model, X_train, train_y, apply_temp_scaling=False):
        import pandas as pd
        import numpy as np
        from sklearn.metrics import mean_absolute_error
        from scipy import stats
        
        print("Calibrating Quantile Regressor thresholds...")
        
        # 1. Extract Meta-Features and Predictions
        train_q_f = self._extract_features(sense_model, X_train, apply_temp_scaling=apply_temp_scaling)
        model_uncer = train_q_f[:, 1]  # Assuming Epistemic Uncertainty is at index 1
        
        test_quants = np.round(np.linspace(0.01, 0.99, 99), 2).tolist()
        y_pred_test = self.qrf.predict(train_q_f, quantiles=test_quants)
        
        # ---------------------------------------------------------
        # 2. CALIBRATE BASE QUANTILE (Cumulative Sum MAE)
        # ---------------------------------------------------------
        quantile_df = pd.DataFrame(y_pred_test, columns=test_quants)
        quantile_df['target'] = train_y
        quantile_sum = quantile_df.cumsum()
        
        min_error = float('inf')
        best_quant = 0.5
        
        for i in range(len(test_quants)):
            mae_error = mean_absolute_error(quantile_sum['target'].values, quantile_sum[test_quants[i]].values)
            # print(test_quants[i], mae_error) # Uncomment to debug base calibration
            if mae_error <= min_error:
                min_error = mae_error
                best_quant = test_quants[i]
        
        self.best_quant = best_quant
        
        # ---------------------------------------------------------
        # 3. CALIBRATE UPPER QUANTILE (Weighted Loss MAE)
        # ---------------------------------------------------------
        df = pd.DataFrame(y_pred_test, columns=test_quants)
        df['y'] = train_y
        df['epis'] = model_uncer
        
        # Map epistemic values to their percentiles
        q = []
        for i in range(len(df)):
            val = df.loc[i, 'epis']
            if hasattr(self, 'quantile') and hasattr(self.quantile, 'find_quantile'):
                q.append(self.quantile.find_quantile(val))
            else:
                # Fallback if self.quantile object isn't present
                q.append(stats.percentileofscore(model_uncer, val) / 100.0)
                
        df['epis_quant'] = q
        df['vote'] = df[test_quants].sum(axis=1)
        
        # Threshold logic
        self.quant_thresh = df.loc[(df['y'] == 1) & (df['epis_quant'] >= 0.1)]['epis_quant'].mean()
        self.min_votes = df.loc[(df['y'] < 1) & (df['epis_quant'] > self.quant_thresh)]['vote'].mean()

        min_error = float('inf')
        best_upper_quant = self.best_quant
        
        for i in range(len(test_quants)):
            # Weighted MAE: Target=1 vs Target<1
            mae_error = mean_absolute_error(df.loc[(df['y'] == 1)]['y'].values, df.loc[(df['y'] == 1)][test_quants[i]].values)
            mae_error2 = mean_absolute_error(df.loc[(df['y'] < 1)]['y'].values, df.loc[(df['y'] < 1)][test_quants[i]].values)
            
            combined_error = 2 * mae_error + mae_error2
            # print(test_quants[i], mae_error, mae_error2, combined_error) # Uncomment to debug upper calibration
            
            if combined_error <= min_error:
                min_error = combined_error
                best_upper_quant = test_quants[i]
                
        if best_upper_quant > 0.99:
            best_upper_quant = 0.99
            
        self.best_upper_quant = best_upper_quant
        
        # ---------------------------------------------------------
        # 4. CALCULATE INTERMEDIATE QUANTILES
        # ---------------------------------------------------------
        quant_diff = round(float((self.best_upper_quant - self.best_quant) / 3), 3)
        self.mid_quant_1 = self.best_quant + quant_diff
        self.mid_quant_2 = self.best_quant + 2 * quant_diff
        
        print(f"BEST BASE QUANT:  {self.best_quant}")
        print(f"BEST MID 1 QUANT: {self.mid_quant_1}")
        print(f"BEST MID 2 QUANT: {self.mid_quant_2}")
        print(f"BEST UPPER QUANT: {self.best_upper_quant}")

    def predict_tiered(self, sense_model, X,apply_temp_scaling=False):
        """
        Predicts using the 4 calibrated levels.
        Returns a dictionary with predictions for each tier.
        """
        features = self._extract_features(sense_model, X, apply_temp_scaling)
        
        # Predict at the 4 specific levels
        levels = [self.best_quant, self.mid_quant_1, self.mid_quant_2, self.best_upper_quant]
        preds = self.qrf.predict(features, quantiles=levels)
        
        return {
            "base_pred": preds[:, 0],
            "mid1_pred": preds[:, 1],
            "mid2_pred": preds[:, 2],
            "upper_pred": preds[:, 3]
        }
    
    def estimate_uncertainty_drift(self, sense_model, X_ref, X_chrono_train, window=100, tests=200, apply_temp_scaling=False):
        from scipy.spatial import distance
        import numpy as np
        
        # 1. Extract features for both the reference and the chronological train chunk
        ref_q_f = self._extract_features(sense_model, X_ref, apply_temp_scaling=apply_temp_scaling)
        test_q_f = self._extract_features(sense_model, X_chrono_train, apply_temp_scaling=apply_temp_scaling)
        
        ref_eu = ref_q_f[:, 1]
        test_eu_chrono = test_q_f[:, 1] 
        
        # 2. Establish Fixed Global Binning (prevents dynamic bin warping)
        min_val = min(np.min(ref_eu), np.min(test_eu_chrono))
        max_val = max(np.max(ref_eu), np.max(test_eu_chrono))
        self.global_bins = np.linspace(min_val, max_val + 1e-5, 51)
        
        def js_div_fixed(p, q):
            p_hist, _ = np.histogram(p, bins=self.global_bins, density=True)
            q_hist, _ = np.histogram(q, bins=self.global_bins, density=True)
            p_hist, q_hist = p_hist + 1e-10, q_hist + 1e-10
            return distance.jensenshannon(p_hist / np.sum(p_hist), q_hist / np.sum(q_hist))

        # 3. Sample chronological windows to establish baseline drift stats
        dist_avg = []
        N = len(test_eu_chrono)
        rng = np.random.default_rng(getattr(self, 'random_state', 42))

        print(f"Calibrating Chronological Drift on TRAIN DATA over {tests} windows of size {window}...")
        for _ in range(tests):
            start = rng.integers(0, max(1, N - window))
            win = test_eu_chrono[start : start + window]
            dist_avg.append(js_div_fixed(ref_eu, win))

        dist_avg = np.asarray(dist_avg)
        self.ref_epistemic = ref_eu
        self.drift_mean = dist_avg.mean()
        self.drift_std = dist_avg.std()
        
        return {"mean": self.drift_mean, "std": self.drift_std}

    # def estimate_uncertainty_drift(self, sense_model, X_ref, X_test, window=100, tests=200, apply_temp_scaling=False):
    #     """
    #     Estimates the distribution of Epistemic Uncertainty drift between a 
    #     reference set and a test set using Jensen-Shannon Divergence.
        
    #     Args:
    #         sense_model: The ensemble model used for feature extraction.
    #         X_ref: The reference dataset (e.g., training or calibration data).
    #         X_test: The target dataset to check for drift.
    #         window (int): The size of the sliding/random window to sample from X_test.
    #         tests (int): Number of random iterations to perform.
    #     """
    #     # 1. Extract features - Epistemic Uncertainty is at index 1 of the returned stack
    #     ref_features = self._extract_features(sense_model, X_ref,apply_temp_scaling)
    #     test_features = self._extract_features(sense_model, X_test,apply_temp_scaling)
        
    #     ref_eu = ref_features[:, 1]
    #     test_eu = test_features[:, 1]
        
    #     dist_avg = []
    #     N = len(test_eu)
        
    #     # Ensure reproducibility if a random_state was provided to the class
    #     rng = np.random.default_rng(getattr(self, 'random_state', None))

    #     print(f"Calculating Epistemic Drift over {tests} windows of size {window}...")

    #     # 2. Random Sampling Loop
    #     for _ in range(tests):
    #         # Pick a random contiguous slice from the test uncertainty vector
    #         start = rng.integers(0, max(1, N - window))
    #         win = test_eu[start : start + window]
            
    #         # Calculate JS Divergence using the provided util function
    #         js_dist = jensen_shannon_divergence(ref_eu, win, bins=50)
    #         dist_avg.append(js_dist)

    #     dist_avg = np.asarray(dist_avg)

    #     # 3. Store artefacts for deployment comparison
    #     self.ref_epistemic = ref_eu
    #     self.drift_mean = dist_avg.mean()
    #     self.drift_std = dist_avg.std()
        
        
    #     return {"mean": self.drift_mean, "std": self.drift_std, "js_distances": dist_avg}

    def predict_at_quantile(self, sense_model, X, quantile):
        """
        Predicts using the 4 calibrated levels.
        Returns a dictionary with predictions for each tier.
        """
        features = self._extract_features(sense_model, X,apply_temp_scaling=True)
        
        # Predict at the 4 specific levels
        preds = self.qrf.predict(features, quantiles=quantile)
        print("QUANTILE:",preds)
        return preds
    