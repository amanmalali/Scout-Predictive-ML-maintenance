import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import mean_absolute_error

# Clean absolute import
from metamodel.uncertainty import model_uncertainty, entropy, entropy_shannon, model_uncertainty_shannon

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

    def _extract_features(self, sense_model, X):
        """
        Extracts features: [Class 1 Prob, Epistemic Uncertainty, Total Entropy]
        """
        # Check if the model has a scaler loaded to decide on temperature scaling
        apply_temp_scaling = False
        if hasattr(sense_model, 'scaler') and sense_model.scaler is not None:
            apply_temp_scaling = True

        # Use the metamodel's predict method
        if hasattr(sense_model, 'predict'):
            # The metamodel returns a tensor of logits
            pred_logits = sense_model.predict(X, apply_temp=apply_temp_scaling)
        else:
            # Fallback if a raw model is passed (though not expected based on architecture)
            pred_logits = torch.tensor(sense_model.predict(X, output_margin=True))

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

    def fit(self, sense_model, train_x, train_loss):
        """
        Trains the Quantile Random Forest on features extracted from the sense_model.
        """
        print("Extracting features and training Quantile Regressor...")
        features = self._extract_features(sense_model, train_x)
        self.qrf.fit(features, train_loss)
        print("QRF Trained.")

    def calibrate(self, sense_model, val_x, val_y, val_loss):
        """
        Calibrates the 4 quantile levels (Base, Mid1, Mid2, Upper) using logic:
        1. Base Quantile: Minimizes error of Cumulative Sum (Aggregate accuracy).
        2. Upper Quantile: Minimizes Weighted MAE (2x weight on 'Hard' Class 1).
        """
        print("Calibrating Quantiles...")
        features = self._extract_features(sense_model, val_x)
        
        # Predict a wide range of quantiles for testing
        test_quants = [0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 
                       0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        y_pred_test = self.qrf.predict(features, quantiles=test_quants)
        
        # --- Logic Part A: Find Base Quantile via Cumulative Sum MAE ---
        quantile_df = pd.DataFrame(y_pred_test, columns=test_quants)
        quantile_df['target'] = val_loss 
        
        quantile_sum = quantile_df.cumsum()
        
        min_error = 1e9
        best_quant = 0.5 
        
        for q in test_quants:
            # Metric: MAE of the Cumulative Sums
            mae_error = mean_absolute_error(quantile_sum['target'].values, quantile_sum[q].values)
            
            if mae_error <= min_error:
                min_error = mae_error
                best_quant = q
        
        self.best_quant = best_quant
        print(f"Best Base Quantile (CumSum): {self.best_quant}")

        # --- Logic Part B: Find Upper Quantile via Weighted MAE ---
        # Re-using the DF, adding binary labels for splitting
        df = quantile_df.copy()
        df['y'] = val_y # Binary labels (0=Easy, 1=Hard/Error)
        
        min_error = 1e9
        best_upper_quant = 0.9 
        
        for q in test_quants:
            # Metric: 2 * (MAE on Class 1) + (MAE on Class 0)
            mae_error_1 = mean_absolute_error(df.loc[df['y'] == 1]['y'].values, df.loc[df['y'] == 1][q].values)
            mae_error_0 = mean_absolute_error(df.loc[df['y'] < 1]['y'].values, df.loc[df['y'] < 1][q].values)
            
            weighted_error = 2 * mae_error_1 + mae_error_0
            
            if weighted_error <= min_error:
                min_error = weighted_error
                best_upper_quant = q

        if best_upper_quant > 1.0: 
            best_upper_quant = 0.99
            
        self.best_upper_quant = best_upper_quant

        # --- Logic Part C: Interpolate Mid Quantiles ---
        quant_diff = round(float((self.best_upper_quant - self.best_quant) / 3), 3)
        self.mid_quant_1 = self.best_quant + quant_diff
        self.mid_quant_2 = self.best_quant + 2 * quant_diff
        
        print(f"Final Quantiles -> Base: {self.best_quant}, Mid1: {self.mid_quant_1}, Mid2: {self.mid_quant_2}, Upper: {self.best_upper_quant}")

    def predict_tiered(self, sense_model, X):
        """
        Predicts using the 4 calibrated levels.
        Returns a dictionary with predictions for each tier.
        """
        features = self._extract_features(sense_model, X)
        
        # Predict at the 4 specific levels
        levels = [self.best_quant, self.mid_quant_1, self.mid_quant_2, self.best_upper_quant]
        preds = self.qrf.predict(features, quantiles=levels)
        
        return {
            "base_pred": preds[:, 0],
            "mid1_pred": preds[:, 1],
            "mid2_pred": preds[:, 2],
            "upper_pred": preds[:, 3]
        }