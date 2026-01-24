import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from quantile_forest import RandomForestQuantileRegressor
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import interp1d

# Absolute import as discussed
from metamodel.uncertainty import model_uncertainty, entropy, entropy_shannon, model_uncertainty_shannon

class UncertaintyCalibratedRegressor:
    def __init__(self, n_estimators=500, random_state=42, ent_mode="collision", target_coverage=0.90):
        """
        Args:
            target_coverage (float): The base safety margin we want to guarantee (e.g. 0.90).
                                     The calibrator will find the quantile needed to achieve this
                                     for a given uncertainty level.
        """
        self.qrf = RandomForestQuantileRegressor(n_estimators=n_estimators, random_state=random_state)
        self.ent_mode = ent_mode
        self.target_coverage = target_coverage
        
        # The mapper that converts Uncertainty -> Quantile
        self.uncertainty_to_quantile_map = None
        self.max_seen_uncertainty = 0.0

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
            # Fallback if a raw model is passed
            pred_logits = torch.tensor(sense_model.predict(X, output_margin=True))

        # Ensure output is a tensor
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

        features = np.hstack([
            np.expand_dims(qr_prob, 1),
            np.expand_dims(model_uncer, 1),
            np.expand_dims(uncer, 1)
        ])
        
        return features, model_uncer

    def fit(self, sense_model, train_x, train_loss):
        print("Training QRF...")
        features, _ = self._extract_features(sense_model, train_x)
        self.qrf.fit(features, train_loss)
        print("QRF Trained.")

    def calibrate(self, sense_model, val_x, val_loss):
        """
        Learns the relationship: How high must the quantile be to ensure 
        'target_coverage' given a specific level of uncertainty?
        """
        print(f"Calibrating Uncertainty -> Quantile Curve (Target Coverage: {self.target_coverage})...")
        features, val_uncer = self._extract_features(sense_model, val_x)
        self.max_seen_uncertainty = np.max(val_uncer)
        
        # 1. Get predictions for a dense grid of quantiles
        # We need a fine grid to find the "tipping point" for each sample
        quant_grid = np.linspace(0.1, 0.99, 20)
        preds = self.qrf.predict(features, quantiles=quant_grid.tolist()) # shape (N, 20)
        
        # 2. For each sample, find the minimum quantile needed to cover the True Loss
        # i.e., find smallest q such that Pred[q] >= True_Loss
        required_quantiles = []
        
        for i in range(len(val_loss)):
            true_l = val_loss[i]
            # Check which quantiles cover this loss
            covers = preds[i, :] >= true_l
            if np.any(covers):
                # Pick the first quantile that covers the error
                idx = np.argmax(covers)
                required_quantiles.append(quant_grid[idx])
            else:
                # Even the 99th quantile failed -> We need to be max conservative
                required_quantiles.append(0.99)
                
        required_quantiles = np.array(required_quantiles)
        
        # 3. Fit Isotonic Regression (Monotonicity Constraint)
        # Hypothesis: Higher Uncertainty => Higher Required Quantile
        # We smooth this relationship to avoid noise.
        
        # We use a Rolling Window to estimate the quantile needed to achieve 'target_coverage'
        # within that uncertainty bucket.
        
        # Sort by uncertainty
        sort_idx = np.argsort(val_uncer)
        sorted_uncer = val_uncer[sort_idx]
        sorted_req_q = required_quantiles[sort_idx]
        
        # Create bins of uncertainty
        n_bins = 10
        bin_edges = np.linspace(sorted_uncer[0], sorted_uncer[-1], n_bins + 1)
        
        x_points = []
        y_points = []
        
        for i in range(n_bins):
            # Mask for current bin
            mask = (sorted_uncer >= bin_edges[i]) & (sorted_uncer < bin_edges[i+1])
            if np.sum(mask) > 5: # Need min samples
                bin_qs = sorted_req_q[mask]
                
                # We want the quantile that satisfies 'target_coverage' of the points in this bin.
                # Example: If target is 0.9, we take the 90th percentile of the "required quantiles" in this bin.
                # This ensures that for this uncertainty level, 90% of samples are covered.
                safe_q = np.percentile(bin_qs, self.target_coverage * 100)
                
                # Center of bin
                x_points.append(np.mean(sorted_uncer[mask]))
                y_points.append(safe_q)
        
        # Ensure we have boundary points for interpolation
        if len(x_points) > 0 and x_points[0] > 0:
            x_points.insert(0, 0.0)
            y_points.insert(0, y_points[0]) # Flat extension to 0
        elif len(x_points) == 0:
             # Fallback for empty bins (rare edge case)
             x_points = [0.0, 1.0]
             y_points = [0.99, 0.99]
            
        # 4. Create the Interpolator
        # We use Isotonic Regression implicitly by ensuring y_points are sorted (or max-pooling them)
        # But simple linear interp is robust enough if bins are good.
        # Let's enforce monotonicity manually just in case:
        y_points = np.maximum.accumulate(y_points)
        
        self.uncertainty_to_quantile_map = interp1d(
            x_points, y_points, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(y_points[0], 0.99) # Extrapolate: Flat below, 0.99 above max
        )
        
        print("Calibration Complete. Mapping function established.")

    def predict_robust(self, sense_model, X):
        """
        Predicts the loss using a quantile dynamically selected based on uncertainty.
        """
        features, model_uncer = self._extract_features(sense_model, X)
        
        # 1. Determine the dynamic quantile for each sample
        # If uncertainty > training max, this automatically clips to 0.99 (Maximum Safety)
        dynamic_quantiles = self.uncertainty_to_quantile_map(model_uncer)
        
        # 2. QRF prediction
        # QRF usually takes a scalar quantile. Since every point needs a DIFFERENT quantile,
        # we have two options:
        # A) Loop (slow)
        # B) Predict a grid and interpolate (fast). 
        # Since we used a grid of 20 in calibration, let's predict that grid and pick the right one.
        
        quant_grid = np.linspace(0.1, 0.99, 20)
        preds_grid = self.qrf.predict(features, quantiles=quant_grid.tolist()) # (N, 20)
        
        final_preds = []
        for i in range(len(X)):
            target_q = dynamic_quantiles[i]
            
            # Find closest index in grid
            # Or use interpolation for precision
            idx = (np.abs(quant_grid - target_q)).argmin()
            
            # If we want to be strictly conservative, pick the next higher grid point
            if quant_grid[idx] < target_q and idx < 19:
                idx += 1
                
            final_preds.append(preds_grid[i, idx])
            
        return np.array(final_preds), dynamic_quantiles