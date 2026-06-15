import numpy as np
import torch
import torch.nn.functional as F
from quantile_forest import RandomForestQuantileRegressor
from sklearn.isotonic import IsotonicRegression
from metamodel.uncertainty import model_uncertainty, entropy, entropy_shannon, model_uncertainty_shannon

class QuantileErrorEstimator:
    def __init__(self, n_estimators=200, random_state=42, ent_mode="collision"):
        self.qrf = RandomForestQuantileRegressor(n_estimators=n_estimators, random_state=random_state)
        self.ent_mode = ent_mode
        
        self.failure_threshold = None 
        self.max_train_eu = 0.0        
        self.calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        
        # PARAMETER: Zero-Gate Threshold
        # Any raw risk below 5% is treated as noise and clamped to 0.0
        self.ZERO_GATE = 0.1 
        
        self.test_quantiles = np.linspace(0.01, 0.99, 99).tolist()

    def _extract_features(self, sense_model, X, apply_temp_scaling=True):
        if hasattr(sense_model, 'predict'):
            pred_logits = sense_model.predict(X, apply_temp=apply_temp_scaling)
        else:
            pred_logits = torch.tensor(sense_model.predict(X, output_margin=True))

        if not isinstance(pred_logits, torch.Tensor):
            pred_logits = torch.tensor(pred_logits)
        
        probs = F.softmax(pred_logits, dim=-1)
        mean_probs = torch.mean(probs, dim=1)
        
        if self.ent_mode == "shannon":
            try:
                model_uncer = model_uncertainty_shannon(probs).detach().cpu().numpy()
                uncer = entropy_shannon(mean_probs).detach().cpu().numpy()
            except: 
                 model_uncer = model_uncertainty(probs).detach().cpu().numpy()
                 uncer = entropy(mean_probs).detach().cpu().numpy()
        else:
            model_uncer = model_uncertainty(probs).detach().cpu().numpy()
            uncer = entropy(mean_probs).detach().cpu().numpy()

        model_uncer[model_uncer < 0] = 0
        qr_prob = mean_probs[:, 1].detach().cpu().numpy()

        return np.hstack([
            np.expand_dims(qr_prob, 1),
            np.expand_dims(model_uncer, 1),
            np.expand_dims(uncer, 1)
        ])

    def fit(self, sense_model, X, y_loss, train_loss_distribution=None):
        print(f"   [QRF] Extracting features for {len(y_loss)} samples...")
        features = self._extract_features(sense_model, X, apply_temp_scaling=False)
        
        mu = np.mean(y_loss)
        sigma = np.std(y_loss)
        self.failure_threshold = mu + 2.0 * sigma
        
        print(f"   [QRF] Physical Threshold Set: T = {self.failure_threshold:.4f} (mu+2sigma)")
        print(f"   [QRF] Training Forest on Uncapped Loss (Max: {y_loss.max():.4f})...")
        self.qrf.fit(features, y_loss)

    def calibrate(self, sense_model, X_train, y_train_binary):
        print("   [QRF] Calibrating Probabilities & Epistemic Boundaries...")
        features = self._extract_features(sense_model, X_train, apply_temp_scaling=False)
        self.max_train_eu = np.max(features[:, -2])
        
        y_pred_grid = self.qrf.predict(features, quantiles=self.test_quantiles)
        greater_mask = y_pred_grid > self.failure_threshold
        raw_tail_mass = np.mean(greater_mask, axis=1)
        
        # *** APPLY ZERO-GATE BEFORE CALIBRATION ***
        # This prevents the Isotonic Regressor from learning a noise floor
        raw_tail_mass[raw_tail_mass < self.ZERO_GATE] = 0.0
        
        print(f"   [QRF] Fitting Isotonic Calibrator (Zero-Gated < {self.ZERO_GATE})...")
        self.calibrator.fit(raw_tail_mass, y_train_binary)
        
        print(f"   [QRF] Calibration Complete. Max Known EU: {self.max_train_eu:.4f}")

    def predict(self, sense_model, X):
        features = self._extract_features(sense_model, X, apply_temp_scaling=True)
        eu_scores = features[:, -2]
        
        y_pred_grid = self.qrf.predict(features, quantiles=self.test_quantiles)
        greater_mask = y_pred_grid > self.failure_threshold
        raw_tail_mass = np.mean(greater_mask, axis=1)
        
        # *** APPLY ZERO-GATE DURING INFERENCE ***
        raw_tail_mass[raw_tail_mass < self.ZERO_GATE] = 0.0
        
        calibrated_p_fail = self.calibrator.transform(raw_tail_mass)
        
        return {
            "p_fail": calibrated_p_fail,
            "epistemic_uncertainty": eu_scores
        }