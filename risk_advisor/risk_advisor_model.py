import xgboost as xgb
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

# ─────────────────────────────────────────────
#  THRESHOLD HELPERS
# ─────────────────────────────────────────────

def best_fbeta_threshold(y_true, scores, beta: float = 1.0):
    """
    Pick the score cut-off that maximises Fβ on a validation set.

    Parameters
    ----------
    y_true : 1-D array-like  Ground-truth binary labels (0/1).
    scores : 1-D array-like  Continuous prediction scores (larger => more positive).
    beta   : float           β > 1 favours recall, β < 1 favours precision.

    Returns
    -------
    t_best : float   Threshold achieving highest Fβ.
    f_best : float   Value of Fβ at t_best.
    """
    p, r, thr = precision_recall_curve(y_true, scores)
    fbeta = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-12)
    i_best = np.nanargmax(fbeta)
    if i_best == len(thr):
        t_best = scores.min() - 1e-12
    else:
        t_best = thr[i_best]
    return float(t_best), float(fbeta[i_best])


def find_best_thresh(sense_model, calib_x, calib_y, beta: float = 1.0):
    """
    Compute per-sample risk scores from the risk advisor and find the optimal
    F-beta threshold on a calibration set.

    Risk score = Shannon entropy of mean ensemble probs + prob(class=1).

    Parameters
    ----------
    sense_model : riskadvisor_model   Trained risk advisor (sense_model populated).
    calib_x     : np.ndarray          Scaled calibration features.
    calib_y     : np.ndarray          Binary ground-truth labels.
    beta        : float               F-beta beta parameter (default 1.0).

    Returns
    -------
    threshold : float   Best risk-score threshold.
    f_score   : float   F-beta at that threshold.
    """
    from metamodel.uncertainty import entropy_shannon

    logits = sense_model.gen_logits(calib_x)          # (N, num_models, 2)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    probs = F.softmax(logits_t, dim=-1)                # (N, num_models, 2)
    mean_probs = torch.mean(probs, dim=1)              # (N, 2)
    total_uncer = entropy_shannon(mean_probs).detach().numpy()  # (N,)
    p1 = mean_probs[:, 1].detach().numpy()             # (N,)
    risk_scores = total_uncer + p1                     # (N,)

    return best_fbeta_threshold(np.array(calib_y), risk_scores, beta=beta)


# ─────────────────────────────────────────────
#  HYPERPARAMETERS
# ─────────────────────────────────────────────

_PARAMETERS = {
    "n_estimators":    [400, 600, 1000, 600,  800, 400, 600, 1000, 600,  800],
    "lr":              [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "max_depth":       [10,  10,  10,  5,   5,   10,  10,  10,  5,   5],
    "subsample":       [0.7, 0.6, 0.7, 0.5, 0.7, 0.7, 0.6, 0.7, 0.5, 0.7],
    "grow_policy":     [
        'depthwise', 'lossguide', 'depthwise', 'lossguide', 'lossguide',
        'depthwise', 'lossguide', 'depthwise', 'lossguide', 'lossguide',
    ],
    "min_child_weight": [5, 5, 0.5, 5, 0.2, 5, 5, 0.5, 5, 0.2],
}


# ─────────────────────────────────────────────
#  RISK ADVISOR MODEL
# ─────────────────────────────────────────────

class riskadvisor_model:
    """
    Ensemble of 10 XGBoost classifiers that predicts per-sample failure risk
    for classification tasks.

    Labels: 0 if prediction == true_y, 1 if prediction != true_y.
    Trained WITHOUT class-weight balancing.

    After training, call find_best_thresh() to obtain the operating threshold.
    """

    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.xgb_models = self._create_models()
        self.sense_model = None   # Populated after train_sense_model()
        self.lin_model = None     # Unused (reserved for future temp scaling)
        self.scaler = None        # Unused (reserved for future temp scaling)
        self.data_scaler = None   # StandardScaler fitted on training features

    def _create_models(self, num_models: int = 10):
        models = []
        for i in range(num_models):
            clf = xgb.XGBClassifier(
                device=self.device,
                n_estimators=_PARAMETERS["n_estimators"][i],
                learning_rate=_PARAMETERS["lr"][i],
                max_depth=_PARAMETERS["max_depth"][i],
                subsample=_PARAMETERS["subsample"][i],
                tree_method='hist',
                grow_policy=_PARAMETERS["grow_policy"][i],
                min_child_weight=_PARAMETERS["min_child_weight"][i],
                n_jobs=-1,
                objective='multi:softmax',
                num_class=2,
            )
            models.append(("clf" + str(i), clf))
        return models

    def gen_calib_data(self, train_x, train_y, train_loss, test_size: float = 0.1):
        """
        Split (train_x, train_y, train_loss) into a training portion and an
        unbalanced calibration portion used for threshold finding.

        Returns
        -------
        train_x, train_y, train_loss, calib_un_x, calib_un_y, calib_un_loss
        """
        train_y = train_y.reshape(-1, 1)
        train_loss = train_loss.reshape(-1, 1)
        train_labels = np.concatenate([train_y, train_loss], axis=1)

        train_x, calib_un_x, train_labels, calib_un_labels = train_test_split(
            train_x, train_labels, test_size=test_size
        )

        train_y = train_labels[:, 0]
        train_loss = train_labels[:, 1]
        calib_un_y = calib_un_labels[:, 0]
        calib_un_loss = calib_un_labels[:, 1]

        return train_x, train_y, train_loss, calib_un_x, calib_un_y, calib_un_loss

    def gen_logits(self, input_x):
        """
        Collect raw margin logits from every ensemble member.

        Returns
        -------
        logits_preds : np.ndarray, shape (N, num_models, num_classes)
        """
        num_models = len(self.sense_model)
        model_logits = []
        for n in range(num_models):
            logits = np.array(
                self.sense_model[n][1].predict(input_x, output_margin=True)
            )
            model_logits.append(logits)
        model_logits = np.array(model_logits)     # (num_models, N, num_classes)

        logits_preds = []
        for row in range(len(input_x)):
            logits_preds.append(model_logits[:, row, :])
        return np.array(logits_preds)             # (N, num_models, num_classes)

    def train_sense_model(self, train_x, train_y):
        """
        Fit all ensemble members.  No class-weight balancing.
        Populates self.sense_model.
        """
        for i in range(len(self.xgb_models)):
            print(f"Training risk advisor model {i} ...")
            self.xgb_models[i][1].fit(train_x, train_y)
        self.sense_model = np.array(self.xgb_models)

    def inf(self, X, apply_temp: bool = False):
        """
        Run inference and return ensemble logits as a torch.Tensor.

        Parameters
        ----------
        X          : np.ndarray, shape (N, D)
        apply_temp : bool  Unused; kept for API compatibility.

        Returns
        -------
        scaled_logits : torch.Tensor, shape (N, num_models, num_classes)
        """
        logits = self.gen_logits(X)
        return torch.tensor(logits, dtype=torch.float32)
