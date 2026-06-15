"""
DDLAModel
=========
Implements the DDLA (Decision-path Data-Loss Anomaly) technique from
"Efficiently Mitigating the Impact of Data Drift on Machine Learning Pipelines".

Core idea
---------
1. Train a DecisionTreeClassifier on (features, classification-error labels).
2. Extract every root-to-leaf path that leads to class=1 (misclassification).
3. At inference time, compute the fraction of an incoming batch whose samples
   fall into ANY of those high-error paths  →  ``batch_proportion``.
4. The baseline ``train_proportion`` is the same fraction over training data.

Deviation from the legacy code
-------------------------------
* Path conditions are stored as integer (feature-index, operator, threshold)
  tuples rather than string (feature-name, operator, threshold) tuples.
* Matching is fully vectorised with NumPy — the original per-row pandas
  ``apply()`` is O(N × paths × depth) and far too slow for stream batches.
* An optional grid-search constructor (``fit_grid_search``) is exposed for
  tuning max_depth / min_samples_leaf.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


class DDLAModel:
    """
    Decision-path based misclassification rate estimator.

    Attributes
    ----------
    decision_paths  : list of paths; each path is a list of
                      (feature_index: int, operator: str, threshold: float)
    train_proportion: float — fraction of training rows that matched any path
    n_features      : int  — dimensionality of the feature space
    """

    def __init__(self) -> None:
        self.decision_paths:  list  = []
        self.train_proportion: float = 0.0
        self.n_features:       int  = 0

    # ── PUBLIC API ─────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            use_grid_search: bool = False) -> "DDLAModel":
        """
        Fit the DDLA decision tree and extract high-error decision paths.

        Parameters
        ----------
        X_train         : (N, D) feature array (no scaling required).
        y_train         : (N,) binary labels — 1 = misclassified, 0 = correct.
        use_grid_search : bool  Use GridSearchCV to tune max_depth & min_samples_leaf.

        Returns
        -------
        self (for chaining).
        """
        self.n_features = X_train.shape[1]

        prop0 = float(np.mean(y_train == 0))
        prop1 = float(np.mean(y_train == 1))
        print(f"  [DDLAModel] Label distribution: correct={prop0:.3f}  error={prop1:.3f}")

        if use_grid_search:
            tree = self._fit_grid_search(X_train, y_train)
        else:
            tree = DecisionTreeClassifier()
            tree.fit(X_train, y_train)

        self.decision_paths = self._extract_paths(tree)
        print(f"  [DDLAModel] Extracted {len(self.decision_paths)} high-error paths "
              f"from tree with {tree.tree_.node_count} nodes.")

        self.train_proportion = self._compute_proportion(X_train)
        print(f"  [DDLAModel] train_proportion = {self.train_proportion:.4f}")

        return self

    def predict_proportion(self, X_batch: np.ndarray) -> float:
        """
        Returns the fraction of rows in X_batch that match any high-error path.

        Parameters
        ----------
        X_batch : (N, D) feature array (same space as training features).

        Returns
        -------
        proportion : float in [0, 1].
        """
        if len(self.decision_paths) == 0:
            return 0.0
        matches = self._batch_matches_any(X_batch)
        return float(matches.mean())

    def predict_failures(self, X_batch: np.ndarray) -> float:
        """
        Returns the estimated failure count for X_batch.

        Returns
        -------
        failures : float — batch_proportion × len(X_batch).
        """
        return self.predict_proportion(X_batch) * len(X_batch)

    # ── INTERNALS ──────────────────────────────────────────────────────────

    def _fit_grid_search(self, X, y):
        """Grid search over max_depth and min_samples_leaf (mirrors ddla_build_grid_search)."""
        param_grid = {
            "max_depth":        [5, 10, 15, 20, None],
            "min_samples_leaf": [1, 5, 10, 20],
        }
        gs = GridSearchCV(
            DecisionTreeClassifier(), param_grid, cv=3,
            scoring="balanced_accuracy", n_jobs=-1
        )
        gs.fit(X, y)
        print(f"  [DDLAModel] Grid search best params: {gs.best_params_}  "
              f"score={gs.best_score_:.3f}")
        return gs.best_estimator_

    def _extract_paths(self, tree: DecisionTreeClassifier) -> list:
        """
        Recursively extract all root-to-leaf paths that predict class=1.

        Paths stored as lists of (feature_index, operator, threshold).
        Operator is either "<=" (left child) or ">" (right child).
        """
        t      = tree.tree_
        paths  = []

        def recurse(node, current_path):
            if t.feature[node] < 0:                      # leaf
                # class-1 leaf: more class-1 samples than class-0
                if t.value[node][0][1] > t.value[node][0][0]:
                    paths.append(current_path[:])        # copy
                return

            feat = int(t.feature[node])
            thr  = float(t.threshold[node])
            recurse(t.children_left[node],  current_path + [(feat, "<=", thr)])
            recurse(t.children_right[node], current_path + [(feat, ">",  thr)])

        recurse(0, [])
        return paths

    def _batch_matches_any(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorised match: returns boolean array of shape (N,), True where
        the sample satisfies ALL conditions of at least ONE path.
        """
        any_match = np.zeros(len(X), dtype=bool)
        for path in self.decision_paths:
            mask = np.ones(len(X), dtype=bool)
            for feat_idx, op, thr in path:
                col = X[:, feat_idx]
                if op == "<=":
                    mask &= col <= thr
                else:
                    mask &= col > thr
                if not mask.any():     # early exit if nothing left
                    break
            any_match |= mask
            if any_match.all():        # early exit if everything already matched
                break
        return any_match

    def _compute_proportion(self, X: np.ndarray) -> float:
        if len(self.decision_paths) == 0:
            return 0.0
        return float(self._batch_matches_any(X).mean())
