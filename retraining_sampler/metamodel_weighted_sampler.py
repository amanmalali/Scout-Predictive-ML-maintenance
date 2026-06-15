import numpy as np

from .base_sampler import BaseSampler

VALID_WEIGHT_TYPES = ("eu", "p_fail", "tu_entropy", "tu_quantile")


class MetamodelWeightedSampler(BaseSampler):
    """
    Sampling weighted by a Scout metamodel uncertainty/failure quantity.

    The sampler expects pre-computed metamodel features and quantile predictions
    for the **combined** [train; stream] pool to be passed as keyword arguments.
    Pre-computation is the caller's responsibility so the sampler remains
    dataset-agnostic (CIFAR needs ViT embeddings; census uses raw features, etc.)

    Required keyword arguments
    --------------------------
    meta_features : ndarray, shape (N_combined, 3)
        Columns: [p_fail (col 0), eu (col 1), tu_entropy (col 2)]
        Produced by  QuantileErrorEstimator._extract_features().
    q_preds : ndarray, shape (N_combined, 99)
        Per-sample quantile predictions across quantiles [0.01, 0.02, ..., 0.99]
        from  q_estimator.qrf.predict(..., quantiles=QUANTILES).
        Used to compute tu_quantile = q_preds[:, -1] - q_preds[:, 0].
    weight_type : str
        One of 'eu', 'p_fail', 'tu_entropy', 'tu_quantile'.

    Sampling
    --------
    Raw weights are taken from the chosen column, floored to 0, then
    normalised to a probability distribution.  A small epsilon is added to
    every weight before normalisation so that samples with weight = 0 still
    have a tiny but non-zero probability of being selected (avoids degenerate
    cases where an entire region of feature space is excluded).
    """

    def sample(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        stream_x: np.ndarray,
        stream_y: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
        **kwargs,
    ) -> np.ndarray:
        meta_features = kwargs.get("meta_features")
        q_preds = kwargs.get("q_preds")
        weight_type = kwargs.get("weight_type", "eu")

        if meta_features is None or q_preds is None:
            raise ValueError(
                "MetamodelWeightedSampler requires 'meta_features' and 'q_preds' "
                "keyword arguments."
            )
        if weight_type not in VALID_WEIGHT_TYPES:
            raise ValueError(
                f"weight_type must be one of {VALID_WEIGHT_TYPES}, got '{weight_type}'."
            )

        n_total = len(train_x) + len(stream_x)
        if len(meta_features) != n_total:
            raise ValueError(
                f"meta_features has {len(meta_features)} rows but combined pool "
                f"has {n_total} samples."
            )

        if weight_type == "p_fail":
            raw_weights = meta_features[:, 0]
        elif weight_type == "eu":
            raw_weights = meta_features[:, 1]
        elif weight_type == "tu_entropy":
            raw_weights = meta_features[:, 2]
        elif weight_type == "tu_quantile":
            raw_weights = q_preds[:, -1] - q_preds[:, 0]

        # Floor at 0 (negative values can appear for tu_quantile in rare edge cases)
        raw_weights = np.clip(raw_weights, 0.0, None)

        # Add a small epsilon so every sample has a non-zero selection probability
        eps = 1e-8
        raw_weights = raw_weights + eps

        weights = raw_weights / raw_weights.sum()

        if n_samples > n_total:
            raise ValueError(
                f"n_samples ({n_samples}) > total pool size ({n_total})."
            )

        return rng.choice(n_total, size=n_samples, replace=False, p=weights)
