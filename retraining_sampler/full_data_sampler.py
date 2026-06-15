import numpy as np
from .base_sampler import BaseSampler


class FullDataSampler(BaseSampler):
    """
    Returns ALL indices from the combined [train; stream] pool.

    Unlike other samplers, this one ignores `n_samples` and selects every
    available sample — the entire original training set plus every stream
    sample that has ground truth available.  This replicates the 'full data'
    retraining strategy but within the sampler interface so it can be compared
    directly against size-constrained strategies.
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
        return np.arange(len(train_x) + len(stream_x))
