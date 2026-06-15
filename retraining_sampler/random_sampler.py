import numpy as np

from .base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    """
    Uniformly random sampling from the combined [train; stream] pool.

    Draws `n_samples` indices without replacement with equal probability
    across the full combined pool.
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
        n_total = len(train_x) + len(stream_x)
        if n_samples > n_total:
            raise ValueError(
                f"n_samples ({n_samples}) > total pool size ({n_total}). "
                "Consider using replace=True or reducing n_samples."
            )
        return rng.choice(n_total, size=n_samples, replace=False)
