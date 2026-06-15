import numpy as np

from .base_sampler import BaseSampler


class TemporalBiasedSampler(BaseSampler):
    """
    Temporally-biased sampling that favours more recently arrived data.

    Implements the exponential-decay scheme from:
        "Temporally-Biased Sampling Schemes for Online Model Management"

    Timestamps
    ----------
    - Original training set: all samples receive timestamp t = 0, meaning
      they are treated as having arrived one step *before* the simulation
      begins.
    - Stream sample at 0-based position j receives timestamp t = j + 1, so
      the most recently arrived stream sample has the highest timestamp.

    Weight function
    ---------------
    Given the maximum timestamp T (= N_stream, the number of stream samples
    with GT available), the weight for a sample with timestamp t is:

        w(t) = decay_rate ^ (T - t)

    With decay_rate in (0, 1]:
    - decay_rate = 1.0  ->  uniform sampling (all weights equal)
    - decay_rate -> 0   ->  only the most recent samples are selected

    Weights are then normalised to a valid probability distribution before
    sampling without replacement.

    Parameters
    ----------
    decay_rate : float, default 0.99
        Controls how quickly older samples are down-weighted.  Must be in
        (0, 1].
    """

    def __init__(self, decay_rate: float = 0.99):
        if not (0.0 < decay_rate <= 1.0):
            raise ValueError(f"decay_rate must be in (0, 1], got {decay_rate}.")
        self.decay_rate = decay_rate

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
        n_train = len(train_x)
        n_stream = len(stream_x)
        n_total = n_train + n_stream

        if n_samples > n_total:
            raise ValueError(
                f"n_samples ({n_samples}) > total pool size ({n_total})."
            )

        # Build timestamp array:
        #   training set -> t = 0  (arrived one step before simulation start)
        #   stream sample j (0-based) -> t = j + 1
        timestamps = np.concatenate([
            np.zeros(n_train, dtype=np.float64),
            np.arange(1, n_stream + 1, dtype=np.float64),
        ])

        # T = latest timestamp in the available stream window
        T = float(n_stream)

        # Exponential decay: w_i = decay_rate ^ (T - t_i)
        # Equivalent to: exp((T - t_i) * log(decay_rate))
        # Using log for numerical stability when T is large
        log_r = np.log(self.decay_rate)
        log_weights = (T - timestamps) * log_r  # all <= 0 since T >= t_i
        # Shift so max log_weight = 0 before exp (prevents underflow)
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights /= weights.sum()

        # With aggressive decay rates the oldest training samples (t=0) can
        # underflow to exactly 0.0, leaving fewer non-zero entries than
        # n_samples and causing rng.choice(..., replace=False) to fail.
        # Fix: select all non-zero-weight entries first, then fill the
        # remainder uniformly from the zero-weight pool (old training data).
        nonzero_mask = weights > 0.0
        n_nonzero = int(nonzero_mask.sum())

        if n_nonzero >= n_samples:
            return rng.choice(n_total, size=n_samples, replace=False, p=weights)

        # All "live" (non-zero weight) entries are included; fill the gap
        # with a uniform random draw from the zero-weight (oldest) entries.
        live_indices  = np.where(nonzero_mask)[0]
        stale_indices = np.where(~nonzero_mask)[0]
        n_fill = n_samples - n_nonzero
        fill = rng.choice(stale_indices, size=n_fill, replace=False)
        return np.concatenate([live_indices, fill])
