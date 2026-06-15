from abc import ABC, abstractmethod
import numpy as np


class BaseSampler(ABC):
    """
    Abstract base class for retraining set samplers.

    All samplers select a fixed-size retraining set from the combined pool of
    original training data and streaming data with ground truth available.

    Subclasses implement `sample()` and return a 1-D array of integer indices
    into the concatenated array  [train_x ; stream_x]  (i.e., indices 0 ..
    N_train-1 refer to training samples, indices N_train .. N_train+N_stream-1
    refer to stream samples).  Returning indices instead of sliced arrays keeps
    the sampler dataset-agnostic: the caller can apply the same indices to any
    extra aligned arrays (e.g. image embeddings for CIFAR-10).
    """

    @abstractmethod
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
        """
        Select `n_samples` indices from the combined [train; stream] pool.

        Parameters
        ----------
        train_x : ndarray, shape (N_train, ...)
            Original training features.
        train_y : ndarray, shape (N_train,)
            Original training labels.
        stream_x : ndarray, shape (N_stream, ...)
            Streaming features with ground truth available (up to gt_delay
            cutoff).
        stream_y : ndarray, shape (N_stream,)
            Ground-truth labels for the streaming samples.
        n_samples : int
            Number of samples to select.  Typically equal to
            len(original full_train_x) so the retraining set is the same
            size as the original training set.
        rng : np.random.Generator
            Seeded random generator for reproducibility.
        **kwargs
            Sampler-specific extra arguments (e.g. metamodel features,
            decay rate).

        Returns
        -------
        indices : ndarray of int, shape (n_samples,)
            Indices into the combined array np.vstack([train_x, stream_x]).
            Values in [0, N_train) correspond to training samples; values in
            [N_train, N_train + N_stream) correspond to stream samples.
        """
