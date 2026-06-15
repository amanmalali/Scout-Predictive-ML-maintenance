"""
UPFModel — Uncertainty-Performance Forecaster
==============================================
Implements the UPF retraining decision algorithm from:

    "When to retrain a machine learning model" (arXiv:2505.14903v1)

The model tracks observed per-batch performance values and uses ElasticNetCV
linear regression to forecast future performance.  A Monte Carlo comparison at
the 95th-percentile cost level (fixed δ=0.95 per paper) decides whether
retraining is beneficial.

The cost-to-performance ratio α (paper Eq. 3) is supplied via the ``alpha``
parameter.  The total-retrain budget (maximum number of allowed retraining
events) is enforced externally in the simulation loop — this class only
makes the per-batch retrain decision.

Usage (sketch)
--------------
upf = UPFModel(task_type="classification", T_total=500, alpha=0.05, delta_r=10)

# Offline initialisation (meta_train chunks):
for k, (X_chunk, y_chunk) in enumerate(offline_chunks):
    perf = compute_chunk_perf(X_chunk, y_chunk)
    upf.set_z_shift(z_shift_value)
    upf.observe(t_norm=(k + 1) / T_total, perf_value=perf)

# Online stream loop (called from simulation script):
upf.set_z_shift(current_batch_z_shift)
upf.observe(t_norm=batch_j / T_total, perf_value=batch_perf)
should_retrain, info = upf.decide(t_now_norm=batch_j / T_total)
if should_retrain:
    do_retrain()
    upf.reset_after_retrain(t_retrain_norm=batch_j / T_total)
"""

import warnings as _warnings

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler


class UPFModel:
    """
    Uncertainty-Performance Forecaster (UPF) retraining decision model.

    Parameters
    ----------
    task_type : str
        "classification" → Beta distribution on 1-accuracy.
        "regression"     → LogNormal distribution on mean MSE.
    T_total : int
        Total number of online stream batches.  Used to normalise time indices.
    alpha : float
        Cost-to-performance ratio α = c/(eN) from the paper (Eq. 3).
        Added to the retrain-scenario Monte Carlo samples:
            cost_retrain = alpha + perf_samples_retrain
        Analogous to ReCEC's --alpha parameter.
    delta_r : int
        Forecast horizon in number of batches (≡ ΔR in the paper).
    n_mc : int
        Number of Monte Carlo samples drawn for each scenario.  Default: 10 000.
    min_obs : int
        Minimum number of GT-observed batches required before a decision is made.
        Returns ``(False, {'reason': 'insufficient_obs', ...})`` until reached.
    rng_seed : int or None
        Optional random seed for reproducible MC sampling.

    Notes
    -----
    The quantile level δ is fixed at 95% per the paper ("We set the confidence
    threshold of our UPF algorithm to δ=95%, as it is a standard value used for
    confidence intervals.").  It is NOT a configurable parameter.

    The maximum number of allowed retraining events (budget count) is enforced
    externally in the simulation loop; this class only makes the per-batch
    retrain/keep decision.
    """

    def __init__(
        self,
        task_type: str,
        T_total: int,
        alpha: float,
        delta_r: int,
        n_mc: int = 10_000,
        min_obs: int = 5,
        min_samples: int = None,
        rng_seed: int = None,
    ):
        """
        Parameters
        ----------
        min_obs : int
            Minimum number of GT-observed *batches* before making a decision.
            Kept for backward compatibility; ignored when ``min_samples`` is set.
        min_samples : int or None
            Minimum number of GT-observed *data points* (samples) before making
            a decision.  When set, this overrides ``min_obs`` — use this to align
            the readiness threshold with RECEC's ``n_req`` (both in samples).
            Each call to ``observe()`` must pass ``n_samples`` for this to work.
        """
        assert task_type in ("classification", "regression"), (
            f"task_type must be 'classification' or 'regression', got '{task_type}'"
        )
        self.task_type     = task_type
        self.T_total       = int(T_total)
        self.alpha         = float(alpha)
        self.delta_r       = int(delta_r)
        self.delta_r_norm  = delta_r / T_total  # pre-computed fractional horizon
        self.n_mc          = int(n_mc)
        self.min_obs       = int(min_obs)
        self.min_samples   = int(min_samples) if min_samples is not None else None
        self._n_samples    = 0   # cumulative sample count (used with min_samples)
        self.rng           = np.random.default_rng(rng_seed)

        # Observation store: list of (r_ij [4-vector], a_ij scalar) pairs
        self.observations: list[tuple[list[float], float]] = []

        # Normalised time at which the current model was trained.
        # 0.0 for V1 (trained before the stream begins).
        self.t_model_norm: float = 0.0

        # Current-batch z_shift (set externally via set_z_shift before observe/decide)
        self._z_shift: float = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    #  Public setters
    # ─────────────────────────────────────────────────────────────────────────

    def set_z_shift(self, z_shift: float) -> None:
        """
        Set the z_shift for the next observe() or decide() call.

        z_shift is the L1 distance between the current batch's mean feature
        vector and the previous batch's mean feature vector (raw, not scaled).
        Must be ≥ 0; set to 0.0 for the very first batch (no previous batch).
        """
        self._z_shift = max(0.0, float(z_shift))

    def set_model_train_norm(self, t_norm: float) -> None:
        """
        Update the normalised training time of the active model.

        Call this after a retraining event with t_norm = retrain_batch / T_total.
        (Alternatively, use reset_after_retrain which calls this internally.)
        """
        self.t_model_norm = float(t_norm)

    # ─────────────────────────────────────────────────────────────────────────
    #  Core methods
    # ─────────────────────────────────────────────────────────────────────────

    def observe(self, t_j_norm: float, perf_value: float,
                n_samples: int = None) -> None:
        """
        Record one GT-observed performance value.

        Parameters
        ----------
        t_j_norm : float
            Normalised time of this observation: j / T_total.
        perf_value : float
            Observed performance:
            - Classification: mean(misclassification indicators) = 1 - accuracy
            - Regression: mean(per-sample MSE losses)
        n_samples : int or None
            Number of data points this observation represents (i.e. batch size).
            Required when ``min_samples`` is set on the UPFModel; ignored otherwise.

        Feature vector layout (4-D):
            r_ij[0] = i/T       — when the model was trained (t_model_norm)
            r_ij[1] = (j-i)/T   — model age at observation time  ← position 1 so
            r_ij[2] = j/T       — absolute observation time         ElasticNet coord-
            r_ij[3] = z_shift   — distribution shift signal         descent assigns
                                                                     its coefficient
                                                                     to the age feature

        Note: Before the first retrain all observations share t_model_norm = 0,
        making r_ij[0] (i/T) zero-variance and r_ij[2] (j/T) perfectly collinear
        with r_ij[1] ((j-i)/T).  sklearn's coordinate descent updates features in
        index order, so whichever of the collinear pair sits at index 1 receives
        the regression coefficient.  Placing (j-i)/T at index 1 ensures the age
        signal drives the keep-vs-retrain comparison (larger age → worse predicted
        performance → retrain is beneficial).  StandardScaler in decide() prevents
        raw z_shift magnitudes from over-regularising the model.
        """
        r_ij = [
            self.t_model_norm,                      # [0] i/T — when model was trained
            float(t_j_norm) - self.t_model_norm,   # [1] (j-i)/T — model age  ← index 1
            float(t_j_norm),                        # [2] j/T — when perf was measured
            self._z_shift,                          # [3] distribution shift signal
        ]
        self.observations.append((r_ij, float(perf_value)))
        if n_samples is not None:
            self._n_samples += int(n_samples)

    def decide(self, t_now_norm: float) -> tuple[bool, dict]:
        """
        Make a retraining decision.

        Parameters
        ----------
        t_now_norm : float
            Current normalised time: batch_index / T_total.

        Returns
        -------
        should_retrain : bool
        info : dict
            Keys: 'reason', 'n_obs', 'mu_keep', 'mu_retrain', 'sigma2',
                  'q95_keep', 'q95_retrain'
        """
        n_obs = len(self.observations)

        # Readiness check: prefer sample-based threshold (min_samples) for
        # parity with RECEC's n_req; fall back to batch-count (min_obs).
        if self.min_samples is not None:
            not_ready = self._n_samples < self.min_samples
        else:
            not_ready = n_obs < self.min_obs

        if not_ready:
            return False, {
                "reason":      "insufficient_obs",
                "n_obs":       n_obs,
                "n_samples":   self._n_samples,
                "mu_keep":     float("nan"),
                "mu_retrain":  float("nan"),
                "sigma2":      float("nan"),
                "q95_keep":    float("nan"),
                "q95_retrain": float("nan"),
            }

        # ── Fit ElasticNetCV on accumulated observations ──────────────────────
        R = np.array([r for r, _ in self.observations], dtype=np.float64)   # (n_obs, 4)
        A = np.array([a for _, a in self.observations], dtype=np.float64)   # (n_obs,)

        # Scale features so that raw z_shift magnitudes (which can be orders of
        # magnitude larger than the normalised time features) do not cause
        # ElasticNetCV to select excessive regularisation and zero out the age
        # and time coefficients.  StandardScaler gracefully handles zero-variance
        # columns (e.g. i/T = 0 for all V1 observations) by setting their scale
        # to 1, effectively mean-centring them to 0 — no crash or NaN.
        scaler = StandardScaler()
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")   # suppress zero-variance column warning
            R_scaled = scaler.fit_transform(R)

        n_splits = max(2, min(5, n_obs))
        enet = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
            cv=n_splits,
            max_iter=10_000,
            n_jobs=1,
        )
        enet.fit(R_scaled, A)

        # Residual variance σ² from training fit (on scaled features)
        A_hat  = enet.predict(R_scaled)
        sigma2 = float(np.mean((A - A_hat) ** 2))
        sigma2 = max(sigma2, 1e-8)          # numerical floor

        # ── Construct horizon feature vectors ─────────────────────────────────
        t_horizon = t_now_norm + self.delta_r_norm

        # Feature order matches observe(): [i/T, (j-i)/T, j/T, z_shift]
        # "Keep" scenario: current model at horizon
        r_keep = np.array([[
            self.t_model_norm,
            t_horizon - self.t_model_norm,   # [1] age of current model at horizon
            t_horizon,                        # [2] j/T
            self._z_shift,
        ]])

        # "Retrain" scenario: new model trained NOW, evaluated at horizon
        r_retrain_new = np.array([[
            t_now_norm,           # [0] model trained at current time
            self.delta_r_norm,    # [1] model age at horizon = delta_r (fresh)
            t_horizon,            # [2] j/T
            self._z_shift,
        ]])

        # Apply the same scaler fitted on training observations
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            r_keep_s    = scaler.transform(r_keep)
            r_retrain_s = scaler.transform(r_retrain_new)

        mu_keep    = float(enet.predict(r_keep_s)[0])
        mu_retrain = float(enet.predict(r_retrain_s)[0])

        # ── Monte Carlo 95th-percentile cost comparison ───────────────────────
        samples_keep    = self._sample(mu_keep,    sigma2)  # (n_mc,)
        samples_retrain = self._sample(mu_retrain, sigma2)  # (n_mc,)

        cost_keep    = samples_keep                          # C_keep = perf at horizon
        cost_retrain = self.alpha + samples_retrain         # C_retrain = α + perf  (paper Eq. 15)

        q95_keep    = float(np.percentile(cost_keep,    95))
        q95_retrain = float(np.percentile(cost_retrain, 95))

        # Retrain if the 95th-pct retrain cost is lower than the 95th-pct keep cost
        should_retrain = q95_retrain < q95_keep

        return should_retrain, {
            "reason":      "decided",
            "n_obs":       n_obs,
            "mu_keep":     mu_keep,
            "mu_retrain":  mu_retrain,
            "sigma2":      sigma2,
            "q95_keep":    q95_keep,
            "q95_retrain": q95_retrain,
        }

    def reset_after_retrain(self, t_retrain_norm: float) -> None:
        """
        Cold-start: clear all observations and update model training time.

        Called immediately after a retraining event completes.  UPF will
        withhold retrain decisions until min_obs new GT batches are observed.

        Parameters
        ----------
        t_retrain_norm : float
            Normalised time of the retrain event: retrain_batch / T_total.
        """
        self.observations  = []
        self._n_samples    = 0
        self.t_model_norm  = float(t_retrain_norm)
        self._z_shift      = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    #  Distribution sampling helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _sample(self, mu: float, sigma2: float) -> np.ndarray:
        """Dispatch to task-appropriate distribution sampler."""
        if self.task_type == "classification":
            return self._sample_beta(mu, sigma2)
        else:
            return self._sample_lognormal(mu, sigma2)

    def _sample_beta(self, mu: float, sigma2: float) -> np.ndarray:
        """
        Sample from Beta(α, β) parameterised by mean μ and variance σ².

        Paper §3.3:
            α = μ · (μ(1-μ)/σ² - 1)
            β = (1-μ) · (μ(1-μ)/σ² - 1)

        Falls back to Beta(1, 1) (uniform) when parameters are degenerate
        (σ² ≥ μ(1-μ), or α ≤ 0, or β ≤ 0).
        """
        # Clip mean to (0, 1) to guarantee well-defined Beta parameters
        mu = float(np.clip(mu, 1e-6, 1.0 - 1e-6))

        max_var = mu * (1.0 - mu)
        if sigma2 >= max_var:
            # Degenerate case — uninformative prior
            return self.rng.beta(1.0, 1.0, size=self.n_mc)

        scale = max_var / sigma2 - 1.0    # > 0 guaranteed by the guard above
        alpha = mu * scale
        beta_ = (1.0 - mu) * scale

        if alpha <= 0 or beta_ <= 0 or np.isnan(alpha) or np.isnan(beta_):
            return self.rng.beta(1.0, 1.0, size=self.n_mc)

        return self.rng.beta(alpha, beta_, size=self.n_mc)

    def _sample_lognormal(self, mu: float, sigma2: float) -> np.ndarray:
        """
        Sample from LogNormal(m, v²) parameterised by mean μ and variance σ².

        Paper §3.4:
            v = sqrt( log(1 + σ²/μ²) )
            m = log(μ) - v²/2

        Falls back to a point mass at μ when parameters are numerically
        degenerate (μ ≤ 0 or σ²/μ² overflows).
        """
        mu = max(float(mu), 1e-9)

        ratio = sigma2 / (mu ** 2)
        if ratio <= 0 or not np.isfinite(ratio):
            return np.full(self.n_mc, mu)

        v2 = np.log(1.0 + ratio)
        v  = float(np.sqrt(v2))
        m  = float(np.log(mu) - v2 / 2.0)

        if not np.isfinite(v) or not np.isfinite(m):
            return np.full(self.n_mc, mu)

        # scipy lognorm: X = exp(m + v·Z), Z ~ N(0,1)
        return self.rng.lognormal(mean=m, sigma=v, size=self.n_mc)

    # ─────────────────────────────────────────────────────────────────────────
    #  Diagnostic helpers
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"UPFModel(task_type={self.task_type!r}, T_total={self.T_total}, "
            f"alpha={self.alpha}, delta_r={self.delta_r}, "
            f"n_obs={len(self.observations)}, t_model_norm={self.t_model_norm:.4f})"
        )
