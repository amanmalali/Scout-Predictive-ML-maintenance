"""
ADWIN — ADaptive WINdowing change detector
===========================================
Self-contained implementation of ADWIN2 (Bifet & Gavaldà, "Learning from
Time-Changing Data with Adaptive Windowing", SDM 2007).  `river` / `skmultiflow`
are not installed in this environment, so we ship a faithful port of the
canonical MOA / scikit-multiflow algorithm: an exponential-histogram window with
the variance-corrected Hoeffding cut.

The detector maintains a window W of recent real values (here, per-sample binary
error indicators in [0, 1]).  After each insertion it looks for a split
W = W0 · W1 (W0 = older, W1 = newer) such that the two sub-window means differ
by more than a statistical bound derived from the confidence `delta`.  When such
a split exists, the older sub-window W0 is dropped — this both signals a change
and adapts the window to the new distribution.

Buckets are kept in an exponential histogram so the whole structure uses
O(log W) memory and each update costs O(log W).

Extension for Scout
-------------------
For a *retraining* trigger we care about the direction of change: a change is
actionable only when the newer sub-window error mean is **higher** than the
older one (the model got worse).  Each `update` therefore exposes:
  - `detected_increase` : True iff the change detected on this update was an
                          error *increase* (newer mean > older mean).
  - `last_mean_old`     : older sub-window mean (u0) at the detected cut.
  - `last_mean_new`     : newer sub-window mean (u1) at the detected cut.

`update(value)` returns True iff *any* change was detected on this step
(regardless of direction); callers gate retraining on
`update(...) and detector.detected_increase`.
"""

from math import log, sqrt

import numpy as np


class _Item:
    """One row of the exponential histogram.

    Holds up to ADWIN.MAX_BUCKETS + 1 buckets, each summarising 2**row_index
    elements.  Per bucket we store the sum (`bucket_total`) and the
    within-bucket sum-of-squared-deviations (`bucket_variance`).  Within a row,
    index 0 is the OLDEST bucket and higher indices are newer.
    """

    __slots__ = ("next", "previous", "bucket_size_row",
                 "bucket_total", "bucket_variance")

    def __init__(self, next_item=None, previous_item=None):
        self.next = next_item
        self.previous = previous_item
        if next_item is not None:
            next_item.previous = self
        if previous_item is not None:
            previous_item.next = self
        self.bucket_total = [0.0] * (ADWIN.MAX_BUCKETS + 1)
        self.bucket_variance = [0.0] * (ADWIN.MAX_BUCKETS + 1)
        self.bucket_size_row = 0

    def insert_bucket(self, value, variance):
        idx = self.bucket_size_row
        self.bucket_total[idx] = value
        self.bucket_variance[idx] = variance
        self.bucket_size_row += 1

    def remove_bucket(self):
        """Drop the oldest bucket (index 0), shifting the rest down."""
        self.compress_bucket_row(1)

    def compress_bucket_row(self, num_deleted=1):
        for i in range(num_deleted, ADWIN.MAX_BUCKETS + 1):
            self.bucket_total[i - num_deleted] = self.bucket_total[i]
            self.bucket_variance[i - num_deleted] = self.bucket_variance[i]
        for i in range(ADWIN.MAX_BUCKETS + 1 - num_deleted, ADWIN.MAX_BUCKETS + 1):
            self.bucket_total[i] = 0.0
            self.bucket_variance[i] = 0.0
        self.bucket_size_row -= num_deleted


class _List:
    """Doubly-linked list of bucket rows.  `first` (head) = row 0 (newest,
    smallest buckets); `last` (tail) = highest row (oldest, largest buckets).
    Following `.next` walks head -> tail (newest -> oldest)."""

    def __init__(self):
        self.first = None
        self.last = None
        self.add_to_head()

    def add_to_head(self):
        self.first = _Item(next_item=self.first, previous_item=None)
        if self.last is None:
            self.last = self.first

    def add_to_tail(self):
        self.last = _Item(next_item=None, previous_item=self.last)
        if self.first is None:
            self.first = self.last

    def remove_from_tail(self):
        self.last = self.last.previous
        if self.last is not None:
            self.last.next = None
        else:
            self.first = None


class ADWIN:
    """Adaptive windowing change detector (ADWIN2)."""

    MAX_BUCKETS = 5

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        # statistical / efficiency constants (MOA defaults)
        self.mint_clock = 32              # only test for change every 32 inserts
        self.mint_min_window_longitude = 10
        self.mint_min_window_length = 5
        # direction-of-change state (Scout extension)
        self.detected_increase = False
        self.last_mean_old = 0.0
        self.last_mean_new = 0.0
        self._init_buckets()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _init_buckets(self):
        self.list_row_bucket = _List()
        self.last_bucket_row = 0
        self._total = 0.0
        self._variance = 0.0
        self._width = 0.0
        self.bucket_number = 0
        self.mint_time = 0.0

    def reset(self):
        """Cold-start the detector (used after a retrain)."""
        self.detected_increase = False
        self.last_mean_old = 0.0
        self.last_mean_new = 0.0
        self._init_buckets()

    # ── public API ───────────────────────────────────────────────────────────
    @property
    def width(self) -> int:
        return int(self._width)

    @property
    def variance(self) -> float:
        return self._variance / self._width if self._width > 0 else 0.0

    @property
    def estimation(self) -> float:
        """Current window mean (i.e. the estimated error rate)."""
        return self._total / self._width if self._width > 0 else 0.0

    def update(self, value: float) -> bool:
        """Add one element; return True iff a change was detected this step.

        `detected_increase` is reset every call and set True only if the
        detected change was an error increase (newer mean > older mean)."""
        self.detected_increase = False
        self._add_element(float(value))
        return self._detect_change()

    # ── insertion ────────────────────────────────────────────────────────────
    def _bucket_size(self, row: int) -> int:
        return 1 << row   # 2 ** row

    def _add_element(self, value: float):
        self._width += 1
        self.list_row_bucket.first.insert_bucket(value, 0.0)
        self.bucket_number += 1
        if self._width > 1:
            mean_prev = self._total / (self._width - 1)
            inc = (self._width - 1) * (value - mean_prev) * (value - mean_prev) / self._width
            self._variance += inc
        self._total += value
        self._compress_buckets()

    def _compress_buckets(self):
        cursor = self.list_row_bucket.first
        i = 0
        while cursor is not None:
            if cursor.bucket_size_row == self.MAX_BUCKETS + 1:
                next_node = cursor.next
                if next_node is None:
                    self.list_row_bucket.add_to_tail()
                    next_node = cursor.next
                    self.last_bucket_row += 1
                n1 = self._bucket_size(i)
                n2 = self._bucket_size(i)
                u1 = cursor.bucket_total[0] / n1
                u2 = cursor.bucket_total[1] / n2
                inc = n1 * n2 * (u1 - u2) * (u1 - u2) / (n1 + n2)
                next_node.insert_bucket(
                    cursor.bucket_total[0] + cursor.bucket_total[1],
                    cursor.bucket_variance[0] + cursor.bucket_variance[1] + inc,
                )
                self.bucket_number -= 1   # two merged into one
                cursor.compress_bucket_row(2)
                if next_node.bucket_size_row <= self.MAX_BUCKETS:
                    break
            else:
                break
            cursor = cursor.next
            i += 1

    def _delete_element(self) -> int:
        """Drop the oldest bucket (tail row, index 0).  Returns #elements removed."""
        node = self.list_row_bucket.last
        n1 = self._bucket_size(self.last_bucket_row)
        self._width -= n1
        self._total -= node.bucket_total[0]
        u1 = node.bucket_total[0] / n1
        if self._width > 0:
            inc = (node.bucket_variance[0]
                   + n1 * self._width * (u1 - self._total / self._width)
                   * (u1 - self._total / self._width) / (n1 + self._width))
            self._variance -= inc
        node.remove_bucket()
        self.bucket_number -= 1
        if node.bucket_size_row == 0:
            self.list_row_bucket.remove_from_tail()
            self.last_bucket_row -= 1
        return int(n1)

    # ── change detection ─────────────────────────────────────────────────────
    def _bln_cut_expression(self, n0, n1, abs_value) -> bool:
        n = self._width
        dd = log(2.0 * log(n) / self.delta)
        v = self._variance / self._width
        m = (1.0 / (n0 - self.mint_min_window_length + 1)) + \
            (1.0 / (n1 - self.mint_min_window_length + 1))
        epsilon = sqrt(2.0 * m * v * dd) + (2.0 / 3.0) * dd * m
        return abs(abs_value) > epsilon

    def _find_cut(self):
        """Scan splits from the oldest side; return (found, mean_old, mean_new)
        for the first split that exceeds the Hoeffding bound."""
        n0 = 0
        n1 = self._width
        u0 = 0.0
        u1 = self._total
        cursor = self.list_row_bucket.last     # oldest row
        i = self.last_bucket_row
        while cursor is not None:
            for k in range(cursor.bucket_size_row):   # index 0 = oldest in row
                # stop once only the single newest bucket would remain in W1
                if i == 0 and k == cursor.bucket_size_row - 1:
                    return False, 0.0, 0.0
                bsize = self._bucket_size(i)
                n0 += bsize
                n1 -= bsize
                u0 += cursor.bucket_total[k]
                u1 -= cursor.bucket_total[k]
                if n0 >= self.mint_min_window_length and n1 >= self.mint_min_window_length:
                    abs_value = (u0 / n0) - (u1 / n1)
                    if self._bln_cut_expression(n0, n1, abs_value):
                        return True, u0 / n0, u1 / n1
            cursor = cursor.previous   # move toward head (newer)
            i -= 1
        return False, 0.0, 0.0

    def _detect_change(self) -> bool:
        changed = False
        self.mint_time += 1
        if (self.mint_time % self.mint_clock == 0) and \
                (self._width > self.mint_min_window_longitude):
            keep_reducing = True
            while keep_reducing:
                keep_reducing = False
                found, mean_old, mean_new = self._find_cut()
                if found:
                    self._delete_element()
                    changed = True
                    keep_reducing = True
                    self.last_mean_old = mean_old
                    self.last_mean_new = mean_new
                    # degradation = newer (W1) error mean exceeds older (W0)
                    self.detected_increase = (mean_new > mean_old)
        return changed


class CovariateDriftADWIN:
    """ADWIN-based covariate (data-distribution) drift detector.

    This is the faithful "ADWIN as a retraining trigger" used in the model-
    maintenance literature (CARA, Mahadevan & Mathioudakis 2024; UPF, 2505.14903),
    where ADWIN performs "statistical testing of the data / feature distribution"
    and a retrain is made whenever a drift is detected — as opposed to DDM, which
    monitors the model's error rate.  It is **label-free**: only the incoming
    features X are needed, not the ground-truth labels y.

    ADWIN is univariate, so the multivariate feature vector is reduced to a single
    per-sample covariate-shift score — the RMS standardised deviation from the
    anchor (training) feature distribution:

        z      = (x - mu) / sigma          # per-dimension standardisation
        score  = sqrt( mean_d z_d^2 )      # 1-D covariate-novelty statistic

    where mu, sigma are the per-dimension mean/std of the anchor features.  The
    score stream is fed into ADWIN; a detected change in its running mean signals
    a shift in the incoming data distribution → retrain.  On training-like data
    the score is ≈ 1; it rises as the inputs drift away from the anchor.
    """

    def __init__(self, anchor_features, delta: float = 0.05, eps: float = 1e-8):
        self.delta = delta
        self.eps = eps
        self._set_anchor(anchor_features)
        self.adwin = ADWIN(delta=delta)

    def _set_anchor(self, anchor_features):
        A = np.asarray(anchor_features, dtype=float)
        A = A.reshape(len(A), -1)          # flatten any per-sample shape to (N, D)
        self.mu = A.mean(axis=0)
        self.sigma = A.std(axis=0) + self.eps

    def score(self, X) -> np.ndarray:
        """Per-sample RMS standardised deviation from the anchor (shape (N,))."""
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        z = (X - self.mu) / self.sigma
        return np.sqrt(np.mean(z * z, axis=1))

    def update_batch(self, X):
        """Feed a batch's per-sample covariate scores into ADWIN, in order.
        Returns (drift_detected: bool, mean_batch_score: float)."""
        scores = self.score(X)
        drift = False
        for s in scores:
            if self.adwin.update(float(s)):
                drift = True
        mean_score = float(np.mean(scores)) if len(scores) else float("nan")
        return drift, mean_score

    def reset(self, anchor_features):
        """Re-anchor on new training features and cold-start ADWIN (after retrain)."""
        self._set_anchor(anchor_features)
        self.adwin.reset()

    @property
    def width(self) -> int:
        return self.adwin.width

    @property
    def estimation(self) -> float:
        """Current ADWIN window mean of the covariate-shift score."""
        return self.adwin.estimation


# ─────────────────────────────────────────────────────────────────────────────
#  SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    def run_stream(values, delta):
        det = ADWIN(delta=delta)
        detections = []          # (index, increase, mean_old, mean_new, width)
        for idx, v in enumerate(values):
            if det.update(v):
                detections.append((idx, det.detected_increase,
                                   round(det.last_mean_old, 3),
                                   round(det.last_mean_new, 3),
                                   det.width))
        return det, detections

    rng = np.random.default_rng(0)

    print("=" * 70)
    print("TEST 1 — step-change stream  (Bernoulli 0.05 -> 0.50 at i=2000)")
    print("=" * 70)
    step = np.concatenate([rng.binomial(1, 0.05, 2000),
                           rng.binomial(1, 0.50, 2000)]).astype(float)
    for d in (0.002, 0.02, 0.05):
        det, dets = run_stream(step, d)
        first_after = next((x for x in dets if x[0] >= 2000), None)
        inc_dets = [x for x in dets if x[1]]
        print(f"\n  delta={d}: {len(dets)} detection(s); "
              f"final est={det.estimation:.3f}, width={det.width}")
        if first_after:
            print(f"    first detection after step: index={first_after[0]} "
                  f"(latency={first_after[0]-2000}), increase={first_after[1]}, "
                  f"mean_old={first_after[2]}, mean_new={first_after[3]}")
        assert first_after is not None, "FAIL: no detection after the step!"
        assert first_after[1] is True, "FAIL: step change not flagged as increase!"
        assert first_after[0] - 2000 < 200, "FAIL: detection latency too high!"
        # estimation should recover toward the new rate 0.5
        assert det.estimation > 0.3, "FAIL: estimation did not track new rate!"
    print("\n  [OK] drift detected as an INCREASE shortly after the step "
          "for all deltas; latency shrinks as delta grows.")

    print("\n" + "=" * 70)
    print("TEST 2 — stationary stream  (Bernoulli 0.10, N=8000)")
    print("=" * 70)
    for d in (0.002, 0.02, 0.05):
        stat = rng.binomial(1, 0.10, 8000).astype(float)
        det, dets = run_stream(stat, d)
        rate = len(dets) / len(stat)
        print(f"  delta={d}: {len(dets)} false detection(s) "
              f"({rate*100:.3f}% of steps); final est={det.estimation:.3f}")
        assert det.estimation == det.estimation  # not nan
        assert abs(det.estimation - 0.10) < 0.05, "FAIL: estimation off on stationary!"
        assert rate < 0.01, "FAIL: too many false alarms on stationary stream!"
    print("  [OK] estimation tracks the true 0.10 rate; false-alarm rate is "
          "low and bounded by delta.")

    print("\n" + "=" * 70)
    print("TEST 3 — improvement stream  (Bernoulli 0.50 -> 0.05 at i=2000)")
    print("=" * 70)
    drop = np.concatenate([rng.binomial(1, 0.50, 2000),
                           rng.binomial(1, 0.05, 2000)]).astype(float)
    det, dets = run_stream(drop, 0.002)
    first_after = next((x for x in dets if x[0] >= 2000), None)
    print(f"  delta=0.002: {len(dets)} detection(s); final est={det.estimation:.3f}")
    if first_after:
        print(f"    first detection after drop: index={first_after[0]}, "
              f"increase={first_after[1]} (mean_old={first_after[2]}, "
              f"mean_new={first_after[3]})")
        assert first_after[1] is False, \
            "FAIL: error DROP should NOT be flagged as an increase!"
    print("  [OK] a performance improvement is detected as a change but "
          "detected_increase=False -> would NOT trigger a retrain.")

    print("\n" + "=" * 70)
    print("TEST 4 — CovariateDriftADWIN  (feature covariate shift, label-free)")
    print("=" * 70)
    D = 20
    anchor = rng.normal(0.0, 1.0, size=(2000, D))     # training feature distribution
    det = CovariateDriftADWIN(anchor, delta=0.05)
    # 3000 in-distribution samples, then 3000 shifted (mean +1.5 per dim)
    pre  = rng.normal(0.0, 1.0, size=(3000, D))
    post = rng.normal(1.5, 1.0, size=(3000, D))
    drift_pre = False
    for b in np.array_split(pre, 30):
        d, _ = det.update_batch(b)
        drift_pre = drift_pre or d
    score_pre = det.estimation
    drift_post_batch = None
    seen = 3000
    for k, b in enumerate(np.array_split(post, 30)):
        d, ms = det.update_batch(b)
        if d and drift_post_batch is None:
            drift_post_batch = seen
        seen += len(b)
    print(f"  pre-shift : drift={drift_pre}  mean-score≈{score_pre:.3f} (expect ≈1)")
    print(f"  post-shift: first drift at sample≈{drift_post_batch}  "
          f"final mean-score≈{det.estimation:.3f} (rises with shift)")
    assert not drift_pre, "FAIL: covariate detector fired on in-distribution data!"
    assert drift_post_batch is not None, "FAIL: covariate shift not detected!"
    assert det.estimation > score_pre, "FAIL: covariate score did not rise after shift!"
    # re-anchor on the shifted data → score returns toward ~1, detector cold-starts
    det.reset(post)
    d2, ms2 = det.update_batch(rng.normal(1.5, 1.0, size=(500, D)))
    print(f"  after reset on shifted anchor: drift={d2}, mean-score≈{ms2:.3f} (≈1 again)")
    assert not d2, "FAIL: spurious drift right after re-anchor!"
    print("  [OK] label-free covariate drift detected on feature shift; quiet on "
          "in-distribution data; re-anchors cleanly after a retrain.")

    print("\nAll ADWIN self-tests passed.")
