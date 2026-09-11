"""
threshold.py -- reconstruction error computation, threshold candidates
(derived from normal data only), and the ml_anomaly_score definition.

ml_anomaly_score(x) = ECDF_train(reconstruction_error(x))
                     = (# TRAIN reconstruction errors <= error(x)) / N_train

  - Bounded [0, 1], monotonic in reconstruction_error by construction.
  - Deterministic: computed against a FIXED, saved reference distribution
    (the training split's reconstruction errors), never re-fit at inference
    time.
  - Directly interpretable: score=0.99 means "higher error than 99% of
    normal training sequences."
  - Grounded in the real training distribution, not an arbitrary constant
    or unexplained min-max range.
  - Contains no root-cause information -- purely a function of
    reconstruction_error. Root-cause classification is Agent 4's job.
"""
import numpy as np


def compute_reconstruction_error(model, X, batch_size=512):
    """Returns (sequence_level_error, per_feature_error) where
    sequence_level_error is mean squared error over all timesteps & features
    (the primary field used everywhere else), and per_feature_error is a
    dict of {feature_index: per-timestep-mean error array} for diagnostics."""
    X_hat = model.predict(X, batch_size=batch_size, verbose=0)
    sq_err = (X - X_hat) ** 2
    sequence_level_error = sq_err.mean(axis=(1, 2))
    return sequence_level_error, sq_err


def threshold_candidates(normal_errors):
    """Candidate thresholds derived ONLY from a normal (held-out, e.g. val)
    reconstruction-error distribution. Never touches labeled fault data."""
    normal_errors = np.asarray(normal_errors)
    return {
        "p95": float(np.percentile(normal_errors, 95)),
        "p975": float(np.percentile(normal_errors, 97.5)),
        "p99": float(np.percentile(normal_errors, 99)),
        "mean_plus_2std": float(normal_errors.mean() + 2 * normal_errors.std()),
        "mean_plus_3std": float(normal_errors.mean() + 3 * normal_errors.std()),
    }


class MLAnomalyScorer:
    """Deterministic ECDF-based anomaly score against a fixed reference
    (training) reconstruction-error distribution."""

    def __init__(self, train_errors_sorted):
        self.train_errors_sorted = np.asarray(train_errors_sorted)
        self.n_train = len(self.train_errors_sorted)

    @classmethod
    def from_file(cls, path):
        return cls(np.load(path))

    def save(self, path):
        np.save(path, self.train_errors_sorted)

    def score(self, reconstruction_error):
        reconstruction_error = np.atleast_1d(reconstruction_error)
        ranks = np.searchsorted(self.train_errors_sorted, reconstruction_error, side="right")
        return ranks / self.n_train
