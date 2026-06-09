"""
Training Time Estimator

Fits a power-law model in log space using experiment log data:
    time = C * complexity^a * batch_size^b * steps_per_epoch^c * epochs^d * k_folds^e

Falls back to a global per-epoch-per-fold rate when OLS inputs are unavailable.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Oldest version to include; earlier runs used CPU and are ~10× slower
_MIN_VERSION = 253


class TrainingTimeEstimator:
    """
    Two-tier timing estimator using experiment log data.

    Tier 1 — OLS (≥20 valid rows, model_complexity and total_samples available):
        5-feature log-space regression with p75 inflation factor.
    Tier 2 — Global rate (≥3 rows of basic timing data):
        p75 of (time / mean_epochs_trained / k_folds), scaled by predicted epochs
        and k_folds at query time.
    """

    def __init__(self, experiment_log_path, classifier_type):
        self.experiment_log_path = Path(experiment_log_path)
        self.classifier_type = classifier_type

        self.ols_model = None           # (coeffs[5], intercept, inflation)
        self.global_rate_p75 = None     # minutes per epoch per fold
        self.utilisation_by_patience = {}  # {patience: p75_utilisation}
        self._valid_rows = 0
        self._tier = 2                  # 1 = OLS, 2 = global rate

        self._load_experiment_log()

    # ------------------------------------------------------------------
    # Data loading and model fitting
    # ------------------------------------------------------------------

    def _load_experiment_log(self):
        if not self.experiment_log_path.exists():
            return

        df = pd.read_csv(self.experiment_log_path, encoding='utf-8', on_bad_lines='skip')

        # Restrict to GPU-training era
        if 'version' in df.columns:
            version_nums = df['version'].astype(str).str.extract(r'(\d+)')[0].apply(
                pd.to_numeric, errors='coerce')
            df = df[version_nums >= _MIN_VERSION].copy()

        # --- Epoch utilisation by patience (used by both tiers) ---
        ep_cols = ['epochs', 'mean_epochs_trained', 'early_stopping_patience']
        if all(c in df.columns for c in ep_cols):
            ep_df = df[ep_cols].dropna()
            ep_df = ep_df[(ep_df['epochs'] > 0) & (ep_df['mean_epochs_trained'] > 0)].copy()
            ep_df['utilisation'] = ep_df['mean_epochs_trained'] / ep_df['epochs']
            ep_df['patience_int'] = ep_df['early_stopping_patience'].astype(int)
            for pat, group in ep_df.groupby('patience_int'):
                self.utilisation_by_patience[pat] = float(group['utilisation'].quantile(0.75))

        # --- Tier 2: global rate (always fit if basic data available) ---
        rate_cols = ['total_training_time_minutes', 'mean_epochs_trained', 'k_folds']
        if all(c in df.columns for c in rate_cols):
            rate_df = df[rate_cols].dropna()
            rate_df = rate_df[
                (rate_df['total_training_time_minutes'] > 0) &
                (rate_df['mean_epochs_trained'] > 0) &
                (rate_df['k_folds'] > 0)
            ]
            if len(rate_df) >= 3:
                rates = (rate_df['total_training_time_minutes']
                         / rate_df['mean_epochs_trained']
                         / rate_df['k_folds'])
                self.global_rate_p75 = float(rates.quantile(0.75))
                self._tier = 2

        # --- Tier 1: OLS (requires full feature set and ≥20 rows) ---
        required = [
            'model_complexity', 'batch_size', 'total_samples',
            'mean_epochs_trained', 'k_folds', 'total_training_time_minutes'
        ]
        if all(c in df.columns for c in required):
            valid = df[required].dropna()
            valid = valid[
                (valid['total_training_time_minutes'] > 0) &
                (valid['mean_epochs_trained'] > 0) &
                (valid['model_complexity'] > 0) &
                (valid['batch_size'] > 0) &
                (valid['total_samples'] > 0) &
                (valid['k_folds'] > 0)
            ].copy()
            self._valid_rows = len(valid)
            if self._valid_rows >= 20:
                self._fit_ols(valid)

    def _fit_ols(self, valid):
        log_features = np.column_stack([
            np.log(valid['model_complexity'].values.astype(float)),
            np.log(valid['batch_size'].values.astype(float)),
            np.log((valid['total_samples'] / valid['batch_size']).values.astype(float)),
            np.log(valid['mean_epochs_trained'].values.astype(float)),
            np.log(valid['k_folds'].values.astype(float)),
        ])
        log_y = np.log(valid['total_training_time_minutes'].values.astype(float))

        A = np.column_stack([log_features, np.ones(len(log_features))])
        result, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
        coeffs = result[:5]
        intercept = result[5]

        log_predicted = log_features @ coeffs + intercept
        residual_ratios = np.exp(log_y) / np.exp(log_predicted)
        inflation = float(np.percentile(residual_ratios, 75))

        self.ols_model = (coeffs, intercept, inflation)
        self._tier = 1

    # ------------------------------------------------------------------
    # Epoch prediction from patience
    # ------------------------------------------------------------------

    def _predict_epochs(self, config_epochs, early_stopping_patience):
        patience = int(early_stopping_patience) if early_stopping_patience else None

        if patience is not None and patience in self.utilisation_by_patience:
            utilisation = self.utilisation_by_patience[patience]
        elif self.utilisation_by_patience:
            nearest = min(self.utilisation_by_patience.keys(),
                          key=lambda p: abs(p - (patience or 10)))
            utilisation = self.utilisation_by_patience[nearest]
        else:
            utilisation = 0.70

        return config_epochs * utilisation

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def estimate_time(self, config, model_complexity=None):
        """
        Estimate training time in minutes.

        Args:
            config: dict with keys batch_size, k_folds, epochs,
                    early_stopping_patience, total_samples (optional).
            model_complexity: pre-computed complexity score (optional).

        Returns:
            float: estimated minutes.
        """
        batch_size = float(config.get('batch_size', 32))
        k_folds = float(config.get('k_folds', 5))
        epochs = float(config.get('epochs', 50))
        early_stopping_patience = config.get('early_stopping_patience', 10)
        total_samples = config.get('total_samples')

        predicted_epochs = self._predict_epochs(epochs, early_stopping_patience)

        # Tier 1: OLS
        if (self.ols_model is not None
                and model_complexity is not None
                and total_samples is not None):
            coeffs, intercept, inflation = self.ols_model
            steps_per_epoch = float(total_samples) / batch_size
            log_features = np.array([
                np.log(float(model_complexity)),
                np.log(batch_size),
                np.log(steps_per_epoch),
                np.log(max(predicted_epochs, 1.0)),
                np.log(k_folds),
            ])
            log_time = np.dot(log_features, coeffs) + intercept
            return float(np.exp(log_time) * inflation)

        # Tier 2: global rate
        if self.global_rate_p75 is not None:
            return self.global_rate_p75 * max(predicted_epochs, 1.0) * k_folds

        # Last resort: fixed heuristic (no log data available)
        base = 30.0 if self.classifier_type == 'cwt_image' else 20.0
        return base * (k_folds / 5.0)

    def get_stats(self):
        if self._tier == 1 and self.ols_model is not None:
            coeffs, _, inflation = self.ols_model
            names = ['complexity', 'batch_size', 'steps/epoch', 'epochs', 'k_folds']
            exp_str = ', '.join(f"{n}^{c:.2f}" for n, c in zip(names, coeffs))
            theory_ok = all(abs(coeffs[i] - 1.0) <= 0.15 for i in [2, 3, 4])
            flag = "" if theory_ok else " [theory check FAIL]"
            return f"OLS ({self._valid_rows} rows) | {exp_str} | inflation={inflation:.2f}{flag}"

        if self._tier == 2 and self.global_rate_p75 is not None:
            return f"Global rate | rate_p75={self.global_rate_p75:.3f} min/epoch/fold"

        return "Heuristic (no log data)"


def create_timing_estimator(classifier_type):
    if classifier_type == 'cwt_image':
        from config import get_cwt_experiment_log_path
        log_path = get_cwt_experiment_log_path()
    else:
        from config import get_pd_experiment_log_path
        log_path = get_pd_experiment_log_path()
    return TrainingTimeEstimator(log_path, classifier_type)


if __name__ == "__main__":
    import sys
    log_path = (
        sys.argv[1] if len(sys.argv) > 1
        else "D:/ME1573_data_processing/ml/logs/cwt/cwt_experiment_log.csv"
    )
    classifier_type = sys.argv[2] if len(sys.argv) > 2 else "cwt_image"

    est = TrainingTimeEstimator(log_path, classifier_type)

    print("=" * 70)
    print("TRAINING TIME ESTIMATOR")
    print("=" * 70)
    print(f"\nLog: {log_path}")
    print(f"Valid rows: {est._valid_rows}")
    print(f"Active tier: {est._tier}")
    print(f"Stats: {est.get_stats()}")

    if est.utilisation_by_patience:
        print("\nEpoch utilisation by patience:")
        for pat in sorted(est.utilisation_by_patience):
            print(f"  patience={pat:3d}: p75 = {est.utilisation_by_patience[pat]:.2f}")

    print("\n" + "=" * 70)
    print("SAMPLE ESTIMATES")
    print("=" * 70)
    sample_configs = [
        {'batch_size': 16, 'k_folds': 5,  'epochs': 50,  'early_stopping_patience': 10, 'total_samples': 710},
        {'batch_size': 32, 'k_folds': 5,  'epochs': 50,  'early_stopping_patience': 10, 'total_samples': 710},
        {'batch_size': 64, 'k_folds': 5,  'epochs': 100, 'early_stopping_patience': 15, 'total_samples': 710},
        {'batch_size': 32, 'k_folds': 3,  'epochs': 50,  'early_stopping_patience': 10, 'total_samples': 710},
        {'batch_size': 32, 'k_folds': 10, 'epochs': 50,  'early_stopping_patience': 10, 'total_samples': 710},
    ]
    for cfg in sample_configs:
        t = est.estimate_time(cfg, model_complexity=500_000)
        print(f"  BS={cfg['batch_size']:3d}, kf={cfg['k_folds']}, ep={cfg['epochs']}, "
              f"pat={cfg['early_stopping_patience']} → {t:.2f} min")
