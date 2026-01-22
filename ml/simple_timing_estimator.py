"""
Experiment Log Timing Estimator

Simple, transparent timing estimation using experiment log statistics.
Reads directly from experiment_log.csv instead of maintaining separate timing database.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class SimpleTimingEstimator:
    """
    Dead-simple timing estimator using experiment log percentiles.

    Strategy:
    - Read experiment log CSV
    - Build lookup table: batch_size → percentile statistics
    - Return percentile value (default: 75th for conservative estimates)
    - Scale linearly by k_folds
    - Scale linearly by dataset size (relative to baseline)

    IMPORTANT LIMITATION:
    Dataset size scaling uses LINEAR ASSUMPTION due to lack of empirical data.
    Current experiment log contains only 710-sample datasets (baseline).
    Estimates for other dataset sizes are EXTRAPOLATED and may be inaccurate.
    """

    BASELINE_DATASET_SIZE = 710  # Default baseline if not determined from data

    def __init__(self, experiment_log_path, classifier_type):
        """
        Initialize estimator by loading experiment log.

        Args:
            experiment_log_path: Path to experiment_log.csv
            classifier_type: 'cwt_image' or 'pd_signal'
        """
        self.experiment_log_path = Path(experiment_log_path)
        self.classifier_type = classifier_type

        # Load and build lookup table
        self.timing_table = {}
        self.baseline_dataset_size = None  # Will be set from actual data
        self.inferred_dataset_size = None  # Will be set from actual data if available
        self._load_experiment_log()
    
    def _load_experiment_log(self):
        """Load experiment log and build timing lookup table."""
        if not self.experiment_log_path.exists():
            print(f"Warning: Experiment log not found: {self.experiment_log_path}")
            return

        # Load log
        df = pd.read_csv(self.experiment_log_path)

        # Filter for complete timing data
        required_cols = ['batch_size', 'k_folds', 'total_training_time_minutes']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing required columns in {self.experiment_log_path}")
            return

        # Also check for total_samples if available
        has_samples = 'total_samples' in df.columns
        if has_samples:
            required_cols.append('total_samples')

        timing_df = df[required_cols].dropna()

        if len(timing_df) == 0:
            print(f"Warning: No complete timing data in {self.experiment_log_path}")
            return

        # Determine baseline dataset size (most common size in valid data)
        if has_samples:
            # Filter for valid training runs (not test runs)
            valid_runs = timing_df[
                (timing_df['batch_size'].isin([16, 32, 64, 128])) &
                (timing_df['total_training_time_minutes'] > 10)
            ]
            if len(valid_runs) > 0:
                self.baseline_dataset_size = int(valid_runs['total_samples'].mode()[0])
                # Also store the inferred dataset size for configs that don't specify it
                self.inferred_dataset_size = self.baseline_dataset_size
            else:
                self.baseline_dataset_size = self.BASELINE_DATASET_SIZE
                self.inferred_dataset_size = self.BASELINE_DATASET_SIZE
        else:
            self.baseline_dataset_size = self.BASELINE_DATASET_SIZE
            self.inferred_dataset_size = None  # No dataset size info available

        # Build lookup table by batch_size
        for bs in timing_df['batch_size'].unique():
            subset = timing_df[timing_df['batch_size'] == bs]
            times = subset['total_training_time_minutes']

            self.timing_table[int(bs)] = {
                'count': len(times),
                'median': float(times.median()),
                'p75': float(times.quantile(0.75)),
                'p90': float(times.quantile(0.90)),
                'mean': float(times.mean()),
                'std': float(times.std()),
                'min': float(times.min()),
                'max': float(times.max())
            }
    
    
    @property
    def records(self):
        """
        Compatibility property for backward compatibility.
        Returns a list-like object with length equal to number of timing records.
        """
        # Calculate total number of records from timing_table
        total_count = sum(stats['count'] for stats in self.timing_table.values())
        # Return a list-like object that supports len()
        return [None] * total_count
    
    def estimate_time(self, config, real_complexity=None):
        """
        Estimate training time using experiment log statistics.

        Args:
            config: Configuration dict (must have 'batch_size', optionally 'k_folds', 'total_samples')
            real_complexity: Ignored (kept for API compatibility)

        Returns:
            float: Estimated time in minutes
        """
        batch_size = config.get('batch_size', 16)
        k_folds = config.get('k_folds', 5)
        # Use inferred dataset size if not explicitly provided
        total_samples = config.get('total_samples', self.inferred_dataset_size if hasattr(self, 'inferred_dataset_size') else None)

        # No data - use conservative heuristic
        if not self.timing_table:
            base_time = 30.0 if self.classifier_type == 'cwt_image' else 20.0
            time_estimate = base_time * (k_folds / 5.0)
            # Apply dataset scaling if available
            if total_samples and self.baseline_dataset_size:
                dataset_scale = total_samples / self.baseline_dataset_size
                time_estimate *= dataset_scale
            return time_estimate

        # Try exact batch_size match
        if batch_size in self.timing_table:
            stats = self.timing_table[batch_size]
            # Use 75th percentile for conservative estimate
            base_time = stats['p75']
        else:
            # Use nearest batch_size
            available_bs = sorted(self.timing_table.keys())
            nearest_bs = min(available_bs, key=lambda x: abs(x - batch_size))
            stats = self.timing_table[nearest_bs]
            # Use 90th percentile when extrapolating (more conservative)
            base_time = stats['p90']

        # Scale by k_folds (linear relationship)
        time_estimate = base_time * (k_folds / 5.0)

        # Scale by dataset size (linear assumption)
        if total_samples and self.baseline_dataset_size:
            dataset_scale = total_samples / self.baseline_dataset_size

            # Warn if extrapolating beyond baseline
            if abs(dataset_scale - 1.0) > 0.1:  # More than 10% different
                print(f"⚠️  Timing estimate uses LINEAR scaling assumption for dataset size")
                print(f"    Baseline: {self.baseline_dataset_size} samples, Requested: {total_samples} samples ({dataset_scale:.2f}x)")
                print(f"    Estimate may be inaccurate - collect empirical data for this dataset size!")

            time_estimate *= dataset_scale

        return time_estimate
    
    def get_last_strategy(self):
        """Return the strategy used (for compatibility with old API)."""
        return "Experiment log lookup (75th percentile)"
    
    def record_actual_time(self, config, actual_time_minutes, real_complexity):
        """
        Record actual training time.
        
        Note: This is a no-op since we read directly from experiment log.
        The experiment log is updated by log_experiment_to_csv() in config.py.
        """
        pass  # Data is already in experiment log
    
    def get_stats(self):
        """Get timing statistics for debugging."""
        if not self.timing_table:
            return "No timing data available"

        lines = [f"Timing data from: {self.experiment_log_path.name}"]
        lines.append(f"Total batch sizes: {len(self.timing_table)}")

        if self.baseline_dataset_size:
            lines.append(f"Baseline dataset size: {self.baseline_dataset_size} samples")
        else:
            lines.append(f"Baseline dataset size: {self.BASELINE_DATASET_SIZE} samples (default)")

        for bs in sorted(self.timing_table.keys()):
            stats = self.timing_table[bs]
            lines.append(
                f"  BS={bs}: n={stats['count']}, "
                f"median={stats['median']:.1f}min, "
                f"75th={stats['p75']:.1f}min, "
                f"90th={stats['p90']:.1f}min"
            )

        return "\n".join(lines)


def create_simple_estimator(classifier_type):
    """Create timing estimator for given classifier type."""
    if classifier_type == 'cwt_image':
        from config import get_cwt_experiment_log_path
        log_path = get_cwt_experiment_log_path()
    else:
        from config import get_pd_experiment_log_path
        log_path = get_pd_experiment_log_path()
    
    return SimpleTimingEstimator(log_path, classifier_type)


if __name__ == "__main__":
    # Test with CWT experiment log
    estimator = SimpleTimingEstimator(
        "D:/ME1573_data_processing/ml/logs/cwt/cwt_experiment_log.csv",
        "cwt_image"
    )
    
    print("="*70)
    print("EXPERIMENT LOG TIMING ESTIMATOR TEST")
    print("="*70)
    
    print(f"\n{estimator.get_stats()}")
    
    print("\n" + "="*70)
    print("TIME ESTIMATES (k_folds=5)")
    print("="*70)
    
    # Test different batch sizes
    for bs in [16, 32, 64, 128, 256]:
        config = {'batch_size': bs, 'k_folds': 5}
        estimate = estimator.estimate_time(config)
        
        # Show actual data if available
        if bs in estimator.timing_table:
            stats = estimator.timing_table[bs]
            actual_range = f"{stats['min']:.1f}-{stats['max']:.1f}"
            status = f"(actual range: {actual_range} min, n={stats['count']})"
        else:
            nearest = min(estimator.timing_table.keys(), key=lambda x: abs(x - bs))
            status = f"(extrapolated from BS={nearest})"
        
        print(f"BS={bs:3d}: {estimate:5.1f} min  {status}")
    
    print("\n" + "="*70)
    print("SCALING BY K-FOLDS (batch_size=32)")
    print("="*70)
    
    for kf in [3, 5, 7, 10]:
        config = {'batch_size': 32, 'k_folds': kf}
        estimate = estimator.estimate_time(config)
        print(f"k_folds={kf:2d}: {estimate:5.1f} min")

    print("\n" + "="*70)
    print("SCALING BY DATASET SIZE (batch_size=32, k_folds=5)")
    print("="*70)

    # Test different dataset sizes
    for ds in [710, 1500, 3000, 4284]:
        config = {'batch_size': 32, 'k_folds': 5, 'total_samples': ds}
        estimate = estimator.estimate_time(config)
        scale = ds / 710.0
        status = "baseline" if ds == 710 else f"{scale:.2f}x baseline"
        print(f"Dataset size {ds:4d}: {estimate:5.1f} min  ({status})")
