"""
Simple Timing Estimator - Percentile-Based Approach

This module provides reliable timing estimation using historical percentiles.
Uses 75th percentile of actual training times grouped by batch size.
Conservative approach to avoid underestimation.
"""

import json
import datetime
import numpy as np
from pathlib import Path
from collections import defaultdict


class SimpleTimingEstimator:
    """Percentile-based timing estimator using real training time distributions."""

    def __init__(self, timing_db_path, classifier_type):
        self.timing_db_path = Path(timing_db_path)
        self.classifier_type = classifier_type
        self.records = self._load_records()
        self._last_strategy = None

    def _load_records(self):
        """Load timing records, create empty if not exists."""
        if self.timing_db_path.exists():
            try:
                with open(self.timing_db_path, 'r') as f:
                    data = json.load(f)
                return data.get('records', [])
            except:
                pass
        return []

    def estimate_time(self, config, real_complexity=None):
        """
        Estimate training time using percentile-based approach.

        Strategy:
        1. Look for historical data matching batch_size and k_folds
        2. Use 75th percentile (conservative) if exact match found
        3. Apply complexity scaling only if model differs significantly (>50%)
        4. Fall back to 90th percentile of all data if no match

        Args:
            config: Configuration dict (must have 'batch_size' and 'k_folds')
            real_complexity: Real Keras parameter count (optional, for scaling)

        Returns:
            float: Estimated time in minutes
        """
        batch_size = config['batch_size']
        k_folds = config.get('k_folds', 5)

        if len(self.records) < 3:
            fold_scale = k_folds / 5.0
            batch_scale = 16.0 / batch_size
            base_time = 30.0 if self.classifier_type == 'cwt_image' else 20.0
            self._last_strategy = "heuristic (insufficient data)"
            return max(base_time * fold_scale * batch_scale, 1.0)

        by_config = defaultdict(list)
        for r in self.records:
            r_batch_size = int(16.0 / r.get('batch_factor', 1.0))
            r_k_folds = int(r.get('fold_factor', 1.0) * 5.0)
            key = (r_batch_size, r_k_folds)
            by_config[key].append(r)

        exact_match = by_config.get((batch_size, k_folds), [])

        if len(exact_match) >= 3:
            times = [r['time'] for r in exact_match]
            base_estimate = np.percentile(times, 75)

            if real_complexity:
                complexities = [r['complexity'] for r in exact_match]
                avg_complexity = np.mean(complexities)

                if abs(real_complexity - avg_complexity) / avg_complexity > 0.5:
                    complexity_ratio = real_complexity / avg_complexity
                    base_estimate *= complexity_ratio
                    strategy = f"75th percentile (BS={batch_size}, k={k_folds}, n={len(exact_match)}) + complexity scaling"
                else:
                    strategy = f"75th percentile (BS={batch_size}, k={k_folds}, n={len(exact_match)})"
            else:
                strategy = f"75th percentile (BS={batch_size}, k={k_folds}, n={len(exact_match)})"

            self._last_strategy = strategy
            return max(base_estimate, 1.0)

        batch_match = []
        for (r_bs, r_kf), records in by_config.items():
            if r_bs == batch_size:
                batch_match.extend(records)

        if len(batch_match) >= 3:
            times = [r['time'] for r in batch_match]
            base_estimate = np.percentile(times, 75)

            avg_k_folds = np.mean([r.get('fold_factor', 1.0) * 5.0 for r in batch_match])
            fold_scale = k_folds / avg_k_folds
            base_estimate *= fold_scale

            strategy = f"75th percentile (BS={batch_size} only, n={len(batch_match)}) + k_folds scaling"
            self._last_strategy = strategy
            return max(base_estimate, 1.0)

        all_times = [r['time'] for r in self.records]
        base_estimate = np.percentile(all_times, 90)

        avg_k_folds = np.mean([r.get('fold_factor', 1.0) * 5.0 for r in self.records])
        fold_scale = k_folds / avg_k_folds

        avg_batch_size = np.mean([16.0 / r.get('batch_factor', 1.0) for r in self.records])
        batch_scale = avg_batch_size / batch_size

        base_estimate *= fold_scale * batch_scale

        strategy = f"90th percentile fallback (all data, n={len(self.records)}) + scaling"
        self._last_strategy = strategy
        return max(base_estimate, 1.0)

    def get_last_strategy(self):
        """Return the strategy used in the last estimate_time() call."""
        return self._last_strategy or "No estimation performed yet"

    def record_actual_time(self, config, actual_time_minutes, real_complexity):
        """Record actual training time for learning."""
        if not real_complexity or actual_time_minutes <= 0:
            return

        record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'complexity': int(real_complexity),
            'time': float(actual_time_minutes),
            'batch_factor': 16.0 / config['batch_size'],
            'fold_factor': config['k_folds'] / 5.0,
            'batch_size': config['batch_size'],
            'k_folds': config['k_folds'],
            'classifier': self.classifier_type
        }

        self.records.append(record)
        self.records = self.records[-500:]

        self.timing_db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.timing_db_path, 'w') as f:
            json.dump({'records': self.records}, f, indent=2)

    def get_stats(self):
        """Get simple stats for debugging."""
        if len(self.records) < 2:
            return "No sufficient data"

        by_batch = defaultdict(list)
        for r in self.records:
            batch_size = int(16.0 / r.get('batch_factor', 1.0))
            by_batch[batch_size].append(r['time'])

        stats_lines = [f"Total records: {len(self.records)}"]

        for bs in sorted(by_batch.keys()):
            times = by_batch[bs]
            stats_lines.append(
                f"BS={bs}: n={len(times)}, "
                f"median={np.median(times):.1f}min, "
                f"75th={np.percentile(times, 75):.1f}min, "
                f"90th={np.percentile(times, 90):.1f}min"
            )

        return "\n  ".join(stats_lines)


def create_simple_estimator(classifier_type):
    """Create timing estimator for given classifier type."""
    if classifier_type == 'cwt_image':
        from config import get_cwt_timing_database_path
        db_path = get_cwt_timing_database_path()
    else:
        from config import get_pd_timing_database_path
        db_path = get_pd_timing_database_path()

    return SimpleTimingEstimator(db_path, classifier_type)


if __name__ == "__main__":
    estimator = SimpleTimingEstimator(
        "D:/ME1573_data_processing/ml/logs/cwt/hyperopt_results/timing_database.json",
        "cwt_image"
    )

    print("="*60)
    print("TIMING ESTIMATOR TEST")
    print("="*60)

    print(f"\nDatabase stats:\n  {estimator.get_stats()}")

    for bs in [16, 32, 64, 128]:
        config = {'batch_size': bs, 'k_folds': 5}
        estimate = estimator.estimate_time(config, real_complexity=9738609)
        print(f"\nBS={bs}: {estimate:.1f} min")
        print(f"  Strategy: {estimator.get_last_strategy()}")
