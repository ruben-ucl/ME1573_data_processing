"""
Simple Timing Estimator - Replaces the overcomplicated timing system

This module provides dead-simple timing estimation with just power law learning.
No more hundreds of lines of complexity - just the bare essentials.
"""

import json
import datetime
import numpy as np
from pathlib import Path
from scipy.stats import linregress


class SimpleTimingEstimator:
    """Ultra-simple timing estimator using power law on real Keras complexity."""
    
    def __init__(self, timing_db_path, classifier_type):
        self.timing_db_path = Path(timing_db_path)
        self.classifier_type = classifier_type
        self.records = self._load_records()
    
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
        Estimate training time. 
        
        Args:
            config: Configuration dict (for batch_size, k_folds)
            real_complexity: Real Keras parameter count (if available)
            
        Returns:
            float: Estimated time in minutes
        """
        # Simple scaling factors
        batch_factor = 16.0 / config['batch_size']  # Slower for smaller batches
        fold_factor = config['k_folds'] / 5.0       # Scales with k-folds
        
        if len(self.records) >= 3:
            # Use power law from historical data
            complexities = [r['complexity'] for r in self.records]
            times = [r['time'] / (r.get('batch_factor', 1.0) * r.get('fold_factor', 1.0)) for r in self.records]
            
            try:
                # Fit: time = a * complexity^b
                slope, intercept, _, _, _ = linregress(np.log(complexities), np.log(times))
                
                if real_complexity:
                    # Use real complexity
                    base_time = np.exp(intercept) * (real_complexity ** slope)
                else:
                    # Use heuristic complexity
                    heuristic_complexity = 8e6 if self.classifier_type == 'cwt_image' else 1e6
                    base_time = np.exp(intercept) * (heuristic_complexity ** slope)
                
                return max(base_time * batch_factor * fold_factor, 1.0)
                
            except:
                pass  # Fallback to heuristic
        
        # Simple heuristic when no data or power law fails
        if real_complexity:
            # Based on observed CWT data: ~1 second per million parameters per epoch, assume 25 epochs average
            base_time = (real_complexity / 1e6) * 1.0 / 60.0 * 25  # Convert to minutes
        else:
            # Realistic defaults based on typical performance (5 folds, 50 epochs max)
            base_time = 12.0 if self.classifier_type == 'cwt_image' else 8.0
        
        return max(base_time * batch_factor * fold_factor, 1.0)
    
    def record_actual_time(self, config, actual_time_minutes, real_complexity):
        """Record actual training time for learning."""
        if not real_complexity or actual_time_minutes <= 0:
            return  # Skip invalid data
        
        record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'complexity': int(real_complexity),  # Convert numpy int64 to Python int
            'time': float(actual_time_minutes),  # Ensure it's a Python float
            'batch_factor': 16.0 / config['batch_size'],
            'fold_factor': config['k_folds'] / 5.0,
            'classifier': self.classifier_type
        }
        
        self.records.append(record)
        
        # Keep only last 20 records to prevent bloat
        self.records = self.records[-50:]
        
        # Save to disk
        self.timing_db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.timing_db_path, 'w') as f:
            json.dump({'records': self.records}, f, indent=2)
    
    def get_stats(self):
        """Get simple stats for debugging."""
        if len(self.records) < 2:
            return "No sufficient data"
        
        complexities = [r['complexity'] for r in self.records]
        times = [r['time'] for r in self.records]
        
        try:
            slope, intercept, r_value, _, _ = linregress(np.log(complexities), np.log(times))
            return f"Power law: time ∝ complexity^{slope:.2f}, R²={r_value**2:.2f}, n={len(self.records)}"
        except:
            return f"Linear fallback, n={len(self.records)}"


# Simple integration functions for the hyperparameter tuner
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
    # Simple test
    estimator = SimpleTimingEstimator("test_timing.json", "cwt_image")
    
    # Test config
    config = {'batch_size': 16, 'k_folds': 5}
    
    print(f"Initial estimate: {estimator.estimate_time(config):.1f} minutes")
    
    # Simulate some training records
    estimator.record_actual_time(config, 45.0, 9000000)  # 45 min for 9M params
    estimator.record_actual_time(config, 30.0, 6000000)  # 30 min for 6M params
    estimator.record_actual_time(config, 60.0, 12000000) # 60 min for 12M params
    
    print(f"After learning: {estimator.estimate_time(config, 9000000):.1f} minutes")
    print(f"Stats: {estimator.get_stats()}")