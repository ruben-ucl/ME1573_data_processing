"""
Statistical analysis operations for time series.

This module contains all statistical analysis functionality including
correlation analysis, cross-correlation lags, and statistical measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Try to import statsmodels for cross-correlation function
try:
    from statsmodels.tsa.stattools import ccf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class StatisticsMixin:
    """
    Mixin class containing all statistical analysis operations.

    This class expects the following attributes to be available from the parent:
    - self.datasets: List[DatasetConfig]
    - self.processed_data: Dict[str, np.ndarray]
    - self.time_vectors: Dict[str, np.ndarray]
    - self.statistics: Dict
    - self.correlations: Dict
    """

    def calculate_statistics(self) -> None:
        """Calculate comprehensive statistics for all datasets"""
        print("\nCalculating statistics...")
        
        for label, data in self.processed_data.items():
            stats = {
                'mean': np.mean(data),
                'median': np.median(data),
                'std': np.std(data),
                'var': np.var(data),
                'min': np.min(data),
                'max': np.max(data),
                'range': np.ptp(data),
                'skewness': self._calculate_skewness(data),
                'kurtosis': self._calculate_kurtosis(data),
                'rms': np.sqrt(np.mean(data**2)),
                'energy': np.sum(data**2),
                'zero_crossings': self._count_zero_crossings(data)
            }
            self.statistics[label] = stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _count_zero_crossings(self, data: np.ndarray) -> int:
        """Count zero crossings in the signal"""
        return len(np.where(np.diff(np.signbit(data)))[0])
    
    def calculate_correlations(self) -> Dict[str, float]:
        """
        Calculate Pearson and Spearman correlations between all pairs of datasets.

        Results are stored in self.correlations and also returned.
        """
        correlations = {}
        labels = list(self.processed_data.keys())

        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels[i+1:], i+1):
                data1 = self.processed_data[label1]
                data2 = self.processed_data[label2]
                time1 = self.time_vectors[label1]
                time2 = self.time_vectors[label2]

                data1_sync, data2_sync = self._synchronize_time_series(
                    data1, time1, data2, time2
                )

                pearson_corr, pearson_p = pearsonr(data1_sync, data2_sync)
                spearman_corr, spearman_p = spearmanr(data1_sync, data2_sync)

                pair_key = f"{label1} vs {label2}"
                correlations[pair_key] = {
                    'pearson': pearson_corr,
                    'pearson_p': pearson_p,
                    'spearman': spearman_corr,
                    'spearman_p': spearman_p,
                    'n_actual': len(data1_sync),
                }

        self.correlations = correlations
        return correlations
    
    def calculate_differences(self) -> Dict[str, Dict[str, float]]:
        """Calculate various difference metrics between datasets using synchronized time series"""
        differences = {}
        labels = list(self.processed_data.keys())
        
        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels[i+1:], i+1):
                data1 = self.processed_data[label1]
                data2 = self.processed_data[label2]
                time1 = self.time_vectors[label1]
                time2 = self.time_vectors[label2]
                
                # Synchronize time series for difference analysis
                data1_sync, data2_sync = self._synchronize_time_series(
                    data1, time1, data2, time2
                )
                
                # Calculate various metrics
                mse = mean_squared_error(data1_sync, data2_sync)
                mae = mean_absolute_error(data1_sync, data2_sync)
                rmse = np.sqrt(mse)
                
                # Normalized metrics
                range1 = np.ptp(data1_sync)
                range2 = np.ptp(data2_sync)
                avg_range = (range1 + range2) / 2
                normalized_rmse = rmse / avg_range if avg_range > 0 else 0
                
                pair_key = f"{label1} vs {label2}"
                differences[pair_key] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'normalized_rmse': normalized_rmse
                }
        
        return differences

    def calculate_silhouette_scores(self, silhouette_threshold: float = 0.6) -> Dict[str, Dict[str, float]]:
        """
        Calculate silhouette scores for signal pairs using KMeans clustering.

        Tests clustering quality with k=2 and k=3 clusters, then determines
        optimal cluster count based on silhouette scores.

        Parameters:
        -----------
        silhouette_threshold : float, default=0.25
            Threshold below which clustering is considered weak (predicts k=1)

        Returns:
        --------
        silhouette_scores : dict
            Dictionary with silhouette analysis for each pair:
            - 'silhouette_k2': Silhouette score for 2 clusters
            - 'silhouette_k3': Silhouette score for 3 clusters
            - 'optimal_k': Predicted optimal cluster count (1, 2, or 3)
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        if not self.processed_data:
            raise ValueError("No processed data available. Call process_data() first.")

        labels = list(self.processed_data.keys())
        if len(labels) < 2:
            return {}

        silhouette_scores = {}

        for i, label1 in enumerate(labels):
            for label2 in labels[i+1:]:
                # Get synchronized data
                data1 = self.processed_data[label1]
                data2 = self.processed_data[label2]

                # Create feature matrix (N samples × 2 features)
                # Stack signals as columns
                min_len = min(len(data1), len(data2))
                X = np.column_stack([
                    data1[:min_len],
                    data2[:min_len]
                ])

                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Need at least 3 samples for k=3 clustering
                if len(X_scaled) < 3:
                    pair_key = f"{label1} vs {label2}"
                    silhouette_scores[pair_key] = {
                        'silhouette_k2': np.nan,
                        'silhouette_k3': np.nan,
                        'optimal_k': 1
                    }
                    continue

                # Calculate silhouette score for k=2
                try:
                    kmeans_k2 = KMeans(n_clusters=2, random_state=42, n_init=10)
                    labels_k2 = kmeans_k2.fit_predict(X_scaled)
                    score_k2 = silhouette_score(X_scaled, labels_k2)
                except:
                    score_k2 = np.nan

                # Calculate silhouette score for k=3
                try:
                    kmeans_k3 = KMeans(n_clusters=3, random_state=42, n_init=10)
                    labels_k3 = kmeans_k3.fit_predict(X_scaled)
                    score_k3 = silhouette_score(X_scaled, labels_k3)
                except:
                    score_k3 = np.nan

                # Determine optimal k
                if np.isnan(score_k2) and np.isnan(score_k3):
                    optimal_k = 1
                elif np.isnan(score_k3):
                    optimal_k = 2 if score_k2 > silhouette_threshold else 1
                elif np.isnan(score_k2):
                    optimal_k = 3 if score_k3 > silhouette_threshold else 1
                else:
                    # Both scores valid
                    if score_k2 < silhouette_threshold and score_k3 < silhouette_threshold:
                        optimal_k = 1  # No meaningful clustering
                    elif score_k2 > score_k3:
                        optimal_k = 2
                    else:
                        optimal_k = 3

                pair_key = f"{label1} vs {label2}"
                silhouette_scores[pair_key] = {
                    'silhouette_k2': score_k2,
                    'silhouette_k3': score_k3,
                    'optimal_k': optimal_k
                }

        return silhouette_scores

    def calculate_cross_correlation_lags(self, max_lag: Optional[int] = None,
                                        use_statsmodels: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Calculate cross-correlation between all pairs of time series to detect lags

        Parameters:
        -----------
        max_lag : int, optional
            Maximum lag to consider (default: min(len(data)//4, 500))
        use_statsmodels : bool, default=True
            If True and available, use statsmodels CCF; otherwise use numpy

        Returns:
        --------
        lag_info : dict
            Dictionary with lag information for each pair:
            - 'optimal_lag_samples': Lag at maximum correlation (in samples)
            - 'optimal_lag_time': Lag in time units (seconds)
            - 'max_correlation': Maximum correlation value
            - 'correlation_at_zero': Correlation at zero lag
        """
        if not self.processed_data:
            print("No processed data available. Load data first.")
            return {}

        lag_info = {}
        labels = list(self.processed_data.keys())

        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels[i+1:], i+1):
                data1 = self.processed_data[label1]
                data2 = self.processed_data[label2]
                time1 = self.time_vectors[label1]
                time2 = self.time_vectors[label2]

                # Synchronize time series
                data1_sync, data2_sync = self._synchronize_time_series(
                    data1, time1, data2, time2
                )

                # Determine max lag
                if max_lag is None:
                    lag_limit = min(len(data1_sync) // 4, 500)
                else:
                    lag_limit = min(max_lag, len(data1_sync) - 1)

                # Use unified cross-correlation helper (checks both positive and negative lags)
                method_to_use = 'statsmodels' if (use_statsmodels and STATSMODELS_AVAILABLE) else 'numpy'
                lags, cross_corr, optimal_lag = self._compute_cross_correlation(
                    data1_sync, data2_sync,
                    max_lag=lag_limit,
                    method=method_to_use
                )

                # Find correlation values
                optimal_idx = np.where(lags == optimal_lag)[0][0]
                max_corr = cross_corr[optimal_idx]
                zero_idx = np.where(lags == 0)[0]
                corr_at_zero = cross_corr[zero_idx[0]] if len(zero_idx) > 0 else max_corr

                # Convert lag to time units
                if label1 in self.time_vectors and len(self.time_vectors[label1]) > 1:
                    dt = np.mean(np.diff(self.time_vectors[label1]))
                    optimal_lag_time = optimal_lag * dt
                else:
                    optimal_lag_time = float('nan')

                pair_key = f"{label1} vs {label2}"
                lag_info[pair_key] = {
                    'optimal_lag_samples': int(optimal_lag),
                    'optimal_lag_time': optimal_lag_time,
                    'max_correlation': float(max_corr),
                    'correlation_at_zero': float(corr_at_zero),
                    'lag_interpretation': self._interpret_lag(optimal_lag, label1, label2)
                }

        return lag_info

    def _interpret_lag(self, lag: int, label1: str, label2: str) -> str:
        """
        Interpret the meaning of a lag value

        Parameters:
        -----------
        lag : int
            Lag in samples (positive means label2 lags behind label1)
        label1, label2 : str
            Labels of the two series

        Returns:
        --------
        interpretation : str
            Human-readable interpretation of the lag
        """
        if lag == 0:
            return "No lag detected (synchronized)"
        elif lag > 0:
            return f"{label2} lags behind {label1} by {abs(lag)} samples"
        else:
            return f"{label1} lags behind {label2} by {abs(lag)} samples"

