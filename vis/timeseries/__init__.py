"""
Time Series Analysis Package

A comprehensive toolkit for time series comparison, analysis, and visualization.
"""

from .config import DatasetConfig, ProcessingConfig
from .processor import TimeSeriesProcessor
from .comparator import TimeSeriesComparator

__all__ = [
    'DatasetConfig',
    'ProcessingConfig',
    'TimeSeriesProcessor',
    'TimeSeriesComparator',
]

__version__ = '2.1.0'
