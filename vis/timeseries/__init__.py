"""
Time Series Analysis Package

A comprehensive toolkit for time series comparison, analysis, and visualization.
"""

from .config import DatasetConfig, ProcessingConfig
from .processor import TimeSeriesProcessor
from .comparator import TimeSeriesComparator
from .logging import ProcessingLog

__all__ = [
    'DatasetConfig',
    'ProcessingConfig',
    'TimeSeriesProcessor',
    'TimeSeriesComparator',
    'ProcessingLog',
]

__version__ = '2.1.0'
