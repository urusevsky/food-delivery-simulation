# delivery_sim/metrics/entity/aggregate/entity_aggregates.py
"""
Entity aggregation functions for replication-level summaries.

This module provides functions to aggregate individual entity metrics
into replication-level summary statistics for analysis.
"""

import numpy as np


def calculate_summary_statistics(metric_values):
    """
    Calculate summary statistics for a list of metric values.
    
    This function computes descriptive statistics including central tendency,
    dispersion, and key percentiles for a collection of metric values.
    
    Args:
        metric_values: List of numeric values (None values will be filtered out)
        
    Returns:
        dict: Summary statistics containing:
            - count: Number of valid observations
            - mean: Arithmetic mean
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value  
            - p25: 25th percentile
            - p50: 50th percentile (median)
            - p75: 75th percentile
            - p95: 95th percentile
    """
    # Filter out None values
    valid_values = [v for v in metric_values if v is not None]
    
    if not valid_values:
        return {
            'count': 0,
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'p25': None,
            'p50': None,
            'p75': None,
            'p95': None
        }
    
    values_array = np.array(valid_values)
    
    return {
        'count': len(valid_values),
        'mean': np.mean(values_array),
        'std': np.std(values_array),
        'min': np.min(values_array),
        'max': np.max(values_array),
        'p25': np.percentile(values_array, 25),
        'p50': np.percentile(values_array, 50),  # median
        'p75': np.percentile(values_array, 75),
        'p95': np.percentile(values_array, 95)
    }


