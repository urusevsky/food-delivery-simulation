# delivery_sim/analysis_pipeline/experiment_level_aggregation/confidence_intervals.py
"""
Statistical utilities for calculating confidence intervals.

This module provides functions to calculate confidence intervals for different types
of metrics aggregated across replications in simulation experiments.
"""

import numpy as np
from scipy import stats


def calculate_confidence_interval(values, confidence_level=0.95):
    """
    Calculate confidence interval for a list of values using t-distribution.
    
    Uses t-distribution which is appropriate for small sample sizes (n < 30)
    typical in simulation studies with 5-10 replications.
    
    Args:
        values: List of numeric values from replications
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        
    Returns:
        dict: Contains point_estimate, confidence_interval, standard_error, sample_size
              Returns None values if insufficient data
    """
    # Filter out None values
    valid_values = [v for v in values if v is not None]
    
    if len(valid_values) < 2:
        return {
            'point_estimate': None,
            'confidence_interval': [None, None],
            'standard_error': None,
            'sample_size': len(valid_values)
        }
    
    # Convert to numpy array for calculations
    data = np.array(valid_values)
    n = len(data)
    
    # Calculate basic statistics
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation (n-1 denominator)
    standard_error = std / np.sqrt(n)
    
    # Calculate confidence interval using t-distribution
    alpha = 1 - confidence_level
    degrees_freedom = n - 1
    t_critical = stats.t.ppf(1 - alpha/2, degrees_freedom)
    
    margin_error = t_critical * standard_error
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return {
        'point_estimate': mean,
        'confidence_interval': [ci_lower, ci_upper],
        'standard_error': standard_error,
        'sample_size': n
    }


def calculate_proportion_confidence_interval(successes, totals, confidence_level=0.95):
    """
    Calculate confidence interval for proportion metrics (like completion rate).
    
    Uses Wilson score interval which is more robust than normal approximation
    for small sample sizes and proportions near 0 or 1.
    
    Args:
        successes: List of success counts from each replication
        totals: List of total counts from each replication  
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        
    Returns:
        dict: Contains point_estimate, confidence_interval, standard_error, sample_size
    """
    # Filter out None values and ensure paired data
    valid_pairs = [(s, t) for s, t in zip(successes, totals) 
                   if s is not None and t is not None and t > 0]
    
    if len(valid_pairs) < 2:
        return {
            'point_estimate': None,
            'confidence_interval': [None, None],
            'standard_error': None,
            'sample_size': len(valid_pairs)
        }
    
    # Calculate individual proportions for each replication
    proportions = [s / t for s, t in valid_pairs]
    
    # Use standard CI calculation on the proportions
    # This treats each replication's proportion as an independent observation
    return calculate_confidence_interval(proportions, confidence_level)


def calculate_summary_with_ci(values, confidence_level=0.95):
    """
    Calculate complete summary statistics including confidence interval.
    
    Convenience function that combines basic descriptive statistics
    with confidence interval calculation.
    
    Args:
        values: List of numeric values from replications
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        
    Returns:
        dict: Complete summary with descriptives and confidence interval
    """
    # Filter valid values
    valid_values = [v for v in values if v is not None]
    
    if not valid_values:
        return {
            'count': 0,
            'point_estimate': None,
            'confidence_interval': [None, None],
            'standard_error': None,
            'min': None,
            'max': None,
            'std': None
        }
    
    # Get confidence interval
    ci_result = calculate_confidence_interval(valid_values, confidence_level)
    
    # Add descriptive statistics
    data = np.array(valid_values)
    
    return {
        'count': len(valid_values),
        'point_estimate': ci_result['point_estimate'],
        'confidence_interval': ci_result['confidence_interval'],
        'standard_error': ci_result['standard_error'],
        'min': np.min(data),
        'max': np.max(data),
        'std': np.std(data, ddof=1)
    }


def format_confidence_interval(ci_result, decimal_places=3):
    """
    Format confidence interval results for readable output.
    
    Args:
        ci_result: Dictionary from confidence interval calculation
        decimal_places: Number of decimal places for formatting
        
    Returns:
        str: Formatted string like "14.24 ± 0.65 [13.59, 14.89]"
    """
    if ci_result['point_estimate'] is None:
        return "No data available"
    
    point_est = ci_result['point_estimate']
    ci_lower, ci_upper = ci_result['confidence_interval']
    margin = (ci_upper - ci_lower) / 2
    
    return f"{point_est:.{decimal_places}f} ± {margin:.{decimal_places}f} [{ci_lower:.{decimal_places}f}, {ci_upper:.{decimal_places}f}]"