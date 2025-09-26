# analysis_pipeline_redesigned/confidence_intervals.py
"""
Optional confidence interval construction for experiment-level statistics.

This module provides CI construction as a separate, optional step after
descriptive statistics have been computed. Maintains separation between
exploratory analysis (descriptive stats) and inferential analysis (CIs).

Design philosophy:
- CI construction is optional and separate from statistics calculation
- Uses underlying replication values stored during aggregation
- Simple t-distribution approach for independent replication observations
"""

import numpy as np
from scipy import stats
from delivery_sim.utils.logging_system import get_logger
from delivery_sim.analysis_pipeline_redesigned.metric_configurations import (
    get_aggregation_pattern, 
    get_ci_configuration, 
    get_ci_method
)
from delivery_sim.analysis_pipeline_redesigned.statistics_engine import StatisticsEngine

logger = get_logger("analysis_pipeline_redesigned.confidence_intervals")


def construct_confidence_intervals_for_experiment(experiment_statistics, replication_level_metrics, metric_types, confidence_level=0.95):
    """
    Construct confidence intervals for experiment-level statistics.
    
    This is the main CI construction function that adds CIs to existing
    experiment statistics as an optional separate step.
    
    Args:
        experiment_statistics: Results from aggregation processor (descriptive stats)
        replication_level_metrics: Processed replication-level metrics (needed for CI construction)  # ← Updated
        metric_types: List of metric types to process CIs for
        confidence_level: Confidence level for CI construction
        
    Returns:
        dict: Experiment statistics with confidence intervals added
    """
    logger.info(f"Constructing {confidence_level*100}% confidence intervals")
    
    results_with_cis = {}
    
    for metric_type in metric_types:
        if metric_type not in experiment_statistics:
            logger.warning(f"No experiment statistics found for {metric_type}")
            continue
            
        logger.debug(f"Processing CIs for {metric_type}")
        results_with_cis[metric_type] = _construct_cis_for_metric_type(
            experiment_statistics[metric_type],
            replication_level_metrics[metric_type],
            metric_type,
            confidence_level
        )
    
    logger.info("Confidence interval construction completed")
    return results_with_cis


def _construct_cis_for_metric_type(metric_statistics, replication_level_metrics, metric_type, confidence_level):
    """
    Construct CIs for a specific metric type.
    
    Handles both two-level and one-level patterns by re-extracting the
    underlying values that were used to compute the statistics.
    """
    
    pattern = get_aggregation_pattern(metric_type)
    
    if pattern == 'two_level':
        return _construct_cis_for_two_level(metric_statistics, replication_level_metrics, metric_type, confidence_level)
    elif pattern == 'one_level':
        return _construct_cis_for_one_level(metric_statistics, replication_level_metrics, confidence_level)
    else:
        raise ValueError(f"Unknown aggregation pattern: {pattern}")


def _construct_cis_for_two_level(metric_statistics, replication_level_metrics, metric_type, confidence_level):
    """
    Construct CIs for two-level pattern (statistics-of-statistics).
    
    ✅ SIMPLIFIED: Updated for flat structure without entity_type nesting.
    """
    
    statistics_engine = StatisticsEngine()
    ci_configurations = get_ci_configuration(metric_type)
    
    results_with_cis = {}
    
    # ✅ SIMPLIFIED: Direct iteration over metrics, no entity_type loop
    for metric_name, metric_stats in metric_statistics.items():
        results_with_cis[metric_name] = {}
        
        # Copy all descriptive statistics first
        for stat_name, stat_value in metric_stats.items():
            results_with_cis[metric_name][stat_name] = {
                'point_estimate': stat_value,
                'confidence_interval': [None, None]  # Default: no CI
            }
        
        # Add CIs only for configured statistics
        for ci_config in ci_configurations:
            stat_name = ci_config['name']
            
            if stat_name in metric_stats:
                # ✅ SIMPLIFIED: Re-extract values using updated method signature
                extracted_values = statistics_engine.extract_statistic_for_experiment_aggregation(
                    replication_level_metrics, metric_name, ci_config['extract']  # ← Removed entity_type
                )
                
                # Automatically determine CI method from compute field
                ci_method = get_ci_method(metric_type, stat_name)
                
                # Construct CI using automatically determined method
                ci_result = _construct_ci_with_method(
                    extracted_values, confidence_level, ci_config['compute'], ci_method
                )
                
                # Update with CI information
                results_with_cis[metric_name][stat_name] = ci_result
                
                logger.debug(f"Constructed {ci_method} CI for {metric_name}.{stat_name}")
    
    return results_with_cis


def _construct_cis_for_one_level(metric_statistics, replication_level_metrics, metric_type, confidence_level):
    """
    Construct CIs for one-level pattern (system metrics).
    
    Only constructs CIs for metrics that have construct_ci=True in configuration.
    Automatically uses t-distribution since we're always estimating means fo system metrics.
    """

    ci_configurations = get_ci_configuration(metric_type)
    results_with_cis = {}
    
    # Extract metric names from first replication
    if not replication_level_metrics:
        return results_with_cis
    
    metric_names = list(replication_level_metrics[0].keys())
    
    # Start with all metrics as descriptive only
    for metric_name in metric_names:
        scalar_values = [rep_result[metric_name] for rep_result in replication_level_metrics 
                        if metric_name in rep_result]
        point_estimate = np.mean(scalar_values) if scalar_values else None
        
        results_with_cis[metric_name] = {
            'point_estimate': point_estimate,
            'confidence_interval': [None, None]  # Default: no CI
        }
    
    # Add CIs only for configured metrics
    for ci_config in ci_configurations:
        metric_name = ci_config['metric_name']
        
        if metric_name in metric_names:
            # Re-extract scalar values across replications
            scalar_values = [rep_result[metric_name] for rep_result in replication_level_metrics 
                            if metric_name in rep_result]
            
            # System metrics always use t-distribution (estimating mean)
            ci_method = 't_distribution'
            ci_result = _construct_ci_with_method(scalar_values, confidence_level, 'mean', ci_method)
            
            results_with_cis[metric_name] = ci_result
            logger.debug(f"Constructed {ci_method} CI for {metric_name} (system metric - always mean estimation)")
    
    return results_with_cis


def _construct_ci_with_method(values, confidence_level, target_statistic, ci_method):
    """
    Construct CI using specified statistical method.
    
    Args:
        values: List of values from replications
        confidence_level: Confidence level
        target_statistic: Which statistic to compute CI for ('mean', 'std', 'variance')
        ci_method: Statistical method ('t_distribution', 'chi_square')
        
    Returns:
        dict: Contains point_estimate, confidence_interval, and metadata
    """
    valid_values = [v for v in values if v is not None]
    
    if len(valid_values) < 2:
        return {
            'point_estimate': valid_values[0] if valid_values else None,
            'confidence_interval': [None, None],
            'standard_error': None,
            'sample_size': len(valid_values),
            'ci_method': ci_method
        }
    
    data = np.array(valid_values)
    n = len(data)
    
    # Calculate target statistic as point estimate
    if target_statistic == 'mean':
        point_estimate = np.mean(data)
    elif target_statistic == 'std':
        point_estimate = np.std(data, ddof=1)
    elif target_statistic == 'variance':
        point_estimate = np.var(data, ddof=1)
    else:
        logger.warning(f"Unknown target statistic: {target_statistic}, defaulting to mean")
        point_estimate = np.mean(data)
    
    # Construct CI based on method
    if ci_method == 't_distribution':
        return _construct_t_distribution_ci(data, point_estimate, confidence_level)
    elif ci_method == 'chi_square':
        return _construct_chi_square_ci(data, point_estimate, confidence_level)
    else:
        logger.error(f"Unknown CI method: {ci_method}")
        return {
            'point_estimate': float(point_estimate),
            'confidence_interval': [None, None],
            'standard_error': None,
            'sample_size': n,
            'ci_method': ci_method
        }


def _construct_t_distribution_ci(data, point_estimate, confidence_level):
    """Construct CI using t-distribution (for mean estimation)."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    standard_error = std / np.sqrt(n)
    
    # t-distribution CI
    alpha = 1 - confidence_level
    degrees_freedom = n - 1
    t_critical = stats.t.ppf(1 - alpha/2, degrees_freedom)
    
    margin_error = t_critical * standard_error
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return {
        'point_estimate': float(point_estimate),
        'confidence_interval': [float(ci_lower), float(ci_upper)],
        'standard_error': float(standard_error),
        'sample_size': n,
        'ci_method': 't_distribution'
    }


def _construct_chi_square_ci(data, point_estimate, confidence_level):
    """Construct CI using chi-square distribution (for variance/std estimation)."""
    n = len(data)
    sample_variance = np.var(data, ddof=1)
    
    # Chi-square CI for variance
    alpha = 1 - confidence_level
    degrees_freedom = n - 1
    
    chi2_lower = stats.chi2.ppf(alpha/2, degrees_freedom)
    chi2_upper = stats.chi2.ppf(1 - alpha/2, degrees_freedom)
    
    # CI for variance
    var_ci_lower = (degrees_freedom * sample_variance) / chi2_upper
    var_ci_upper = (degrees_freedom * sample_variance) / chi2_lower
    
    # Convert to std CI if needed
    if 'std' in str(point_estimate.__class__).lower():  # Rough check for std
        ci_lower = np.sqrt(var_ci_lower)
        ci_upper = np.sqrt(var_ci_upper)
    else:
        ci_lower = var_ci_lower
        ci_upper = var_ci_upper
    
    return {
        'point_estimate': float(point_estimate),
        'confidence_interval': [float(ci_lower), float(ci_upper)],
        'standard_error': None,  # Not applicable for chi-square
        'sample_size': n,
        'ci_method': 'chi_square'
    }


def format_confidence_interval_result(ci_result, decimal_places=3):
    """
    Format confidence interval result for readable output.
    
    Args:
        ci_result: Result from CI construction
        decimal_places: Number of decimal places
        
    Returns:
        str: Formatted string
    """
    if ci_result['point_estimate'] is None:
        return "No data available"
    
    point_est = ci_result['point_estimate']
    ci_lower, ci_upper = ci_result['confidence_interval']
    
    if ci_lower is None or ci_upper is None:
        return f"{point_est:.{decimal_places}f} (insufficient data for CI)"
    
    margin = (ci_upper - ci_lower) / 2
    return f"{point_est:.{decimal_places}f} ± {margin:.{decimal_places}f} [{ci_lower:.{decimal_places}f}, {ci_upper:.{decimal_places}f}]"