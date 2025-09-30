# analysis_pipeline/confidence_interval_constructor.py
"""
Confidence interval construction (Phase 3).

Single responsibility: Add statistical inference (confidence intervals)
to experiment-level statistics using extracted replication data.
"""

import numpy as np
from scipy import stats
from delivery_sim.utils.logging_system import get_logger
from delivery_sim.analysis_pipeline.extraction_engine import ExtractionEngine

logger = get_logger("analysis_pipeline.confidence_interval_constructor")


def construct_confidence_intervals(metric_statistics, metrics_across_replications, 
                                   config, confidence_level):
    """
    Router: Construct confidence intervals according to aggregation pattern.
    
    Args:
        metric_statistics: Experiment-level descriptive statistics (for ONE metric type)
        metrics_across_replications: List of replication-level data (for ONE metric type)
        config: Metric configuration dictionary
        confidence_level: CI confidence level
        
    Returns:
        dict: Statistics with confidence intervals for this metric type
    """
    pattern = config['aggregation_pattern']
    
    if pattern == 'two_level':
        return _construct_cis_for_two_level(
            metric_statistics, metrics_across_replications, config, confidence_level
        )
    elif pattern == 'one_level':
        return _construct_cis_for_one_level(
            metric_statistics, metrics_across_replications, config, confidence_level
        )
    else:
        raise ValueError(f"Unknown aggregation pattern: {pattern}")


def _construct_cis_for_two_level(metric_statistics, metrics_across_replications, 
                                 config, confidence_level):
    """
    Construct CIs for two-level pattern (statistics-of-statistics).
    
    Args:
        metric_statistics: Experiment-level descriptive statistics (for ONE metric type)
        metrics_across_replications: List of replication-level metrics (for ONE metric type)
        config: Metric configuration
        confidence_level: CI confidence level
        
    Returns:
        dict: Statistics with confidence intervals for this metric type
    """
    extraction_engine = ExtractionEngine()
    ci_configurations = [
        stat for stat in config.get('experiment_stats', []) 
        if stat.get('construct_ci', False)
    ]
    
    results_with_cis = {}
    
    for metric_name, metric_stats in metric_statistics.items():
        results_with_cis[metric_name] = {}
        
        # Copy all descriptive statistics first (with default no-CI structure)
        for stat_name, stat_value in metric_stats.items():
            results_with_cis[metric_name][stat_name] = {
                'point_estimate': stat_value,
                'confidence_interval': [None, None]
            }
        
        # Add CIs for configured statistics
        for ci_config in ci_configurations:
            stat_name = ci_config['name']
            statistic_type = ci_config['extract']      # NEW: Explicit variable
            target_statistic = ci_config['compute']    # NEW: Explicit variable
            
            if stat_name in metric_stats:
                # Extract values across replications
                extracted_values = extraction_engine.extract_for_two_level_pattern(
                    metrics_across_replications, metric_name, statistic_type
                )
                
                # Construct CI
                ci_result = _construct_confidence_interval(
                    extracted_values, confidence_level, target_statistic
                )
                
                results_with_cis[metric_name][stat_name] = ci_result
                logger.debug(f"Constructed CI for {metric_name}.{stat_name}")
    
    return results_with_cis


def _construct_cis_for_one_level(metric_statistics, metrics_across_replications, 
                                 config, confidence_level):
    """
    Construct CIs for one-level pattern (system metrics).
    
    Args:
        metric_statistics: Experiment-level descriptive statistics (for ONE metric type)
        metrics_across_replications: List of scalar dicts (for ONE metric type)
        config: Metric configuration
        confidence_level: CI confidence level
        
    Returns:
        dict: Statistics with confidence intervals for this metric type
    """
    extraction_engine = ExtractionEngine()
    ci_configurations = config.get('ci_config', [])
    results_with_cis = {}
    
    # Initialize all metrics with descriptive statistics (no CI)
    for metric_name, metric_stats in metric_statistics.items():
        results_with_cis[metric_name] = {
            'point_estimate': metric_stats['mean'],
            'confidence_interval': [None, None]
        }
    
    # Add CIs ONLY for configured metrics
    for ci_config in ci_configurations:
        metric_name = ci_config['metric_name']
        target_statistic = ci_config.get('target_statistic', 'mean')
        
        if metric_name in metric_statistics and ci_config.get('construct_ci', False):
            # Extract values only when we need to construct CI
            scalar_values = extraction_engine.extract_for_one_level_pattern(
                metrics_across_replications, metric_name
            )
            
            # Construct CI for the specified target statistic
            ci_result = _construct_confidence_interval(
                scalar_values, confidence_level, target_statistic
            )
            results_with_cis[metric_name] = ci_result
            logger.debug(f"Constructed {target_statistic} CI for {metric_name}")
    
    return results_with_cis


def _construct_confidence_interval(values, confidence_level, target_statistic):
    """
    Unified confidence interval construction.
    
    Handles t-distribution (for means) and chi-square (for variance/std).
    
    Args:
        values: List of numeric values
        confidence_level: Confidence level (e.g., 0.95)
        target_statistic: 'mean', 'std', or 'variance'
        
    Returns:
        dict: CI result with point_estimate, confidence_interval, etc.
    """
    valid_values = [v for v in values if v is not None]
    
    # Handle insufficient data
    if len(valid_values) < 2:
        return {
            'point_estimate': valid_values[0] if valid_values else None,
            'confidence_interval': [None, None],
            'standard_error': None,
            'sample_size': len(valid_values),
            'ci_method': 'insufficient_data'
        }
    
    data = np.array(valid_values)
    n = len(data)
    alpha = 1 - confidence_level
    
    # Calculate point estimate
    if target_statistic == 'mean':
        point_estimate = np.mean(data)
    elif target_statistic == 'std':
        point_estimate = np.std(data, ddof=1)
    elif target_statistic == 'variance':
        point_estimate = np.var(data, ddof=1)
    else:
        point_estimate = np.mean(data)  # Default to mean
    
    # Direct CI calculation based on target statistic
    if target_statistic == 'mean':
        # t-distribution for mean estimation
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        standard_error = std / np.sqrt(n)
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        margin_error = t_critical * standard_error
        
        return {
            'point_estimate': float(point_estimate),
            'confidence_interval': [float(mean - margin_error), float(mean + margin_error)],
            'standard_error': float(standard_error),
            'sample_size': n,
            'ci_method': 't_distribution'
        }
    
    elif target_statistic in ['std', 'variance']:
        # Chi-square for variance/std estimation
        sample_variance = np.var(data, ddof=1)
        degrees_freedom = n - 1
        
        chi2_lower = stats.chi2.ppf(alpha/2, degrees_freedom)
        chi2_upper = stats.chi2.ppf(1 - alpha/2, degrees_freedom)
        
        var_ci_lower = (degrees_freedom * sample_variance) / chi2_upper
        var_ci_upper = (degrees_freedom * sample_variance) / chi2_lower
        
        if target_statistic == 'std':
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
    
    # Fallback for unknown target_statistic
    return {
        'point_estimate': float(point_estimate),
        'confidence_interval': [None, None],
        'standard_error': None,
        'sample_size': n,
        'ci_method': 'unknown_target'
    }