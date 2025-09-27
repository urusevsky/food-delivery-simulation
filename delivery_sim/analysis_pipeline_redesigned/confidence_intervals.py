# analysis_pipeline_redesigned/confidence_intervals.py
"""
Optional confidence interval construction for experiment-level statistics.

Simplified design for research context:
- Direct CI calculation without excessive abstraction
- Unified patterns for two-level and one-level processing
- Inline simple helpers to reduce function count
"""

import numpy as np
from scipy import stats
from delivery_sim.utils.logging_system import get_logger
from delivery_sim.analysis_pipeline_redesigned.statistics_engine import StatisticsEngine

logger = get_logger("analysis_pipeline_redesigned.confidence_intervals")


def construct_confidence_intervals_for_experiment(experiment_statistics, replication_level_metrics, metric_configs, confidence_level=0.95):
    """
    Construct confidence intervals for experiment-level statistics.
    
    Args:
        experiment_statistics: Results from aggregation processor (descriptive stats)
        replication_level_metrics: Processed replication-level metrics (needed for CI construction)
        metric_configs: Dict of {metric_type: config} - passed from pipeline coordinator
        confidence_level: Confidence level for CI construction
        
    Returns:
        dict: Experiment statistics with confidence intervals added
    """
    logger.info(f"Constructing {confidence_level*100}% confidence intervals")
    
    results_with_cis = {}
    
    for metric_type, config in metric_configs.items():
        if metric_type not in experiment_statistics:
            logger.warning(f"No experiment statistics found for {metric_type}")
            continue
            
        logger.debug(f"Processing CIs for {metric_type}")
        
        pattern = config['aggregation_pattern']
        
        if pattern == 'two_level':
            results_with_cis[metric_type] = _construct_cis_for_two_level(
                experiment_statistics[metric_type],
                replication_level_metrics[metric_type],
                config,
                confidence_level
            )
        elif pattern == 'one_level':
            results_with_cis[metric_type] = _construct_cis_for_one_level(
                experiment_statistics[metric_type],
                replication_level_metrics[metric_type],
                config,
                confidence_level
            )
    
    logger.info("Confidence interval construction completed")
    return results_with_cis


def _construct_cis_for_two_level(metric_statistics, replication_level_metrics, config, confidence_level):
    """
    Construct CIs for two-level pattern (statistics-of-statistics).
    
    Uses direct CI calculation to reduce function call overhead.
    """
    statistics_engine = StatisticsEngine()
    ci_configurations = [stat for stat in config.get('experiment_stats', []) 
                        if stat.get('construct_ci', False)]
    
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
            
            if stat_name in metric_stats:
                # Extract values across replications
                extracted_values = statistics_engine.extract_statistic_for_experiment_aggregation(
                    replication_level_metrics, metric_name, ci_config['extract']
                )
                
                # Construct CI directly (inline method determination)
                ci_result = _construct_confidence_interval(
                    extracted_values, 
                    confidence_level, 
                    ci_config['compute']
                )
                
                results_with_cis[metric_name][stat_name] = ci_result
                logger.debug(f"Constructed CI for {metric_name}.{stat_name}")
    
    return results_with_cis


def _construct_cis_for_one_level(metric_statistics, replication_level_metrics, config, confidence_level):
    """
    Construct CIs for one-level pattern (system metrics).
    
    System metrics always use t-distribution since we're estimating means.
    """
    ci_configurations = config.get('ci_config', [])
    results_with_cis = {}
    
    if not replication_level_metrics:
        return results_with_cis
    
    metric_names = list(replication_level_metrics[0].keys())
    
    # Initialize all metrics with descriptive-only structure
    for metric_name in metric_names:
        scalar_values = [rep_result[metric_name] for rep_result in replication_level_metrics 
                        if metric_name in rep_result]
        point_estimate = np.mean(scalar_values) if scalar_values else None
        
        results_with_cis[metric_name] = {
            'point_estimate': point_estimate,
            'confidence_interval': [None, None]
        }
    
    # Add CIs for configured metrics
    for ci_config in ci_configurations:
        metric_name = ci_config['metric_name']
        
        if metric_name in metric_names and ci_config.get('construct_ci', False):
            scalar_values = [rep_result[metric_name] for rep_result in replication_level_metrics 
                            if metric_name in rep_result]
            
            # System metrics always estimate means → use t-distribution
            ci_result = _construct_confidence_interval(scalar_values, confidence_level, 'mean')
            results_with_cis[metric_name] = ci_result
            logger.debug(f"Constructed CI for {metric_name} (system metric)")
    
    return results_with_cis


def _construct_confidence_interval(values, confidence_level, target_statistic):
    """
    Unified confidence interval construction.
    
    Handles both t-distribution (for means) and chi-square (for variance/std) methods
    with direct calculation instead of multiple helper functions.
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