# delivery_sim/warmup_analysis/simple_time_series_preprocessing.py
"""
Simplified Time Series Processing for Warmup Analysis

Clean, transparent processing pipeline that eliminates unnecessary complexity.
Direct path from raw simulation snapshots to Welch's method visualization.
"""

import numpy as np
import pandas as pd
from delivery_sim.utils.logging_system import get_logger


def extract_warmup_time_series(study_results, design_points, 
                              metrics=['active_drivers', 'unassigned_delivery_entities'],
                              moving_average_window=100):
    """
    Extract time series data for warmup analysis from study results.
    
    REFACTORED: study_results values are now direct replication_results lists.
    
    Simple, direct extraction that:
    1. Gets system snapshots from all replications
    2. Calculates cross-replication averages
    3. Applies Welch's method (moving averages)
    4. Adds Little's Law theoretical values
    
    Args:
        study_results: Raw results from experimental execution
        design_points: Dict of DesignPoint instances for Little's Law
        metrics: List of metrics to extract from snapshots
        moving_average_window: Window size for Welch's method smoothing
        
    Returns:
        dict: Time series data by design point name
              Format: {design_name: {metric_name: {time_points, cross_rep_avg, moving_avg, ...}}}
    """
    logger = get_logger("warmup_analysis.time_series_processing")
    logger.info(f"Extracting time series for {len(study_results)} design points")
    
    all_time_series_data = {}  # ✅ Consistent with calling code
    
    for design_name, replication_results in study_results.items():
        logger.debug(f"Processing {design_name}...")
        
        # ✅ REFACTORED: Consistent naming - replication_results throughout codebase
        # Step 1: Extract system snapshots from all replications
        replication_snapshots = _extract_replication_snapshots(replication_results)
        
        if len(replication_snapshots) < 2:
            logger.warning(f"Skipping {design_name}: only {len(replication_snapshots)} replications")
            continue
        
        # Step 2: Process metrics for this design point
        design_point = design_points[design_name]
        time_series_data = _process_design_point_metrics(
            replication_snapshots, 
            design_point,
            metrics, 
            moving_average_window
        )
        
        all_time_series_data[design_name] = time_series_data
        logger.debug(f"Processed {design_name}: {len(time_series_data)} metrics")
    
    logger.info(f"Time series extraction complete for {len(all_time_series_data)} design points")
    return all_time_series_data


def _extract_replication_snapshots(replication_results):
    """
    Extract system snapshots from replication results.
    
    REFACTORED: Consistent parameter naming throughout codebase.
    
    Args:
        replication_results: Direct replication_results list (no wrapper dictionary)
        
    Returns:
        list: List of snapshot lists, one per replication
    """
    replication_snapshots = []
    
    # ✅ REFACTORED: replication_results IS the replication_results list
    for replication_result in replication_results:
        if 'system_snapshots' in replication_result:
            replication_snapshots.append(replication_result['system_snapshots'])
    
    return replication_snapshots


def _process_design_point_metrics(replication_snapshots, design_point, metrics, window):
    """
    Process metrics for one design point through Welch's method pipeline.
    
    Args:
        replication_snapshots: List of snapshot lists from replications
        design_point: DesignPoint instance for Little's Law calculation
        metrics: List of metric names to process
        window: Moving average window size
        
    Returns:
        dict: Processed metrics data
    """
    results = {}
    
    for metric_name in metrics:
        # Step 1: Extract metric values from all replications
        metric_arrays = _extract_metric_from_replications(replication_snapshots, metric_name)
        
        # Step 2: Calculate cross-replication averages
        cross_rep_averages = np.mean(metric_arrays, axis=0)
        
        # Step 3: Apply Welch's method (moving averages)
        moving_averages = _apply_moving_average(cross_rep_averages, window)
        
        # Step 4: Create time points (assuming 1-minute intervals)
        time_points = _create_time_points(len(cross_rep_averages))
        
        # Step 5: Package results
        results[metric_name] = {
            'time_points': time_points,
            'cross_rep_averages': cross_rep_averages.tolist(),
            'moving_averages': moving_averages,
            'replication_count': len(metric_arrays),
            'moving_average_window': window,
            'metric_name': metric_name
        }
        
        # Step 6: Add Little's Law theoretical value for active_drivers
        if metric_name == 'active_drivers':
            theoretical_value = _calculate_littles_law_active_drivers(design_point)
            results[metric_name]['theoretical_value'] = theoretical_value
    
    return results


def _extract_metric_from_replications(replication_snapshots, metric_name):
    """
    Extract one metric from all replications.
    
    Args:
        replication_snapshots: List of snapshot lists
        metric_name: Name of metric to extract
        
    Returns:
        numpy.ndarray: 2D array where rows=replications, cols=time_points
    """
    metric_arrays = []
    
    for rep_snapshots in replication_snapshots:
        # Extract metric values from this replication's snapshots
        values = [snapshot[metric_name] for snapshot in rep_snapshots]
        metric_arrays.append(values)
    
    # Convert to numpy array for easy averaging
    return np.array(metric_arrays)


def _apply_moving_average(data_series, window):
    """
    Apply moving average smoothing (Welch's method) using trailing window.
    
    This uses only historical data (causal), making the representation
    honest: smoothed values at time t use only data from [t-window+1, t].
    """
    if len(data_series) < window:
        return data_series.tolist()
    
    series = pd.Series(data_series)
    # Remove center=True to use trailing (backward-looking) window
    smoothed = series.rolling(window=window, min_periods=1).mean()
    
    return smoothed.tolist()


def _create_time_points(length):
    """Create time points assuming 1-minute collection interval."""
    return list(range(length))


def _calculate_littles_law_active_drivers(design_point):
    """
    Calculate theoretical active drivers using Little's Law.
    
    E[Active Drivers] = λ_driver × E[Service Duration]
    
    Args:
        design_point: DesignPoint instance with operational config
        
    Returns:
        float: Theoretical number of active drivers
    """
    # Driver arrival rate (arrivals per minute)
    mean_inter_arrival = design_point.operational_config.mean_driver_inter_arrival_time
    arrival_rate = 1.0 / mean_inter_arrival
    
    # Mean service duration (minutes)
    mean_service_duration = design_point.operational_config.mean_service_duration
    
    # Little's Law: E[N] = λ × E[S]
    theoretical_active_drivers = arrival_rate * mean_service_duration
    
    return theoretical_active_drivers

