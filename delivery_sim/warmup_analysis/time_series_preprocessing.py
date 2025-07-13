# delivery_sim/warmup_analysis/time_series_preprocessing.py
"""
Time Series Preprocessing for Warmup Analysis

Simplified approach focusing on cross-replication averaging for visual inspection.
Removes the complexity of cumulative smoothing in favor of direct pattern recognition.

This module extracts cross-replication averages that reveal underlying system behavior
patterns by removing replication-specific noise. Human visual inspection of these
averaged time series is used to determine warmup periods.
"""

import numpy as np
from delivery_sim.utils.logging_system import get_logger


class TimeSeriesPreprocessor:
    """
    Simple preprocessor for multi-replication time series data.
    
    Focuses on cross-replication averaging to reveal underlying system patterns
    for visual warmup period determination.
    """
    
    def __init__(self):
        self.logger = get_logger("warmup_analysis.preprocessor")
    
    def extract_cross_replication_averages(self, multi_replication_snapshots, 
                                         metrics=['active_drivers', 'active_delivery_entities'],
                                         collection_interval=1.0):
        """
        Extract cross-replication averages for visual warmup inspection.
        
        This is the core of warmup analysis: averaging across replications reveals
        the underlying system behavior pattern while removing replication-specific noise.
        
        Args:
            multi_replication_snapshots: List of snapshot lists (one per replication)
            metrics: List of metric names to process
            collection_interval: Time interval between snapshots (for time axis)
            
        Returns:
            dict: Simple time series data for each metric ready for plotting
        """
        self.logger.info(f"Processing {len(multi_replication_snapshots)} replications for warmup analysis")
        
        results = {}
        
        for metric_name in metrics:
            self.logger.debug(f"Processing metric: {metric_name}")
            
            # Extract metric data from each replication
            metric_data = self._extract_metric_data(multi_replication_snapshots, metric_name)
            
            if not metric_data:
                self.logger.warning(f"No data found for metric {metric_name}")
                continue
            
            # Calculate cross-replication averages (the key insight!)
            cross_rep_averages = self._calculate_cross_replication_averages(metric_data)
            
            # Prepare time axis
            time_points = [i * collection_interval for i in range(len(cross_rep_averages))]
            
            results[metric_name] = {
                'time_points': time_points,
                'cross_rep_averages': cross_rep_averages,
                'replication_count': len(metric_data),
                'metric_name': metric_name
            }
            
            self.logger.debug(f"Processed {metric_name}: {len(time_points)} time points, {len(metric_data)} replications")
        
        self.logger.info(f"Preprocessing complete: {len(results)} metrics ready for visual inspection")
        return results
    
    def _extract_metric_data(self, multi_replication_snapshots, metric_name):
        """Extract metric data from each replication (no alignment needed)."""
        metric_data = []
        
        for rep_idx, snapshots in enumerate(multi_replication_snapshots):
            if not snapshots:
                self.logger.warning(f"Replication {rep_idx} has no snapshots")
                continue
                
            # Extract metric values for this replication
            metric_values = []
            for snapshot in snapshots:
                if metric_name in snapshot:
                    metric_values.append(snapshot[metric_name])
                else:
                    self.logger.warning(f"Metric {metric_name} missing in snapshot")
                    metric_values.append(0)  # Default for missing values
            
            if metric_values:
                metric_data.append(metric_values)
                self.logger.debug(f"Replication {rep_idx}: {len(metric_values)} data points")
        
        if not metric_data:
            self.logger.error(f"No valid data found for metric {metric_name}")
            return None
        
        # Verify all replications have same length (they should!)
        lengths = [len(series) for series in metric_data]
        if len(set(lengths)) > 1:
            self.logger.warning(f"Replication lengths differ: {lengths}. Using shortest: {min(lengths)}")
            # Truncate to shortest length as fallback
            min_length = min(lengths)
            metric_data = [series[:min_length] for series in metric_data]
        
        return metric_data
    
    def _calculate_cross_replication_averages(self, metric_data):
        """
        Calculate cross-replication averages for each time point.
        
        This is the core operation: Ȳ.j = (1/R) × Σ(r=1 to R) Y_rj
        where R is the number of replications and j is the time point.
        
        This averaging removes replication-specific randomness and reveals
        the underlying system behavior pattern.
        """
        data_array = np.array(metric_data)  # Shape: (replications, time_points)
        cross_rep_averages = np.mean(data_array, axis=0)  # Average across replications
        return cross_rep_averages.tolist()


def extract_time_series_for_inspection(multi_replication_snapshots, 
                                     metrics=['active_drivers', 'active_delivery_entities'],
                                     collection_interval=0.5):
    """
    Convenience function for extracting time series data for warmup inspection.
    
    Args:
        multi_replication_snapshots: List of snapshot lists from simulation results
        metrics: List of metric names to analyze
        collection_interval: Time between snapshots (should match SystemDataCollector)
        
    Returns:
        dict: Time series data ready for visual inspection
    """
    preprocessor = TimeSeriesPreprocessor()
    return preprocessor.extract_cross_replication_averages(
        multi_replication_snapshots, metrics, collection_interval
    )