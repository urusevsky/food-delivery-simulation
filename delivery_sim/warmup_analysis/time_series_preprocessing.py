# delivery_sim/warmup_analysis/time_series_preprocessing.py
"""
Enhanced Time Series Preprocessing for Warmup Analysis

Implements Welch's graphical method using moving averages and Little's Law 
theoretical validation for robust warmup period determination.
"""

import numpy as np
from delivery_sim.utils.logging_system import get_logger


class TimeSeriesPreprocessor:
    """
    Enhanced preprocessor implementing Welch's graphical method for warmup analysis.
    
    Combines cross-replication averaging with moving average smoothing and
    Little's Law theoretical validation for principled warmup detection.
    """
    
    def __init__(self):
        self.logger = get_logger("warmup_analysis.preprocessor")
    
    def extract_cross_replication_averages(self, multi_replication_snapshots, 
                                        metrics=['active_drivers', 'unassigned_delivery_entities'],
                                        moving_average_window=50):
        """
        Extract cross-replication averages with auto-detected collection interval.
        
        Args:
            multi_replication_snapshots: List of snapshot lists (one per replication)
            metrics: List of metric names to process
            moving_average_window: Window size for moving average smoothing
            
        Returns:
            dict: Enhanced time series data with auto-detected time axis
        """
        self.logger.info(f"Processing {len(multi_replication_snapshots)} replications with Welch's method")

        # Auto-detect collection interval from actual timestamps
        collection_interval = self._detect_collection_interval(multi_replication_snapshots)
    
        
        results = {}
        
        for metric_name in metrics:
            self.logger.debug(f"Processing metric: {metric_name}")
            
            # Extract metric data from each replication
            metric_data = self._extract_metric_data(multi_replication_snapshots, metric_name)
            
            if not metric_data:
                self.logger.warning(f"No data found for metric {metric_name}")
                continue
            
            # Calculate cross-replication averages
            cross_rep_averages = self._calculate_cross_replication_averages(metric_data)
            
            # Apply Welch's method: moving average smoothing
            moving_averages = self._calculate_moving_averages(cross_rep_averages, moving_average_window)
            
            # Prepare time axis
            time_points = [i * collection_interval for i in range(len(cross_rep_averages))]
            
            results[metric_name] = {
                'time_points': time_points,
                'cross_rep_averages': cross_rep_averages,
                'moving_averages': moving_averages,
                'replication_count': len(metric_data),
                'metric_name': metric_name,
                'moving_average_window': moving_average_window
            }
            
            self.logger.debug(f"Processed {metric_name}: {len(time_points)} time points, "
                            f"{len(metric_data)} replications, Welch's smoothing applied")
        
        self.logger.info(f"Welch's method preprocessing complete: {len(results)} metrics ready")
        return results
    
    def add_little_law_theoretical_values(self, time_series_data, design_points_dict):
        """
        Add Little's Law theoretical values for active drivers validation.
        
        Args:
            time_series_data: Dict of time series data by design point name
            design_points_dict: Dict of DesignPoint instances by name
            
        Returns:
            dict: Enhanced time series data with theoretical values
        """
        self.logger.info("Adding Little's Law theoretical validation")
        
        enhanced_data = {}
        
        for design_name, ts_data in time_series_data.items():
            if design_name not in design_points_dict:
                self.logger.warning(f"Design point {design_name} not found in design_points_dict")
                enhanced_data[design_name] = ts_data
                continue
            
            design_point = design_points_dict[design_name]
            enhanced_ts_data = ts_data.copy()
            
            # Calculate Little's Law theoretical value for active drivers
            if 'active_drivers' in ts_data:
                theoretical_active_drivers = self._calculate_little_law_active_drivers(design_point)
                enhanced_ts_data['active_drivers']['theoretical_value'] = theoretical_active_drivers
                
                self.logger.debug(f"{design_name}: Little's Law predicts {theoretical_active_drivers:.1f} active drivers")
            
            enhanced_data[design_name] = enhanced_ts_data
        
        self.logger.info(f"Little's Law theoretical values added for {len(enhanced_data)} design points")
        return enhanced_data
    
    def _calculate_little_law_active_drivers(self, design_point):
        """
        Calculate theoretical active drivers using Little's Law.
        
        E[Active Drivers] = λ_driver × E[Service Duration]
        """
        # Driver arrival rate (per minute)
        driver_arrival_rate = 1.0 / design_point.operational_config.mean_driver_inter_arrival_time
        
        # Mean service duration (minutes)
        mean_service_duration = design_point.operational_config.mean_service_duration
        
        # Little's Law application
        theoretical_active_drivers = driver_arrival_rate * mean_service_duration
        
        return theoretical_active_drivers
    
    def _calculate_moving_averages(self, data_series, window_size):
        """
        Calculate moving averages for Welch's graphical method.
        
        Uses centered moving average where possible, forward-looking for initial values.
        """
        if len(data_series) < window_size:
            self.logger.warning(f"Data series length ({len(data_series)}) < window size ({window_size})")
            return data_series.copy()
        
        moving_averages = []
        
        for i in range(len(data_series)):
            # Determine window bounds
            if i < window_size // 2:
                # Forward-looking window for initial values
                start_idx = 0
                end_idx = min(window_size, len(data_series))
            elif i >= len(data_series) - window_size // 2:
                # Backward-looking window for final values
                start_idx = max(0, len(data_series) - window_size)
                end_idx = len(data_series)
            else:
                # Centered window for middle values
                start_idx = i - window_size // 2
                end_idx = i + window_size // 2 + 1
            
            # Calculate average for this window
            window_data = data_series[start_idx:end_idx]
            moving_avg = np.mean(window_data)
            moving_averages.append(moving_avg)
        
        return moving_averages
    
    def _extract_metric_data(self, multi_replication_snapshots, metric_name):
        """Extract metric data from each replication."""
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
        
        # Verify all replications have same length
        lengths = [len(series) for series in metric_data]
        if len(set(lengths)) > 1:
            self.logger.warning(f"Replication lengths differ: {lengths}. Using shortest: {min(lengths)}")
            min_length = min(lengths)
            metric_data = [series[:min_length] for series in metric_data]
        
        return metric_data
    
    def _calculate_cross_replication_averages(self, metric_data):
        """Calculate cross-replication averages for each time point."""
        data_array = np.array(metric_data)
        cross_rep_averages = np.mean(data_array, axis=0)
        return cross_rep_averages.tolist()
    
    def _detect_collection_interval(self, multi_replication_snapshots):
        for replication_snapshots in multi_replication_snapshots:
            if len(replication_snapshots) >= 2:
                # Calculate first interval
                interval = replication_snapshots[1]['timestamp'] - replication_snapshots[0]['timestamp']
                
                # Verify all intervals are the same (optional validation)
                # for i in range(1, len(replication_snapshots) - 1):
                #     next_interval = replication_snapshots[i+1]['timestamp'] - replication_snapshots[i]['timestamp']
                #     if abs(next_interval - interval) > 1e-6:
                #         self.logger.warning(f"Inconsistent intervals detected: {interval} vs {next_interval}")
                
                self.logger.info(f"Detected collection interval: {interval:.2f} minutes")
                return interval
        
        return 1.0  # fallback


def extract_time_series_for_welch_analysis(multi_replication_snapshots, 
                                          design_points_dict=None,
                                          metrics=['active_drivers', 'unassigned_delivery_entities'],
                                          collection_interval=0.5,
                                          moving_average_window=50):
    """
    Convenience function for extracting enhanced time series data with Welch's method.
    
    Args:
        multi_replication_snapshots: List of snapshot lists from simulation results
        design_points_dict: Optional dict of DesignPoint instances for Little's Law validation
        metrics: List of metric names to analyze
        collection_interval: Time between snapshots
        moving_average_window: Window size for moving average smoothing
        
    Returns:
        dict: Enhanced time series data ready for Welch's method visualization
    """
    preprocessor = TimeSeriesPreprocessor()
    
    # Extract cross-replication averages with moving average smoothing
    time_series_data = preprocessor.extract_cross_replication_averages(
        multi_replication_snapshots, metrics, collection_interval, moving_average_window
    )
    
    # Add Little's Law theoretical values if design points provided
    if design_points_dict:
        time_series_data = preprocessor.add_little_law_theoretical_values(
            time_series_data, design_points_dict
        )
    
    return time_series_data