import numpy as np
from delivery_sim.utils.logging_system import get_logger

class WelchAnalyzer:
    """
    Simple implementation of Welch's method for warmup detection.
    
    Follows the textbook approach:
    1. Vertical averaging across replications 
    2. Cumulative average smoothing
    3. Visual inspection of convergence
    """
    
    def __init__(self):
        self.logger = get_logger("warmup_analysis.welch_analyzer")
    
    def analyze_warmup_convergence(self, multi_replication_snapshots, 
                                  metrics=['active_drivers', 'active_delivery_entities'],
                                  collection_interval=1.0):
        """
        Apply Welch's method: vertical averaging + cumulative smoothing.
        
        Args:
            multi_replication_snapshots: List of snapshot lists (one per replication)
            metrics: List of metric names to analyze
            collection_interval: Time interval between snapshots
            
        Returns:
            dict: Simple results for each metric with time series data
        """
        self.logger.info(f"Starting Welch analysis for {len(multi_replication_snapshots)} replications")
        
        results = {}
        
        for metric_name in metrics:
            self.logger.debug(f"Processing metric: {metric_name}")
            
            # Step 1: Extract metric data from each replication
            metric_data = self._extract_metric_data(multi_replication_snapshots, metric_name)
            
            if not metric_data:
                self.logger.warning(f"No data found for metric {metric_name}")
                continue
            
            # Step 2: Vertical averaging - calculate Ȳ.j across replications
            cross_rep_averages = self._calculate_cross_replication_averages(metric_data)
            
            # Step 3: Horizontal smoothing - cumulative average only (per textbook)
            cumulative_average = self._calculate_cumulative_average(cross_rep_averages)
            
            # Step 4: Prepare time axis
            time_points = [i * collection_interval for i in range(len(cross_rep_averages))]
            
            results[metric_name] = {
                'time_points': time_points,
                'cross_rep_averages': cross_rep_averages,  # Raw Ȳ.j (blue line)
                'cumulative_average': cumulative_average,  # Smoothed (red line)
                'replication_count': len(metric_data)
            }
            
            self.logger.debug(f"Completed analysis for {metric_name}: {len(time_points)} time points")
        
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
        Calculate Ȳ.j = (1/R) × Σ(r=1 to R) Y_rj for each time point j.
        
        This is the vertical averaging step of Welch's method.
        """
        data_array = np.array(metric_data)  # Shape: (replications, time_points)
        cross_rep_averages = np.mean(data_array, axis=0)  # Average across replications
        return cross_rep_averages.tolist()
    
    def _calculate_cumulative_average(self, cross_rep_averages):
        """
        Calculate cumulative average for smoothing (per textbook approach).
        
        This is the horizontal smoothing step: cumulative average of Ȳ.j series.
        """
        data = np.array(cross_rep_averages)
        cumulative_average = np.cumsum(data) / np.arange(1, len(data) + 1)
        return cumulative_average.tolist()