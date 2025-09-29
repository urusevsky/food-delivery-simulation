# analysis_pipeline/extraction_engine.py
"""
Centralized extraction engine for replication-level data.

Single responsibility: Extract values from replication_level_metrics
for downstream statistical processing and CI construction.
"""

from delivery_sim.utils.logging_system import get_logger


class ExtractionEngine:
    """
    Centralized engine for extracting values from replication-level metrics.
    
    Handles both aggregation patterns with consistent interface.
    """
    
    def __init__(self):
        self.logger = get_logger("analysis_pipeline.extraction_engine")
    
    def extract_for_two_level_pattern(self, replication_level_metrics, metric_name, statistic_type):
        """
        Extract specific statistic across replications for two-level pattern.
        
        Args:
            replication_level_metrics: List of replication dictionaries
            metric_name: Metric name (e.g., 'assignment_time')
            statistic_type: Statistic to extract (e.g., 'mean', 'std', 'p95')
            
        Returns:
            list: Statistic values across replications
            
        Example:
            means = engine.extract_for_two_level_pattern(
                replication_data, 'assignment_time', 'mean'
            )
            # Result: [12.47, 13.12, 11.85] - one mean per replication
        """
        values = []
        
        for rep_summary in replication_level_metrics:
            metric_data = rep_summary[metric_name]
            stat_value = metric_data[statistic_type]
            values.append(float(stat_value))
        
        return values
    
    def extract_for_one_level_pattern(self, replication_level_metrics):
        """
        Extract all scalar values across replications for one-level pattern.
        
        Args:
            replication_level_metrics: List of replication dictionaries
            
        Returns:
            dict: {metric_name: [values_across_replications]}
        """
        scalar_data = {}
        
        if not replication_level_metrics:
            return scalar_data
            
        # Get all metric names from first replication
        metric_names = list(replication_level_metrics[0].keys())
        
        for metric_name in metric_names:
            scalar_data[metric_name] = [
                rep_result[metric_name] 
                for rep_result in replication_level_metrics 
                if metric_name in rep_result
            ]
        
        return scalar_data
    
    def extract_scalar_values_for_metric(self, replication_level_metrics, metric_name):
        """
        Extract scalar values for a specific metric (one-level pattern).
        
        Convenience method for CI construction on specific metrics.
        
        Args:
            replication_level_metrics: List of replication dictionaries
            metric_name: Name of the metric to extract
            
        Returns:
            list: Scalar values across replications for this metric
        """
        return [
            rep_result[metric_name] 
            for rep_result in replication_level_metrics 
            if metric_name in rep_result
        ]