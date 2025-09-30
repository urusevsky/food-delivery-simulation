# analysis_pipeline/extraction_engine.py
"""
Centralized extraction engine for replication-level data.

Single responsibility: Extract values from metrics_across_replications
for downstream statistical processing and CI construction.
"""

from delivery_sim.utils.logging_system import get_logger


class ExtractionEngine:
    """
    Centralized engine for extracting values from replication-level metrics.
    
    Provides focused extraction methods for both aggregation patterns.
    """
    
    def __init__(self):
        self.logger = get_logger("analysis_pipeline.extraction_engine")
    
    def extract_for_two_level_pattern(self, metrics_across_replications, 
                                      metric_name, statistic_type):
        """
        Extract specific statistic across replications for two-level pattern.
        
        Args:
            metrics_across_replications: List of replication metric dicts
            metric_name: Metric name (e.g., 'assignment_time')
            statistic_type: Statistic to extract (e.g., 'mean', 'std', 'p95')
            
        Returns:
            list: Statistic values across replications
            
        Example:
            means = engine.extract_for_two_level_pattern(
                metrics_data, 'assignment_time', 'mean'
            )
            # Result: [12.47, 13.12, 11.85] - one mean per replication
        """
        return [
            float(rep_metrics[metric_name][statistic_type])
            for rep_metrics in metrics_across_replications
        ]
    
    def extract_for_one_level_pattern(self, metrics_across_replications, metric_name):
        """
        Extract scalar values for a specific metric (one-level pattern).
        
        Args:
            metrics_across_replications: List of replication scalar dicts
            metric_name: Name of the metric to extract
            
        Returns:
            list: Scalar values across replications for this metric
            
        Example:
            values = engine.extract_for_one_level_pattern(
                metrics_data, 'avg_active_drivers'
            )
            # Result: [45.2, 46.1, 44.8]
        """
        return [
            rep_metrics[metric_name] 
            for rep_metrics in metrics_across_replications 
            if metric_name in rep_metrics
        ]