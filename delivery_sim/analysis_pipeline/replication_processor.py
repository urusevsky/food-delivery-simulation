# analysis_pipeline/replication_processor.py
"""
Replication-level metric processing (Phase 1).

Single responsibility: Transform raw simulation data into standardized
replication-level metrics, handling pattern-specific processing internally.
"""

import importlib
from delivery_sim.utils.logging_system import get_logger
from delivery_sim.analysis_pipeline.statistics_engine import StatisticsEngine


class ReplicationProcessor:
    """
    Phase 1: Process raw simulation data into replication-level metrics.
    
    Handles both aggregation patterns:
    - Two-level: Raw data → Individual metrics → Statistics (first aggregation)
    - One-level: Raw data → Direct calculation (no aggregation at this phase)
    """
    
    def __init__(self):
        self.logger = get_logger("analysis_pipeline.replication_processor")
        self.statistics_engine = StatisticsEngine()
        self._metric_function_cache = {}
    
    def process_replication(self, analysis_data, metric_config):
        """
        Process single replication according to metric configuration.
        
        Args:
            analysis_data: Prepared analysis data from data_preparation module
            metric_config: Metric configuration dictionary for a specific metric type
            
        Returns:
            dict: Replication-level metrics (format depends on pattern)
        """
        pattern = metric_config['aggregation_pattern']
        
        if pattern == 'two_level':
            return self._process_two_level_replication(analysis_data, metric_config)
        elif pattern == 'one_level':
            return self._process_one_level_replication(analysis_data, metric_config)
        else:
            raise ValueError(f"Unknown aggregation pattern: {pattern}")
    
    def _process_two_level_replication(self, analysis_data, metric_config):
        """
        Two-level pattern: Items → Metrics → Statistics.
        
        Handles both entity-based metrics and system state metrics through
        identical processing structure despite different timing of calculation.
        
        PROCESSING FLOW:
        1. Get items (entities or snapshots) from analysis_data
        2. Apply metric_function to each item
        - For entity metrics: CALCULATES metrics from entity attributes
        - For state metrics: EXTRACTS pre-calculated metrics from snapshot
        3. Aggregate metrics across items to statistics
        
        CONCRETE EXAMPLES:
        
        Entity-based metrics (order_metrics):
            items = [Order1, Order2, ...]  # From cohort_completed_orders
            metric_function = calculate_all_order_metrics
            individual_metrics = [
                {'assignment_time': 12.3, 'fulfillment_time': 25.7},  # Calculated from Order1 attributes
                {'assignment_time': 15.1, 'fulfillment_time': 28.2},  # Calculated from Order2 attributes
                ...
            ]
            → Aggregate to statistics: {'assignment_time': {mean: 13.7, std: 2.1, ...}, ...}
        
        System state metrics (system_state_metrics):
            items = [Snapshot1, Snapshot2, ...]  # From post_warmup_snapshots
            metric_function = extract_snapshot_metrics
            individual_metrics = [
                {'available_drivers': 5, 'active_drivers': 20, ...},  # Extracted from Snapshot1
                {'available_drivers': 7, 'active_drivers': 22, ...},  # Extracted from Snapshot2
                ...
            ]
            → Aggregate to statistics: {'available_drivers': {mean: 6.2, std: 1.5, ...}, ...}
        
        Args:
            analysis_data: AnalysisData with filtered populations and snapshots
            metric_config: Configuration specifying data_key and metric_function
            
        Returns:
            dict: Statistics for each metric {metric_name: {mean, std, min, max, ...}}
        """
        # Get items from analysis data
        data_key = metric_config['data_key']
        items = getattr(analysis_data, data_key, [])
        
        if not items:
            self.logger.warning(f"No items found for {data_key}")
            return {}
        
        # Get/calculate metrics for each item
        # For entity metrics: function CALCULATES from attributes
        # For state metrics: function EXTRACTS pre-calculated values
        metric_function = self._get_metric_function(metric_config)
        individual_metrics = [metric_function(item) for item in items]
        
        # Aggregate individual metric values to statistics (first aggregation)
        # Identical process regardless of whether metrics were calculated or extracted
        return self._aggregate_individual_values(individual_metrics)
    
    def _process_one_level_replication(self, analysis_data, metric_config):
        """
        One-level pattern: Direct calculation (already replication-level).
        
        No aggregation at this phase - just calculation.
        """
        metric_function = self._get_metric_function(metric_config)
        return metric_function(analysis_data)
    
    def _aggregate_individual_values(self, individual_metrics):
        """
        Aggregate individual entity values into statistics objects.
        
        Used by two-level pattern for first aggregation within replication.
        """
        if not individual_metrics:
            return {}
        
        metric_names = list(individual_metrics[0].keys())
        results = {}
        
        for metric_name in metric_names:
            # Extract values for this metric across all entities
            values = [
                metrics[metric_name] 
                for metrics in individual_metrics 
                if metric_name in metrics and metrics[metric_name] is not None
            ]
            
            if values:
                results[metric_name] = self.statistics_engine.calculate_statistics(values)
        
        return results
    
    def _get_metric_function(self, metric_config):
        """Import and cache metric calculation function."""
        module_path = metric_config['metric_module']
        function_name = metric_config['metric_function']
        cache_key = f"{module_path}.{function_name}"
        
        if cache_key not in self._metric_function_cache:
            module = importlib.import_module(module_path)
            self._metric_function_cache[cache_key] = getattr(module, function_name)
        
        return self._metric_function_cache[cache_key]