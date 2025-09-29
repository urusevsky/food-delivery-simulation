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
    
    def process_replication(self, analysis_data, config):
        """
        Process single replication according to metric configuration.
        
        Args:
            analysis_data: Prepared analysis data from data_preparation module
            config: Metric configuration dictionary
            
        Returns:
            dict: Replication-level metrics (format depends on pattern)
        """
        pattern = config['aggregation_pattern']
        
        if pattern == 'two_level':
            return self._process_two_level_replication(analysis_data, config)
        elif pattern == 'one_level':
            return self._process_one_level_replication(analysis_data, config)
        else:
            raise ValueError(f"Unknown aggregation pattern: {pattern}")
    
    def _process_two_level_replication(self, analysis_data, config):
        """
        Two-level pattern: Individual entities → Statistics objects.
        
        First aggregation happens here (within replication).
        """
        # Get entities from analysis data
        entity_data_key = config['entity_data_key']
        entities = getattr(analysis_data, entity_data_key, [])
        
        if not entities:
            self.logger.warning(f"No entities found for {entity_data_key}")
            return {}
        
        # Calculate individual metrics for each entity
        metric_function = self._get_metric_function(config)
        individual_metrics = [metric_function(entity) for entity in entities]
        
        # Aggregate individual values to statistics objects (first aggregation)
        return self._aggregate_individual_values(individual_metrics)
    
    def _process_one_level_replication(self, analysis_data, config):
        """
        One-level pattern: Direct calculation (already replication-level).
        
        No aggregation at this phase - just calculation.
        """
        metric_function = self._get_metric_function(config)
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
    
    def _get_metric_function(self, config):
        """Import and cache metric calculation function."""
        module_path = config['metric_module']
        function_name = config['metric_function']
        cache_key = f"{module_path}.{function_name}"
        
        if cache_key not in self._metric_function_cache:
            module = importlib.import_module(module_path)
            self._metric_function_cache[cache_key] = getattr(module, function_name)
        
        return self._metric_function_cache[cache_key]