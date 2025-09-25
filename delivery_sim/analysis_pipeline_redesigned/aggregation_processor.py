# analysis_pipeline_redesigned/aggregation_processor.py
"""
Aggregation processor for pattern-based metric processing.

Clean design using data structure-based aggregation methods:
- _aggregate_individual_values: List of individual metrics → Statistics objects
- _aggregate_scalar_values: List of scalar values → Statistics objects  
- _aggregate_statistics_objects: Statistics objects → Statistics-of-statistics

Two-level pattern:
  Replication: Individual entities → Statistics objects (via individual values aggregation)
  Experiment: Statistics objects → Statistics-of-statistics

One-level pattern:
  Replication: AnalysisData → Direct calculation (already replication level)  
  Experiment: Scalar values → Statistics objects (via scalar values aggregation)
"""

import importlib
from delivery_sim.utils.logging_system import get_logger
from delivery_sim.analysis_pipeline_redesigned.statistics_engine import StatisticsEngine
from delivery_sim.analysis_pipeline_redesigned.metric_configurations import (
    get_metric_configuration,
    get_experiment_statistics
)


class AggregationProcessor:
    """
    Handles aggregation using data structure-based methods for clarity.
    """
    
    def __init__(self):
        self.logger = get_logger("analysis_pipeline_redesigned.aggregation_processor")
        self.statistics_engine = StatisticsEngine()
        self._metric_function_cache = {}
    
    # ==============================================================================
    # PROCESSING METHODS (ORCHESTRATION)
    # ==============================================================================
    
    def process_replication_level(self, analysis_data, metric_type):
        """Process metrics at replication level according to aggregation pattern."""
        config = get_metric_configuration(metric_type)
        pattern = config['aggregation_pattern']
        
        if pattern == 'two_level':
            return self._process_two_level_replication(analysis_data, config, metric_type)
        elif pattern == 'one_level':
            return self._process_one_level_replication(analysis_data, config)
        else:
            raise ValueError(f"Unknown aggregation pattern: {pattern}")
    
    def process_experiment_level(self, replication_results, metric_type):
        """Process metrics at experiment level according to aggregation pattern."""
        config = get_metric_configuration(metric_type)
        pattern = config['aggregation_pattern']
        
        if pattern == 'two_level':
            return self._process_two_level_experiment(replication_results, metric_type)
        elif pattern == 'one_level':
            return self._process_one_level_experiment(replication_results)
        else:
            raise ValueError(f"Unknown aggregation pattern: {pattern}")
    
    def _process_two_level_replication(self, analysis_data, config, metric_type):
        """Individual entities → Statistics objects (via individual values aggregation)."""
        # Get entities
        entity_data_key = config['entity_data_key']
        entities = getattr(analysis_data, entity_data_key, [])
        
        if not entities:
            self.logger.warning(f"No entities found for {metric_type}")
            return {}
        
        # Calculate individual metrics
        metric_function = self._get_metric_function(config)
        individual_metrics = [metric_function(entity) for entity in entities]
        
        # Aggregate individual values to statistics objects
        # ✅ SIMPLIFIED: Remove entity_type extraction and passing
        return self._aggregate_individual_values(individual_metrics)
    
    def _process_one_level_replication(self, analysis_data, config):
        """Direct calculation (already at replication level)."""
        metric_function = self._get_metric_function(config)
        return metric_function(analysis_data)
    
    def _process_two_level_experiment(self, replication_results, metric_type):
        """Statistics objects → Statistics-of-statistics."""
        if not replication_results or not replication_results[0]:
            return {}
        
        experiment_stats_config = get_experiment_statistics(metric_type)
        return self._aggregate_statistics_objects(replication_results, experiment_stats_config)
    
    def _process_one_level_experiment(self, replication_results):
        """Scalar values → Statistics objects (via scalar values aggregation)."""
        if not replication_results:
            return {}
        
        # Extract scalar values for aggregation
        scalar_data = self._extract_scalar_data(replication_results)
        return self._aggregate_scalar_values(scalar_data)
    
    # ==============================================================================
    # AGGREGATION METHODS (CATEGORIZED BY DATA STRUCTURE)
    # ==============================================================================
    
    def _aggregate_individual_values(self, individual_metrics):
        """
        Aggregate individual entity values into statistics objects.
        
        Input: List of individual metric dictionaries from entities
        Output: Direct statistics structure {metric_name: stats}  # ← Updated comment
        Context: Within replication, across entities
        """
        if not individual_metrics:
            return {}
        
        metric_names = list(individual_metrics[0].keys())
        results = {}
        
        for metric_name in metric_names:
            # Extract values for this metric across all entities
            values = [metrics[metric_name] for metrics in individual_metrics 
                    if metric_name in metrics and metrics[metric_name] is not None]
            
            if values:
                results[metric_name] = self.statistics_engine.calculate_statistics(values)
        
        return results  # ✅ SIMPLIFIED: Remove entity_type wrapper
    
    def _aggregate_scalar_values(self, scalar_data):
        """
        Aggregate scalar values into statistics objects.
        
        Input: Dict of metric_name -> list of scalar values
        Output: Flat statistics structure {metric_name: stats}
        Context: Across replications, for system metrics
        """
        results = {}
        
        for metric_name, values in scalar_data.items():
            if values:
                results[metric_name] = self.statistics_engine.calculate_basic_statistics(values)
        
        return results
    
    def _aggregate_statistics_objects(self, replication_results, experiment_stats_config):
        """
        Aggregate statistics objects into statistics-of-statistics.
        
        Input: List of replication results with statistics objects
        Output: Experiment-level statistics-of-statistics
        Context: Across replications, for entity metrics (second order)
        """
        if not replication_results or not replication_results[0]:
            return {}
        
        # ✅ SIMPLIFIED: Direct metric access, no entity_type navigation
        metric_names = list(replication_results[0].keys())
        results = {}
        
        for metric_name in metric_names:
            results[metric_name] = {}
            
            # Calculate each configured statistics-of-statistics
            for stat_config in experiment_stats_config:
                stat_name = stat_config['name']
                extract_stat = stat_config['extract']
                compute_stat = stat_config['compute']
                
                # ✅ SIMPLIFIED: Extract base statistic values across replications
                extracted_values = self.statistics_engine.extract_statistic_for_experiment_aggregation(
                    replication_results, metric_name, extract_stat  # ← Removed entity_type parameter
                )
                
                if extracted_values:
                    # Compute target statistic on extracted values
                    if compute_stat == 'mean':
                        stats = self.statistics_engine.calculate_basic_statistics(extracted_values)
                        results[metric_name][stat_name] = stats['mean']
                    elif compute_stat == 'std':
                        stats = self.statistics_engine.calculate_basic_statistics(extracted_values)
                        results[metric_name][stat_name] = stats['std']
                    elif compute_stat == 'variance':
                        stats = self.statistics_engine.calculate_basic_statistics(extracted_values)
                        results[metric_name][stat_name] = stats['variance']
                    elif compute_stat == 'full_stats':
                        results[metric_name][stat_name] = \
                            self.statistics_engine.calculate_basic_statistics(extracted_values)
                    else:
                        self.logger.warning(f"Unknown compute statistic: {compute_stat}")
                else:
                    self.logger.warning(f"No values for {metric_name}.{extract_stat}")
        
        return results
    
    # ==============================================================================
    # HELPER METHODS
    # ==============================================================================
    
    def _extract_scalar_data(self, replication_results):
        """Extract scalar data from replication results for scalar aggregation."""
        metric_names = list(replication_results[0].keys())
        scalar_data = {}
        
        for metric_name in metric_names:
            values = [rep_result[metric_name] for rep_result in replication_results 
                     if metric_name in rep_result]
            scalar_data[metric_name] = values
        
        return scalar_data
    
    def _get_metric_function(self, config):
        """Import and cache metric function."""
        module_path = config['metric_module']
        function_name = config['metric_function']
        cache_key = f"{module_path}.{function_name}"
        
        if cache_key not in self._metric_function_cache:
            module = importlib.import_module(module_path)
            self._metric_function_cache[cache_key] = getattr(module, function_name)
        
        return self._metric_function_cache[cache_key]