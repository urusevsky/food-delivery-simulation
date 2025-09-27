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
    
    def process_replication_level(self, analysis_data, config):  # 🎯 CHANGED: metric_type → config
        """
        Process metrics at replication level according to aggregation pattern.
        
        Args:
            analysis_data: Prepared analysis data from data_preparation module  
            config: Metric configuration dictionary (from get_metric_configuration)  # 🎯 UPDATED
            
        Returns:
            dict: Replication-level metrics (format depends on aggregation pattern)
        """
        pattern = config['aggregation_pattern']  # 🎯 CHANGED: Direct access instead of get_metric_configuration()
        
        if pattern == 'two_level':
            return self._process_two_level_replication(analysis_data, config)
        elif pattern == 'one_level':
            return self._process_one_level_replication(analysis_data, config)
        else:
            raise ValueError(f"Unknown aggregation pattern: {pattern}")
    
    def process_experiment_level(self, processed_replications, config):  # 🎯 CHANGED: metric_type → config
        """
        Process metrics at experiment level according to aggregation pattern.
        
        Args:
            processed_replications: List of replication-level metric results
            config: Metric configuration dictionary (from get_metric_configuration)  # 🎯 UPDATED
            
        Returns:
            dict: Experiment-level aggregated statistics
        """
        pattern = config['aggregation_pattern']  # 🎯 CHANGED: Direct access instead of get_metric_configuration()
        
        if pattern == 'two_level':
            return self._process_two_level_experiment(processed_replications, config)  # 🎯 CHANGED: Pass config
        elif pattern == 'one_level':
            return self._process_one_level_experiment(processed_replications)
        else:
            raise ValueError(f"Unknown aggregation pattern: {pattern}")
    
    def _process_two_level_replication(self, analysis_data, config):  # ✅ Removed metric_type
        """Individual entities → Statistics objects (via individual values aggregation)."""
        # Get entities
        entity_data_key = config['entity_data_key']
        entities = getattr(analysis_data, entity_data_key, [])
        
        if not entities:
            self.logger.warning(f"No entities found for {entity_data_key}")  # ✅ Use entity_data_key from config
            return {}
        
        # Calculate individual metrics
        metric_function = self._get_metric_function(config)
        individual_metrics = [metric_function(entity) for entity in entities]
        
        # Aggregate individual values to statistics objects
        return self._aggregate_individual_values(individual_metrics)
    
    def _process_one_level_replication(self, analysis_data, config):
        """Direct calculation (already at replication level)."""
        metric_function = self._get_metric_function(config)
        return metric_function(analysis_data)
    
    def _process_two_level_experiment(self, processed_replications, config):  # 🎯 CHANGED: metric_type → config
        """
        Statistics objects → Statistics-of-statistics.
        
        Args:
            processed_replications: List of replication results with statistics objects
            config: Metric configuration dictionary containing experiment_stats  # 🎯 UPDATED
            
        Returns:
            dict: Experiment-level statistics-of-statistics
        """
        if not processed_replications or not processed_replications[0]:
            return {}
        
        experiment_stats_config = config.get('experiment_stats', [])  # 🎯 CHANGED: Direct access instead of get_experiment_statistics()
        return self._aggregate_statistics_objects(processed_replications, experiment_stats_config)
    
    def _process_one_level_experiment(self, processed_replications):  # ✅ Updated parameter
        """Scalar values → Statistics objects (via scalar values aggregation)."""
        if not processed_replications:  # ✅ Updated
            return {}
        
        # Extract scalar values for aggregation
        scalar_data = self._extract_scalar_data(processed_replications)  # ✅ Updated
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
                results[metric_name] = self.statistics_engine.calculate_statistics(values, include_percentiles=False)
        
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
                        stats = self.statistics_engine.calculate_statistics(extracted_values, include_percentiles=False)
                        results[metric_name][stat_name] = stats['mean']
                    elif compute_stat == 'std':
                        stats = self.statistics_engine.calculate_statistics(extracted_values, include_percentiles=False)
                        results[metric_name][stat_name] = stats['std']
                    elif compute_stat == 'variance':
                        stats = self.statistics_engine.calculate_statistics(extracted_values, include_percentiles=False)
                        results[metric_name][stat_name] = stats['variance']
                    elif compute_stat == 'full_stats':
                        results[metric_name][stat_name] = \
                            self.statistics_engine.calculate_statistics(extracted_values, include_percentiles=False)
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