# analysis_pipeline/experiment_aggregator.py
"""
Experiment-level aggregation across replications (Phase 2).

Single responsibility: Aggregate replication-level metrics into
experiment-level statistics, handling pattern-specific logic internally.
"""

from delivery_sim.utils.logging_system import get_logger
from delivery_sim.analysis_pipeline.statistics_engine import StatisticsEngine
from delivery_sim.analysis_pipeline.extraction_engine import ExtractionEngine


class ExperimentAggregator:
    """
    Phase 2: Aggregate replication-level metrics across all replications.
    
    Handles both aggregation patterns:
    - Two-level: Statistics → Statistics-of-statistics (second aggregation)
    - One-level: Scalars → Statistics (first and only aggregation)
    """
    
    def __init__(self):
        self.logger = get_logger("analysis_pipeline.experiment_aggregator")
        self.statistics_engine = StatisticsEngine()
        self.extraction_engine = ExtractionEngine()
    
    def aggregate_experiment(self, metrics_across_replications, config):
        """
        Aggregate across all replications according to metric configuration.
        
        Args:
            metrics_across_replications: List of metric results, one per replication
                                         (for a SINGLE metric type)
            config: Metric configuration dictionary
            
        Returns:
            dict: Experiment-level aggregated statistics
        """
        # Guard clause and common setup
        if not metrics_across_replications:
            return {}
        
        # Extract metric names once at router level
        metric_names = list(metrics_across_replications[0].keys())
        
        # Route to pattern-specific aggregation
        pattern = config['aggregation_pattern']
        
        if pattern == 'two_level':
            return self._aggregate_two_level_experiment(
                metrics_across_replications, metric_names, config
            )
        elif pattern == 'one_level':
            return self._aggregate_one_level_experiment(
                metrics_across_replications, metric_names
            )
        else:
            raise ValueError(f"Unknown aggregation pattern: {pattern}")
    
    def _aggregate_two_level_experiment(self, metrics_across_replications, 
                                       metric_names, config):
        """
        Two-level pattern: Statistics → Statistics-of-statistics.
        
        Second aggregation happens here (across replications).
        
        Args:
            metrics_across_replications: List of replication-level metric dicts
            metric_names: List of metric names to process
            config: Metric configuration
        """
        experiment_stats_config = config.get('experiment_stats', [])
        results = {}
        
        for metric_name in metric_names:
            results[metric_name] = {}
            
            # Calculate each configured statistics-of-statistics
            for stat_config in experiment_stats_config:
                stat_name = stat_config['name']
                extract_stat = stat_config['extract']
                compute_stat = stat_config['compute']
                
                # Extract base statistic values across replications
                extracted_values = self.extraction_engine.extract_for_two_level_pattern(
                    metrics_across_replications, metric_name, extract_stat
                )
                
                if extracted_values:
                    # Compute target statistic on extracted values
                    if compute_stat == 'mean':
                        stats = self.statistics_engine.calculate_statistics(
                            extracted_values, include_percentiles=False
                        )
                        results[metric_name][stat_name] = stats['mean']
                    elif compute_stat == 'std':
                        stats = self.statistics_engine.calculate_statistics(
                            extracted_values, include_percentiles=False
                        )
                        results[metric_name][stat_name] = stats['std']
                    elif compute_stat == 'variance':
                        stats = self.statistics_engine.calculate_statistics(
                            extracted_values, include_percentiles=False
                        )
                        results[metric_name][stat_name] = stats['variance']
                    elif compute_stat == 'full_stats':
                        results[metric_name][stat_name] = \
                            self.statistics_engine.calculate_statistics(
                                extracted_values, include_percentiles=False
                            )
                    else:
                        self.logger.warning(f"Unknown compute statistic: {compute_stat}")
                else:
                    self.logger.warning(f"No values for {metric_name}.{extract_stat}")
        
        return results
    
    def _aggregate_one_level_experiment(self, metrics_across_replications, metric_names):
            """
            One-level pattern: Scalars → Statistics.
            
            First and only aggregation happens here (across replications).
            
            Args:
                metrics_across_replications: List of scalar dicts, one per replication
                metric_names: List of metric names to process
            """
            results = {}
            
            for metric_name in metric_names:
                values = self.extraction_engine.extract_for_one_level_pattern(
                    metrics_across_replications, metric_name
                )
                if values:
                    results[metric_name] = self.statistics_engine.calculate_statistics(
                        values, include_percentiles=False
                    )
            
            return results