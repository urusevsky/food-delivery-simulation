# delivery_sim/analysis_pipeline/pipeline_coordinator.py
"""
Pipeline coordinator for end-to-end experiment analysis.

This module orchestrates the complete analysis pipeline from raw simulation results
to experiment-level summaries with confidence intervals.
"""

from delivery_sim.analysis_pipeline.data_preparation import filter_entities_for_analysis
from delivery_sim.analysis_pipeline.replication_level_aggregation.entity_replication_summaries import aggregate_entity_metrics
from delivery_sim.metrics.system.entity_derived_metrics import calculate_all_entity_derived_system_metrics
from delivery_sim.analysis_pipeline.experiment_level_aggregation.experiment_summaries import create_complete_experiment_summary
from delivery_sim.utils.logging_system import get_logger


class ExperimentAnalysisPipeline:
    """
    Coordinates the complete analysis pipeline for simulation experiments.
    
    Takes raw simulation results (multiple replications) and produces
    experiment-level summaries with confidence intervals.
    """
    
    def __init__(self, warmup_period, confidence_level=0.95):
        """
        Initialize the analysis pipeline.
        
        Args:
            warmup_period: Duration to exclude from analysis (simulation time units)
            confidence_level: Confidence level for intervals (default: 0.95)
        """
        self.warmup_period = warmup_period
        self.confidence_level = confidence_level
        self.logger = get_logger("analysis_pipeline.coordinator")
        
        self.logger.info(f"Pipeline initialized with warmup_period={warmup_period}, confidence_level={confidence_level}")
    
    def analyze_experiment(self, simulation_results):
        """
        Run complete analysis pipeline on simulation results.
        
        Args:
            simulation_results: Results from simulation_runner.run_experiment()
                Expected structure: {
                    'replication_results': [repo_dict1, repo_dict2, ...],
                    'infrastructure_characteristics': {...},
                    'config_summary': str,
                    'num_replications': int
                }
                
        Returns:
            dict: Complete experiment summary with confidence intervals
        """
        replication_results = simulation_results['replication_results']
        num_replications = len(replication_results)
        
        self.logger.info(f"Starting analysis pipeline for {num_replications} replications")
        
        # Process each replication through the pipeline
        entity_summaries = []
        system_summaries = []
        
        for i, repositories in enumerate(replication_results):
            self.logger.debug(f"Processing replication {i+1}/{num_replications}")
            
            # Step 1: Data preparation (filtering)
            filtered_entities = filter_entities_for_analysis(repositories, self.warmup_period)
            
            # Step 2: Entity metrics (if any filtered entities exist)
            entity_summary = self._process_entity_metrics(filtered_entities)
            if entity_summary:
                entity_summaries.append(entity_summary)
            
            # Step 3: System metrics
            system_summary = self._process_system_metrics(repositories, filtered_entities)
            system_summaries.append(system_summary)
        
        self.logger.info(f"Processed {len(entity_summaries)} entity summaries and {len(system_summaries)} system summaries")
        
        # Step 4: Experiment-level aggregation
        experiment_summary = create_complete_experiment_summary(
            entity_summaries, 
            system_summaries, 
            self.confidence_level
        )
        
        # Add metadata
        experiment_summary.update({
            'warmup_period': self.warmup_period,
            'infrastructure_characteristics': simulation_results.get('infrastructure_characteristics'),
            'config_summary': simulation_results.get('config_summary')
        })
        
        self.logger.info("Analysis pipeline completed successfully")
        return experiment_summary
    
    def _process_entity_metrics(self, filtered_entities):
        """
        Process entity metrics for a single replication.
        
        Args:
            filtered_entities: Dictionary of filtered entity lists
            
        Returns:
            dict: Entity replication summary or empty dict if no entities
        """
        if not filtered_entities:
            self.logger.debug("No filtered entities - skipping entity metrics")
            return {}
        
        # Import metric functions
        from delivery_sim.metrics.entity.order_metrics import calculate_all_order_metrics
        from delivery_sim.metrics.entity.delivery_unit_metrics import calculate_all_delivery_unit_metrics
        
        entity_summary = {}
        
        # Process orders if available
        if 'order' in filtered_entities and filtered_entities['order']:
            order_metrics = {
                'assignment_time': lambda order: calculate_all_order_metrics(order)['waiting_time'],
                'fulfillment_time': lambda order: calculate_all_order_metrics(order)['fulfillment_time']
            }
            entity_summary['orders'] = aggregate_entity_metrics(
                filtered_entities['order'], 
                order_metrics
            )
            
            self.logger.debug(f"Processed {len(filtered_entities['order'])} orders")
        
        # Process delivery units if available
        if 'delivery_unit' in filtered_entities and filtered_entities['delivery_unit']:
            delivery_unit_metrics = {
                'total_distance': lambda unit: calculate_all_delivery_unit_metrics(unit)['total_distance']
            }
            entity_summary['delivery_units'] = aggregate_entity_metrics(
                filtered_entities['delivery_unit'],
                delivery_unit_metrics
            )
            
            self.logger.debug(f"Processed {len(filtered_entities['delivery_unit'])} delivery units")
        
        return entity_summary
    
    def _process_system_metrics(self, repositories, filtered_entities):
        """
        Process system metrics for a single replication.
        
        Args:
            repositories: Dictionary of entity repositories
            filtered_entities: Dictionary of filtered entity lists
            
        Returns:
            dict: System metrics for this replication
        """
        # Calculate system metrics directly
        system_metrics = calculate_all_entity_derived_system_metrics(
            repositories, 
            filtered_entities, 
            self.warmup_period
        )
        
        self.logger.debug(f"Calculated system metrics: {list(system_metrics.keys())}")
        return system_metrics


def analyze_single_configuration(simulation_results, warmup_period, confidence_level=0.95):
    """
    Convenience function for analyzing a single configuration experiment.
    
    Args:
        simulation_results: Results from simulation_runner.run_experiment()
        warmup_period: Duration to exclude from analysis
        confidence_level: Confidence level for intervals (default: 0.95)
        
    Returns:
        dict: Complete experiment summary with confidence intervals
    """
    pipeline = ExperimentAnalysisPipeline(warmup_period, confidence_level)
    return pipeline.analyze_experiment(simulation_results)


def quick_summary(experiment_summary, metrics_of_interest=None):
    """
    Generate a quick summary of key metrics for thesis writing.
    
    Args:
        experiment_summary: Results from analyze_single_configuration()
        metrics_of_interest: List of metric names to focus on (default: all)
        
    Returns:
        dict: Simplified summary with just point estimates and CI widths
    """
    if metrics_of_interest is None:
        metrics_of_interest = ['completion_rate', 'assignment_time', 'total_distance']
    
    quick_results = {}
    
    # System metrics
    system_metrics = experiment_summary.get('system_metrics', {})
    for metric_name in metrics_of_interest:
        if metric_name in system_metrics:
            result = system_metrics[metric_name]
            if result['point_estimate'] is not None:
                ci_lower, ci_upper = result['confidence_interval']
                ci_width = ci_upper - ci_lower
                quick_results[metric_name] = {
                    'value': result['point_estimate'],
                    'ci_width': ci_width,
                    'formatted': f"{result['point_estimate']:.3f} ± {ci_width/2:.3f}"
                }
    
    # Entity metrics (focusing on means)
    entity_metrics = experiment_summary.get('entity_metrics', {})
    for entity_type, metrics in entity_metrics.items():
        for metric_name, stats in metrics.items():
            if metric_name in metrics_of_interest and 'mean' in stats:
                result = stats['mean']
                if result['point_estimate'] is not None:
                    ci_lower, ci_upper = result['confidence_interval']
                    ci_width = ci_upper - ci_lower
                    quick_results[f"{entity_type}_{metric_name}_mean"] = {
                        'value': result['point_estimate'],
                        'ci_width': ci_width,
                        'formatted': f"{result['point_estimate']:.3f} ± {ci_width/2:.3f}"
                    }
    
    return quick_results