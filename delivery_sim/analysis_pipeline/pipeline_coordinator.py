# delivery_sim/analysis_pipeline/pipeline_coordinator.py
"""
Pipeline coordinator for end-to-end experiment analysis.

This module orchestrates the complete analysis pipeline from raw simulation results
to experiment-level summaries with confidence intervals.

Updated to use the new centralized data preparation approach with AnalysisData.
"""

from delivery_sim.analysis_pipeline.data_preparation import prepare_analysis_data
from delivery_sim.analysis_pipeline.replication_level_aggregation.entity_replication_summaries import aggregate_entity_metrics
from delivery_sim.metrics.system.entity_derived_metrics import calculate_all_entity_derived_system_metrics
from delivery_sim.analysis_pipeline.experiment_level_aggregation.experiment_summaries import create_complete_experiment_summary
from delivery_sim.utils.logging_system import get_logger


class ExperimentAnalysisPipeline:
    """
    Coordinates the complete analysis pipeline for simulation experiments.
    
    Takes raw simulation results (multiple replications) and produces
    experiment-level summaries with confidence intervals.
    
    Updated to use centralized data preparation with AnalysisData objects.
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
                    'typical_distance': float,
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
        
        for i, replication_result in enumerate(replication_results):
            self.logger.debug(f"Processing replication {i+1}/{num_replications}")

            # Extract repositories from enhanced structure
            repositories = replication_result['repositories']
            
            # Step 1: Centralized data preparation (NEW APPROACH)
            analysis_data = prepare_analysis_data(repositories, self.warmup_period)
            
            self.logger.debug(f"Prepared analysis data with {len(analysis_data.cohort_orders)} cohort orders")
            
            # Step 2: Entity metrics (if any completed entities exist)
            entity_summary = self._process_entity_metrics(analysis_data)
            if entity_summary:
                entity_summaries.append(entity_summary)
            
            # Step 3: System metrics (using clean AnalysisData interface)
            system_summary = self._process_system_metrics(analysis_data)
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
            'typical_distance': simulation_results.get('typical_distance'),
            'config_summary': simulation_results.get('config_summary'),
            'num_replications': simulation_results.get('num_replications')
        })
        
        self.logger.info("Analysis pipeline completed successfully")
        return experiment_summary
    
    def _process_entity_metrics(self, analysis_data):
        """
        Process entity metrics for a single replication using AnalysisData.
        
        Args:
            analysis_data: AnalysisData object with pre-filtered populations
            
        Returns:
            dict: Entity replication summary or empty dict if no entities
        """
        # Check if we have any completed entities to process (for performance metrics)
        if not (analysis_data.cohort_completed_orders or analysis_data.cohort_completed_delivery_units):
            self.logger.debug("No completed entities in cohort - skipping entity metrics")
            return {}
        
        # Import metric functions
        from delivery_sim.metrics.entity.order_metrics import calculate_all_order_metrics
        from delivery_sim.metrics.entity.delivery_unit_metrics import calculate_all_delivery_unit_metrics
        
        entity_summary = {}
        
        # Process completed orders if available (for performance averages)
        if analysis_data.cohort_completed_orders:
            order_metrics = {
                'assignment_time': lambda order: calculate_all_order_metrics(order)['waiting_time'],
                'travel_time': lambda order: calculate_all_order_metrics(order)['travel_time'],
                'fulfillment_time': lambda order: calculate_all_order_metrics(order)['fulfillment_time']
            }
            
            entity_summary['orders'] = aggregate_entity_metrics(
                analysis_data.cohort_completed_orders, 
                order_metrics
            )
            
            self.logger.debug(f"Processed {len(analysis_data.cohort_completed_orders)} completed orders")
        
        # Process completed delivery units if available (for performance averages)
        if analysis_data.cohort_completed_delivery_units:
            delivery_unit_metrics = {
                'total_distance': lambda unit: calculate_all_delivery_unit_metrics(unit)['total_distance']
            }
            entity_summary['delivery_units'] = aggregate_entity_metrics(
                analysis_data.cohort_completed_delivery_units,
                delivery_unit_metrics
            )
            
            self.logger.debug(f"Processed {len(analysis_data.cohort_completed_delivery_units)} completed delivery units")
        
        return entity_summary
    
    def _process_system_metrics(self, analysis_data):
        """
        Process system metrics for a single replication using clean AnalysisData interface.
        
        Args:
            analysis_data: AnalysisData object with pre-filtered populations
            
        Returns:
            dict: System metrics for this replication
        """
        # Calculate system metrics using the new clean interface
        system_metrics = calculate_all_entity_derived_system_metrics(analysis_data)
        
        self.logger.debug(f"Calculated system metrics: {list(system_metrics.keys())}")
        return system_metrics


def analyze_single_configuration(simulation_results, warmup_period, confidence_level=0.95):
    """
    Convenience function for analyzing a single configuration experiment.
    
    Updated to use the new pipeline coordinator approach.
    
    Args:
        simulation_results: Results from simulation run containing repositories
        warmup_period: Warmup period for analysis
        confidence_level: Confidence level for statistical analysis
        
    Returns:
        dict: Complete analysis results for the configuration
    """
    pipeline = ExperimentAnalysisPipeline(warmup_period, confidence_level)
    return pipeline.analyze_experiment(simulation_results)