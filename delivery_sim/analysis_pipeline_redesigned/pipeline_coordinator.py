# analysis_pipeline_redesigned/pipeline_coordinator.py
"""
Configuration-driven pipeline coordinator for experiment analysis.

This module orchestrates the complete analysis pipeline using the pattern-based
architecture. It replaces hard-coded metric-specific processing with generic,
configuration-driven processing that can handle any metric type.

Clean, focused design:
- No convenience functions or unnecessary complexity
- Metric types determined upfront, not modified ad-hoc  
- Fails fast on errors for research debugging
- Separates descriptive statistics from confidence interval construction
"""

from delivery_sim.utils.logging_system import get_logger
from delivery_sim.analysis_pipeline_redesigned.data_preparation import prepare_analysis_data
from delivery_sim.analysis_pipeline_redesigned.aggregation_processor import AggregationProcessor
from delivery_sim.analysis_pipeline_redesigned.confidence_intervals import construct_confidence_intervals_for_experiment


class ExperimentAnalysisPipeline:
    """
    Configuration-driven orchestrator for end-to-end experiment analysis.
    
    Takes raw simulation results and produces experiment-level summaries
    using pattern-based processing instead of hard-coded metric types.
    """
    
    def __init__(self, warmup_period, enabled_metric_types, confidence_level=0.95):
        """
        Initialize the analysis pipeline.
        
        Args:
            warmup_period: Duration to exclude from analysis (simulation time units)
            enabled_metric_types: List of metric types to process (determined upfront)
            confidence_level: Confidence level for intervals (default: 0.95)
        """
        self.warmup_period = warmup_period
        self.enabled_metric_types = enabled_metric_types
        self.confidence_level = confidence_level
        
        self.logger = get_logger("analysis_pipeline_redesigned.coordinator")
        self.aggregation_processor = AggregationProcessor()
        
        self.logger.info(f"Pipeline initialized: warmup_period={warmup_period}, "
                        f"confidence_level={confidence_level}, "
                        f"enabled_metrics={enabled_metric_types}")
    
    def analyze_experiment(self, replication_results):
        """
        Process replication results and create complete statistical summary.
        
        REFACTORED: Parameter directly receives replication_results list.
        
        This is the main entry point that processes all replications and produces
        experiment-level results using configuration-driven processing.
        
        Processing flow:
        1. Replication-level processing (descriptive stats per replication)
        2. Experiment-level aggregation (descriptive stats across replications) 
        3. Confidence interval construction (granular control via configuration)
        4. Metadata addition and finalization
        
        Args:
            replication_results: Direct replication results list from simulation_runner.run_experiment()
                
        Returns:
            dict: Complete experiment summary with statistics (and CIs for configured metrics)
        """
        # ✅ REFACTORED: Direct parameter usage, no redundant assignment
        num_replications = len(replication_results)
        
        self.logger.info(f"Starting analysis for {num_replications} replications")
        self.logger.info(f"Processing metric types: {self.enabled_metric_types}")
        
        # Step 1: Process replication-level for all metric types
        all_processed_replications = self._compute_replication_level_metrics(replication_results)  # ✅ Updated
        
        # Step 2: Process experiment-level aggregation (descriptive statistics)
        experiment_statistics = self._compute_experiment_level_statistics(all_processed_replications)  # ✅ Updated method name and variable
        
        # Step 3: Construct confidence intervals (granular control via configuration)
        experiment_with_cis = self._construct_confidence_intervals(experiment_statistics, all_processed_replications)  # ✅ Updated
        
        # Step 4: Add metadata and return
        return self._finalize_experiment_summary(experiment_with_cis, replication_results)
    
    def _compute_replication_level_metrics(self, replication_results):
        """
        Process replication-level analysis for all metric types across all replications.
        
        Returns:
            dict: metric_type -> list of processed replications
        """
        all_processed_replications = {metric_type: [] for metric_type in self.enabled_metric_types}  # ✅ Updated
        
        for i, replication_result in enumerate(replication_results):
            self.logger.debug(f"Processing replication {i+1}/{len(replication_results)}")
            
            # Prepare analysis data using existing data preparation
            repositories = replication_result['repositories']
            analysis_data = prepare_analysis_data(repositories, self.warmup_period)
            
            # Process each configured metric type
            for metric_type in self.enabled_metric_types:
                processed_replication = self.aggregation_processor.process_replication_level(  # ✅ Updated
                    analysis_data, metric_type
                )
                
                if processed_replication:  # ✅ Updated
                    all_processed_replications[metric_type].append(processed_replication)  # ✅ Updated
                    self.logger.debug(f"Processed {metric_type} for replication {i+1}")
                else:
                    self.logger.warning(f"Empty results from {metric_type} for replication {i+1} - investigate upstream")
        
        # Log processing summary
        for metric_type, processed_replications in all_processed_replications.items():  # ✅ Updated
            self.logger.info(f"Collected {len(processed_replications)} processed replications for {metric_type}")  # ✅ Updated
        
        return all_processed_replications  # ✅ Updated
    
    def _compute_experiment_level_statistics(self, all_processed_replications):  # ✅ Renamed method
        """
        Aggregate experiment-level statistics for all metric types.
        
        This step produces descriptive statistics at experiment level.
        
        Returns:
            dict: Experiment-level statistics (without confidence intervals)
        """
        experiment_statistics = {}
        
        for metric_type, processed_replications in all_processed_replications.items():  # ✅ Updated
            if not processed_replications:  # ✅ Updated
                self.logger.warning(f"No processed replications for {metric_type} - check replication processing")
                continue
            
            experiment_result = self.aggregation_processor.process_experiment_level(
                processed_replications, metric_type  # ✅ Updated
            )
            
            if experiment_result:
                experiment_statistics[metric_type] = experiment_result
                self.logger.debug(f"Computed experiment statistics for {metric_type}")
            else:
                self.logger.warning(f"No experiment statistics generated for {metric_type} - investigate aggregation")
        
        return experiment_statistics
    
    def _construct_confidence_intervals(self, experiment_statistics, all_processed_replications):  # ✅ Updated parameter
        """
        Construct confidence intervals based on granular configuration.
        Uses configuration to determine which specific statistics/metrics get CIs
        and which statistical method to use for each one.
        
        Args:
            experiment_statistics: Experiment-level statistics from previous step
            all_processed_replications: Original processed replication data needed for CI construction  # ✅ Updated
            
        Returns:
            dict: Experiment results with CIs added for configured statistics/metrics
        """
        
        return construct_confidence_intervals_for_experiment(
            experiment_statistics,
            all_processed_replications,  # ✅ Updated
            self.enabled_metric_types,
            self.confidence_level
        )
    
    def _finalize_experiment_summary(self, experiment_with_cis, replication_results):
        """
        Add metadata and finalize experiment summary.
        
        REFACTORED: replication_results parameter directly receives the list.
        Metadata that was previously in wrapper dictionary is no longer available.
        
        Returns:
            dict: Complete experiment summary
        """
        experiment_summary = {
            'num_replications': len(replication_results),
            'warmup_period': self.warmup_period,
            'confidence_level': self.confidence_level,
            'processed_metric_types': self.enabled_metric_types,
            'results': experiment_with_cis
        }
        
        self.logger.info("Analysis pipeline completed successfully")
        return experiment_summary