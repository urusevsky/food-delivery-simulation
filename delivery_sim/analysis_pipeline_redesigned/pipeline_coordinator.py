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
from delivery_sim.analysis_pipeline_redesigned.metric_configurations import get_metric_configuration

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
        Transform raw simulation data into comprehensive experiment-level statistical summary.
        
        This is the main pipeline orchestrator that processes all replications through
        configuration-driven, pattern-aware aggregation to produce experiment-level results.
        
        Processing pipeline:
        1. Replication-level metric computation (pattern-dependent processing)
        - Two-level patterns: Raw data → Individual metrics → Statistical summaries  
        - One-level patterns: Raw data → Direct metric calculations
        
        2. Experiment-level statistical aggregation (across all replications)
        - Two-level patterns: Statistics-of-statistics (second-order aggregation)
        - One-level patterns: Standard statistics from scalar values (first-order aggregation)
        
        3. Confidence interval construction (configuration-driven, selective application)
        
        4. Metadata enrichment and result finalization
        
        Args:
            replication_results: List of raw simulation outputs from simulation_runner.run_experiment()
                            Each element contains repositories, events, and simulation state
                    
        Returns:
            dict: Complete experiment summary containing:
                - Descriptive statistics for all enabled metric types
                - Confidence intervals for configured metrics/statistics  
                - Pipeline metadata (num_replications, warmup_period, etc.)
        """
        num_replications = len(replication_results)
        
        self.logger.info(f"Starting analysis for {num_replications} replications")
        self.logger.info(f"Processing metric types: {self.enabled_metric_types}")
        
        # 🎯 NEW: Create metric configs once at entry point
        metric_configs = {
            metric_type: get_metric_configuration(metric_type) 
            for metric_type in self.enabled_metric_types
        }
        
        # Step 1: Transform raw simulation data into replication-level metrics
        # Output varies by pattern: statistics objects (two-level) or direct values (one-level)
        replication_level_metrics = self._compute_replication_level_metrics(
            replication_results, metric_configs  # 🎯 NEW: Pass metric_configs
        )
        
        # Step 2: Aggregate replication-level metrics into experiment-level statistics
        # Two-level: statistics-of-statistics, One-level: standard statistical aggregation
        experiment_statistics = self._compute_experiment_level_statistics(
            replication_level_metrics, metric_configs  # 🎯 NEW: Pass metric_configs
        )
        
        # Step 3: Construct confidence intervals using configuration-driven selection
        experiment_with_cis = self._construct_confidence_intervals(
            experiment_statistics, replication_level_metrics, metric_configs  # 🎯 NEW: Pass metric_configs
        )
        
        # Step 4: Enrich with metadata and finalize experiment summary
        return self._finalize_experiment_summary(experiment_with_cis, replication_results)
    
    def _compute_replication_level_metrics(self, replication_results, metric_configs):  # 🎯 NEW: Add metric_configs parameter
        """
        Transform raw simulation data into replication-level metrics for all enabled metric types.
        
        This method applies warmup period filtering and computes metrics according to each
        metric type's aggregation pattern:
        
        Two-level patterns (e.g., entity-based metrics):
        - Raw data → Individual entity metrics → Statistical summaries (mean, std, etc.)
        - Output: Statistics objects representing entity distributions within each replication
        
        One-level patterns (e.g., system-wide metrics):  
        - Raw data → Direct metric calculation
        - Output: Scalar values or simple data structures
        
        Args:
            replication_results: List of raw simulation outputs (repositories, events, etc.)
            metric_configs: Dict of {metric_type: config} - created once at entry point  # 🎯 NEW: Document new parameter
            
        Returns:
            dict: {metric_type: [replication_metrics, ...]}
                Where replication_metrics varies by pattern:
                - Two-level: Statistics objects (with mean, std, count, etc.)
                - One-level: Scalar values or simple structures
        """
        all_processed_replications = {metric_type: [] for metric_type in metric_configs.keys()}  # 🎯 CHANGED: Use metric_configs.keys()
        
        for i, replication_result in enumerate(replication_results):
            self.logger.debug(f"Processing replication {i+1}/{len(replication_results)}")
            
            # Prepare analysis data using existing data preparation
            repositories = replication_result['repositories']
            analysis_data = prepare_analysis_data(repositories, self.warmup_period)
            
            # Process each configured metric type
            for metric_type, config in metric_configs.items():  # 🎯 CHANGED: Iterate over metric_configs.items()
                processed_replication = self.aggregation_processor.process_replication_level(
                    analysis_data, config  # 🎯 CHANGED: Pass config instead of metric_type
                )
                
                if processed_replication:
                    all_processed_replications[metric_type].append(processed_replication)
                    self.logger.debug(f"Processed {metric_type} for replication {i+1}")
                else:
                    self.logger.warning(f"Empty results from {metric_type} for replication {i+1} - investigate upstream")
        
        # Log processing summary
        for metric_type, processed_replications in all_processed_replications.items():
            self.logger.info(f"Collected {len(processed_replications)} processed replications for {metric_type}")
        
        return all_processed_replications
    
    def _compute_experiment_level_statistics(self, all_processed_replications, metric_configs):  # 🎯 NEW: Add metric_configs parameter
        """
        Aggregate replication-level metrics into experiment-level statistical summaries.
        
        This method computes descriptive statistics across replications, with different
        approaches based on aggregation pattern:
        
        Two-level patterns:
        - Input: Statistics objects from each replication
        - Process: Statistics-of-statistics (second-order aggregation)
        - Output: Mean-of-means, std-of-means, mean-of-stds, etc.
        
        One-level patterns:
        - Input: Scalar values from each replication  
        - Process: Standard statistical aggregation (first-order)
        - Output: Mean, std, min, max across replications
        
        Args:
            all_processed_replications: Output from _compute_replication_level_metrics()
            metric_configs: Dict of {metric_type: config} - passed from entry point  # 🎯 NEW: Document new parameter
            
        Returns:
            dict: {metric_type: experiment_statistics}
                Where experiment_statistics contains descriptive statistics
                summarizing behavior across all replications (no confidence intervals yet)
        """
        experiment_statistics = {}
        
        for metric_type, config in metric_configs.items():  # 🎯 CHANGED: Iterate over metric_configs.items()
            processed_replications = all_processed_replications.get(metric_type, [])  # 🎯 CHANGED: Use .get() for safety
            
            if not processed_replications:
                self.logger.warning(f"No processed replications for {metric_type} - check replication processing")
                continue
            
            experiment_result = self.aggregation_processor.process_experiment_level(
                processed_replications, config  # 🎯 CHANGED: Pass config instead of metric_type
            )
            
            if experiment_result:
                experiment_statistics[metric_type] = experiment_result
                self.logger.debug(f"Computed experiment statistics for {metric_type}")
            else:
                self.logger.warning(f"No experiment statistics generated for {metric_type} - investigate aggregation")
        
        return experiment_statistics
    
    def _construct_confidence_intervals(self, experiment_statistics, all_processed_replications, metric_configs):  # 🎯 NEW: Add metric_configs parameter
        """
        Construct confidence intervals based on granular configuration.
        Uses configuration to determine which specific statistics/metrics get CIs
        and which statistical method to use for each one.
        
        Args:
            experiment_statistics: Experiment-level statistics from previous step
            all_processed_replications: Original processed replication data needed for CI construction
            metric_configs: Dict of {metric_type: config} - passed from entry point  # 🎯 NEW: Document new parameter
            
        Returns:
            dict: Experiment results with CIs added for configured statistics/metrics
        """
        
        return construct_confidence_intervals_for_experiment(
            experiment_statistics,
            all_processed_replications,
            metric_configs,  # 🎯 CHANGED: Pass metric_configs instead of self.enabled_metric_types
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