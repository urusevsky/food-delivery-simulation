# analysis_pipeline/pipeline_coordinator.py
"""
Phase-based pipeline coordinator for experiment analysis.

Single responsibility: Orchestrate the analysis pipeline through
clearly defined phases, delegating work to specialized processors.
"""

from delivery_sim.utils.logging_system import get_logger
from delivery_sim.analysis_pipeline.data_preparation import prepare_analysis_data
from delivery_sim.analysis_pipeline.replication_processor import ReplicationProcessor
from delivery_sim.analysis_pipeline.experiment_aggregator import ExperimentAggregator
from delivery_sim.analysis_pipeline.confidence_interval_constructor import construct_confidence_intervals
from delivery_sim.analysis_pipeline.metric_configurations import get_metric_configuration


class ExperimentAnalysisPipeline:
    """
    Phase-based orchestrator for end-to-end experiment analysis.
    
    Coordinates four distinct phases:
    Phase 0: Data preparation (warmup filtering)
    Phase 1: Replication-level processing
    Phase 2: Experiment-level aggregation
    Phase 3: Confidence interval construction
    """
    
    def __init__(self, warmup_period, enabled_metric_types, confidence_level=0.95):
        """
        Initialize the analysis pipeline.
        
        Args:
            warmup_period: Duration to exclude from analysis (simulation time units)
            enabled_metric_types: List of metric types to process
            confidence_level: Confidence level for intervals (default: 0.95)
        """
        self.warmup_period = warmup_period
        self.enabled_metric_types = enabled_metric_types
        self.confidence_level = confidence_level
        
        # Initialize phase processors
        self.replication_processor = ReplicationProcessor()
        self.experiment_aggregator = ExperimentAggregator()
        
        self.logger = get_logger("analysis_pipeline.coordinator")
        self.logger.info(f"Pipeline initialized: warmup_period={warmup_period}, "
                        f"confidence_level={confidence_level}, "
                        f"enabled_metrics={enabled_metric_types}")
    
    def analyze_experiment(self, replication_results):
        """
        Transform raw simulation data into comprehensive experiment-level summary.
        
        Clear phase-by-phase processing:
        Phase 0: Data preparation (warmup filtering)
        Phase 1: Replication-level processing
        Phase 2: Experiment-level aggregation
        Phase 3: Confidence interval construction
        Phase 4: Metadata enrichment and finalization
        """
        num_replications = len(replication_results)
        self.logger.info(f"Starting analysis for {num_replications} replications")
        
        # Prepare metric configurations once
        metric_configs = {
            metric_type: get_metric_configuration(metric_type) 
            for metric_type in self.enabled_metric_types
        }
        
        # Phase 0: Prepare analysis data (warmup filtering)
        prepared_analysis_data = self._phase0_prepare_analysis_data(replication_results)
        
        # Phase 1: Process each replication
        replication_level_metrics = self._phase1_process_replications(
            prepared_analysis_data, metric_configs
        )
        
        # Phase 2: Aggregate across replications
        experiment_statistics = self._phase2_aggregate_experiment(
            replication_level_metrics, metric_configs
        )
        
        # Phase 3: Construct confidence intervals
        experiment_with_cis = self._phase3_construct_confidence_intervals(
            experiment_statistics, replication_level_metrics, metric_configs
        )
        
        # Phase 4: Finalize with metadata
        return self._phase4_finalize_results(experiment_with_cis, replication_results)
    
    def _phase0_prepare_analysis_data(self, replication_results):
        """
        Phase 0: Prepare analysis data by filtering warmup period.
        
        Transforms raw simulation outputs into analysis-ready data by:
        - Extracting repositories from each replication
        - Applying warmup period filtering
        - Creating analytical populations
        
        Args:
            replication_results: List of raw simulation outputs
            
        Returns:
            list: Analysis data objects (one per replication) ready for metric processing
        """
        self.logger.info("Phase 0: Preparing analysis data (warmup filtering)")
        
        prepared_data = []
        
        for i, replication_result in enumerate(replication_results):
            repositories = replication_result['repositories']
            analysis_data = prepare_analysis_data(repositories, self.warmup_period)
            prepared_data.append(analysis_data)
            
            self.logger.debug(
                f"Prepared analysis data for replication {i+1}/{len(replication_results)}"
            )
        
        self.logger.info(f"Phase 0 complete: Prepared {len(prepared_data)} replications")
        return prepared_data
        
    def _phase1_process_replications(self, prepared_analysis_data, metric_configs):
        """
        Phase 1: Transform analysis data into replication-level metrics.
        
        For each prepared analysis data:
        - Process according to metric pattern
        - Two-level: Calculate individual metrics → Aggregate to statistics
        - One-level: Calculate directly (no aggregation at replication level)
        
        Args:
            prepared_analysis_data: List of analysis data objects from Phase 0
            metric_configs: Dict of metric configurations
            
        Returns:
            dict: {metric_type: [replication_metrics, ...]}
        """
        self.logger.info("Phase 1: Processing replication-level metrics")
        
        all_processed = {metric_type: [] for metric_type in metric_configs.keys()}
        
        for i, analysis_data in enumerate(prepared_analysis_data):
            self.logger.debug(
                f"Processing replication {i+1}/{len(prepared_analysis_data)}"
            )
            
            # Process each metric type
            for metric_type, config in metric_configs.items():
                processed = self.replication_processor.process_replication(
                    analysis_data, config
                )
                
                if processed:
                    all_processed[metric_type].append(processed)
                    self.logger.debug(f"Processed {metric_type} for replication {i+1}")
                else:
                    self.logger.warning(
                        f"Empty results for {metric_type} in replication {i+1}"
                    )
        
        # Log phase summary
        for metric_type, processed_reps in all_processed.items():
            self.logger.info(
                f"Phase 1 complete: {len(processed_reps)} replications for {metric_type}"
            )
        
        return all_processed
    
    def _phase2_aggregate_experiment(self, replication_level_metrics, metric_configs):
        """
        Phase 2: Aggregate replication-level metrics into experiment-level statistics.
        """
        self.logger.info("Phase 2: Aggregating across replications")
        
        experiment_statistics = {}
        
        for metric_type, config in metric_configs.items():
            replication_results = replication_level_metrics.get(metric_type, [])
            
            if not replication_results:
                self.logger.warning(f"No replication results for {metric_type}")
                continue
            
            experiment_stats = self.experiment_aggregator.aggregate_experiment(
                replication_results, config
            )
            
            if experiment_stats:
                experiment_statistics[metric_type] = experiment_stats
                self.logger.debug(f"Aggregated {metric_type}")
            else:
                self.logger.warning(f"No experiment statistics for {metric_type}")
        
        self.logger.info(f"Phase 2 complete: {len(experiment_statistics)} metric types")
        return experiment_statistics
    
    def _phase3_construct_confidence_intervals(self, experiment_statistics, 
                                              replication_level_metrics, metric_configs):
        """
        Phase 3: Add statistical inference (confidence intervals).
        """
        self.logger.info("Phase 3: Constructing confidence intervals")
        
        experiment_with_cis = construct_confidence_intervals(
            experiment_statistics,
            replication_level_metrics,
            metric_configs,
            self.confidence_level
        )
        
        self.logger.info("Phase 3 complete: Confidence intervals added")
        return experiment_with_cis
    
    def _phase4_finalize_results(self, experiment_with_cis, replication_results):
        """
        Phase 4: Add metadata and finalize experiment summary.
        """
        self.logger.info("Phase 4: Finalizing experiment summary")
        
        experiment_summary = {
            'num_replications': len(replication_results),
            'warmup_period': self.warmup_period,
            'confidence_level': self.confidence_level,
            'processed_metric_types': self.enabled_metric_types,
            'results': experiment_with_cis
        }
        
        self.logger.info("Analysis pipeline completed successfully")
        return experiment_summary