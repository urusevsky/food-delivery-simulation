# delivery_sim/analysis_pipeline/experiment_level_aggregation/experiment_summaries.py
"""
Experiment-level aggregation functions.

This module aggregates replication-level data into experiment-level summaries
with confidence intervals for both entity metrics and system metrics.
"""

from delivery_sim.analysis_pipeline.experiment_level_aggregation.confidence_intervals import (
    calculate_summary_with_ci, 
    calculate_proportion_confidence_interval
)


def aggregate_entity_metrics_across_replications(replication_summaries, confidence_level=0.95):
    """
    Aggregate entity metrics across multiple replications.
    
    Takes replication summaries (each containing mean, std, p95, etc.) and
    creates experiment-level summaries with confidence intervals.
    
    Args:
        replication_summaries: List of dictionaries, each containing entity summaries for one replication
        confidence_level: Confidence level for intervals (default: 0.95)
        
    Returns:
        dict: Experiment-level entity metrics with confidence intervals
    """
    if not replication_summaries:
        return {}
    
    # Identify all entity types and metrics from first replication
    sample_summary = replication_summaries[0]
    
    experiment_results = {}
    
    # Process each entity type (orders, pairs, delivery_units, etc.)
    for entity_type, entity_data in sample_summary.items():
        experiment_results[entity_type] = {}
        
        # Process each metric for this entity type (mean, std, p95, etc.)
        for metric_name, metric_summary in entity_data.items():
            if isinstance(metric_summary, dict) and 'mean' in metric_summary:
                # This is a summary dict - aggregate the summary statistics
                experiment_results[entity_type][metric_name] = _aggregate_summary_statistic(
                    replication_summaries, entity_type, metric_name, confidence_level
                )
    
    return experiment_results


def aggregate_system_metrics_across_replications(replication_data, confidence_level=0.95):
    """
    Aggregate system metrics across multiple replications.
    
    Takes system metric values (single values per replication) and creates
    experiment-level summaries with confidence intervals.
    
    Args:
        replication_data: List of dictionaries, each containing system metrics for one replication
        confidence_level: Confidence level for intervals (default: 0.95)
        
    Returns:
        dict: Experiment-level system metrics with confidence intervals
    """
    if not replication_data:
        return {}
    
    # Identify all system metrics from first replication
    sample_data = replication_data[0]
    
    experiment_results = {}
    
    # Process each system metric
    for metric_name in sample_data.keys():
        # Extract values across all replications
        metric_values = [rep_data.get(metric_name) for rep_data in replication_data]
        
        # Standard numeric metric (including completion rates)
        experiment_results[metric_name] = calculate_summary_with_ci(
            metric_values, confidence_level
        )
    
    return experiment_results


def create_complete_experiment_summary(entity_summaries, system_summaries, confidence_level=0.95):
    """
    Create complete experiment summary combining entity and system metrics.
    
    Args:
        entity_summaries: List of replication-level entity summaries
        system_summaries: List of replication-level system metric dictionaries
        confidence_level: Confidence level for intervals (default: 0.95)
        
    Returns:
        dict: Complete experiment summary with both entity and system metrics
    """
    experiment_summary = {
        'num_replications': len(entity_summaries) if entity_summaries else len(system_summaries),
        'confidence_level': confidence_level,
        'entity_metrics': {},
        'system_metrics': {}
    }
    
    # Aggregate entity metrics if provided
    if entity_summaries:
        experiment_summary['entity_metrics'] = aggregate_entity_metrics_across_replications(
            entity_summaries, confidence_level
        )
    
    # Aggregate system metrics if provided
    if system_summaries:
        experiment_summary['system_metrics'] = aggregate_system_metrics_across_replications(
            system_summaries, confidence_level
        )
    
    return experiment_summary


def _aggregate_summary_statistic(replication_summaries, entity_type, metric_name, confidence_level):
    """
    Helper function to aggregate a specific summary statistic across replications.
    
    Args:
        replication_summaries: List of replication summaries
        entity_type: Entity type (e.g., 'orders', 'delivery_units')
        metric_name: Metric name (e.g., 'assignment_time', 'total_distance')
        confidence_level: Confidence level for CI calculation
        
    Returns:
        dict: Aggregated results for specific statistic types (mean, p95, etc.)
    """
    # Extract the summary statistics we want to aggregate
    stats_to_aggregate = ['mean', 'std', 'p95', 'p50']  # Focus on key statistics
    
    aggregated_stats = {}
    
    for stat_type in stats_to_aggregate:
        # Extract this statistic across all replications
        stat_values = []
        for rep_summary in replication_summaries:
            entity_data = rep_summary.get(entity_type, {})
            metric_data = entity_data.get(metric_name, {})
            stat_value = metric_data.get(stat_type)
            stat_values.append(stat_value)
        
        # Calculate confidence interval for this statistic
        aggregated_stats[f'{stat_type}'] = calculate_summary_with_ci(
            stat_values, confidence_level
        )
    
    return aggregated_stats


def format_experiment_summary(experiment_summary, decimal_places=3):
    """
    Format experiment summary for readable output.
    
    Args:
        experiment_summary: Dictionary from create_complete_experiment_summary
        decimal_places: Number of decimal places for formatting
        
    Returns:
        str: Formatted summary string
    """
    from delivery_sim.analysis_pipeline.experiment_level_aggregation.confidence_intervals import format_confidence_interval
    
    lines = []
    lines.append(f"Experiment Summary ({experiment_summary['num_replications']} replications)")
    lines.append("=" * 60)
    
    # Entity metrics
    if experiment_summary['entity_metrics']:
        lines.append("\nEntity Metrics:")
        for entity_type, metrics in experiment_summary['entity_metrics'].items():
            lines.append(f"  {entity_type.title()}:")
            for metric_name, stats in metrics.items():
                if 'mean' in stats:
                    mean_result = stats['mean']
                    lines.append(f"    {metric_name} (mean): {format_confidence_interval(mean_result, decimal_places)}")
    
    # System metrics  
    if experiment_summary['system_metrics']:
        lines.append("\nSystem Metrics:")
        for metric_name, result in experiment_summary['system_metrics'].items():
            formatted_name = metric_name.replace('_', ' ').title()
            lines.append(f"  {formatted_name}: {format_confidence_interval(result, decimal_places)}")
    
    return "\n".join(lines)