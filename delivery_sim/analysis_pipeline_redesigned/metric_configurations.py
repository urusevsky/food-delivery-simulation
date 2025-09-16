# analysis_pipeline_redesigned/metric_configurations.py
"""
Metric configuration system for pattern-based analysis pipeline.

This module defines how different metric types should be processed through
the analysis pipeline. It acts as the central registry that declares:
- Which aggregation pattern each metric type uses
- Which existing metric functions to call
- Which experiment-level statistics to compute

Design philosophy:
- Point directly to existing metric functions (no wrapper layers)
- Simple declarative configuration 
- Clear separation between two-level and one-level patterns
"""

from delivery_sim.utils.logging_system import get_logger

logger = get_logger("analysis_pipeline_redesigned.configurations")

# ==============================================================================
# CORE METRIC CONFIGURATIONS
# ==============================================================================

METRIC_CONFIGURATIONS = {
    
    # Two-level aggregation pattern
    'order_metrics': {
        'aggregation_pattern': 'two_level',
        'metric_module': 'delivery_sim.metrics.entity.order_metrics',
        'metric_function': 'calculate_all_order_metrics',
        'entity_data_key': 'cohort_completed_orders',
        'experiment_stats': [
            {
                'name': 'mean_of_means', 
                'extract': 'mean', 
                'compute': 'mean',
                'construct_ci': True,
                'description': 'Average of assignment time means across replications'
            },
            {
                'name': 'std_of_means', 
                'extract': 'mean', 
                'compute': 'std',
                'construct_ci': False,  # Descriptive only - insufficient replications
                'description': 'Variability of means across replications'
            },
            {
                'name': 'mean_of_variances', 
                'extract': 'variance', 
                'compute': 'mean',
                'construct_ci': True,
                'description': 'Average within-replication variance'
            }
        ],
        'description': 'Individual order performance metrics'
    },
    
    'delivery_unit_metrics': {
        'aggregation_pattern': 'two_level', 
        'metric_module': 'delivery_sim.metrics.entity.delivery_unit_metrics',
        'metric_function': 'calculate_all_delivery_unit_metrics',
        'entity_data_key': 'cohort_completed_delivery_units',
        'experiment_stats': [
            {
                'name': 'mean_of_means', 
                'extract': 'mean', 
                'compute': 'mean',
                'construct_ci': True
            },
            {
                'name': 'std_of_means', 
                'extract': 'mean', 
                'compute': 'std',
                'construct_ci': False  # Descriptive only
            }
        ],
        'description': 'Delivery unit performance metrics'
    },
    
    # One-level aggregation pattern
    'system_metrics': {
        'aggregation_pattern': 'one_level',
        'metric_module': 'delivery_sim.metrics.system.entity_derived_metrics', 
        'metric_function': 'calculate_all_entity_derived_system_metrics',
        'entity_data_key': None,  # Function takes full AnalysisData object
        'ci_config': [
            {
                'metric_name': 'completion_rate',
                'construct_ci': True,
                'description': 'System completion rate with CI'
            },
            {
                'metric_name': 'pairing_rate', 
                'construct_ci': True,
                'description': 'System pairing rate with CI'
            }
        ],
        'description': 'System-wide performance metrics'
    }
}
# ==============================================================================
# CONFIGURATION ACCESS FUNCTIONS
# ==============================================================================

def get_metric_configuration(metric_type):
    """
    Get configuration for a specific metric type.
    
    Args:
        metric_type: Metric type key (e.g., 'order_metrics', 'system_metrics')
        
    Returns:
        dict: Configuration dictionary for the metric type
        
    Raises:
        KeyError: If metric type is not configured
    """
    if metric_type not in METRIC_CONFIGURATIONS:
        available_types = list(METRIC_CONFIGURATIONS.keys())
        logger.error(f"Unknown metric type: {metric_type}. Available types: {available_types}")
        raise KeyError(f"Metric type '{metric_type}' not found in configurations")
    
    return METRIC_CONFIGURATIONS[metric_type]


def get_aggregation_pattern(metric_type):
    """
    Get the aggregation pattern for a metric type.
    
    Returns:
        str: Either 'two_level' or 'one_level'
    """
    config = get_metric_configuration(metric_type)
    return config['aggregation_pattern']


def get_metric_function_info(metric_type):
    """
    Get the module and function information for a metric type.
    
    Returns:
        tuple: (module_path, function_name)
    """
    config = get_metric_configuration(metric_type)
    return config['metric_module'], config['metric_function']


def get_entity_data_key(metric_type):
    """
    Get the AnalysisData attribute key for entity data.
    
    Returns:
        str or None: Attribute name for entity data, or None for full AnalysisData
    """
    config = get_metric_configuration(metric_type)
    return config['entity_data_key']


def get_experiment_statistics(metric_type):
    """
    Get the list of experiment-level statistics to compute for a metric type.
    
    Only applicable for two-level aggregation patterns.
    
    Returns:
        list: Statistics to compute (e.g., ['mean_of_means', 'std_of_means'])
        
    Raises:
        ValueError: If called on one-level pattern (no experiment stats defined)
    """
    config = get_metric_configuration(metric_type)
    
    if config['aggregation_pattern'] == 'one_level':
        raise ValueError(f"Experiment statistics not applicable for one-level pattern: {metric_type}")
    
    return config.get('experiment_stats', ['mean_of_means'])  # Default to mean_of_means


def list_available_metric_types():
    """
    Get list of all configured metric types.
    
    Returns:
        list: Available metric type keys
    """
    return list(METRIC_CONFIGURATIONS.keys())


def list_metric_types_by_pattern(pattern):
    """
    Get list of metric types that use a specific aggregation pattern.
    
    Args:
        pattern: Aggregation pattern ('two_level' or 'one_level')
        
    Returns:
        list: Metric types using the specified pattern
    """
    return [
        metric_type for metric_type, config in METRIC_CONFIGURATIONS.items()
        if config['aggregation_pattern'] == pattern
    ]


# ==============================================================================
# CONFIGURATION VALIDATION
# ==============================================================================

def validate_configurations():
    """
    Validate that all metric configurations are properly structured.
    
    This function can be called during pipeline initialization to catch
    configuration errors early.
    
    Returns:
        bool: True if all configurations are valid
        
    Raises:
        ValueError: If any configuration is invalid
    """
    required_fields = {
        'two_level': ['aggregation_pattern', 'metric_module', 'metric_function', 
                      'entity_data_key', 'experiment_stats'],
        'one_level': ['aggregation_pattern', 'metric_module', 'metric_function']
    }
    
    for metric_type, config in METRIC_CONFIGURATIONS.items():
        pattern = config.get('aggregation_pattern')
        
        if pattern not in ['two_level', 'one_level']:
            raise ValueError(f"Invalid aggregation pattern for {metric_type}: {pattern}")
        
        # Check required fields
        required = required_fields[pattern]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in {metric_type} configuration")
        
        # Pattern-specific validation
        if pattern == 'two_level':
            if config['entity_data_key'] is None:
                raise ValueError(f"Two-level pattern {metric_type} must specify entity_data_key")
            if not config.get('experiment_stats'):
                raise ValueError(f"Two-level pattern {metric_type} must specify experiment_stats")
        
        elif pattern == 'one_level':
            if 'experiment_stats' in config:
                logger.warning(f"One-level pattern {metric_type} should not specify experiment_stats")
    
    logger.info(f"Configuration validation passed for {len(METRIC_CONFIGURATIONS)} metric types")
    return True


# ==============================================================================
# CONFIGURATION SUMMARY
# ==============================================================================

def print_configuration_summary():
    """
    Print a summary of all configured metric types for debugging/documentation.
    """
    print("Metric Configuration Summary")
    print("=" * 50)
    
    two_level = list_metric_types_by_pattern('two_level')
    one_level = list_metric_types_by_pattern('one_level')
    
    print(f"\nTwo-Level Pattern ({len(two_level)} types):")
    for metric_type in two_level:
        config = get_metric_configuration(metric_type)
        print(f"  {metric_type}: {config['metric_function']} -> {config['experiment_stats']}")
    
    print(f"\nOne-Level Pattern ({len(one_level)} types):")
    for metric_type in one_level:
        config = get_metric_configuration(metric_type)
        print(f"  {metric_type}: {config['metric_function']}")
    
    print(f"\nTotal: {len(METRIC_CONFIGURATIONS)} configured metric types")


# Run validation on import to catch configuration errors early
if __name__ != "__main__":
    try:
        validate_configurations()
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise