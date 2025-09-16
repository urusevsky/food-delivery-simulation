# analysis_pipeline_redesigned/metric_configurations.py
"""
Metric configuration system for pattern-based analysis pipeline.

This module defines how different metric types should be processed through
the analysis pipeline. Uses robust configuration format for experiment statistics
instead of brittle string parsing.
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
    """Get configuration for a specific metric type."""
    if metric_type not in METRIC_CONFIGURATIONS:
        available_types = list(METRIC_CONFIGURATIONS.keys())
        logger.error(f"Unknown metric type: {metric_type}. Available types: {available_types}")
        raise KeyError(f"Metric type '{metric_type}' not found in configurations")
    
    return METRIC_CONFIGURATIONS[metric_type]


def get_aggregation_pattern(metric_type):
    """Get the aggregation pattern for a metric type."""
    config = get_metric_configuration(metric_type)
    return config['aggregation_pattern']


def get_metric_function_info(metric_type):
    """Get the module and function information for a metric type."""
    config = get_metric_configuration(metric_type)
    return config['metric_module'], config['metric_function']


def get_entity_data_key(metric_type):
    """Get the AnalysisData attribute key for entity data."""
    config = get_metric_configuration(metric_type)
    return config['entity_data_key']


def get_experiment_statistics(metric_type):
    """
    Get the experiment statistics configuration for a metric type.
    
    Returns:
        list: List of experiment statistics configurations for aggregation processor
    """
    config = get_metric_configuration(metric_type)
    
    if config['aggregation_pattern'] == 'one_level':
        raise ValueError(f"Experiment statistics not applicable for one-level pattern: {metric_type}")
    
    return config.get('experiment_stats', [])


def get_ci_configuration(metric_type):
    """
    Get confidence interval configuration for a metric type.
    
    Returns:
        list: CI configuration for each statistic/metric that should have CIs constructed
    """
    config = get_metric_configuration(metric_type)
    pattern = config['aggregation_pattern']
    
    if pattern == 'two_level':
        # Extract CI config from experiment_stats
        ci_configs = []
        for stat_config in config.get('experiment_stats', []):
            if stat_config.get('construct_ci', False):
                ci_configs.append(stat_config)
        return ci_configs
    
    elif pattern == 'one_level':
        # Get CI config from ci_config field
        return config.get('ci_config', [])
    
    else:
        raise ValueError(f"Unknown aggregation pattern: {pattern}")


def get_statistics_requiring_ci(metric_type):
    """
    Get list of statistics that require CI construction for a metric type.
    
    Returns:
        list: Names of statistics that need CIs
    """
    ci_configs = get_ci_configuration(metric_type)
    
    pattern = get_aggregation_pattern(metric_type)
    
    if pattern == 'two_level':
        return [config['name'] for config in ci_configs]
    elif pattern == 'one_level':
        return [config['metric_name'] for config in ci_configs]
    else:
        return []


def get_ci_method(metric_type, statistic_name):
    """
    Automatically determine the CI construction method based on compute field.
    
    Args:
        metric_type: Metric type key
        statistic_name: Name of the statistic (or metric name for one-level)
        
    Returns:
        str: CI method ('t_distribution', 'chi_square') based on what's being estimated
    """
    ci_configs = get_ci_configuration(metric_type)
    pattern = get_aggregation_pattern(metric_type)
    
    if pattern == 'two_level':
        # Find by statistic name and determine method from compute field
        for config in ci_configs:
            if config['name'] == statistic_name:
                compute_type = config.get('compute')
                return _determine_ci_method_from_compute(compute_type)
    
    elif pattern == 'one_level':
        # For system metrics, we're always estimating the mean
        return 't_distribution'
    
    return 't_distribution'  # Default fallback


def _determine_ci_method_from_compute(compute_type):
    """
    Automatically determine CI method based on what statistic we're computing.
    
    Args:
        compute_type: The 'compute' field value ('mean', 'std', 'variance')
        
    Returns:
        str: Appropriate CI method
    """
    if compute_type == 'mean':
        return 't_distribution'  # Estimating mean -> use t-distribution
    elif compute_type in ['std', 'variance']:
        return 'chi_square'      # Estimating variance/std -> use chi-square
    else:
        logger.warning(f"Unknown compute type: {compute_type}, defaulting to t-distribution")
        return 't_distribution'


def list_available_metric_types():
    """Get list of all configured metric types."""
    return list(METRIC_CONFIGURATIONS.keys())


def list_metric_types_by_pattern(pattern):
    """Get list of metric types that use a specific aggregation pattern."""
    return [
        metric_type for metric_type, config in METRIC_CONFIGURATIONS.items()
        if config['aggregation_pattern'] == pattern
    ]


# ==============================================================================
# EXPERIMENT STATISTICS HELPER FUNCTIONS
# ==============================================================================

def add_experiment_statistic(metric_type, name, extract, compute):
    """
    Dynamically add an experiment statistic to a metric type configuration.
    
    Args:
        metric_type: Metric type key
        name: Display name for the statistic
        extract: Base statistic to extract ('mean', 'std', 'variance', etc.)
        compute: Statistic to compute on extracted values ('mean', 'std', etc.)
    """
    config = get_metric_configuration(metric_type)
    
    if config['aggregation_pattern'] != 'two_level':
        raise ValueError(f"Cannot add experiment statistics to one-level pattern: {metric_type}")
    
    new_stat = {'name': name, 'extract': extract, 'compute': compute}
    config['experiment_stats'].append(new_stat)
    
    logger.info(f"Added experiment statistic to {metric_type}: {new_stat}")


def validate_experiment_statistic_config(stat_config):
    """
    Validate that an experiment statistic configuration is properly formatted.
    
    Args:
        stat_config: Dictionary with 'name', 'extract', 'compute' keys
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['name', 'extract', 'compute']
    
    if not isinstance(stat_config, dict):
        raise ValueError("Experiment statistic configuration must be a dictionary")
    
    for key in required_keys:
        if key not in stat_config:
            raise ValueError(f"Missing required key '{key}' in experiment statistic configuration")
    
    # Validate that extract and compute refer to valid statistics
    valid_statistics = ['mean', 'std', 'variance', 'min', 'max', 'p25', 'p50', 'p75', 'p95', 'full_stats']
    
    if stat_config['extract'] not in valid_statistics:
        raise ValueError(f"Invalid extract statistic: {stat_config['extract']}. Must be one of {valid_statistics}")
    
    if stat_config['compute'] not in valid_statistics:
        raise ValueError(f"Invalid compute statistic: {stat_config['compute']}. Must be one of {valid_statistics}")
    
    return True


# ==============================================================================
# CONFIGURATION VALIDATION
# ==============================================================================

def validate_configurations():
    """Validate that all metric configurations are properly structured."""
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
            
            # Validate experiment statistics configurations
            for stat_config in config['experiment_stats']:
                validate_experiment_statistic_config(stat_config)
                
                # Validate compute field for CI construction
                if stat_config.get('construct_ci', False):
                    compute_type = stat_config.get('compute')
                    if compute_type not in ['mean', 'std', 'variance']:
                        raise ValueError(f"Invalid compute type '{compute_type}' in {metric_type}. "
                                       f"Supported types: mean, std, variance")
        
        elif pattern == 'one_level':
            # Validate CI configuration if present
            if 'ci_config' in config:
                for ci_config in config['ci_config']:
                    required_ci_fields = ['metric_name', 'construct_ci']
                    for field in required_ci_fields:
                        if field not in ci_config:
                            raise ValueError(f"Missing CI config field '{field}' in {metric_type}")
            
            if 'experiment_stats' in config:
                logger.warning(f"One-level pattern {metric_type} should not specify experiment_stats")
    
    logger.info(f"Configuration validation passed for {len(METRIC_CONFIGURATIONS)} metric types")
    return True


# ==============================================================================
# CONFIGURATION SUMMARY
# ==============================================================================

def print_configuration_summary():
    """Print a summary of all configured metric types."""
    print("Metric Configuration Summary")
    print("=" * 50)
    
    two_level = list_metric_types_by_pattern('two_level')
    one_level = list_metric_types_by_pattern('one_level')
    
    print(f"\nTwo-Level Pattern ({len(two_level)} types):")
    for metric_type in two_level:
        config = get_metric_configuration(metric_type)
        print(f"  {metric_type}: {config['metric_function']}")
        for stat in config['experiment_stats']:
            ci_status = "CI" if stat.get('construct_ci', False) else "descriptive only"
            if stat.get('construct_ci', False):
                ci_method = _determine_ci_method_from_compute(stat.get('compute'))
                print(f"    → {stat['name']}: {stat['compute']} of {stat['extract']} ({ci_status} - {ci_method})")
            else:
                print(f"    → {stat['name']}: {stat['compute']} of {stat['extract']} ({ci_status})")
    
    print(f"\nOne-Level Pattern ({len(one_level)} types):")
    for metric_type in one_level:
        config = get_metric_configuration(metric_type)
        print(f"  {metric_type}: {config['metric_function']}")
        if 'ci_config' in config:
            for ci_config in config['ci_config']:
                ci_status = "CI (t-distribution)" if ci_config.get('construct_ci', False) else "descriptive only"
                print(f"    → {ci_config['metric_name']}: {ci_status}")
    
    print(f"\nTotal: {len(METRIC_CONFIGURATIONS)} configured metric types")
    print("\nCI Methods automatically determined:")
    print("  - compute='mean' → t-distribution")
    print("  - compute='std' or 'variance' → chi-square")
    print("  - system metrics → t-distribution (always estimating mean)")



# Run validation on import to catch configuration errors early
if __name__ != "__main__":
    try:
        validate_configurations()
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise