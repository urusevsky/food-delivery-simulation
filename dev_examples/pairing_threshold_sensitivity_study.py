# pairing_threshold_sensitivity_study.py
"""
Threshold Sensitivity Study: Validation Pairs Across Pairing Threshold Configurations

Research Question: How do different pairing threshold configurations affect 
pairing effectiveness and system performance across various load ratios and absolute scales?

Threshold Sensitivity Design:
- Threshold Configurations: Conservative (2.0, 1.5), Moderate (4.0, 3.0), Liberal (6.0, 4.5), Ultra-liberal (8.0, 6.0)
- Load Ratios: [3.0, 5.0, 7.0, 8.0] with validation pairs
- Baseline: (1.0, LR) vs 2x Baseline: (2.0, 2×LR) for each threshold × load ratio combination
- Pairing Only: Focus on pairing-enabled configurations to understand threshold sensitivity

This reveals:
1. How threshold liberality affects pairing effectiveness across regimes
2. Whether threshold effects are consistent across different load ratios  
3. Scale dependency of threshold effectiveness
4. Optimal threshold selection for different operational contexts
"""

# %% Enable Autoreload (ALWAYS put this at the top of research scripts)
%load_ext autoreload
%autoreload 2

# %% Step 1: Setup and Imports
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from delivery_sim.simulation.configuration import (
    StructuralConfig, OperationalConfig, ExperimentConfig, 
    LoggingConfig, ScoringConfig
)
from delivery_sim.infrastructure.infrastructure import Infrastructure
from delivery_sim.infrastructure.infrastructure_analyzer import InfrastructureAnalyzer
from delivery_sim.experimental.design_point import DesignPoint
from delivery_sim.experimental.experimental_runner import ExperimentalRunner
from delivery_sim.utils.logging_system import configure_logging

print("="*80)
print("THRESHOLD SENSITIVITY STUDY: PAIRING THRESHOLD CONFIGURATIONS")
print("="*80)
print("Research Focus: Threshold × Load Ratio × Absolute Scale Interaction Effects")

# %% Step 2: Logging Configuration
logging_config = LoggingConfig(
    console_level="INFO",
    component_levels={
        "services": "ERROR", "entities": "ERROR", "repositories": "ERROR",
        "utils": "ERROR", "system_data": "ERROR",
        "simulation.runner": "INFO", "infrastructure": "INFO", 
        "experimental.runner": "INFO",
    }
)
configure_logging(logging_config)
print("✓ Clean logging configured")

# %% Step 3: Infrastructure Setup (Consistent with Previous Studies)
print("\n" + "="*50)
print("INFRASTRUCTURE SETUP")
print("="*50)

structural_config = StructuralConfig(
    delivery_area_size=10,
    num_restaurants=10,
    driver_speed=0.5
)

infrastructure = Infrastructure(structural_config, master_seed=42)
print(f"✓ Infrastructure created: {structural_config.delivery_area_size}km × {structural_config.delivery_area_size}km")
print(f"✓ Restaurants: {structural_config.num_restaurants}")
print(f"✓ Driver speed: {structural_config.driver_speed} km/min")

analyzer = InfrastructureAnalyzer(infrastructure)
analysis = analyzer.analyze_complete_infrastructure()
print(f"✓ Infrastructure analyzed: typical_distance = {analysis['typical_distance']:.2f}km")

# %% Step 4: Threshold Sensitivity Design Points
print("\n" + "="*50)
print("THRESHOLD SENSITIVITY DESIGN POINTS")
print("="*50)

# Scoring configuration (using default weights)
scoring_config = ScoringConfig()

# Threshold configurations for sensitivity analysis
threshold_configurations = {
    'conservative': {
        'restaurants_proximity_threshold': 2.0,
        'customers_proximity_threshold': 1.5
    },
    'moderate': {
        'restaurants_proximity_threshold': 4.0,
        'customers_proximity_threshold': 3.0
    },
    'liberal': {
        'restaurants_proximity_threshold': 6.0,
        'customers_proximity_threshold': 4.5
    },
    'ultra_liberal': {
        'restaurants_proximity_threshold': 8.0,
        'customers_proximity_threshold': 6.0
    }
}

print(f"Threshold Configurations for Sensitivity Analysis:")
for config_name, thresholds in threshold_configurations.items():
    print(f"  • {config_name.replace('_', '-').capitalize()}: "
          f"restaurants={thresholds['restaurants_proximity_threshold']}km, "
          f"customers={thresholds['customers_proximity_threshold']}km")

# Base operational parameters
base_params = {
    'mean_service_duration': 100,
    'service_duration_std_dev': 60,
    'min_service_duration': 30,
    'max_service_duration': 200,
}

# Target load ratios for threshold sensitivity analysis
target_load_ratios = [3.0, 5.0, 7.0, 8.0]

print(f"\nCreating threshold sensitivity design points...")
print(f"Pattern: Threshold Config × Load Ratio × Validation Pairs")
print(f"Total combinations: {len(threshold_configurations)} configs × {len(target_load_ratios)} ratios × 2 validation = {len(threshold_configurations) * len(target_load_ratios) * 2}")

design_points = {}

for threshold_name, threshold_params in threshold_configurations.items():
    
    # Create pairing parameters with this threshold configuration
    pairing_params = {
        'pairing_enabled': True,
        **threshold_params
    }
    
    for load_ratio in target_load_ratios:
        
        # === Baseline Design Point ===
        baseline_name = f"threshold_{threshold_name}_load_ratio_{load_ratio:.1f}_baseline"
        design_points[baseline_name] = DesignPoint(
            infrastructure=infrastructure,
            operational_config=OperationalConfig(
                mean_order_inter_arrival_time=1.0,
                mean_driver_inter_arrival_time=load_ratio,
                **base_params,
                **pairing_params
            ),
            scoring_config=scoring_config,
            name=baseline_name
        )
        
        # === 2x Baseline Design Point ===
        double_baseline_name = f"threshold_{threshold_name}_load_ratio_{load_ratio:.1f}_2x_baseline"
        design_points[double_baseline_name] = DesignPoint(
            infrastructure=infrastructure,
            operational_config=OperationalConfig(
                mean_order_inter_arrival_time=2.0,
                mean_driver_inter_arrival_time=2.0 * load_ratio,
                **base_params,
                **pairing_params
            ),
            scoring_config=scoring_config,
            name=double_baseline_name
        )
        
    print(f"  ✓ {threshold_name.replace('_', '-').capitalize()}: {len(target_load_ratios)} load ratios × 2 validation = {len(target_load_ratios) * 2} points")

print(f"\n✓ Created {len(design_points)} threshold sensitivity design points")
print(f"✓ Design enables analysis of: Threshold × Load Ratio × Scale interactions")

# %% Step 5: Experiment Configuration
print("\n" + "="*50)
print("EXPERIMENT CONFIGURATION")
print("="*50)

experiment_config = ExperimentConfig(
    simulation_duration=2000,
    num_replications=5,
    master_seed=42
)

print(f"✓ Duration: {experiment_config.simulation_duration} minutes")
print(f"✓ Replications: {experiment_config.num_replications}")
print(f"✓ Total simulation runs: {len(design_points)} × {experiment_config.num_replications} = {len(design_points) * experiment_config.num_replications}")

# %% Step 6: Execute Threshold Sensitivity Study
print("\n" + "="*50)
print("THRESHOLD SENSITIVITY EXECUTION")
print("="*50)

runner = ExperimentalRunner()
print("✓ ExperimentalRunner initialized")

print(f"\nExecuting threshold sensitivity study...")
print("Focus: How do threshold configurations affect pairing across load ratios and scales?")
study_results = runner.run_experimental_study(design_points, experiment_config)

print(f"\n✅ THRESHOLD SENSITIVITY STUDY COMPLETE!")
print(f"✓ All {len(design_points)} design points executed")
print(f"✓ Results contain threshold × load ratio × scale interaction data")

# %% Step 7: Warmup Period (From Previous Study)
print("\n" + "="*50)
print("WARMUP PERIOD (FROM PREVIOUS STUDY)")
print("="*50)

uniform_warmup_period = 500

print(f"✓ Using verified warmup period: {uniform_warmup_period} minutes")
print(f"✓ Based on visual inspection from load_ratio_driven_supply_demand_study.py")
print(f"✓ Streamlined approach - no warmup detection needed")

# %%
# ==================================================================================
# STEP 8: EXPERIMENTAL ANALYSIS USING ANALYSIS PIPELINE
# ==================================================================================

print(f"\n{'='*80}")
print("STEP 8: EXPERIMENTAL ANALYSIS USING ANALYSIS PIPELINE")
print(f"{'='*80}\n")

from delivery_sim.analysis_pipeline.pipeline_coordinator import ExperimentAnalysisPipeline

# Initialize pipeline
pipeline = ExperimentAnalysisPipeline(
    warmup_period=uniform_warmup_period,
    enabled_metric_types=['order_metrics', 'system_metrics'],
    confidence_level=0.95
)

# Process each design point
design_analysis_results = {}

print(f"Processing {len(study_results)} design points through analysis pipeline...")
print(f"Warmup period: {uniform_warmup_period} minutes")
print(f"Confidence level: 95%")
print(f"Metrics: Assignment Time + Pairing Rate + Completion Rate\n")

for i, (design_name, raw_replication_results) in enumerate(study_results.items(), 1):
    print(f"[{i:2d}/{len(study_results)}] Analyzing {design_name}...")
    
    analysis_result = pipeline.analyze_experiment(raw_replication_results)
    design_analysis_results[design_name] = analysis_result
    
    print(f"    ✓ Processed {analysis_result['num_replications']} replications")

print(f"\n✓ Completed analysis for all {len(design_analysis_results)} design points")
print("Analysis results stored in 'design_analysis_results'")

# %%
# ==================================================================================
# STEP 9: EXTRACT AND PRESENT THRESHOLD SENSITIVITY METRICS
# ==================================================================================

print(f"\n{'='*80}")
print("STEP 9: THRESHOLD SENSITIVITY METRICS EXTRACTION")
print(f"{'='*80}\n")

import re

def extract_threshold_design_info(design_name):
    """Extract threshold config, load ratio, and interval type from design name."""
    # Pattern: threshold_conservative_load_ratio_3.0_baseline
    # Pattern: threshold_ultra_liberal_load_ratio_5.0_2x_baseline
    
    pattern = r"threshold_(\w+)_load_ratio_(\d+\.?\d*)_(.*)"
    match = re.match(pattern, design_name)
    
    if match:
        threshold_config = match.group(1)
        load_ratio = float(match.group(2))
        interval_suffix = match.group(3)
        
        # Format threshold config name
        if threshold_config == "ultra_liberal":
            threshold_display = "Ultra-liberal"
        else:
            threshold_display = threshold_config.capitalize()
        
        # Format interval type
        if interval_suffix == "baseline":
            interval_type = "Baseline"
        elif interval_suffix == "2x_baseline":
            interval_type = "2x Baseline"
        else:
            interval_type = interval_suffix.replace("_", " ").title()
        
        return threshold_config, threshold_display, load_ratio, interval_type
    else:
        return None, None, None, None

def format_metric_value(value, decimal_places=2):
    """Format metric value for display."""
    if value is None:
        return "N/A"
    return f"{value:.{decimal_places}f}"

def format_ci_value(point_estimate, ci_bounds, decimal_places=2):
    """Format value with confidence interval."""
    if point_estimate is None:
        return "N/A"
    
    if ci_bounds and ci_bounds[0] is not None and ci_bounds[1] is not None:
        lower, upper = ci_bounds
        margin = (upper - lower) / 2
        return f"{point_estimate:.{decimal_places}f} ± {margin:.{decimal_places}f}"
    else:
        return f"{point_estimate:.{decimal_places}f}"

# Extract metrics from analysis results
results_table = []

for design_name, analysis_result in design_analysis_results.items():
    try:
        # Access metrics via statistics_with_cis
        order_metrics_ci = analysis_result['statistics_with_cis']['order_metrics']['assignment_time']
        system_metrics_ci = analysis_result['statistics_with_cis']['system_metrics']
        
        # Extract assignment time metrics (two-level pattern)
        mean_of_means_data = order_metrics_ci['mean_of_means']
        std_of_means_data = order_metrics_ci['std_of_means']
        mean_of_stds_data = order_metrics_ci['mean_of_stds']
        
        mean_of_means = mean_of_means_data['point_estimate']
        mean_of_means_ci = mean_of_means_data['confidence_interval']
        std_of_means = std_of_means_data['point_estimate']
        mean_of_stds = mean_of_stds_data['point_estimate']
        
        # Extract system metrics (one-level pattern)
        completion_rate_data = system_metrics_ci['system_completion_rate']
        pairing_rate_data = system_metrics_ci['system_pairing_rate']
        
        completion_rate = completion_rate_data['point_estimate']
        completion_rate_ci = completion_rate_data['confidence_interval']
        pairing_rate = pairing_rate_data['point_estimate']
        pairing_rate_ci = pairing_rate_data['confidence_interval']
        
        # Parse design point information
        threshold_config, threshold_display, load_ratio, interval_type = extract_threshold_design_info(design_name)
        
        # Store results
        results_table.append({
            'design_name': design_name,
            'threshold_config': threshold_config,
            'threshold_display': threshold_display,
            'load_ratio': load_ratio,
            'interval_type': interval_type,
            'mean_of_means': mean_of_means,
            'mean_of_means_ci': mean_of_means_ci,
            'std_of_means': std_of_means,
            'mean_of_stds': mean_of_stds,
            'pairing_rate': pairing_rate,
            'pairing_rate_ci': pairing_rate_ci,
            'completion_rate': completion_rate,
            'completion_rate_ci': completion_rate_ci
        })
        
    except KeyError as e:
        print(f"⚠ Warning: Could not extract metrics from {design_name}: {e}")

# Sort and display results table
# Define threshold order for sorting
threshold_order = {'conservative': 0, 'moderate': 1, 'liberal': 2, 'ultra_liberal': 3}
results_table.sort(key=lambda x: (x['load_ratio'], threshold_order.get(x['threshold_config'], 99), x['interval_type']))

print("🎯 THRESHOLD SENSITIVITY: THRESHOLD CONFIG × LOAD RATIO × SCALE INTERACTION")
print("=" * 160)
print(f"{'Threshold':>14} {'Load':>5} {'Interval':>12} {'Pairing':>12} {'Mean of Means':>20} {'Std of':>10} {'Mean of':>10} {'Completion Rate':>25}")
print(f"{'Config':>14} {'Ratio':>5} {'Type':>12} {'Rate':>12} {'(Assignment Time)':>20} {'Means':>10} {'Stds':>10} {'(with 95% CI)':>25}")
print("=" * 160)

for result in results_table:
    threshold_display = result['threshold_display'][:14] if result['threshold_display'] else "N/A"
    load_ratio = format_metric_value(result['load_ratio'], 1) if result['load_ratio'] else "N/A"
    interval_type = result['interval_type'][:12] if result['interval_type'] else "N/A"
    
    # Pairing rate (percentage with CI)
    if result['pairing_rate'] is not None:
        pairing_pct = result['pairing_rate'] * 100
        if result['pairing_rate_ci'] and result['pairing_rate_ci'][0] is not None:
            ci_lower = result['pairing_rate_ci'][0] * 100
            ci_upper = result['pairing_rate_ci'][1] * 100
            margin = (ci_upper - ci_lower) / 2
            pairing_rate_formatted = f"{pairing_pct:.1f}% ± {margin:.1f}"
        else:
            pairing_rate_formatted = f"{pairing_pct:.1f}%"
    else:
        pairing_rate_formatted = "N/A"
    
    # Assignment time metrics
    mean_of_means_formatted = format_ci_value(
        result['mean_of_means'], 
        result['mean_of_means_ci'], 
        decimal_places=2
    )
    std_of_means_formatted = format_metric_value(result['std_of_means'], 2)
    mean_of_stds_formatted = format_metric_value(result['mean_of_stds'], 2)
    
    # Completion rate (percentage with CI)
    if result['completion_rate'] is not None:
        comp_pct = result['completion_rate'] * 100
        if result['completion_rate_ci'] and result['completion_rate_ci'][0] is not None:
            ci_lower = result['completion_rate_ci'][0] * 100
            ci_upper = result['completion_rate_ci'][1] * 100
            margin = (ci_upper - ci_lower) / 2
            completion_rate_formatted = f"{comp_pct:.1f}% ± {margin:.1f}"
        else:
            completion_rate_formatted = f"{comp_pct:.1f}%"
    else:
        completion_rate_formatted = "N/A"
    
    print(f"{threshold_display:>14} {load_ratio:>5} {interval_type:>12} {pairing_rate_formatted:>12} "
          f"{mean_of_means_formatted:>20} {std_of_means_formatted:>10} {mean_of_stds_formatted:>10} "
          f"{completion_rate_formatted:>25}")

print("=" * 160)
print(f"✓ Extracted and displayed metrics from {len(results_table)} design points")
print("Results stored in 'results_table' for further analysis")
print("\nColumn Interpretations:")
print("• Threshold Config: Pairing threshold configuration (Conservative → Ultra-liberal)")
print("• Pairing Rate: Percentage of cohort orders that were paired (with 95% CI)")
print("• Mean of Means: Average assignment time across replications (with 95% CI)")
print("• Std of Means: System consistency between replications (lower = more consistent)")
print("• Mean of Stds: Average within-replication volatility (service predictability)")
print("• Completion Rate: Proportion of orders successfully completed (with 95% CI)")
print("\nResearch Questions:")
print("1. How does threshold liberality affect pairing rate across regimes?")
print("2. Are threshold effects consistent across different load ratios?")
print("3. Do validation pairs show robust threshold sensitivity patterns?")
print("4. Which threshold configuration optimizes performance balance?")