# arrival_interval_ratio_study.py
"""
Arrival Interval Ratio Study: Systematic Validation Design

Research Question: How does the ratio of driver to order arrival intervals 
affect system performance and operational regime characteristics?

Arrival Interval Ratio Definition:
    arrival_interval_ratio = mean_driver_inter_arrival_time / mean_order_inter_arrival_time

Interpretation:
    - ratio = 3.0 → drivers arrive 3× slower than orders
    - ratio = 0.8 → drivers arrive 0.8× as slow (faster than orders)
    - Higher ratio → less driver supply relative to order demand

Design Pattern: For each arrival_interval_ratio R:
- Baseline Interval: (1.0, R) → "Higher intensity" (1.0 orders/min, 1/R drivers/min)
- 2x Baseline: (2.0, 2R) → "Half intensity" (0.5 orders/min, 1/2R drivers/min)

Research Hypothesis:
"Operational regime characteristics are determined by the interaction between 
arrival interval ratio and absolute operational intensity, with driver capacity 
serving as the primary system bottleneck."
"""

# %% CELL 1: Enable Autoreload
%load_ext autoreload 
%autoreload 2

# %% CELL 2: Setup and Imports
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
print("ARRIVAL INTERVAL RATIO STUDY: SYSTEMATIC VALIDATION DESIGN")
print("="*80)
print("Research Question: How does arrival interval ratio affect system performance?")
print("Hypothesis: Regime determined by ratio × absolute intensity interaction")

# %% CELL 3: Logging Configuration
logging_config = LoggingConfig(
    console_level="INFO",
    component_levels={
        "services": "ERROR",
        "entities": "ERROR",
        "repositories": "ERROR",
        "utils": "ERROR",
        "system_data": "ERROR",
        "simulation.runner": "INFO",
        "infrastructure": "INFO",
        "experimental.runner": "INFO",
    }
)
configure_logging(logging_config)
print("✓ Logging configured")

# %% CELL 4: Infrastructure Configuration(s)
"""
OPERATIONAL STUDY: Single fixed infrastructure.
Focus is on varying operational parameters, not infrastructure.
"""

infrastructure_configs = [
    {
        'name': 'baseline',
        'config': StructuralConfig(
            delivery_area_size=10,
            num_restaurants=10,
            driver_speed=0.5
        )
    }
]

print(f"✓ Defined {len(infrastructure_configs)} infrastructure configuration")
for config in infrastructure_configs:
    struct_config = config['config']
    density = struct_config.num_restaurants / (struct_config.delivery_area_size ** 2)
    print(f"  • {config['name']}: {struct_config.num_restaurants} restaurants, "
          f"area={struct_config.delivery_area_size}km, density={density:.4f}/km²")

# %% CELL 5: Structural Seeds
"""
OPERATIONAL STUDY: Single seed.
Layout variation is not the focus of this study.
"""

structural_seeds = [42]

print(f"✓ Structural seeds: {structural_seeds} (fixed layout for operational study)")

# %% CELL 6: Operational Configuration(s)
"""
OPERATIONAL STUDY: Multiple configurations varying arrival interval ratios.

For each target ratio, create validation pair:
- Baseline: (1.0, ratio) → higher intensity
- 2x Baseline: (2.0, 2×ratio) → half intensity

This tests whether ratio alone determines regime or if absolute scale matters.
"""

# Target arrival interval ratios to test
target_arrival_interval_ratios = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]

# Fixed pairing configuration (consistent across all design points)
FIXED_PAIRING_CONFIG = {
    'pairing_enabled': True,
    'restaurants_proximity_threshold': 4.0,
    'customers_proximity_threshold': 3.0
}

# Fixed service duration configuration
FIXED_SERVICE_CONFIG = {
    'mean_service_duration': 120,
    'service_duration_std_dev': 60,
    'min_service_duration': 30,
    'max_service_duration': 240
}

# Build operational configs
operational_configs = []

for ratio in target_arrival_interval_ratios:
    # Baseline configuration: (1.0, ratio)
    operational_configs.append({
        'name': f'ratio_{ratio:.1f}_baseline',
        'config': OperationalConfig(
            mean_order_inter_arrival_time=1.0,
            mean_driver_inter_arrival_time=ratio,
            **FIXED_PAIRING_CONFIG,
            **FIXED_SERVICE_CONFIG
        )
    })
    
    # 2x Baseline configuration: (2.0, 2×ratio)
    operational_configs.append({
        'name': f'ratio_{ratio:.1f}_2x_baseline',
        'config': OperationalConfig(
            mean_order_inter_arrival_time=2.0,
            mean_driver_inter_arrival_time=2.0 * ratio,
            **FIXED_PAIRING_CONFIG,
            **FIXED_SERVICE_CONFIG
        )
    })

print(f"✓ Defined {len(operational_configs)} operational configurations")
print(f"✓ Testing {len(target_arrival_interval_ratios)} arrival interval ratios")
print(f"✓ Each ratio has 2 validation pairs (baseline + 2x_baseline)")

# Display configurations
for config in operational_configs:
    op_config = config['config']
    ratio = op_config.mean_driver_inter_arrival_time / op_config.mean_order_inter_arrival_time
    print(f"  • {config['name']}: "
          f"order_interval={op_config.mean_order_inter_arrival_time:.1f}min, "
          f"driver_interval={op_config.mean_driver_inter_arrival_time:.1f}min, "
          f"ratio={ratio:.1f}")

# %% CELL 7: Design Points Creation
"""
Universal design points creation for operational study.
1 infrastructure × 1 seed × N operational configs = N design points
"""

design_points = {}

print("\n" + "="*50)
print("DESIGN POINTS CREATION")
print("="*50)

for infra_config in infrastructure_configs:
    for structural_seed in structural_seeds:
        
        # Create infrastructure instance
        print(f"\n📍 Creating infrastructure: {infra_config['name']}, seed={structural_seed}")
        infrastructure = Infrastructure(
            infra_config['config'],
            structural_seed
        )
     
        # Analyze infrastructure
        analyzer = InfrastructureAnalyzer(infrastructure)
        analysis_results = analyzer.analyze_complete_infrastructure()
        
        print(f"  ✓ Infrastructure analyzed")
        print(f"    • Typical distance: {analysis_results['typical_distance']:.3f}km")
        print(f"    • Restaurant density: {analysis_results['restaurant_density']:.4f}/km²")
        
        # Create design point for each operational configuration
        for op_config in operational_configs:
            
            # Generate design point name (no need for seed since it's fixed)
            design_name = op_config['name']
            
            # Create design point
            design_points[design_name] = DesignPoint(
                infrastructure=infrastructure,
                operational_config=op_config['config'],
                scoring_config=scoring_config,
                name=design_name
            )
            
            print(f"  ✓ Design point: {design_name}")

print(f"\n{'='*50}")
print(f"✓ Created {len(design_points)} design points")
print(f"✓ Breakdown: {len(infrastructure_configs)} infra × "
      f"{len(structural_seeds)} seeds × {len(operational_configs)} operational")
print(f"{'='*50}")

# %% CELL 8: Experiment Configuration
experiment_config = ExperimentConfig(
    simulation_duration=2000,  # Extended duration for regime pattern analysis
    num_replications=5,
    operational_master_seed=42,
    collection_interval=1.0
)

total_runs = len(design_points) * experiment_config.num_replications
estimated_time = total_runs * 5

print(f"✓ Experiment configuration:")
print(f"  • Simulation duration: {experiment_config.simulation_duration} minutes")
print(f"  • Replications per design point: {experiment_config.num_replications}")
print(f"  • Operational master seed: {experiment_config.operational_master_seed}")
print(f"  • Collection interval: {experiment_config.collection_interval} minutes")
print(f"\n✓ Execution plan:")
print(f"  • Total simulation runs: {total_runs}")
print(f"  • Estimated time: ~{estimated_time:.0f} seconds (~{estimated_time/60:.1f} minutes)")

# %% CELL 9: Execute Experimental Study
print("\n" + "="*50)
print("EXECUTING EXPERIMENTAL STUDY")
print("="*50)

runner = ExperimentalRunner()
study_results = runner.run_experimental_study(design_points, experiment_config)

print(f"\n{'='*50}")
print("✅ EXPERIMENTAL STUDY COMPLETE")
print(f"{'='*50}")
print(f"✓ Executed {len(design_points)} design points")
print(f"✓ Total simulations: {total_runs}")

# %% CELL 10: Time Series Data Processing for Warmup Analysis
print("\n" + "="*50)
print("TIME SERIES DATA PROCESSING FOR WARMUP ANALYSIS")
print("="*50)

from delivery_sim.warmup_analysis.time_series_processing import extract_warmup_time_series

print("Processing time series data for warmup detection...")

all_time_series_data = extract_warmup_time_series(
    study_results=study_results,
    design_points=design_points,
    metrics=['active_drivers', 'unassigned_delivery_entities'],
    moving_average_window=100  # Larger window for 2000-minute simulation
)

print(f"✓ Time series processing complete for {len(all_time_series_data)} design points")
print(f"✓ Metrics extracted: active_drivers, unassigned_delivery_entities")
print(f"✓ Ready for warmup analysis visualization")

# %% CELL 11: Warmup Analysis Visualization
print("\n" + "="*50)
print("WARMUP ANALYSIS VISUALIZATION")
print("="*50)

from delivery_sim.warmup_analysis.visualization import WelchMethodVisualization
import matplotlib.pyplot as plt

print("Creating warmup analysis plots...")

# Initialize visualization
viz = WelchMethodVisualization(figsize=(16, 10))

# Group design points by arrival interval ratio for organized display
ratio_groups = {}
for design_name in all_time_series_data.keys():
    # Extract ratio from design name (e.g., "ratio_3.0_baseline")
    ratio_str = design_name.split('_')[1]  # "3.0"
    ratio = float(ratio_str)
    
    if ratio not in ratio_groups:
        ratio_groups[ratio] = []
    ratio_groups[ratio].append(design_name)

print(f"✓ Grouped {len(all_time_series_data)} design points by {len(ratio_groups)} ratios")

# Create plots systematically by ratio
plot_count = 0
for ratio in sorted(ratio_groups.keys()):
    print(f"\nRatio {ratio:.1f} (Driver intervals {ratio:.1f}× order intervals):")
    
    for design_name in sorted(ratio_groups[ratio]):
        plot_title = f"Warmup Analysis: {design_name}"
        time_series_data = all_time_series_data[design_name]
        fig = viz.create_warmup_analysis_plot(time_series_data, title=plot_title)
        
        plt.show()
        print(f"    ✓ {design_name} plot displayed")
        plot_count += 1

print(f"\n✓ Warmup analysis visualization complete")
print(f"✓ Created {plot_count} warmup analysis plots")
print(f"✓ Organized by {len(ratio_groups)} arrival interval ratios")

# %% CELL 12: Warmup Period Determination
print("\n" + "="*50)
print("WARMUP PERIOD DETERMINATION")
print("="*50)

# Set warmup period based on visual inspection of Cell 11 plots
uniform_warmup_period = 500  # UPDATE THIS based on visual inspection

print(f"✓ Warmup period set: {uniform_warmup_period} minutes")
print(f"✓ Based on visual inspection of active drivers oscillation around Little's Law values")
print(f"✓ Analysis window: {experiment_config.simulation_duration - uniform_warmup_period} minutes of post-warmup data")

# %% CELL 13: Process Through Analysis Pipeline
print("\n" + "="*80)
print("PROCESSING THROUGH ANALYSIS PIPELINE")
print("="*80)

from delivery_sim.analysis_pipeline.pipeline_coordinator import ExperimentAnalysisPipeline

# Initialize pipeline
pipeline = ExperimentAnalysisPipeline(
    warmup_period=uniform_warmup_period,
    enabled_metric_types=['order_metrics', 'system_metrics'],
    confidence_level=0.95
)

# Process each design point
design_analysis_results = {}

print(f"\nProcessing {len(study_results)} design points...")
print(f"Warmup period: {uniform_warmup_period} minutes")
print(f"Confidence level: 95%\n")

for i, (design_name, replication_results) in enumerate(study_results.items(), 1):
    print(f"[{i:2d}/{len(study_results)}] Analyzing {design_name}...")
    
    analysis_result = pipeline.analyze_experiment(replication_results)
    design_analysis_results[design_name] = analysis_result
    
    print(f"    ✓ Processed {analysis_result['num_replications']} replications")

print(f"\n✓ Analysis pipeline complete for all {len(design_analysis_results)} design points")
print(f"✓ Results stored in 'design_analysis_results'")

# %% CELL 14: Extract and Present Key Metrics
print("\n" + "="*80)
print("KEY PERFORMANCE METRICS EXTRACTION AND PRESENTATION")
print("="*80)

import re

def extract_ratio_and_type(design_name):
    """Extract arrival interval ratio and interval type from design point name."""
    # Pattern: ratio_3.0_baseline or ratio_3.0_2x_baseline
    match = re.match(r'ratio_([\d.]+)_(baseline|2x_baseline)', design_name)
    if match:
        ratio = float(match.group(1))
        interval_type = match.group(2)
        return ratio, interval_type
    return None, None

# Extract key metrics for each design point
metrics_data = []

for design_name, analysis_result in design_analysis_results.items():
    ratio, interval_type = extract_ratio_and_type(design_name)
    if ratio is None:
        continue
    
    # Extract order metrics
    order_metrics = analysis_result.get('order_metrics', {})
    assignment_time = order_metrics.get('assignment_time_statistics', {})
    
    # Extract system metrics
    system_metrics = analysis_result.get('system_metrics', {})
    completion_rate = system_metrics.get('completion_rate_statistics', {})
    
    # Get mean values with confidence intervals
    metrics_data.append({
        'design_name': design_name,
        'ratio': ratio,
        'interval_type': interval_type,
        'assignment_time_mean': assignment_time.get('mean', {}).get('experiment_mean'),
        'assignment_time_ci_lower': assignment_time.get('mean', {}).get('ci_lower'),
        'assignment_time_ci_upper': assignment_time.get('mean', {}).get('ci_upper'),
        'completion_rate_mean': completion_rate.get('mean', {}).get('experiment_mean'),
        'completion_rate_ci_lower': completion_rate.get('mean', {}).get('ci_lower'),
        'completion_rate_ci_upper': completion_rate.get('mean', {}).get('ci_upper'),
    })

# Create summary table
import pandas as pd
df_metrics = pd.DataFrame(metrics_data)
df_metrics = df_metrics.sort_values(['ratio', 'interval_type'])

print("\n📊 KEY PERFORMANCE METRICS SUMMARY")
print("="*80)
print("\nAssignment Time (minutes) with 95% CI:")
print("-"*80)

for ratio in sorted(df_metrics['ratio'].unique()):
    print(f"\nRatio {ratio:.1f} (Driver intervals {ratio:.1f}× order intervals):")
    ratio_data = df_metrics[df_metrics['ratio'] == ratio]
    
    for _, row in ratio_data.iterrows():
        mean = row['assignment_time_mean']
        ci_lower = row['assignment_time_ci_lower']
        ci_upper = row['assignment_time_ci_upper']
        
        print(f"  {row['interval_type']:12s}: {mean:6.2f} min  [{ci_lower:6.2f}, {ci_upper:6.2f}]")

print("\n" + "="*80)
print("\nCompletion Rate (%) with 95% CI:")
print("-"*80)

for ratio in sorted(df_metrics['ratio'].unique()):
    print(f"\nRatio {ratio:.1f}:")
    ratio_data = df_metrics[df_metrics['ratio'] == ratio]
    
    for _, row in ratio_data.iterrows():
        mean = row['completion_rate_mean'] * 100  # Convert to percentage
        ci_lower = row['completion_rate_ci_lower'] * 100
        ci_upper = row['completion_rate_ci_upper'] * 100
        
        print(f"  {row['interval_type']:12s}: {mean:5.2f}%  [{ci_lower:5.2f}%, {ci_upper:5.2f}%]")

print("\n" + "="*80)
print("✓ Metric extraction complete")
print("✓ Data available in 'df_metrics' DataFrame for further analysis")

# %% CELL 15: Validation Pair Comparison
print("\n" + "="*80)
print("VALIDATION PAIR COMPARISON")
print("="*80)
print("\nTesting hypothesis: Do baseline and 2x_baseline show similar regime behavior?")
print("="*80)

for ratio in sorted(df_metrics['ratio'].unique()):
    ratio_data = df_metrics[df_metrics['ratio'] == ratio]
    baseline = ratio_data[ratio_data['interval_type'] == 'baseline'].iloc[0]
    two_x = ratio_data[ratio_data['interval_type'] == '2x_baseline'].iloc[0]
    
    # Compare assignment times
    baseline_time = baseline['assignment_time_mean']
    two_x_time = two_x['assignment_time_mean']
    time_diff = abs(baseline_time - two_x_time)
    time_pct_diff = (time_diff / baseline_time) * 100
    
    # Compare completion rates
    baseline_comp = baseline['completion_rate_mean'] * 100
    two_x_comp = two_x['completion_rate_mean'] * 100
    comp_diff = abs(baseline_comp - two_x_comp)
    
    print(f"\nRatio {ratio:.1f}:")
    print(f"  Assignment Time Difference: {time_diff:.2f} min ({time_pct_diff:.1f}%)")
    print(f"  Completion Rate Difference: {comp_diff:.2f}%")
    
    # Simple interpretation
    if time_pct_diff < 10 and comp_diff < 5:
        print(f"  → Similar regime behavior (validates ratio hypothesis)")
    else:
        print(f"  → Different behavior (absolute scale matters)")

print("\n" + "="*80)
print("✓ Validation pair analysis complete")

print("\n" + "="*80)
print("ARRIVAL INTERVAL RATIO STUDY COMPLETE")
print("="*80)
print("\nKey findings:")
print(f"✓ Tested {len(target_arrival_interval_ratios)} arrival interval ratios")
print(f"✓ Each ratio validated with baseline vs 2x_baseline pairs")
print(f"✓ Analyzed {len(design_points)} design points × {experiment_config.num_replications} replications")
print(f"✓ Results show relationship between arrival interval ratio and system performance")