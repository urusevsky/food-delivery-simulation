# infrastructure_layout_study.py
"""
Infrastructure Layout Sensitivity Study

Research Question: Do operational regime boundaries depend on specific restaurant 
layouts, or are they robust across different spatial configurations?

Test: Multiple infrastructure seeds (different random restaurant layouts) with 
same structural parameters (10km × 10km, 10 restaurants).

Design: 3 seeds × 3 critical ratios × 2 scales = 18 design points
Total runs: 18 × 5 replications = 90 runs (~6-7 minutes)
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
print("INFRASTRUCTURE LAYOUT SENSITIVITY STUDY")
print("="*80)
print("Research Question: Do regime boundaries hold across different layouts?")

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

# %% CELL 3.5: Research Question
"""
Document research question and its evolution.
"""

print("\n" + "="*80)
print("RESEARCH QUESTION")
print("="*80)

# ==============================================================================
# MAIN RESEARCH QUESTION
# ==============================================================================
research_question = """
Do operational regime boundaries depend on specific restaurant layouts,
or are they robust across different spatial configurations?

Test: Three different random layouts (seeds 42, 100, 200) with same 
structural parameters (10km × 10km, 10 restaurants).

Critical ratios: 3.5 (stable), 5.0 (volatile), 7.0 (failure) based on
arrival_interval_ratio_study.py findings.

If boundaries shift significantly across layouts → spatial configuration matters
If boundaries hold consistently → ratio determines regime independent of layout
"""

# ==============================================================================
# CONTEXT & MOTIVATION
# ==============================================================================
context = """
Previous study (arrival_interval_ratio_study.py) identified operational regime 
transitions around arrival interval ratios 6.0-7.0 using a single fixed 
infrastructure (seed=42).

Validity concern: Are these boundaries specific to that particular restaurant 
layout, or do they represent general principles?

This study tests layout sensitivity by varying only the structural seed while
keeping all structural parameters (area size, restaurant count) identical.

Evolution from broad exploration:
1. Initially: "How does infrastructure affect system performance?" (too vague)
2. Refined: "What aspect of infrastructure - size, density, or layout?" 
3. Current: Start minimal - test layout variation only
4. Future: If layout matters, vary area_size and num_restaurants systematically
"""

# ==============================================================================
# SUB-QUESTIONS & HYPOTHESES
# ==============================================================================
sub_questions = """
Layout robustness hypothesis predicts:

1. Regime boundaries consistent across layouts
   - All three seeds show stable regime at ratio 3.5
   - All three seeds show volatile regime at ratio 5.0
   - All three seeds show failure regime at ratio 7.0

2. Performance levels may vary, but regime classification holds
   - Different layouts may have different typical_distance
   - Absolute assignment times may differ proportionally
   - But qualitative behavior (stable/volatile/failure) should match

3. Scale validation still holds within each layout
   - Baseline vs 2× baseline should show similar patterns
   - Within each seed, ratio determines regime

Falsification scenarios:
- If seed 42 fails at ratio 7.0 but seed 100 remains stable → layout matters critically
- If typical_distance doesn't predict performance variation → need other spatial metrics
"""

# ==============================================================================
# SCOPE & BOUNDARIES
# ==============================================================================
scope = """
Fixed factors:
- Area size: 10km × 10km (same as previous study)
- Restaurant count: 10 (same as previous study)
- Driver speed: 0.5 km/min
- Pairing: Disabled throughout
- Service duration: Fixed distribution (mean=100, std=60, min=30, max=200)

Varied factors:
- Structural seeds: [42, 100, 200] → three different restaurant layouts
- Arrival interval ratios: [3.5, 5.0, 7.0] → critical regime boundaries
- Operational scale: Baseline vs 2× baseline for each ratio

Not varied in this study (future work):
- Area size (affects geographical scale)
- Restaurant count (affects density)
- Pairing configurations

Design: 3 seeds × 3 ratios × 2 scales = 18 design points × 5 replications = 90 runs
"""

# ==============================================================================
# KEY METRICS & ANALYSIS FOCUS
# ==============================================================================
analysis_focus = """
Primary comparison: For each ratio, compare across three seeds

Metrics:
1. Assignment time: Mean and variability
2. Completion rate: System stability indicator
3. Typical distance: Infrastructure characteristic (explains performance differences?)

Analysis approach:
- Group by ratio, compare across seeds
- Check if regime classification (stable/volatile/failure) holds
- Examine if typical_distance predicts performance variation
- Within each seed, verify baseline vs 2× baseline validation

Expected patterns:
- Regime boundaries at same ratios regardless of layout
- Absolute performance may differ by typical_distance
- If patterns diverge significantly → layout has structural effects beyond distance
"""

print(research_question)
print("\n" + "-"*80)
print("CONTEXT & MOTIVATION")
print("-"*80)
print(context)
print("\n" + "-"*80)
print("SUB-QUESTIONS & HYPOTHESES")
print("-"*80)
print(sub_questions)
print("\n" + "-"*80)
print("SCOPE & BOUNDARIES")
print("-"*80)
print(scope)
print("\n" + "-"*80)
print("KEY METRICS & ANALYSIS FOCUS")
print("-"*80)
print(analysis_focus)
print("\n" + "="*80)
print("✓ Research question documented - reference this to guide analysis decisions")
print("="*80)

# %% CELL 4: Infrastructure Configuration(s)
"""
INFRASTRUCTURE STUDY: Testing layout sensitivity with multiple structural seeds.

Same structural parameters, different random layouts.
"""

infrastructure_configs = [
    {
        'name': 'area_10km',
        'config': StructuralConfig(
            delivery_area_size=10,
            num_restaurants=10,
            driver_speed=0.5
        )
    }
]

print(f"✓ Defined {len(infrastructure_configs)} infrastructure configuration(s)")
for config in infrastructure_configs:
    print(f"  • {config['name']}: {config['config']}")

# %% CELL 5: Structural Seeds
"""
CRITICAL CHANGE: Test multiple structural seeds for layout sensitivity.

Previous study used single seed (42).
This study tests three different layouts with identical structural parameters.
"""

structural_seeds = [42, 100, 200]

print(f"✓ Structural seeds: {structural_seeds}")
print(f"✓ Testing layout sensitivity with {len(structural_seeds)} different restaurant configurations")

# %% CELL 6: Create Infrastructure Instances
"""
Create and analyze infrastructure instances.
Store analyzer for reuse in visualization.
"""

infrastructure_instances = []

print("\n" + "="*50)
print("INFRASTRUCTURE INSTANCES CREATION")
print("="*50)

for infra_config in infrastructure_configs:
    for structural_seed in structural_seeds:
        
        # Create infrastructure instance
        instance_name = f"{infra_config['name']}_seed{structural_seed}"
        print(f"\n📍 Creating infrastructure: {instance_name}")
        
        infrastructure = Infrastructure(
            infra_config['config'],
            structural_seed
        )
        
        # Analyze infrastructure
        analyzer = InfrastructureAnalyzer(infrastructure)
        analysis_results = analyzer.analyze_complete_infrastructure()
        
        # Store instance with metadata
        infrastructure_instances.append({
            'name': instance_name,
            'infrastructure': infrastructure,
            'analyzer': analyzer,  # Store analyzer for reuse
            'analysis': analysis_results,
            'config_name': infra_config['name'],
            'seed': structural_seed
        })
        
        print(f"  ✓ Infrastructure created and analyzed")
        print(f"    • Typical distance: {analysis_results['typical_distance']:.3f}km")
        print(f"    • Restaurant density: {analysis_results['restaurant_density']:.4f}/km²")

print(f"\n{'='*50}")
print(f"✓ Created {len(infrastructure_instances)} infrastructure instance(s)")
print(f"✓ Breakdown: {len(infrastructure_configs)} configs × {len(structural_seeds)} seeds")
print(f"{'='*50}")

# Display typical distance comparison
print("\n📊 Infrastructure Comparison:")
for instance in infrastructure_instances:
    print(f"  {instance['name']:25s}: typical_distance={instance['analysis']['typical_distance']:.3f}km")

# %% CELL 6.5: Visualize Restaurant Layouts
"""
OPTIONAL: Visualize infrastructure instances to understand layout differences.

For layout sensitivity study, this is particularly valuable to see how
restaurant spatial configurations differ across seeds.
"""

print("\n" + "="*50)
print("INFRASTRUCTURE LAYOUT VISUALIZATION")
print("="*50)

import matplotlib.pyplot as plt

print(f"\nVisualizing {len(infrastructure_instances)} different restaurant layouts...")
print("Compare spatial patterns to understand what 'layout sensitivity' means.\n")

for instance in infrastructure_instances:
    print(f"{'='*50}")
    print(f"Layout: {instance['name']}")
    print(f"Typical Distance: {instance['analysis']['typical_distance']:.3f}km")
    print(f"Restaurant Density: {instance['analysis']['restaurant_density']:.4f}/km²")
    print(f"{'='*50}")
    
    # Visualize using stored analyzer
    instance['analyzer'].visualize_infrastructure()
    
    # Add custom header to distinguish layouts
    fig = plt.gcf()  # Get current figure
    seed = instance['seed']
    typical_dist = instance['analysis']['typical_distance']
    
    # Create informative title
    custom_title = (f"Infrastructure Layout: Seed {seed}\n"
                   f"Typical Distance: {typical_dist:.3f}km | "
                   f"Area: 10×10km | Restaurants: 10")
    
    fig.suptitle(custom_title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ {instance['name']} visualized\n")

print(f"{'='*50}")
print("✓ All layouts visualized")
print("✓ Compare spatial patterns:")
print("  - Are restaurants clustered or dispersed?")
print("  - Do some layouts have outlier restaurants?")
print("  - How might these patterns affect delivery performance?")
print(f"{'='*50}")

# %% CELL 7: Scoring Configuration(s)
"""
Single baseline scoring configuration for this study.
"""

scoring_configs = [
    {
        'name': 'baseline',
        'config': ScoringConfig()  # Use defaults
    }
]

print(f"\n✓ Defined {len(scoring_configs)} scoring configuration(s)")
for config in scoring_configs:
    print(f"  • {config['name']}")

# %% CELL 8: Operational Configuration(s)
"""
OPERATIONAL STUDY: Focus on critical ratios that define regime boundaries.

Reduced from 10 ratios to 3 critical ratios based on previous study findings:
- 3.5: Stable regime
- 5.0: Volatile regime  
- 7.0: Failure regime

For each ratio, create validation pair (baseline + 2× baseline).
"""

# CRITICAL CHANGE: Focus on regime boundary ratios only
target_arrival_interval_ratios = [3.5, 5.0, 7.0]

# Fixed pairing configuration
FIXED_PAIRING_CONFIG = {
    'pairing_enabled': False,
    'restaurants_proximity_threshold': None,
    'customers_proximity_threshold': None
}

# Fixed service duration configuration
FIXED_SERVICE_CONFIG = {
    'mean_service_duration': 100,
    'service_duration_std_dev': 60,
    'min_service_duration': 30,
    'max_service_duration': 200
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
print(f"✓ Testing {len(target_arrival_interval_ratios)} critical arrival interval ratios")
print(f"✓ Each ratio has 2 validation pairs (baseline + 2× baseline)")

# Display configurations
for config in operational_configs:
    op_config = config['config']
    ratio = op_config.mean_driver_inter_arrival_time / op_config.mean_order_inter_arrival_time
    print(f"  • {config['name']}: "
          f"order_interval={op_config.mean_order_inter_arrival_time:.1f}min, "
          f"driver_interval={op_config.mean_driver_inter_arrival_time:.1f}min, "
          f"ratio={ratio:.1f}")

# %% CELL 9: Design Point Creation
"""
Create design points from all combinations.

Design: 3 seeds × 3 ratios × 2 scales = 18 design points
"""

design_points = {}

print("\n" + "="*50)
print("DESIGN POINTS CREATION")
print("="*50)

for infra_instance in infrastructure_instances:
    for op_config in operational_configs:
        for scoring_config_dict in scoring_configs:
            
            # Generate design point name including seed info
            design_name = f"{infra_instance['name']}_{op_config['name']}"
            
            # Create design point
            design_points[design_name] = DesignPoint(
                infrastructure=infra_instance['infrastructure'],
                operational_config=op_config['config'],
                scoring_config=scoring_config_dict['config'],
                name=design_name
            )
            
            print(f"  ✓ Design point: {design_name}")

print(f"\n{'='*50}")
print(f"✓ Created {len(design_points)} design points")
print(f"✓ Breakdown: {len(infrastructure_instances)} infra × "
      f"{len(operational_configs)} operational × {len(scoring_configs)} scoring")
print(f"{'='*50}")

# %% CELL 10: Experiment Configuration
experiment_config = ExperimentConfig(
    simulation_duration=2000,
    num_replications=5,
    operational_master_seed=100,
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

# %% CELL 11: Execute Experimental Study
"""
Run all design points through the experimental runner.
"""

print("\n" + "="*50)
print("EXPERIMENTAL EXECUTION")
print("="*50)

runner = ExperimentalRunner()
study_results = runner.run_experimental_study(design_points, experiment_config)

print(f"\n✅ INFRASTRUCTURE LAYOUT STUDY COMPLETE!")
print(f"✓ Executed {len(study_results)} design points")
print(f"✓ Each with {experiment_config.num_replications} replications")

# %% CELL 12: Time Series Data Processing for Warmup Analysis
print("\n" + "="*50)
print("TIME SERIES DATA PROCESSING FOR WARMUP ANALYSIS")
print("="*50)

from delivery_sim.warmup_analysis.time_series_processing import extract_warmup_time_series

print("Processing time series data for warmup detection...")

all_time_series_data = extract_warmup_time_series(
    study_results=study_results,
    design_points=design_points,
    metrics=['active_drivers', 'available_drivers', 'unassigned_delivery_entities'],  # ← ADDED available_drivers
    moving_average_window=100
)

print(f"✓ Time series processing complete for {len(all_time_series_data)} design points")
print(f"✓ Metrics extracted: active_drivers, available_drivers, unassigned_delivery_entities")  # ← UPDATED
print(f"✓ Ready for warmup analysis visualization")

# %% CELL 13: Warmup Analysis Visualization
print("\n" + "="*50)
print("WARMUP ANALYSIS VISUALIZATION")
print("="*50)

from delivery_sim.warmup_analysis.visualization import WelchMethodVisualization
import matplotlib.pyplot as plt

print("Creating warmup analysis plots...")

# Initialize visualization
viz = WelchMethodVisualization(figsize=(16, 10))

# Group design points by seed for organized display
seed_groups = {}
for design_name in all_time_series_data.keys():
    # Extract seed from design name (e.g., "area_10km_seed42_ratio_3.5_baseline")
    parts = design_name.split('_')
    seed_str = parts[2].replace('seed', '')  # "42"
    seed = int(seed_str)
    
    if seed not in seed_groups:
        seed_groups[seed] = []
    seed_groups[seed].append(design_name)

print(f"✓ Grouped {len(all_time_series_data)} design points by {len(seed_groups)} seeds")

# Create plots systematically by seed
plot_count = 0
for seed in sorted(seed_groups.keys()):
    print(f"\nSeed {seed}:")
    
    for design_name in sorted(seed_groups[seed]):
        plot_title = f"Warmup Analysis: {design_name}"
        time_series_data = all_time_series_data[design_name]
        fig = viz.create_warmup_analysis_plot(time_series_data, title=plot_title)
        
        plt.show()
        print(f"    ✓ {design_name} plot displayed")
        plot_count += 1

print(f"\n✓ Warmup analysis visualization complete")
print(f"✓ Created {plot_count} warmup analysis plots")
print(f"✓ Organized by {len(seed_groups)} seeds")

# %% CELL 14: Warmup Period Determination
print("\n" + "="*50)
print("WARMUP PERIOD DETERMINATION")
print("="*50)

# Set warmup period based on visual inspection of Cell 13 plots
uniform_warmup_period = 500  # UPDATE THIS based on visual inspection

print(f"✓ Warmup period set: {uniform_warmup_period} minutes")
print(f"✓ Based on visual inspection of active drivers oscillation around Little's Law values")
print(f"✓ Analysis window: {experiment_config.simulation_duration - uniform_warmup_period} minutes of post-warmup data")

# %% CELL 15: Process Through Analysis Pipeline
print("\n" + "="*80)
print("PROCESSING THROUGH ANALYSIS PIPELINE")
print("="*80)

from delivery_sim.analysis_pipeline.pipeline_coordinator import ExperimentAnalysisPipeline

# Initialize pipeline with delivery_unit_metrics enabled
pipeline = ExperimentAnalysisPipeline(
    warmup_period=uniform_warmup_period,
    enabled_metric_types=['order_metrics', 'system_metrics', 'delivery_unit_metrics'],  # ← ADDED
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
# %% CELL 16: Extract and Present Key Metrics (TABLE FORMAT WITH SEED)
print("\n" + "="*80)
print("KEY PERFORMANCE METRICS EXTRACTION AND PRESENTATION")
print("="*80)

import re

def extract_seed_ratio_and_type(design_name):
    """Extract seed, ratio, and interval type from design point name."""
    # Pattern: area_10km_seed42_ratio_3.5_baseline
    match = re.match(r'.*_seed(\d+)_ratio_([\d.]+)_(baseline|2x_baseline)', design_name)
    if match:
        seed = int(match.group(1))
        ratio = float(match.group(2))
        interval_type = match.group(3)
        return seed, ratio, interval_type
    return None, None, None

# Extract comprehensive metrics
metrics_data = []

for design_name, analysis_result in design_analysis_results.items():
    seed, ratio, interval_type = extract_seed_ratio_and_type(design_name)
    if seed is None:
        continue
    
    stats_with_cis = analysis_result.get('statistics_with_cis', {})
    
    # Extract assignment time (from order_metrics)
    order_metrics = stats_with_cis.get('order_metrics', {})
    assignment_time = order_metrics.get('assignment_time', {})
    
    mean_of_means = assignment_time.get('mean_of_means', {})
    mom_estimate = mean_of_means.get('point_estimate', 0)
    mom_ci = mean_of_means.get('confidence_interval', [0, 0])
    mom_ci_width = (mom_ci[1] - mom_ci[0]) / 2 if mom_ci[0] is not None else 0
    
    std_of_means = assignment_time.get('std_of_means', {})
    som_estimate = std_of_means.get('point_estimate', 0)
    
    mean_of_stds = assignment_time.get('mean_of_stds', {})
    mos_estimate = mean_of_stds.get('point_estimate', 0)
    
    # Extract completion rate (from system_metrics)
    system_metrics = stats_with_cis.get('system_metrics', {})
    completion_rate = system_metrics.get('system_completion_rate', {})
    
    comp_estimate = completion_rate.get('point_estimate', 0)
    comp_ci = completion_rate.get('confidence_interval', [0, 0])
    comp_ci_width = (comp_ci[1] - comp_ci[0]) / 2 if comp_ci[0] is not None else 0
    
    # ==================== NEW: Extract total distance (from delivery_unit_metrics) ====================
    delivery_unit_metrics = stats_with_cis.get('delivery_unit_metrics', {})
    total_distance_stats = delivery_unit_metrics.get('total_distance', {})
    
    distance_mean_of_means = total_distance_stats.get('mean_of_means', {})
    distance_estimate = distance_mean_of_means.get('point_estimate', 0)
    distance_ci = distance_mean_of_means.get('confidence_interval', [0, 0])
    distance_ci_width = (distance_ci[1] - distance_ci[0]) / 2 if distance_ci[0] is not None else 0
    # ================================================================================================
    
    metrics_data.append({
        'seed': seed,
        'ratio': ratio,
        'interval_type': interval_type,
        'mom_estimate': mom_estimate,
        'mom_ci_width': mom_ci_width,
        'som_estimate': som_estimate,
        'mos_estimate': mos_estimate,
        'comp_estimate': comp_estimate,
        'comp_ci_width': comp_ci_width,
        'distance_estimate': distance_estimate,      # NEW
        'distance_ci_width': distance_ci_width,      # NEW
    })

# Sort by ratio, then seed, then interval type
metrics_data.sort(key=lambda x: (x['ratio'], x['seed'], x['interval_type']))

# Print formatted table with seed column AND total distance
print("\n🎯 KEY PERFORMANCE METRICS: LAYOUT SENSITIVITY WITH DISTANCE")
print("="*180)
print("  Seed   Ratio    Interval        Mean of Means     Std of    Mean of           Completion Rate      Mean Total Distance")
print("                      Type    (Assignment Time)      Means       Stds             (with 95% CI)            (with 95% CI)")
print("="*180)

for row in metrics_data:
    seed = row['seed']
    ratio = row['ratio']
    interval_display = "2x Baseline" if row['interval_type'] == '2x_baseline' else "Baseline"
    
    mom_str = f"{row['mom_estimate']:5.2f} ± {row['mom_ci_width']:5.2f}"
    som_str = f"{row['som_estimate']:5.2f}"
    mos_str = f"{row['mos_estimate']:5.2f}"
    comp_str = f"{row['comp_estimate']:.3f} ± {row['comp_ci_width']:.3f}"
    dist_str = f"{row['distance_estimate']:5.2f} ± {row['distance_ci_width']:5.2f}"  # NEW
    
    print(f"  {seed:4d}   {ratio:3.1f}  {interval_display:12s}     {mom_str:>16s}    {som_str:>7s}    {mos_str:>7s}          {comp_str:>15s}          {dist_str:>18s}")

print("="*180)

# Alternative view: Group by ratio, then seed
print("\n\nAlternative view grouped by ratio:")
print("🎯 PERFORMANCE COMPARISON ACROSS LAYOUTS")
print("="*180)

for ratio in sorted(set(row['ratio'] for row in metrics_data)):
    print(f"\nRatio {ratio:.1f} - Comparing layouts:")
    print("-"*180)
    print("  Seed   Interval        Mean of Means     Std of    Mean of           Completion Rate      Mean Total Distance")
    print("             Type    (Assignment Time)      Means       Stds             (with 95% CI)            (with 95% CI)")
    print("-"*180)
    
    ratio_data = [row for row in metrics_data if row['ratio'] == ratio]
    for row in ratio_data:
        seed = row['seed']
        interval_display = "2x Baseline" if row['interval_type'] == '2x_baseline' else "Baseline"
        
        mom_str = f"{row['mom_estimate']:5.2f} ± {row['mom_ci_width']:5.2f}"
        som_str = f"{row['som_estimate']:5.2f}"
        mos_str = f"{row['mos_estimate']:5.2f}"
        comp_str = f"{row['comp_estimate']:.3f} ± {row['comp_ci_width']:.3f}"
        dist_str = f"{row['distance_estimate']:5.2f} ± {row['distance_ci_width']:5.2f}"
        
        print(f"  {seed:4d}  {interval_display:12s}     {mom_str:>16s}    {som_str:>7s}    {mos_str:>7s}          {comp_str:>15s}          {dist_str:>18s}")

print("\n" + "="*180)

# Alternative view: Group by seed
print("\n\nAlternative view grouped by seed:")
print("🎯 LAYOUT-SPECIFIC PERFORMANCE")
print("="*180)

for seed in sorted(set(row['seed'] for row in metrics_data)):
    # Get typical distance for this seed
    instance_name = f"area_10km_seed{seed}"
    typical_dist = next(i['analysis']['typical_distance'] for i in infrastructure_instances if i['name'] == instance_name)
    
    print(f"\nSeed {seed} (typical_distance={typical_dist:.3f}km):")
    print("-"*180)
    print(" Ratio    Interval        Mean of Means     Std of    Mean of           Completion Rate      Mean Total Distance")
    print("              Type    (Assignment Time)      Means       Stds             (with 95% CI)            (with 95% CI)")
    print("-"*180)
    
    seed_data = [row for row in metrics_data if row['seed'] == seed]
    for row in seed_data:
        ratio = row['ratio']
        interval_display = "2x Baseline" if row['interval_type'] == '2x_baseline' else "Baseline"
        
        mom_str = f"{row['mom_estimate']:5.2f} ± {row['mom_ci_width']:5.2f}"
        som_str = f"{row['som_estimate']:5.2f}"
        mos_str = f"{row['mos_estimate']:5.2f}"
        comp_str = f"{row['comp_estimate']:.3f} ± {row['comp_ci_width']:.3f}"
        dist_str = f"{row['distance_estimate']:5.2f} ± {row['distance_ci_width']:5.2f}"
        
        print(f"  {ratio:3.1f}  {interval_display:12s}     {mom_str:>16s}    {som_str:>7s}    {mos_str:>7s}          {comp_str:>15s}          {dist_str:>18s}")

print("\n" + "="*180)
print("✓ Metric extraction complete with layout sensitivity analysis and distance data")


