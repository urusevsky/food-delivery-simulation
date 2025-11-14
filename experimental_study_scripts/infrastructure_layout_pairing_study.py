# infrastructure_layout_pairing_study.py
"""
Infrastructure Layout × Pairing Interaction Study

Research Question: Does infrastructure layout sensitivity persist when pairing 
is enabled, or does pairing's capacity boost mask/mitigate the spatial disadvantage 
of clustered layouts?

Test: Multiple infrastructure seeds (seeds 42, 100, 200) with pairing enabled vs disabled

Design: 3 seeds × 2 ratios (5.0, 7.0) × 2 pairing configs × 2 scales = 24 design points
Total runs: 24 × 5 replications = 120 runs (~8-10 minutes)
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
print("INFRASTRUCTURE LAYOUT × PAIRING INTERACTION STUDY")
print("="*80)
print("Research Question: Does pairing mitigate layout sensitivity?")

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
Does infrastructure layout sensitivity persist when pairing is enabled, or does 
pairing's capacity boost mask/mitigate the spatial disadvantage of clustered layouts?

Building on infrastructure_layout_study.py findings:
- Seed 200 (clustered) showed 81% worse performance at ratio 5.0 (single delivery)
- Mechanism: clustered layout → longer distances → fewer idle drivers → lower capacity
- Question: Does pairing change this dynamic?

Test: Three layouts (seeds 42, 100, 200) × Pairing (enabled vs disabled)
Focus ratios: 5.0 (volatile), 7.0 (failure) - where pairing matters most
"""

context = """
Previous findings established that:
1. Infrastructure layout affects performance through idle capacity mechanism
2. Clustered layouts (seed 200) create spatial disadvantages (~9% longer distance → 81% worse performance)
3. Pairing provides dramatic capacity improvements (72% at ratio 5.0, changes regime at 7.0)

Open question: How do these two effects interact?
- Clustered layouts might benefit MORE from pairing (more pairing opportunities)
- OR clustered layouts might benefit LESS from pairing (longer paired routes)
- OR pairing might equalize performance across layouts entirely
"""

sub_questions = """
1. Does layout sensitivity magnitude change with pairing?
   - No pairing: 81% performance difference between layouts at ratio 5.0
   - With pairing: Does this gap widen, narrow, or disappear?

2. Do different layouts have different pairing rates?
   - Clustered layouts: More pairing opportunities (closer restaurants)?
   - Dispersed layouts: Fewer pairing opportunities but shorter routes?

3. Does pairing change which layout is "best"?
   - Possible reversal: clustered layouts might excel with pairing
   - Need empirical evidence to determine actual interaction effect
"""

scope = """
INCLUDED in this study:
- Same 3 layouts from previous study (seeds 42, 100, 200)
- Ratios 5.0 and 7.0 only (where pairing effects are strong)
- Pairing enabled vs disabled comparison
- Validation pairs (baseline + 2× baseline) for each condition

NOT included:
- Ratio 3.5 (pairing has minimal effect in stable regime)
- Different pairing thresholds (use standard 4km restaurant, 3km customer)
- Statistical hypothesis testing (exploratory/descriptive focus)
"""

analysis_focus = """
Key metrics:
1. Assignment time: Performance comparison across layouts × pairing
2. Pairing rate: Mechanism (do layouts differ in pairing opportunity?)
3. Available drivers: Capacity mechanism check
4. Completion rate: System stability

Analysis approach:
- Compare layout differences WITH vs WITHOUT pairing
- Examine if pairing rate correlates with layout characteristics
- Check if capacity mechanism (idle drivers) still explains differences
- Identify interaction patterns: Does pairing amplify or reduce layout effects?
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
PAIRING INTERACTION STUDY: Test layout sensitivity with and without pairing.

Focus on ratios 5.0 and 7.0 where pairing effects are strongest.
For each ratio, create 4 configurations:
- No pairing × baseline
- No pairing × 2× baseline
- Pairing enabled × baseline
- Pairing enabled × 2× baseline
"""

# Critical ratios where pairing matters most
target_arrival_interval_ratios = [5.0, 7.0]

# Pairing configurations
pairing_params = {
    'pairing_enabled': True,
    'restaurants_proximity_threshold': 4.0,
    'customers_proximity_threshold': 3.0,
}

no_pairing_params = {
    'pairing_enabled': False,
    'restaurants_proximity_threshold': None,
    'customers_proximity_threshold': None,
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
    for pairing_config, pairing_name in [(no_pairing_params, 'no_pairing'), (pairing_params, 'pairing')]:
        # Baseline configuration: (1.0, ratio)
        operational_configs.append({
            'name': f'ratio_{ratio:.1f}_{pairing_name}_baseline',
            'config': OperationalConfig(
                mean_order_inter_arrival_time=1.0,
                mean_driver_inter_arrival_time=ratio,
                **pairing_config,
                **FIXED_SERVICE_CONFIG
            )
        })
        
        # 2x Baseline configuration: (2.0, 2×ratio)
        operational_configs.append({
            'name': f'ratio_{ratio:.1f}_{pairing_name}_2x_baseline',
            'config': OperationalConfig(
                mean_order_inter_arrival_time=2.0,
                mean_driver_inter_arrival_time=2.0 * ratio,
                **pairing_config,
                **FIXED_SERVICE_CONFIG
            )
        })

print(f"✓ Defined {len(operational_configs)} operational configurations")
print(f"✓ Testing {len(target_arrival_interval_ratios)} critical arrival interval ratios")
print(f"✓ Each ratio has 2 pairing configs × 2 validation pairs = 4 configs")

# Display configurations
print("\nConfiguration breakdown:")
for config in operational_configs:
    op_config = config['config']
    ratio = op_config.mean_driver_inter_arrival_time / op_config.mean_order_inter_arrival_time
    pairing_status = "PAIRING" if op_config.pairing_enabled else "NO_PAIRING"
    print(f"  • {config['name']}: "
          f"ratio={ratio:.1f}, {pairing_status}, "
          f"order_interval={op_config.mean_order_inter_arrival_time:.1f}min, "
          f"driver_interval={op_config.mean_driver_inter_arrival_time:.1f}min")

# %% CELL 9: Design Point Creation
"""
Create design points from all combinations.

Design: 3 seeds × 2 ratios × 2 pairing configs × 2 scales = 24 design points
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

print(f"\n✅ INFRASTRUCTURE LAYOUT × PAIRING STUDY COMPLETE!")
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
    metrics=['active_drivers', 'available_drivers', 'unassigned_delivery_entities'],
    moving_average_window=100
)

print(f"✓ Time series processing complete for {len(all_time_series_data)} design points")
print(f"✓ Metrics extracted: active_drivers, available_drivers, unassigned_delivery_entities")
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
    # Extract seed from design name (e.g., "area_10km_seed42_ratio_5.0_pairing_baseline")
    parts = design_name.split('_')
    seed_idx = parts.index('seed') if 'seed' in parts else -1
    if seed_idx >= 0 and seed_idx + 1 < len(parts):
        seed = parts[seed_idx + 1]
        if seed not in seed_groups:
            seed_groups[seed] = []
        seed_groups[seed].append(design_name)

# Create plots grouped by seed
for seed, design_names in sorted(seed_groups.items()):
    print(f"\nCreating warmup plots for seed {seed}...")
    
    # Prepare data for this seed
    seed_time_series = {name: all_time_series_data[name] for name in design_names}
    
    # Create combined plot
    fig = viz.create_warmup_comparison_plot(
        time_series_data=seed_time_series,
        title=f"Warmup Analysis - Seed {seed}",
        metrics=['active_drivers', 'available_drivers', 'unassigned_delivery_entities']
    )
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Warmup plots created for seed {seed}")

print(f"\n{'='*50}")
print("✓ All warmup visualizations complete")
print("✓ Review plots to determine appropriate warmup period")
print(f"{'='*50}")

# %% CELL 14: Warmup Period Determination
"""
Based on visual inspection of warmup plots, set uniform warmup period.

Update the value below after inspecting Cell 13 plots.
"""

print("\n" + "="*50)
print("WARMUP PERIOD DETERMINATION")
print("="*50)

# ⚠️ UPDATE THIS VALUE based on visual inspection of Cell 13
uniform_warmup_period = 500  # minutes

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
    enabled_metric_types=['order_metrics', 'system_metrics', 'delivery_unit_metrics', 'system_state_metrics'],
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
# %% CELL 16: Extract and Present Key Metrics (TABLE FORMAT WITH PAIRING)
print("\n" + "="*80)
print("KEY PERFORMANCE METRICS EXTRACTION AND PRESENTATION")
print("="*80)

import re

def extract_seed_ratio_pairing_and_type(design_name):
    """Extract seed, ratio, pairing status, and interval type from design point name."""
    # Pattern: area_10km_seed42_ratio_5.0_pairing_baseline or area_10km_seed42_ratio_5.0_no_pairing_baseline
    match = re.match(r'.*_seed(\d+)_ratio_([\d.]+)_(pairing|no_pairing)_(baseline|2x_baseline)', design_name)
    if match:
        seed = int(match.group(1))
        ratio = float(match.group(2))
        pairing_status = match.group(3)
        interval_type = match.group(4)
        return seed, ratio, pairing_status, interval_type
    return None, None, None, None

# Extract comprehensive metrics including pairing rate
metrics_data = []

for design_name, analysis_result in design_analysis_results.items():
    seed, ratio, pairing_status, interval_type = extract_seed_ratio_pairing_and_type(design_name)
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
    
    # Extract pairing rate (from system_metrics) - NOTE: metric name is 'system_pairing_rate'
    pairing_rate_stats = system_metrics.get('system_pairing_rate', {})
    pairing_rate_estimate = pairing_rate_stats.get('point_estimate', 0)
    pairing_rate_ci = pairing_rate_stats.get('confidence_interval', [0, 0])
    pairing_rate_ci_width = (pairing_rate_ci[1] - pairing_rate_ci[0]) / 2 if pairing_rate_ci[0] is not None else 0
    
    # Extract total distance (from delivery_unit_metrics)
    delivery_unit_metrics = stats_with_cis.get('delivery_unit_metrics', {})
    total_distance_stats = delivery_unit_metrics.get('total_distance', {})
    
    distance_mean_of_means = total_distance_stats.get('mean_of_means', {})
    distance_estimate = distance_mean_of_means.get('point_estimate', 0)
    distance_ci = distance_mean_of_means.get('confidence_interval', [0, 0])
    distance_ci_width = (distance_ci[1] - distance_ci[0]) / 2 if distance_ci[0] is not None else 0
    
    # Extract available_drivers (from system_state_metrics)
    state_metrics = stats_with_cis.get('system_state_metrics', {})
    available_drivers_stats = state_metrics.get('available_drivers', {})
    
    avail_mean_of_means = available_drivers_stats.get('mean_of_means', {})
    avail_mom_estimate = avail_mean_of_means.get('point_estimate', 0)
    avail_mom_ci = avail_mean_of_means.get('confidence_interval', [0, 0])
    avail_mom_ci_width = (avail_mom_ci[1] - avail_mom_ci[0]) / 2 if avail_mom_ci[0] is not None else 0
    
    avail_std_of_means = available_drivers_stats.get('std_of_means', {})
    avail_som_estimate = avail_std_of_means.get('point_estimate', 0)
    
    avail_mean_of_stds = available_drivers_stats.get('mean_of_stds', {})
    avail_mos_estimate = avail_mean_of_stds.get('point_estimate', 0)
    
    metrics_data.append({
        'seed': seed,
        'ratio': ratio,
        'pairing_status': pairing_status,
        'interval_type': interval_type,
        'mom_estimate': mom_estimate,
        'mom_ci_width': mom_ci_width,
        'som_estimate': som_estimate,
        'mos_estimate': mos_estimate,
        'comp_estimate': comp_estimate,
        'comp_ci_width': comp_ci_width,
        'pairing_rate_estimate': pairing_rate_estimate,
        'pairing_rate_ci_width': pairing_rate_ci_width,
        'distance_estimate': distance_estimate,
        'distance_ci_width': distance_ci_width,
        'avail_mom_estimate': avail_mom_estimate,
        'avail_mom_ci_width': avail_mom_ci_width,
        'avail_som_estimate': avail_som_estimate,
        'avail_mos_estimate': avail_mos_estimate,
    })

# Sort by ratio, seed, pairing status, then interval type
# This groups results by seed so no_pairing and pairing are juxtaposed for easy comparison
metrics_data.sort(key=lambda x: (x['ratio'], x['seed'], x['pairing_status'], x['interval_type']))

# ========================================
# VIEW 1: Complete table with pairing
# ========================================
print("\n🎯 KEY PERFORMANCE METRICS: LAYOUT × PAIRING INTERACTION")
print("="*240)
print("  Seed   Ratio   Pairing      Interval        Mean of Means     Std of    Mean of           Completion Rate      Pairing Rate         Mean Total Distance      Available Drivers (mean_of_means)    Avail     Avail")
print("                 Status           Type    (Assignment Time)      Means       Stds             (with 95% CI)        (with 95% CI)            (with 95% CI)                  (with 95% CI)          Std of   Mean of")
print("                                                                                                                                                                                                     Means     Stds")
print("="*240)

for row in metrics_data:
    seed = row['seed']
    ratio = row['ratio']
    pairing_display = "Pairing" if row['pairing_status'] == 'pairing' else "No Pairing"
    interval_display = "2x Baseline" if row['interval_type'] == '2x_baseline' else "Baseline"
    
    mom_str = f"{row['mom_estimate']:5.2f} ± {row['mom_ci_width']:5.2f}"
    som_str = f"{row['som_estimate']:5.2f}"
    mos_str = f"{row['mos_estimate']:5.2f}"
    comp_str = f"{row['comp_estimate']:.3f} ± {row['comp_ci_width']:.3f}"
    pairing_rate_str = f"{row['pairing_rate_estimate']*100:4.1f}% ± {row['pairing_rate_ci_width']*100:4.1f}%"
    dist_str = f"{row['distance_estimate']:5.2f} ± {row['distance_ci_width']:5.2f}"
    avail_mom_str = f"{row['avail_mom_estimate']:5.2f} ± {row['avail_mom_ci_width']:5.2f}"
    avail_som_str = f"{row['avail_som_estimate']:5.2f}"
    avail_mos_str = f"{row['avail_mos_estimate']:5.2f}"
    
    print(f"  {seed:4d}   {ratio:3.1f}  {pairing_display:10s}  {interval_display:12s}     {mom_str:>16s}    {som_str:>7s}    {mos_str:>7s}          {comp_str:>15s}          {pairing_rate_str:>18s}          {dist_str:>18s}          {avail_mom_str:>22s}    {avail_som_str:>7s}   {avail_mos_str:>7s}")

print("="*240)

# ========================================
# VIEW 2: Grouped by ratio and pairing
# ========================================
print("\n\nAlternative view - grouped by ratio and pairing status:")
print("🎯 PAIRING EFFECT ON LAYOUT SENSITIVITY")
print("="*240)

for ratio in sorted(set(row['ratio'] for row in metrics_data)):
    for pairing_status in ['no_pairing', 'pairing']:
        pairing_display = "WITH PAIRING" if pairing_status == 'pairing' else "NO PAIRING"
        print(f"\nRatio {ratio:.1f} - {pairing_display}:")
        print("-"*240)
        print("  Seed   Interval        Mean of Means     Std of    Mean of           Completion Rate      Pairing Rate         Mean Total Distance      Available Drivers (mean_of_means)    Avail     Avail")
        print("             Type    (Assignment Time)      Means       Stds             (with 95% CI)        (with 95% CI)            (with 95% CI)                  (with 95% CI)          Std of   Mean of")
        print("                                                                                                                                                                                     Means     Stds")
        print("-"*240)
        
        filtered_data = [row for row in metrics_data if row['ratio'] == ratio and row['pairing_status'] == pairing_status]
        for row in filtered_data:
            seed = row['seed']
            interval_display = "2x Baseline" if row['interval_type'] == '2x_baseline' else "Baseline"
            
            mom_str = f"{row['mom_estimate']:5.2f} ± {row['mom_ci_width']:5.2f}"
            som_str = f"{row['som_estimate']:5.2f}"
            mos_str = f"{row['mos_estimate']:5.2f}"
            comp_str = f"{row['comp_estimate']:.3f} ± {row['comp_ci_width']:.3f}"
            pairing_rate_str = f"{row['pairing_rate_estimate']*100:4.1f}% ± {row['pairing_rate_ci_width']*100:4.1f}%"
            dist_str = f"{row['distance_estimate']:5.2f} ± {row['distance_ci_width']:5.2f}"
            avail_mom_str = f"{row['avail_mom_estimate']:5.2f} ± {row['avail_mom_ci_width']:5.2f}"
            avail_som_str = f"{row['avail_som_estimate']:5.2f}"
            avail_mos_str = f"{row['avail_mos_estimate']:5.2f}"
            
            print(f"  {seed:4d}  {interval_display:12s}     {mom_str:>16s}    {som_str:>7s}    {mos_str:>7s}          {comp_str:>15s}          {pairing_rate_str:>18s}          {dist_str:>18s}          {avail_mom_str:>22s}    {avail_som_str:>7s}   {avail_mos_str:>7s}")

print("\n" + "="*240)
print("✓ Metric extraction complete with layout × pairing interaction analysis")
print("✓ Compare pairing rates and performance across layouts to understand interaction effects")


# %%