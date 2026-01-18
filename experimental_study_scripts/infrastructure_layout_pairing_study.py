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
    seed_str = parts[2].replace('seed', '')  # Get "seed42" from index 2, extract "42"
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

# Initialize pipeline with delivery_unit_metrics, pair_metrics, and queue_dynamics_metrics enabled
pipeline = ExperimentAnalysisPipeline(
    warmup_period=uniform_warmup_period,
    enabled_metric_types=['order_metrics', 'system_metrics', 'delivery_unit_metrics', 
                         'system_state_metrics', 'queue_dynamics_metrics'],
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

# Extract comprehensive metrics
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
    
    # Extract pairing rate (from system_metrics) - NOTE: metric name is 'system_pairing_rate'
    system_metrics = stats_with_cis.get('system_metrics', {})
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
    
    # Extract growth rate (from queue_dynamics_metrics)
    queue_dynamics_metrics = stats_with_cis.get('queue_dynamics_metrics', {})
    growth_rate_metric = queue_dynamics_metrics.get('unassigned_entities_growth_rate', {})
    
    growth_rate_estimate = growth_rate_metric.get('point_estimate', 0)
    growth_rate_ci = growth_rate_metric.get('confidence_interval', [0, 0])
    growth_rate_ci_width = (growth_rate_ci[1] - growth_rate_ci[0]) / 2 if growth_rate_ci[0] is not None else 0
    
    metrics_data.append({
        'seed': seed,
        'ratio': ratio,
        'pairing_status': pairing_status,
        'interval_type': interval_type,
        'mom_estimate': mom_estimate,
        'mom_ci_width': mom_ci_width,
        'som_estimate': som_estimate,
        'mos_estimate': mos_estimate,
        'pairing_rate_estimate': pairing_rate_estimate,
        'pairing_rate_ci_width': pairing_rate_ci_width,
        'distance_estimate': distance_estimate,
        'distance_ci_width': distance_ci_width,
        'growth_rate_estimate': growth_rate_estimate,
        'growth_rate_ci_width': growth_rate_ci_width,
    })

# Sort by seed, ratio, pairing_status, then interval type
metrics_data.sort(key=lambda x: (x['seed'], x['ratio'], x['pairing_status'], x['interval_type']))

# =========================================================================
# PRINT FORMATTED TABLE - Main View
# =========================================================================
print("\n🎯 KEY PERFORMANCE METRICS: LAYOUT × PAIRING INTERACTION")
print("="*200)
print("  Seed   Ratio    Pairing      Interval        Mean of Means     Std of    Mean of         Pairing Rate         Mean Total Distance       Growth Rate")
print("                   Status          Type    (Assignment Time)      Means       Stds         (with 95% CI)            (with 95% CI)         (entities/min)")
print("="*200)

for row in metrics_data:
    seed = row['seed']
    ratio = row['ratio']
    pairing_display = "WITH PAIRING" if row['pairing_status'] == 'pairing' else "NO PAIRING"
    interval_display = "2x Baseline" if row['interval_type'] == '2x_baseline' else "Baseline"
    
    mom_str = f"{row['mom_estimate']:5.2f} ± {row['mom_ci_width']:5.2f}"
    som_str = f"{row['som_estimate']:5.2f}"
    mos_str = f"{row['mos_estimate']:5.2f}"
    pairing_rate_str = f"{row['pairing_rate_estimate']*100:4.1f}% ± {row['pairing_rate_ci_width']*100:4.1f}%"
    dist_str = f"{row['distance_estimate']:5.2f} ± {row['distance_ci_width']:5.2f}"
    growth_rate_str = f"{row['growth_rate_estimate']:7.4f} ± {row['growth_rate_ci_width']:7.4f}"
    
    print(f"  {seed:4d}   {ratio:3.1f}  {pairing_display:12s}  {interval_display:12s}     {mom_str:>16s}    {som_str:>7s}    {mos_str:>7s}          {pairing_rate_str:>18s}          {dist_str:>18s}          {growth_rate_str:>21s}")

print("="*200)

# ========================================
# VIEW 2: Grouped by ratio and pairing
# ========================================
print("\n\nAlternative view - grouped by ratio and pairing status:")
print("🎯 PAIRING EFFECT ON LAYOUT SENSITIVITY")
print("="*200)

for ratio in sorted(set(row['ratio'] for row in metrics_data)):
    for pairing_status in ['no_pairing', 'pairing']:
        pairing_display = "WITH PAIRING" if pairing_status == 'pairing' else "NO PAIRING"
        print(f"\nRatio {ratio:.1f} - {pairing_display}:")
        print("-"*200)
        print("  Seed   Interval        Mean of Means     Std of    Mean of         Pairing Rate         Mean Total Distance       Growth Rate")
        print("             Type    (Assignment Time)      Means       Stds         (with 95% CI)            (with 95% CI)         (entities/min)")
        print("-"*200)
        
        filtered_data = [row for row in metrics_data if row['ratio'] == ratio and row['pairing_status'] == pairing_status]
        for row in filtered_data:
            seed = row['seed']
            interval_display = "2x Baseline" if row['interval_type'] == '2x_baseline' else "Baseline"
            
            mom_str = f"{row['mom_estimate']:5.2f} ± {row['mom_ci_width']:5.2f}"
            som_str = f"{row['som_estimate']:5.2f}"
            mos_str = f"{row['mos_estimate']:5.2f}"
            pairing_rate_str = f"{row['pairing_rate_estimate']*100:4.1f}% ± {row['pairing_rate_ci_width']*100:4.1f}%"
            dist_str = f"{row['distance_estimate']:5.2f} ± {row['distance_ci_width']:5.2f}"
            growth_rate_str = f"{row['growth_rate_estimate']:7.4f} ± {row['growth_rate_ci_width']:7.4f}"
            
            print(f"  {seed:4d}  {interval_display:12s}     {mom_str:>16s}    {som_str:>7s}    {mos_str:>7s}          {pairing_rate_str:>18s}          {dist_str:>18s}          {growth_rate_str:>21s}")

# ========================================
# VIEW 3: Grouped by seed
# ========================================
print("\n\nAlternative view - grouped by seed:")
print("🎯 LAYOUT-SPECIFIC PERFORMANCE ACROSS PAIRING CONDITIONS")
print("="*200)

for seed in sorted(set(row['seed'] for row in metrics_data)):
    # Get typical distance for this seed
    instance_name = f"area_10km_seed{seed}"
    typical_dist = next(i['analysis']['typical_distance'] for i in infrastructure_instances if i['name'] == instance_name)
    
    print(f"\nSeed {seed} (typical_distance={typical_dist:.3f}km):")
    print("-"*200)
    print(" Ratio    Pairing      Interval        Mean of Means     Std of    Mean of         Pairing Rate         Mean Total Distance       Growth Rate")
    print("           Status          Type    (Assignment Time)      Means       Stds         (with 95% CI)            (with 95% CI)         (entities/min)")
    print("-"*200)
    
    seed_data = [row for row in metrics_data if row['seed'] == seed]
    for row in seed_data:
        ratio = row['ratio']
        pairing_display = "WITH PAIRING" if row['pairing_status'] == 'pairing' else "NO PAIRING"
        interval_display = "2x Baseline" if row['interval_type'] == '2x_baseline' else "Baseline"
        
        mom_str = f"{row['mom_estimate']:5.2f} ± {row['mom_ci_width']:5.2f}"
        som_str = f"{row['som_estimate']:5.2f}"
        mos_str = f"{row['mos_estimate']:5.2f}"
        pairing_rate_str = f"{row['pairing_rate_estimate']*100:4.1f}% ± {row['pairing_rate_ci_width']*100:4.1f}%"
        dist_str = f"{row['distance_estimate']:5.2f} ± {row['distance_ci_width']:5.2f}"
        growth_rate_str = f"{row['growth_rate_estimate']:7.4f} ± {row['growth_rate_ci_width']:7.4f}"
        
        print(f"  {ratio:3.1f}  {pairing_display:12s}  {interval_display:12s}     {mom_str:>16s}    {som_str:>7s}    {mos_str:>7s}          {pairing_rate_str:>18s}          {dist_str:>18s}          {growth_rate_str:>21s}")

print("\n" + "="*200)
print("✓ Metric extraction complete with pairing interaction analysis")

# %% CELL 17: [AD HOC] Single Replication Time Series - Pairing Rescue Effect
"""
EXPLORATORY ANALYSIS: Visualize how pairing "rescues" the system from failure.

Focus: Ratio 5.0, Seed 200 (worst-performing layout without pairing)
Compare: No Pairing vs Pairing enabled, using Replication 1

Expected patterns:
- No pairing: High queue levels, few idle drivers, system stressed
- With pairing: Lower queue levels, more idle drivers, system recovered

This shows the "rescue" dynamics in real action - single replication trajectories
are much more revealing than cross-replication averages.
"""

print("\n" + "="*80)
print("AD HOC ANALYSIS: PAIRING RESCUE EFFECT (SINGLE REPLICATION)")
print("="*80)

from delivery_sim.visualization.time_series_plots import TimeSeriesVisualizer
import matplotlib.pyplot as plt

# Initialize visualizer
viz = TimeSeriesVisualizer(figsize=(16, 14))

# ========================================
# Setup: Compare pairing vs no pairing at ratio 5.0, seed 200
# ========================================
print("\nFocus: Ratio 5.0, Seed 200 (Baseline interval)")
print("Comparison: No Pairing vs Pairing Enabled")
print("Replication: 1 (single trajectory for clarity)")

design_no_pairing = 'area_10km_seed200_ratio_5.0_no_pairing_baseline'
design_with_pairing = 'area_10km_seed200_ratio_5.0_pairing_baseline'

# Get replication 1 data
rep_no_pairing = study_results[design_no_pairing][0]['system_snapshots']
rep_with_pairing = study_results[design_with_pairing][0]['system_snapshots']

print(f"\nNo Pairing design: {design_no_pairing}")
print(f"  • Total snapshots: {len(rep_no_pairing)}")
print(f"\nWith Pairing design: {design_with_pairing}")
print(f"  • Total snapshots: {len(rep_with_pairing)}")

# ========================================
# Part 1: Side-by-Side Enhanced Stacked View
# ========================================
print("\n" + "-"*80)
print("PART 1: SIDE-BY-SIDE COMPARISON - ENHANCED STACKED VIEW")
print("-"*80)

print("\nCreating color-coded comparison plots...")
print("Order: Active Drivers (top) → Available Drivers (middle) → Queue (bottom)")

# Define colors for each metric
colors = {
    'active_drivers': '#2E86AB',          # Blue - working drivers
    'available_drivers': '#06A77D',       # Green - idle capacity
    'unassigned_delivery_entities': '#D62828'  # Red - queue pressure
}

# Metric order (top to bottom)
metrics_ordered = ['active_drivers', 'available_drivers', 'unassigned_delivery_entities']
metric_labels = {
    'active_drivers': 'Active Drivers',
    'available_drivers': 'Available Drivers',
    'unassigned_delivery_entities': 'Unassigned Delivery Entities'
}

def plot_enhanced_stacked_view(snapshots, title):
    """Create enhanced stacked view with custom colors and ordering."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    timestamps = [s['timestamp'] for s in snapshots]
    
    for idx, metric in enumerate(metrics_ordered):
        values = [s[metric] for s in snapshots]
        
        # Clean line plot - no fill for discrete state snapshots
        axes[idx].plot(timestamps, values, 
                      color=colors[metric], 
                      linewidth=1.5, 
                      alpha=0.85)
        axes[idx].set_ylabel(metric_labels[metric], 
                           fontsize=12, 
                           fontweight='bold',
                           color=colors[metric])
        axes[idx].tick_params(axis='y', labelcolor=colors[metric])
        axes[idx].grid(True, alpha=0.3)
        
        # Add horizontal line at y=0 for reference
        axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    axes[-1].set_xlabel('Simulation Time (minutes)', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, axes

# Plot 1: No Pairing
print("\n1. NO PAIRING - Seed 200, Ratio 5.0, Baseline")
fig1, axes1 = plot_enhanced_stacked_view(
    rep_no_pairing,
    'NO PAIRING: Seed 200, Ratio 5.0 - Replication 1 (Stressed System)'
)
plt.show()

# Plot 2: With Pairing
print("\n2. WITH PAIRING - Seed 200, Ratio 5.0, Baseline")
fig2, axes2 = plot_enhanced_stacked_view(
    rep_with_pairing,
    'WITH PAIRING: Seed 200, Ratio 5.0 - Replication 1 (Rescued System)'
)
plt.show()

print("\n✓ Enhanced stacked view plots created")
print("\nColor Coding:")
print("  • Blue (top):   Active Drivers - system workload")
print("  • Green (middle): Available Drivers - idle capacity buffer")
print("  • Red (bottom):  Unassigned Entities - queue pressure")
print("\nKey Observations to Look For:")
print("  • Green panel: Higher peaks and more frequent idle periods with pairing")
print("  • Red panel: Much lower queue levels with pairing")
print("  • Blue panel: Similar active driver levels but different system health")

# ========================================
# Part 2: Direct Overlay Comparison
# ========================================
print("\n" + "-"*80)
print("PART 2: DIRECT OVERLAY - QUANTIFYING THE RESCUE EFFECT")
print("-"*80)

print("\nCreating 3-panel direct overlay comparison...")
print("  Panel 1: Active drivers (similar between conditions)")
print("  Panel 2: Available drivers (pairing creates buffer)")
print("  Panel 3: Queue length (pairing reduces pressure)")

fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

# Extract data
timestamps_no_pair = [s['timestamp'] for s in rep_no_pairing]
active_no_pair = [s['active_drivers'] for s in rep_no_pairing]
available_no_pair = [s['available_drivers'] for s in rep_no_pairing]
queue_no_pair = [s['unassigned_delivery_entities'] for s in rep_no_pairing]

timestamps_pair = [s['timestamp'] for s in rep_with_pairing]
active_pair = [s['active_drivers'] for s in rep_with_pairing]
available_pair = [s['available_drivers'] for s in rep_with_pairing]
queue_pair = [s['unassigned_delivery_entities'] for s in rep_with_pairing]

# Panel 1: Active Drivers Comparison
axes[0].plot(timestamps_no_pair, active_no_pair, 
            color='#D62828', linewidth=1.5, alpha=0.7,
            label='No Pairing (Stressed)')
axes[0].plot(timestamps_pair, active_pair, 
            color='#06A77D', linewidth=2.0, alpha=0.85,
            label='With Pairing (Rescued)')
axes[0].set_ylabel('Active Drivers', fontsize=12, fontweight='bold')
axes[0].legend(loc='upper right', fontsize=11, framealpha=0.9)
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Active Drivers: Similar Workload Despite Different Outcomes', 
                  fontsize=13, fontweight='bold')

# Panel 2: Available Drivers Comparison
axes[1].plot(timestamps_no_pair, available_no_pair, 
            color='#D62828', linewidth=1.5, alpha=0.7,
            label='No Pairing (Stressed)')
axes[1].plot(timestamps_pair, available_pair, 
            color='#06A77D', linewidth=2.0, alpha=0.85,
            label='With Pairing (Rescued)')
axes[1].set_ylabel('Available Drivers', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=11, framealpha=0.9)
axes[1].grid(True, alpha=0.3)
axes[1].set_title('Available Drivers: Pairing Creates Idle Capacity Buffer', 
                  fontsize=13, fontweight='bold')

# Panel 3: Queue Comparison
axes[2].plot(timestamps_no_pair, queue_no_pair, 
            color='#D62828', linewidth=1.5, alpha=0.7,
            label='No Pairing (Stressed)')
axes[2].plot(timestamps_pair, queue_pair, 
            color='#06A77D', linewidth=2.0, alpha=0.85,
            label='With Pairing (Rescued)')
axes[2].set_ylabel('Unassigned Delivery Entities', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Simulation Time (minutes)', fontsize=12)
axes[2].legend(loc='upper right', fontsize=11, framealpha=0.9)
axes[2].grid(True, alpha=0.3)
axes[2].set_title('Queue Dynamics: Pairing Reduces Queue Pressure', 
                  fontsize=13, fontweight='bold')

fig.suptitle('PAIRING RESCUE EFFECT: Seed 200, Ratio 5.0 (Baseline) - Replication 1', 
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n✓ Direct overlay comparison created")

# ========================================
# Summary Statistics for This Replication
# ========================================
print("\n" + "-"*80)
print("SUMMARY STATISTICS FOR THIS REPLICATION")
print("-"*80)

import numpy as np

# Calculate summary stats
avg_active_no_pair = np.mean(active_no_pair)
avg_active_pair = np.mean(active_pair)
avg_avail_no_pair = np.mean(available_no_pair)
avg_avail_pair = np.mean(available_pair)
avg_queue_no_pair = np.mean(queue_no_pair)
avg_queue_pair = np.mean(queue_pair)

max_queue_no_pair = np.max(queue_no_pair)
max_queue_pair = np.max(queue_pair)

zero_avail_pct_no_pair = 100 * sum(1 for x in available_no_pair if x == 0) / len(available_no_pair)
zero_avail_pct_pair = 100 * sum(1 for x in available_pair if x == 0) / len(available_pair)

print(f"\n{'Metric':<30} {'No Pairing':<15} {'With Pairing':<15} {'Improvement'}")
print("="*75)
print(f"{'Average Active Drivers':<30} {avg_active_no_pair:>10.2f}     {avg_active_pair:>10.2f}     {((avg_active_pair/avg_active_no_pair - 1)*100):>+6.1f}%")
print(f"{'Average Available Drivers':<30} {avg_avail_no_pair:>10.2f}     {avg_avail_pair:>10.2f}     {((avg_avail_pair/avg_avail_no_pair - 1)*100):>+6.1f}%")
print(f"{'Average Queue Length':<30} {avg_queue_no_pair:>10.2f}     {avg_queue_pair:>10.2f}     {((avg_queue_pair/avg_queue_no_pair - 1)*100):>+6.1f}%")
print(f"{'Maximum Queue Length':<30} {max_queue_no_pair:>10.0f}     {max_queue_pair:>10.0f}     {((max_queue_pair/max_queue_no_pair - 1)*100):>+6.1f}%")
print(f"{'% Time Zero Idle Drivers':<30} {zero_avail_pct_no_pair:>9.1f}%     {zero_avail_pct_pair:>9.1f}%     {(zero_avail_pct_pair - zero_avail_pct_no_pair):>+6.1f}pp")
print("="*75)

print("\n✓ Ratio 5.0 analysis complete")
print("\nKey Insight:")
print("  Ratio 5.0 shows IMPROVEMENT - both systems operational, one performs better:")
print("  • Similar active driver levels (same driver supply)")
print("  • Doubles idle capacity (available drivers)")
print("  • Cuts queue length by 70-80%")
print("  • Reduces time spent at zero capacity")

# ========================================
# Part 4: Ratio 7.0 - THE REAL RESCUE EFFECT
# ========================================
print("\n" + "="*80)
print("PART 3: RATIO 7.0 - PAIRING RESCUES FAILING SYSTEM")
print("="*80)

print("\nNow examining ratio 7.0 where TRUE rescue occurs:")
print("  • No pairing: System failure (83% completion, zero idle capacity)")
print("  • With pairing: System viable (97% completion, some buffer exists)")
print("  • This is REGIME CHANGE, not just improvement\n")

design_no_pairing_70 = 'area_10km_seed200_ratio_7.0_no_pairing_baseline'
design_with_pairing_70 = 'area_10km_seed200_ratio_7.0_pairing_baseline'

# Get replication 1 data
rep_no_pairing_70 = study_results[design_no_pairing_70][0]['system_snapshots']
rep_with_pairing_70 = study_results[design_with_pairing_70][0]['system_snapshots']

print(f"No Pairing design: {design_no_pairing_70}")
print(f"  • Total snapshots: {len(rep_no_pairing_70)}")
print(f"\nWith Pairing design: {design_with_pairing_70}")
print(f"  • Total snapshots: {len(rep_with_pairing_70)}")

# ========================================
# Ratio 7.0: Enhanced Stacked Views
# ========================================
print("\n" + "-"*80)
print("VISUALIZATION: SIDE-BY-SIDE COMPARISON AT RATIO 7.0")
print("-"*80)

print("\nCreating color-coded comparison plots for ratio 7.0...")

# Plot 1: No Pairing at Ratio 7.0
print("\n1. NO PAIRING - Seed 200, Ratio 7.0, Baseline (SYSTEM FAILURE)")
fig1_70, axes1_70 = plot_enhanced_stacked_view(
    rep_no_pairing_70,
    'NO PAIRING: Seed 200, Ratio 7.0 - Replication 1 (SYSTEM FAILURE)'
)
plt.show()

# Plot 2: With Pairing at Ratio 7.0
print("\n2. WITH PAIRING - Seed 200, Ratio 7.0, Baseline (SYSTEM RESCUED)")
fig2_70, axes2_70 = plot_enhanced_stacked_view(
    rep_with_pairing_70,
    'WITH PAIRING: Seed 200, Ratio 7.0 - Replication 1 (SYSTEM RESCUED)'
)
plt.show()

print("\n✓ Ratio 7.0 enhanced stacked views created")
print("\nExpected Patterns:")
print("  • Blue (Active): Similar between conditions (same driver supply)")
print("  • Green (Available): No pairing flatlined at ZERO, pairing shows rare spikes")
print("  • Red (Queue): No pairing shows RUNAWAY GROWTH, pairing shows controlled oscillation")

# ========================================
# Ratio 7.0: Direct Overlay Comparison
# ========================================
print("\n" + "-"*80)
print("DIRECT OVERLAY: RATIO 7.0 - FAILURE VS RESCUE")
print("-"*80)

print("\nCreating 3-panel direct overlay for ratio 7.0...")

fig_70, axes_70 = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

# Extract data
timestamps_no_pair_70 = [s['timestamp'] for s in rep_no_pairing_70]
active_no_pair_70 = [s['active_drivers'] for s in rep_no_pairing_70]
available_no_pair_70 = [s['available_drivers'] for s in rep_no_pairing_70]
queue_no_pair_70 = [s['unassigned_delivery_entities'] for s in rep_no_pairing_70]

timestamps_pair_70 = [s['timestamp'] for s in rep_with_pairing_70]
active_pair_70 = [s['active_drivers'] for s in rep_with_pairing_70]
available_pair_70 = [s['available_drivers'] for s in rep_with_pairing_70]
queue_pair_70 = [s['unassigned_delivery_entities'] for s in rep_with_pairing_70]

# Panel 1: Active Drivers Comparison
axes_70[0].plot(timestamps_no_pair_70, active_no_pair_70, 
            color='#D62828', linewidth=1.5, alpha=0.7,
            label='No Pairing (FAILURE)')
axes_70[0].plot(timestamps_pair_70, active_pair_70, 
            color='#06A77D', linewidth=2.0, alpha=0.85,
            label='With Pairing (RESCUED)')
axes_70[0].set_ylabel('Active Drivers', fontsize=12, fontweight='bold')
axes_70[0].legend(loc='upper right', fontsize=11, framealpha=0.9)
axes_70[0].grid(True, alpha=0.3)
axes_70[0].set_title('Active Drivers: Same Supply, Different Outcomes', 
                  fontsize=13, fontweight='bold')

# Panel 2: Available Drivers Comparison
axes_70[1].plot(timestamps_no_pair_70, available_no_pair_70, 
            color='#D62828', linewidth=1.5, alpha=0.7,
            label='No Pairing (FAILURE)')
axes_70[1].plot(timestamps_pair_70, available_pair_70, 
            color='#06A77D', linewidth=2.0, alpha=0.85,
            label='With Pairing (RESCUED)')
axes_70[1].set_ylabel('Available Drivers', fontsize=12, fontweight='bold')
axes_70[1].legend(loc='upper right', fontsize=11, framealpha=0.9)
axes_70[1].grid(True, alpha=0.3)
axes_70[1].set_title('Available Drivers: Zero vs Rare Buffer Spikes', 
                  fontsize=13, fontweight='bold')

# Panel 3: Queue Comparison - THE CRITICAL DIFFERENCE
axes_70[2].plot(timestamps_no_pair_70, queue_no_pair_70, 
            color='#D62828', linewidth=1.5, alpha=0.7,
            label='No Pairing (FAILURE)')
axes_70[2].plot(timestamps_pair_70, queue_pair_70, 
            color='#06A77D', linewidth=2.0, alpha=0.85,
            label='With Pairing (RESCUED)')
axes_70[2].set_ylabel('Unassigned Delivery Entities', fontsize=12, fontweight='bold')
axes_70[2].set_xlabel('Simulation Time (minutes)', fontsize=12)
axes_70[2].legend(loc='upper left', fontsize=11, framealpha=0.9)
axes_70[2].grid(True, alpha=0.3)
axes_70[2].set_title('Queue Dynamics: RUNAWAY GROWTH vs CONTROLLED OSCILLATION', 
                  fontsize=13, fontweight='bold')

fig_70.suptitle('REGIME CHANGE: Seed 200, Ratio 7.0 (Baseline) - Replication 1', 
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n✓ Ratio 7.0 direct overlay created")

# ========================================
# Ratio 7.0: Summary Statistics
# ========================================
print("\n" + "-"*80)
print("SUMMARY STATISTICS: RATIO 7.0 - FAILURE VS RESCUE")
print("-"*80)

import numpy as np

# Calculate summary stats for ratio 7.0
avg_active_no_pair_70 = np.mean(active_no_pair_70)
avg_active_pair_70 = np.mean(active_pair_70)
avg_avail_no_pair_70 = np.mean(available_no_pair_70)
avg_avail_pair_70 = np.mean(available_pair_70)
avg_queue_no_pair_70 = np.mean(queue_no_pair_70)
avg_queue_pair_70 = np.mean(queue_pair_70)

max_queue_no_pair_70 = np.max(queue_no_pair_70)
max_queue_pair_70 = np.max(queue_pair_70)

zero_avail_pct_no_pair_70 = 100 * sum(1 for x in available_no_pair_70 if x == 0) / len(available_no_pair_70)
zero_avail_pct_pair_70 = 100 * sum(1 for x in available_pair_70 if x == 0) / len(available_pair_70)

print(f"\n{'Metric':<30} {'No Pairing':<15} {'With Pairing':<15} {'Change'}")
print("="*75)
print(f"{'Average Active Drivers':<30} {avg_active_no_pair_70:>10.2f}     {avg_active_pair_70:>10.2f}     {((avg_active_pair_70/avg_active_no_pair_70 - 1)*100):>+6.1f}%")
print(f"{'Average Available Drivers':<30} {avg_avail_no_pair_70:>10.2f}     {avg_avail_pair_70:>10.2f}     {'N/A (0→pos)'}")
print(f"{'Average Queue Length':<30} {avg_queue_no_pair_70:>10.2f}     {avg_queue_pair_70:>10.2f}     {((avg_queue_pair_70/avg_queue_no_pair_70 - 1)*100):>+6.1f}%")
print(f"{'Maximum Queue Length':<30} {max_queue_no_pair_70:>10.0f}     {max_queue_pair_70:>10.0f}     {((max_queue_pair_70/max_queue_no_pair_70 - 1)*100):>+6.1f}%")
print(f"{'% Time Zero Idle Drivers':<30} {zero_avail_pct_no_pair_70:>9.1f}%     {zero_avail_pct_pair_70:>9.1f}%     {(zero_avail_pct_pair_70 - zero_avail_pct_no_pair_70):>+6.1f}pp")
print("="*75)

print("\n✓ Ratio 7.0 analysis complete")

# ========================================
# Final Summary: Ratio 5.0 vs 7.0 Comparison
# ========================================
print("\n" + "="*80)
print("FINAL SUMMARY: IMPROVEMENT (5.0) VS RESCUE (7.0)")
print("="*80)

print("\nRATIO 5.0 - IMPROVEMENT EFFECT:")
print("  • No pairing: Stressed but operational (14 avg queue, 97% completion)")
print("  • With pairing: Healthy operation (3 avg queue, 98% completion)")
print("  • Interpretation: Both work, pairing performs 4× better")
print("  • System state: Volatile regime → Stable regime")

print("\nRATIO 7.0 - RESCUE EFFECT (REGIME CHANGE):")
print("  • No pairing: SYSTEM FAILURE (runaway queue growth, 83% completion)")
print("  • With pairing: System viable (controlled oscillation, 97% completion)")
print("  • Interpretation: One fails, one survives")
print("  • System state: Failure regime → Manageable regime")

print("\n" + "="*80)
print("KEY INSIGHT:")
print("="*80)
print("Ratio 5.0 demonstrates pairing's PERFORMANCE BENEFITS")
print("Ratio 7.0 demonstrates pairing's OPERATIONAL NECESSITY")
print("\nPairing isn't just an optimization - at high load ratios, it's the difference")
print("between a functioning system and a failing one. This is the 'rescue effect.'")
print("="*80)

# %%