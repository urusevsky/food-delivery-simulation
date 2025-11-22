# density_study.py
"""
Restaurant Density Effects Study

Research Question: How does restaurant density affect delivery system performance
and operational regime boundaries? Do density effects interact with pairing policies?

Test: Three density levels (0.05, 0.10, 0.15 restaurants/km²) by varying number of
restaurants in fixed 10×10km area, with pairing enabled vs disabled.

Design: 3 densities × 3 seeds × 2 ratios × 2 pairing × 2 scales = 72 design points
Total runs: 72 × 5 replications = 360 runs (~25-30 minutes)
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
print("RESTAURANT DENSITY EFFECTS STUDY")
print("="*80)
print("Research Question: How does density affect performance and regime boundaries?")

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
How does restaurant density affect delivery system performance and operational 
regime boundaries? Do density effects interact with pairing policies?

Building on previous findings:
- Layout study: Spatial structure affects performance through distance/capacity
- Pairing study: Throughput doubling is the kingpin for system viability
- Question: Does systematically varying density reveal design principles?

Test: Three density levels (0.05, 0.10, 0.15 restaurants/km²) by varying 
number of restaurants (5, 10, 15) in fixed 10×10km delivery area.
Focus ratios: 5.0 (volatile), 7.0 (critical) - where pairing effects are strongest.
"""

context = """
Previous studies established:
1. Infrastructure layout matters - clustered layouts (seed 200) showed 81% worse 
   performance at ratio 5.0 without pairing, 28% worse with pairing
2. Pairing's throughput doubling is fundamental - enables system viability at 
   high load ratios (regime change at 7.0)
3. Mechanism: Distance → capacity → performance, with pairing mitigating effects

Open questions about density:
- Higher density intuitively means shorter distances → better performance
- But does this capacity benefit persist with pairing's throughput boost?
- Do higher densities enable higher pairing rates (more nearby restaurants)?
- At what density does performance plateau or show diminishing returns?
"""

sub_questions = """
1. Density main effect:
   - How much does doubling density (0.05 → 0.10) improve performance?
   - Is the relationship linear, logarithmic, or diminishing returns?

2. Density × pairing interaction:
   - Does density benefit persist equally with and without pairing?
   - Do dense layouts pair more effectively (higher pairing rates)?

3. Density × ratio interaction:
   - Do density effects differ between stable (5.0) and critical (7.0) regimes?
   - Does density affect regime boundary locations?

4. Infrastructure design principles:
   - What density provides optimal performance given resource constraints?
   - Is there a "sweet spot" for restaurant density?
"""

scope = """
INCLUDED in this study:
- 3 density levels: 0.05, 0.10, 0.15 restaurants/km²
  (Implemented as 5, 10, 15 restaurants in 10×10km area)
- 3 random layout seeds (42, 100, 200) per density for robustness
- 2 critical ratios (5.0, 7.0) - volatile and failure regimes
- Pairing enabled vs disabled comparison
- Validation pairs (baseline + 2× baseline) for scale effects

NOT included:
- Lower densities (<0.05) - too sparse for practical delivery markets
- Higher densities (>0.15) - resource/time constraints
- More ratios - 5.0 and 7.0 capture key phenomena
- Area scaling - density tested via restaurant count, not area variation

Methodological note:
Seeds (42, 100, 200) are infrastructure replication factors for robustness,
not categorical "pattern types". Any interesting seed-specific effects will
be characterized post-hoc using spatial metrics (typical distance, etc.).
"""

analysis_focus = """
Key metrics:
1. Assignment time: Core performance indicator
2. Completion rate: System viability check
3. Pairing rate: Mechanism (does density enable more pairing?)
4. Available drivers: Capacity buffer indicator
5. Mean total distance: Infrastructure characteristic (explains mechanism)

Analysis approach:
- Main effects: Aggregate across seeds, compare density levels
- Interactions: Density × pairing, density × ratio
- Robustness: Between-seed variation at each density
- Mechanism: Use distance/pairing rate to explain performance differences
- Design principles: Identify optimal density ranges for different operational contexts

Expected patterns:
- Higher density → shorter distances → better performance
- Density benefits may be larger without pairing (distance matters more)
- High density might increase pairing rates (more nearby restaurants)
- Diminishing returns possible at highest densities
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
DENSITY STUDY: Systematically vary restaurant density.

Density = num_restaurants / delivery_area_size²
Test by varying num_restaurants with fixed delivery_area_size = 10km.

Density levels:
- Low (0.05/km²): 5 restaurants in 10×10km
- Medium (0.10/km²): 10 restaurants in 10×10km (baseline from previous studies)
- High (0.15/km²): 15 restaurants in 10×10km
"""

infrastructure_configs = [
    {
        'name': 'density_0.05',
        'config': StructuralConfig(
            delivery_area_size=10,
            num_restaurants=5,
            driver_speed=0.5
        ),
        'density': 0.05
    },
    {
        'name': 'density_0.10',
        'config': StructuralConfig(
            delivery_area_size=10,
            num_restaurants=10,
            driver_speed=0.5
        ),
        'density': 0.10
    },
    {
        'name': 'density_0.15',
        'config': StructuralConfig(
            delivery_area_size=10,
            num_restaurants=15,
            driver_speed=0.5
        ),
        'density': 0.15
    }
]

print(f"✓ Defined {len(infrastructure_configs)} density configurations")
for config in infrastructure_configs:
    struct_config = config['config']
    density = config['density']
    print(f"  • {config['name']}: {struct_config.num_restaurants} restaurants, "
          f"{struct_config.delivery_area_size}×{struct_config.delivery_area_size}km, "
          f"density={density:.3f}/km²")

# %% CELL 5: Structural Seeds
"""
Infrastructure replication: Test robustness across random layout realizations.

Use same seeds (42, 100, 200) as previous studies for continuity and CRN.
"""

structural_seeds = [42, 100, 200]

print(f"\n✓ Structural seeds: {structural_seeds}")
print(f"✓ Testing density effects with {len(structural_seeds)} random layouts per density")
print(f"✓ Seeds provide infrastructure robustness, maintain CRN across conditions")

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
            'density': infra_config['density'],
            'seed': structural_seed
        })
        
        print(f"  ✓ Infrastructure created and analyzed")
        print(f"    • Density: {infra_config['density']:.3f} restaurants/km²")
        print(f"    • Restaurants: {infra_config['config'].num_restaurants}")
        print(f"    • Typical distance: {analysis_results['typical_distance']:.3f}km")
        print(f"    • Restaurant density: {analysis_results['restaurant_density']:.4f}/km²")

print(f"\n{'='*50}")
print(f"✓ Created {len(infrastructure_instances)} infrastructure instance(s)")
print(f"✓ Breakdown: {len(infrastructure_configs)} densities × {len(structural_seeds)} seeds")
print(f"{'='*50}")

# Display infrastructure comparison by density
print("\n📊 Infrastructure Comparison by Density:")
for density in sorted(set(inst['density'] for inst in infrastructure_instances)):
    print(f"\nDensity {density:.3f} restaurants/km²:")
    density_instances = [inst for inst in infrastructure_instances if inst['density'] == density]
    for inst in density_instances:
        print(f"  Seed {inst['seed']}: typical_distance={inst['analysis']['typical_distance']:.3f}km")

# %% CELL 6.5: Visualize Restaurant Layouts
"""
OPTIONAL: Visualize infrastructure instances to understand density differences.

For density study, this shows how spatial coverage changes with restaurant count.
"""

print("\n" + "="*50)
print("INFRASTRUCTURE LAYOUT VISUALIZATION")
print("="*50)

import matplotlib.pyplot as plt

print(f"\nVisualizing {len(infrastructure_instances)} infrastructure instances...")
print("Compare spatial coverage across density levels.\n")

for instance in infrastructure_instances:
    print(f"{'='*50}")
    print(f"Layout: {instance['name']}")
    print(f"Density: {instance['density']:.3f} restaurants/km²")
    print(f"Restaurants: {instance['infrastructure'].structural_config.num_restaurants}")
    print(f"Typical Distance: {instance['analysis']['typical_distance']:.3f}km")
    print(f"{'='*50}")
    
    # Visualize using stored analyzer
    instance['analyzer'].visualize_infrastructure()
    
    # Add custom header
    fig = plt.gcf()
    seed = instance['seed']
    density = instance['density']
    n_restaurants = instance['infrastructure'].structural_config.num_restaurants
    typical_dist = instance['analysis']['typical_distance']
    
    custom_title = (f"Infrastructure Layout: Density {density:.3f}/km² (Seed {seed})\n"
                   f"{n_restaurants} Restaurants | Typical Distance: {typical_dist:.3f}km | "
                   f"Area: 10×10km")
    
    fig.suptitle(custom_title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ {instance['name']} visualized\n")

print(f"{'='*50}")
print("✓ All layouts visualized")
print("✓ Observe spatial coverage patterns:")
print("  - Low density (0.05): Sparse coverage, potentially longer distances")
print("  - Medium density (0.10): Baseline coverage from previous studies")
print("  - High density (0.15): Dense coverage, shorter distances expected")
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
OPERATIONAL STUDY: Focus on critical ratios with pairing comparison.

Ratios 5.0 and 7.0 capture volatile and critical regimes where:
- Density effects likely strongest (high load conditions)
- Pairing benefits are substantial (established in previous study)
- Interaction effects most interesting

For each ratio, create validation pair (baseline + 2× baseline) for both
pairing enabled and disabled conditions.
"""

# Critical ratios where density effects likely matter most
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

Design: 3 densities × 3 seeds × 2 ratios × 2 pairing × 2 scales = 72 design points
"""

design_points = {}

print("\n" + "="*50)
print("DESIGN POINTS CREATION")
print("="*50)

for infra_instance in infrastructure_instances:
    for op_config in operational_configs:
        for scoring_config_dict in scoring_configs:
            
            # Generate design point name including density and seed info
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

print(f"\n✅ DENSITY STUDY COMPLETE!")
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

# Group design points by density for organized display
density_groups = {}
for design_name in all_time_series_data.keys():
    # Extract density from design name (e.g., "density_0.10_seed42_ratio_5.0_pairing_baseline")
    parts = design_name.split('_')
    density_str = parts[1]  # Get "0.10" from "density_0.10"
    
    if density_str not in density_groups:
        density_groups[density_str] = []
    density_groups[density_str].append(design_name)

print(f"✓ Grouped {len(all_time_series_data)} design points by {len(density_groups)} density levels")

# Create plots systematically by density
plot_count = 0
for density_str in sorted(density_groups.keys()):
    print(f"\nDensity {density_str} restaurants/km²:")
    
    for design_name in sorted(density_groups[density_str]):
        plot_title = f"Warmup Analysis: {design_name}"
        time_series_data = all_time_series_data[design_name]
        fig = viz.create_warmup_analysis_plot(time_series_data, title=plot_title)
        
        plt.show()
        print(f"    ✓ {design_name} plot displayed")
        plot_count += 1

print(f"\n✓ Warmup analysis visualization complete")
print(f"✓ Created {plot_count} warmup analysis plots")
print(f"✓ Organized by {len(density_groups)} density levels")

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

# Initialize pipeline with metric type enabled
pipeline = ExperimentAnalysisPipeline(
    warmup_period=uniform_warmup_period,
    enabled_metric_types=[
        'order_metrics', 
        'system_metrics', 
        'delivery_unit_metrics', 
        'system_state_metrics',
        'pair_metrics'  # ADD THIS
    ],
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

# %% CELL 16: Extract and Present Key Metrics (TABLE FORMAT WITH DENSITY)
print("\n" + "="*80)
print("KEY PERFORMANCE METRICS EXTRACTION AND PRESENTATION")
print("="*80)

import re

def extract_density_seed_ratio_pairing_and_type(design_name):
    """Extract density, seed, ratio, pairing status, and interval type from design point name."""
    # Pattern: density_0.10_seed42_ratio_5.0_pairing_baseline or density_0.10_seed42_ratio_5.0_no_pairing_baseline
    match = re.match(r'density_([\d.]+)_seed(\d+)_ratio_([\d.]+)_(pairing|no_pairing)_(baseline|2x_baseline)', design_name)
    if match:
        density = float(match.group(1))
        seed = int(match.group(2))
        ratio = float(match.group(3))
        pairing_status = match.group(4)
        interval_type = match.group(5)
        return density, seed, ratio, pairing_status, interval_type
    return None, None, None, None, None

# Extract comprehensive metrics including pairing rate and pair formation time
metrics_data = []

for design_name, analysis_result in design_analysis_results.items():
    density, seed, ratio, pairing_status, interval_type = extract_density_seed_ratio_pairing_and_type(design_name)
    if density is None:
        continue
    
    stats_with_cis = analysis_result.get('statistics_with_cis', {})
    
    # Extract assignment time (from order_metrics)
    order_metrics = stats_with_cis.get('order_metrics', {})
    assignment_time = order_metrics.get('assignment_time', {})
    
    mean_of_means = assignment_time.get('mean_of_means', {})
    mom_estimate = mean_of_means.get('point_estimate', 0)
    mom_ci = mean_of_means.get('confidence_interval', [0, 0])
    mom_ci_width = (mom_ci[1] - mom_ci[0]) / 2 if mom_ci[0] is not None else 0
    
    # Extract completion rate (from system_metrics)
    system_metrics = stats_with_cis.get('system_metrics', {})
    completion_rate = system_metrics.get('system_completion_rate', {})
    
    comp_estimate = completion_rate.get('point_estimate', 0)
    comp_ci = completion_rate.get('confidence_interval', [0, 0])
    comp_ci_width = (comp_ci[1] - comp_ci[0]) / 2 if comp_ci[0] is not None else 0
    
    # Extract pairing rate (from system_metrics)
    pairing_rate_stats = system_metrics.get('system_pairing_rate', {})
    pairing_rate_estimate = pairing_rate_stats.get('point_estimate', 0)
    pairing_rate_ci = pairing_rate_stats.get('confidence_interval', [0, 0])
    pairing_rate_ci_width = (pairing_rate_ci[1] - pairing_rate_ci[0]) / 2 if pairing_rate_ci[0] is not None else 0
    
    # Extract mean total distance (from delivery_unit_metrics)
    delivery_unit_metrics = stats_with_cis.get('delivery_unit_metrics', {})
    total_distance = delivery_unit_metrics.get('total_distance', {})
    distance_mom = total_distance.get('mean_of_means', {})
    distance_estimate = distance_mom.get('point_estimate', 0)
    distance_ci = distance_mom.get('confidence_interval', [0, 0])
    distance_ci_width = (distance_ci[1] - distance_ci[0]) / 2 if distance_ci[0] is not None else 0
    
    # Extract available drivers (from system_state_metrics)
    system_state_metrics = stats_with_cis.get('system_state_metrics', {})
    available_drivers = system_state_metrics.get('available_drivers', {})
    avail_mom = available_drivers.get('mean_of_means', {})
    avail_mom_estimate = avail_mom.get('point_estimate', 0)
    avail_mom_ci = avail_mom.get('confidence_interval', [0, 0])
    avail_mom_ci_width = (avail_mom_ci[1] - avail_mom_ci[0]) / 2 if avail_mom_ci[0] is not None else 0
    
    # Extract pair formation time (from pair_metrics) - only available when pairing enabled
    pair_metrics = stats_with_cis.get('pair_metrics', {})
    formation_time = pair_metrics.get('formation_time', {})
    formation_mom = formation_time.get('mean_of_means', {})
    formation_estimate = formation_mom.get('point_estimate', None)
    formation_ci = formation_mom.get('confidence_interval', [None, None])
    formation_ci_width = (formation_ci[1] - formation_ci[0]) / 2 if formation_ci[0] is not None else None
    
    metrics_data.append({
        'density': density,
        'seed': seed,
        'ratio': ratio,
        'pairing_status': pairing_status,
        'interval_type': interval_type,
        'mom_estimate': mom_estimate,
        'mom_ci_width': mom_ci_width,
        'comp_estimate': comp_estimate,
        'comp_ci_width': comp_ci_width,
        'pairing_rate_estimate': pairing_rate_estimate,
        'pairing_rate_ci_width': pairing_rate_ci_width,
        'distance_estimate': distance_estimate,
        'distance_ci_width': distance_ci_width,
        'avail_mom_estimate': avail_mom_estimate,
        'avail_mom_ci_width': avail_mom_ci_width,
        'formation_estimate': formation_estimate,
        'formation_ci_width': formation_ci_width,
    })

# Sort by: interval_type, pairing_status, ratio, seed, density
# This groups all pairing conditions together, then by ratio, seed, with density varying
metrics_data.sort(key=lambda x: (x['interval_type'], x['pairing_status'], x['ratio'], x['seed'], x['density']))

# ========================================
# MAIN TABLE: Density effects by seed
# ========================================
print("\n🎯 KEY PERFORMANCE METRICS: DENSITY × PAIRING × RATIO EFFECTS")
print("="*220)
print(" Density  Seed   Ratio   Pairing      Interval       Assignment Time       Completion Rate        Pairing Rate       Pair Formation Time      Mean Total Distance      Available Drivers")
print(" (/km²)                  Status           Type       (mean_of_means)         (with 95% CI)         (with 95% CI)         (mean_of_means)           (with 95% CI)          (mean_of_means)")
print("="*220)

for row in metrics_data:
    density = row['density']
    seed = row['seed']
    ratio = row['ratio']
    pairing_display = "Pairing" if row['pairing_status'] == 'pairing' else "No Pairing"
    interval_display = "2x Baseline" if row['interval_type'] == '2x_baseline' else "Baseline"
    
    mom_str = f"{row['mom_estimate']:5.2f} ± {row['mom_ci_width']:5.2f}"
    comp_str = f"{row['comp_estimate']:.3f} ± {row['comp_ci_width']:.3f}"
    pairing_rate_str = f"{row['pairing_rate_estimate']*100:4.1f}% ± {row['pairing_rate_ci_width']*100:4.1f}%"
    dist_str = f"{row['distance_estimate']:5.2f} ± {row['distance_ci_width']:5.2f}"
    avail_mom_str = f"{row['avail_mom_estimate']:5.2f} ± {row['avail_mom_ci_width']:5.2f}"
    
    # Pair formation time - N/A for no_pairing
    if row['formation_estimate'] is not None:
        formation_str = f"{row['formation_estimate']:5.2f} ± {row['formation_ci_width']:5.2f}"
    else:
        formation_str = "N/A".center(16)
    
    print(f"  {density:5.3f}  {seed:4d}   {ratio:3.1f}  {pairing_display:10s}  {interval_display:12s}     {mom_str:>16s}       {comp_str:>15s}       {pairing_rate_str:>18s}       {formation_str:>16s}          {dist_str:>18s}        {avail_mom_str:>18s}")

print("="*220)
print("\n✓ Metric extraction complete with density × pairing × ratio analysis")
print("✓ Sorted by: interval_type → pairing_status → ratio → seed → density")
print("✓ Compare density effects within same seed group to understand infrastructure design principles")

# %%
