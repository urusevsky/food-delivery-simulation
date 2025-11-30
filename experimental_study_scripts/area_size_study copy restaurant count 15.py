# area_size_study.py
"""
Delivery Area Size Effects Study (Fixed Restaurant Count)

Research Question: How does delivery area size affect system performance when 
the number of restaurants is held constant? Do area size effects interact with 
pairing policies?

Test: Three area sizes (5×5, 10×10, 15×15 km) with fixed 10 restaurants, with 
pairing enabled vs disabled.

Design: 3 areas × 3 seeds × 2 ratios × 2 pairing × 2 scales = 72 design points
Total runs: 72 × 5 replications = 360 runs (~25-30 minutes)

Scope: This study tests area size effects ONLY. Restaurant count is held constant 
at 10. Effects of restaurant count are not tested in this study.
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
print("DELIVERY AREA SIZE EFFECTS STUDY (FIXED RESTAURANT COUNT)")
print("="*80)
print("Research Question: How does area size affect performance with fixed restaurants?")

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
Document research question and scope.
"""

print("\n" + "="*80)
print("RESEARCH QUESTION")
print("="*80)

# ==============================================================================
# MAIN RESEARCH QUESTION
# ==============================================================================
research_question = """
How does delivery area size affect system performance and operational regime 
boundaries when restaurant count is held constant? Do area size effects 
interact with pairing policies?

Building on previous findings:
- Restaurant count study: More restaurants don't improve performance (temporal dilution)
- Pairing study: Throughput doubling is the kingpin for system viability
- Question: Does area size affect performance through distance mechanism?

Test: Three area sizes (5×5, 10×10, 15×15 km) with fixed 10 restaurants.
Focus ratios: 5.0 (volatile), 7.0 (critical) - where effects are most pronounced.

SCOPE: This study varies area size ONLY. Restaurant count is held constant 
at 10. We make no claims about restaurant count effects, as count is not 
varied in this study.
"""

context = """
Previous studies established:
1. Restaurant count effects: Adding restaurants to fixed area shows no systematic 
   improvement due to temporal dilution under uniform selection
2. Pairing rescue effect: Pairing transforms failing systems into viable ones 
   at high load ratios through throughput doubling
3. Operational regime boundaries: Load ratio determines system behavior regimes

Open question:
- Fundamental geometry: Larger areas → longer distances → longer service times
- Hypothesis: Performance should degrade with area size due to distance mechanism
- But: Does pairing mitigate or amplify area size effects?
- Need empirical evidence to quantify effect size
"""

sub_questions = """
1. Does increasing area size degrade system performance?
   - Hypothesis: Larger areas → longer distances → worse assignment times
   - Expected: Clear monotonic relationship (unlike restaurant count study)

2. Do area size effects interact with pairing policies?
   - With pairing: More travel time, but pairs share return trip cost?
   - Without pairing: Direct distance effect on performance?

3. Are area size effects consistent across different spatial layouts?
   - Test with 3 random seeds (42, 100, 200)
   - Check if effects robust or seed-dependent

4. How does effect size compare to restaurant count effects?
   - Restaurant count: No systematic effect (temporal dilution)
   - Area size: Expected to have clear effect (distance mechanism)
"""

scope = """
INCLUDED in this study:
- Area sizes: 5×5, 10×10, 15×15 km (fixed 10 restaurants)
- Random layout robustness: 3 structural seeds per area
- Pairing interaction: Both enabled and disabled
- Load regimes: Ratios 5.0 (volatile) and 7.0 (critical)
- Operational intensity validation: Baseline + 2× baseline

NOT included (explicit limitations):
- Restaurant count variation: Fixed at 10 restaurants
- Other area sizes: Limited to three levels for computational efficiency
- Different count-to-area ratios: Only one combination per area tested
- Restaurant popularity distributions: Uniform random selection only

SCOPE CLARIFICATION:
This study tests: "Does area size affect performance with 10 restaurants?"
This study does NOT test: "How does restaurant count affect performance?"
The latter was addressed in the restaurant count study.
"""

analysis_focus = """
Key metrics:
1. Assignment time: Core performance indicator
2. Completion rate: System viability check
3. Pairing rate: Mechanism (does area affect pairing feasibility?)
4. Pair formation time: Temporal dynamics across different scales
5. Available drivers: Capacity buffer indicator
6. Mean total distance: Infrastructure characteristic (direct mechanism evidence)

Analysis approach:
- Main effects: Compare area sizes, aggregate across seeds
- Interactions: Area × pairing, area × ratio
- Robustness: Between-seed variation at each area level
- Mechanism: Use distance/pairing rate to explain performance
- Comparison: Contrast with restaurant count study findings

Expected patterns (hypotheses to test):
- Larger areas → longer distances (geometric necessity)
- Larger areas → worse performance (distance drives service time)
- Effect size should be substantial (unlike restaurant count)
- With pairing: Effect may be amplified (pairs need compatible distances)
- Without pairing: Direct monotonic relationship expected
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
AREA SIZE STUDY: Systematically vary delivery area size with fixed restaurant count.

Restaurant count: 10 (fixed)
Test by varying delivery_area_size while holding num_restaurants constant.

Area size levels:
- Small: 5×5 km with 10 restaurants
- Medium: 10×10 km with 10 restaurants (baseline from previous studies)
- Large: 15×15 km with 10 restaurants

Note: We describe by area size, not density, as restaurant count is not varied.
Resulting densities (0.40, 0.10, 0.044/km²) are consequences, not experimental factors.
"""

infrastructure_configs = [
    {
        'name': 'area_5',
        'config': StructuralConfig(
            delivery_area_size=5,
            num_restaurants=15,
            driver_speed=0.5
        ),
        'area_size': 5
    },
    {
        'name': 'area_10',
        'config': StructuralConfig(
            delivery_area_size=10,
            num_restaurants=15,
            driver_speed=0.5
        ),
        'area_size': 10
    },
    {
        'name': 'area_15',
        'config': StructuralConfig(
            delivery_area_size=15,
            num_restaurants=15,
            driver_speed=0.5
        ),
        'area_size': 15
    }
]

print(f"✓ Defined {len(infrastructure_configs)} area size configurations")
for config in infrastructure_configs:
    struct_config = config['config']
    area_size = config['area_size']
    print(f"  • {config['name']}: {area_size}×{area_size}km area, "
          f"{struct_config.num_restaurants} restaurants")

# %% CELL 5: Structural Seeds
"""
Infrastructure replication: Test robustness across random layout realizations.

Use same seeds (42, 100, 200) as previous studies for continuity and CRN.
"""

structural_seeds = [42, 100, 200]

print(f"\n✓ Structural seeds: {structural_seeds}")
print(f"✓ Testing area size effects with {len(structural_seeds)} random layouts per area")
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
            'area_size': infra_config['area_size'],
            'seed': structural_seed
        })
        
        print(f"  ✓ Infrastructure created and analyzed")
        print(f"    • Area size: {infra_config['area_size']}×{infra_config['area_size']} km")
        print(f"    • Restaurant count: 10 (fixed)")
        print(f"    • Typical distance: {analysis_results['typical_distance']:.3f}km")

print(f"\n{'='*50}")
print(f"✓ Created {len(infrastructure_instances)} infrastructure instance(s)")
print(f"✓ Breakdown: {len(infrastructure_configs)} area sizes × {len(structural_seeds)} seeds")
print(f"{'='*50}")

# Display infrastructure comparison by area size
print("\n📊 Infrastructure Comparison by Area Size:")
for area_size in sorted(set(inst['area_size'] for inst in infrastructure_instances)):
    print(f"\n{area_size}×{area_size} km:")
    area_instances = [inst for inst in infrastructure_instances if inst['area_size'] == area_size]
    for inst in area_instances:
        print(f"  Seed {inst['seed']}: typical_distance={inst['analysis']['typical_distance']:.3f}km")

# %% CELL 6.5: Visualize Restaurant Layouts
"""
OPTIONAL: Visualize infrastructure instances to understand spatial scale.

For area size study, this shows how spatial scale changes with area size.
"""

print("\n" + "="*50)
print("INFRASTRUCTURE LAYOUT VISUALIZATION")
print("="*50)

import matplotlib.pyplot as plt

print(f"\nVisualizing {len(infrastructure_instances)} infrastructure instances...")
print("Compare spatial scale across area sizes.\n")

for instance in infrastructure_instances:
    print(f"{'='*50}")
    print(f"Layout: {instance['name']}")
    print(f"Area size: {instance['area_size']}×{instance['area_size']} km")
    print(f"Typical Distance: {instance['analysis']['typical_distance']:.3f}km")
    print(f"{'='*50}")
    
    # Visualize using stored analyzer
    instance['analyzer'].visualize_infrastructure()
    
    # Add custom header
    fig = plt.gcf()
    seed = instance['seed']
    area_size = instance['infrastructure'].structural_config.delivery_area_size
    typical_dist = instance['analysis']['typical_distance']
    
    custom_title = (f"Infrastructure Layout: {area_size}×{area_size}km Area (Seed {seed})\n"
                   f"10 Restaurants | Typical Distance: {typical_dist:.3f}km")
    
    fig.suptitle(custom_title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ {instance['name']} visualized\n")

print(f"{'='*50}")
print("✓ All layouts visualized")
print("✓ Observe spatial scale patterns:")
print("  - 5×5 km: Compact area, shorter typical distances")
print("  - 10×10 km: Medium area (baseline from previous studies)")
print("  - 15×15 km: Large area, longer typical distances expected")
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
- Area size effects likely most pronounced (high load conditions)
- Pairing benefits are substantial (established in previous study)
- Interaction effects most interesting

For each ratio, create validation pair (baseline + 2× baseline) for both
pairing enabled and disabled conditions.
"""

# Critical ratios where area size effects likely matter most
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

# %% CELL 9: Create Design Points
"""
Combine infrastructure × operational × scoring configurations.

Design: 3 areas × 3 seeds × 2 ratios × 2 pairing × 2 scales = 72 design points
"""

design_points = {}

print("\n" + "="*50)
print("DESIGN POINTS CREATION")
print("="*50)

for infra_instance in infrastructure_instances:
    for op_config in operational_configs:
        for scoring_config_dict in scoring_configs:
            
            # Generate design point name including area size and seed info
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

print(f"\n✅ AREA SIZE STUDY COMPLETE!")
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

# Group design points by area size for organized display
area_groups = {}
for design_name in all_time_series_data.keys():
    # Extract area size from design name (e.g., "area_5_seed42_ratio_5.0_pairing_baseline")
    parts = design_name.split('_')
    area_str = parts[1]  # Get "5" from "area_5"
    
    if area_str not in area_groups:
        area_groups[area_str] = []
    area_groups[area_str].append(design_name)

print(f"✓ Grouped {len(all_time_series_data)} design points by {len(area_groups)} area sizes")

# Create plots systematically by area size
plot_count = 0
for area_str in sorted(area_groups.keys(), key=int):
    print(f"\n{area_str}×{area_str} km:")
    
    for design_name in sorted(area_groups[area_str]):
        plot_title = f"Warmup Analysis: {design_name}"
        time_series_data = all_time_series_data[design_name]
        fig = viz.create_warmup_analysis_plot(time_series_data, title=plot_title)
        
        plt.show()
        print(f"    ✓ {design_name} plot displayed")
        plot_count += 1

print(f"\n✓ Warmup analysis visualization complete")
print(f"✓ Created {plot_count} warmup analysis plots")
print(f"✓ Organized by {len(area_groups)} area size levels")

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

# Initialize pipeline with all metric types including pair_metrics
pipeline = ExperimentAnalysisPipeline(
    warmup_period=uniform_warmup_period,
    enabled_metric_types=['order_metrics', 'system_metrics', 'delivery_unit_metrics', 'system_state_metrics', 'pair_metrics'],
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

# %% CELL 16: Extract and Present Key Metrics (TABLE FORMAT)
print("\n" + "="*80)
print("KEY PERFORMANCE METRICS EXTRACTION AND PRESENTATION")
print("="*80)

import re

def extract_area_seed_ratio_pairing_and_type(design_name):
    """Extract area size, seed, ratio, pairing status, and interval type from design point name."""
    # Pattern: area_5_seed42_ratio_5.0_pairing_baseline or area_5_seed42_ratio_5.0_no_pairing_baseline
    match = re.match(r'area_(\d+)_seed(\d+)_ratio_([\d.]+)_(pairing|no_pairing)_(baseline|2x_baseline)', design_name)
    if match:
        area_size = int(match.group(1))
        seed = int(match.group(2))
        ratio = float(match.group(3))
        pairing_status = match.group(4)
        interval_type = match.group(5)
        return area_size, seed, ratio, pairing_status, interval_type
    return None, None, None, None, None

# Extract comprehensive metrics
metrics_data = []

for design_name, analysis_result in design_analysis_results.items():
    area_size, seed, ratio, pairing_status, interval_type = extract_area_seed_ratio_pairing_and_type(design_name)
    if area_size is None:
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
        'area_size': area_size,
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

# Sort by: interval_type, pairing_status, ratio, seed, area_size
metrics_data.sort(key=lambda x: (x['interval_type'], x['pairing_status'], x['ratio'], x['seed'], x['area_size']))

# ========================================
# MAIN TABLE: Area size effects
# ========================================
print("\n🎯 KEY PERFORMANCE METRICS: AREA SIZE × PAIRING × RATIO EFFECTS")
print("="*220)
print("Delivery    Seed   Ratio   Pairing      Interval       Assignment Time       Completion Rate        Pairing Rate       Pair Formation Time      Mean Total Distance      Available Drivers")
print("Area Size                   Status           Type       (mean_of_means)         (with 95% CI)         (with 95% CI)         (mean_of_means)           (with 95% CI)          (mean_of_means)")
print("="*220)

for row in metrics_data:
    area_size = row['area_size']
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
    
    print(f" {area_size:2d}×{area_size:2d} km   {seed:4d}   {ratio:3.1f}  {pairing_display:10s}  {interval_display:12s}     {mom_str:>16s}       {comp_str:>15s}       {pairing_rate_str:>18s}       {formation_str:>16s}          {dist_str:>18s}        {avail_mom_str:>18s}")

print("="*220)
print("\n✓ Metric extraction complete with area size × pairing × ratio analysis")
print("✓ Sorted by: interval_type → pairing_status → ratio → seed → area_size")
print("✓ Compare area size effects within same seed group")
print("\nSCOPE REMINDER: Results specific to 10 restaurants. ")
print("Effects of restaurant count not tested in this study.")

# %% AD HOC ANALYSIS: Eligible Restaurant Pairs by Area Size
"""
Verify hypothesis: Larger delivery areas have fewer eligible restaurant pairs
within the fixed 4.0 km pairing threshold.

This addresses the gap identified in analysis: we need empirical evidence that
spatial dilution actually reduces pairing opportunities as area size increases.
"""

print("\n" + "="*80)
print("AD HOC ANALYSIS: ELIGIBLE RESTAURANT PAIRS BY AREA SIZE")
print("="*80)

from delivery_sim.utils.location_utils import calculate_distance

# Fixed pairing threshold from operational configs
RESTAURANT_PAIRING_THRESHOLD = 4.0  # km

# Analyze each infrastructure instance
eligible_pairs_data = []

print(f"\nAnalyzing eligible restaurant pairs with threshold = {RESTAURANT_PAIRING_THRESHOLD} km")
print(f"Total infrastructure instances: {len(infrastructure_instances)}\n")

for instance in infrastructure_instances:
    area_size = instance['area_size']
    seed = instance['seed']
    infrastructure = instance['infrastructure']
    
    # Get all restaurants from repository
    restaurants = infrastructure.restaurant_repository.find_all()
    num_restaurants = len(restaurants)
    
    # Calculate all pairwise distances
    eligible_pairs = []
    all_distances = []
    
    for i, rest1 in enumerate(restaurants):
        for j, rest2 in enumerate(restaurants[i+1:], i+1):
            distance = calculate_distance(rest1.location, rest2.location)
            all_distances.append(distance)
            
            if distance <= RESTAURANT_PAIRING_THRESHOLD:
                eligible_pairs.append({
                    'restaurant_1': rest1.restaurant_id,
                    'restaurant_2': rest2.restaurant_id,
                    'distance': distance
                })
    
    # Calculate statistics
    total_possible_pairs = len(all_distances)
    num_eligible_pairs = len(eligible_pairs)
    eligible_percentage = (num_eligible_pairs / total_possible_pairs * 100) if total_possible_pairs > 0 else 0
    
    # Distance statistics
    min_distance = min(all_distances) if all_distances else 0
    max_distance = max(all_distances) if all_distances else 0
    mean_distance = sum(all_distances) / len(all_distances) if all_distances else 0
    
    eligible_pairs_data.append({
        'area_size': area_size,
        'seed': seed,
        'num_restaurants': num_restaurants,
        'total_possible_pairs': total_possible_pairs,
        'num_eligible_pairs': num_eligible_pairs,
        'eligible_percentage': eligible_percentage,
        'min_distance': min_distance,
        'max_distance': max_distance,
        'mean_distance': mean_distance
    })
    
    print(f"Area {area_size}×{area_size} km (Seed {seed}):")
    print(f"  Total possible pairs: {total_possible_pairs}")
    print(f"  Eligible pairs: {num_eligible_pairs} ({eligible_percentage:.1f}%)")
    print(f"  Distance range: {min_distance:.2f} - {max_distance:.2f} km (mean: {mean_distance:.2f})")
    print()

# Create summary table by area size
print("="*80)
print("SUMMARY: ELIGIBLE PAIRS BY AREA SIZE")
print("="*80)

# Group by area size
from collections import defaultdict
area_summaries = defaultdict(list)

for data in eligible_pairs_data:
    area_summaries[data['area_size']].append(data)

print(f"\n{'Area Size':<12} {'Seed':<6} {'Total Pairs':<12} {'Eligible':<10} {'Eligible %':<12} {'Mean Dist (km)':<15}")
print("-"*80)

for area_size in sorted(area_summaries.keys()):
    for data in sorted(area_summaries[area_size], key=lambda x: x['seed']):
        print(f"{area_size:>4}×{area_size:<4} km   {data['seed']:<6} {data['total_possible_pairs']:<12} "
              f"{data['num_eligible_pairs']:<10} {data['eligible_percentage']:>7.1f}%      {data['mean_distance']:>7.2f}")
    
    # Calculate mean across seeds for this area size
    area_data = area_summaries[area_size]
    mean_eligible_pct = sum(d['eligible_percentage'] for d in area_data) / len(area_data)
    mean_mean_dist = sum(d['mean_distance'] for d in area_data) / len(area_data)
    
    print(f"{'  Mean':<12} {'':>6} {'':>12} {'':>10} {mean_eligible_pct:>7.1f}%      {mean_mean_dist:>7.2f}")
    print("-"*80)

print("\n✓ Analysis complete")
print("\nINTERPRETATION:")
print("  • If eligible percentage decreases with area size → spatial dilution confirmed")
print("  • Fixed threshold (4.0 km) captures fewer pairs as restaurants spread out")
print("  • This provides mechanistic explanation for area size performance effects")
# %%
