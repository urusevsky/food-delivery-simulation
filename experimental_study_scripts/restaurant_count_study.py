# restaurant_count_study.py
"""
Restaurant Count Effects Study (Fixed Service Area)

Research Question: How does the number of restaurants affect delivery system 
performance in a fixed service area? Do restaurant count effects interact with 
pairing policies?

Test: Three restaurant counts (5, 10, 15) in fixed 10×10km area, with pairing 
enabled vs disabled.

Design: 3 counts × 3 seeds × 2 ratios × 2 pairing × 2 scales = 72 design points
Total runs: 72 × 5 replications = 360 runs (~25-30 minutes)

Scope: This study tests restaurant count effects ONLY. Service area size is held 
constant at 10×10km. Effects of area size are not tested in this study.
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
print("RESTAURANT COUNT EFFECTS STUDY (FIXED SERVICE AREA)")
print("="*80)
print("Research Question: How does restaurant count affect performance in fixed areas?")

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
How does the number of restaurants affect delivery system performance and 
operational regime boundaries in a fixed service area? Do restaurant count 
effects interact with pairing policies?

Building on previous findings:
- Layout study: Spatial structure affects performance through distance/capacity
- Pairing study: Throughput doubling is the kingpin for system viability
- Question: Does increasing restaurant count in a fixed area improve performance?

Test: Three restaurant counts (5, 10, 15) in fixed 10×10km delivery area.
Focus ratios: 5.0 (volatile), 7.0 (critical) - where effects are most pronounced.

SCOPE: This study varies restaurant count ONLY. Service area is held constant 
at 10×10km. We make no claims about density (N/A²) effects, as area size is 
not varied in this study.
"""

context = """
Previous studies established:
1. Infrastructure layout sensitivity: Seed-specific spatial configurations 
   affect performance through distance and capacity mechanisms
2. Pairing rescue effect: Pairing transforms failing systems into viable ones 
   at high load ratios through throughput doubling
3. Operational regime boundaries: Load ratio determines system behavior regimes

Open question:
- Common platform strategy: Add more restaurant partners to existing markets
- Intuitive hypothesis: More restaurants → shorter distances → better performance
- But: Orders spread across more restaurants under uniform selection
- Need empirical evidence for this specific scenario
"""

sub_questions = """
1. Does increasing restaurant count improve system performance?
   - Hypothesis: More restaurants → shorter pickup distances → faster service
   - Alternative: Benefits offset by temporal dilution (orders spread across more locations)

2. Do restaurant count effects interact with pairing policies?
   - With pairing: More restaurants → more pairing opportunities?
   - Without pairing: Count effects through distance mechanism only?

3. Are restaurant count effects consistent across different spatial layouts?
   - Test with 3 random seeds (42, 100, 200)
   - Check if count effects robust or seed-dependent
"""

scope = """
INCLUDED in this study:
- Restaurant counts: 5, 10, 15 (in 10×10km fixed area)
- Random layout robustness: 3 structural seeds per count
- Pairing interaction: Both enabled and disabled
- Load regimes: Ratios 5.0 (volatile) and 7.0 (critical)
- Operational intensity validation: Baseline + 2× baseline

NOT included (explicit limitations):
- Service area size variation: Fixed at 10×10km
- Different area-to-count ratios: Only one combination per count tested
- Density (N/A²) as experimental factor: Would require varying area size
- Restaurant popularity distributions: Uniform random selection only

SCOPE CLARIFICATION:
This study tests: "Does adding restaurants to a 10×10km market improve performance?"
This study does NOT test: "Does density (N/A²) determine performance?"
The latter would require varying both N and A to isolate density effects.
"""

analysis_focus = """
Key metrics:
1. Assignment time: Core performance indicator
2. Completion rate: System viability check
3. Pairing rate: Mechanism (does count enable more pairing?)
4. Pair formation time: Temporal dynamics of pairing opportunities
5. Available drivers: Capacity buffer indicator
6. Mean total distance: Infrastructure characteristic (explains mechanism)

Analysis approach:
- Main effects: Compare restaurant counts, aggregate across seeds
- Interactions: Count × pairing, count × ratio
- Robustness: Between-seed variation at each count level
- Mechanism: Use distance/pairing rate/formation time to explain performance
- Limitations: Acknowledge findings specific to 10×10km areas

Expected patterns (hypotheses to test):
- More restaurants → shorter distances (geometric benefit)
- But: Orders spread across more restaurants (temporal dilution)
- Net effect: Spatial vs temporal trade-off
- With pairing: May see count-invariant performance if effects cancel
- Without pairing: More direct distance effect, but seed-dependent
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
RESTAURANT COUNT STUDY: Systematically vary restaurant count in fixed area.

Service area: 10×10 km (fixed)
Test by varying num_restaurants while holding area constant.

Restaurant count levels:
- Low: 5 restaurants in 10×10km
- Medium: 10 restaurants in 10×10km (baseline from previous studies)
- High: 15 restaurants in 10×10km

Note: We describe by restaurant count, not density, as area is not varied.
Resulting densities (0.05, 0.10, 0.15/km²) are consequences, not experimental factors.
"""

infrastructure_configs = [
    {
        'name': 'restaurants_5',
        'config': StructuralConfig(
            delivery_area_size=10,
            num_restaurants=5,
            driver_speed=0.5
        ),
        'num_restaurants': 5
    },
    {
        'name': 'restaurants_10',
        'config': StructuralConfig(
            delivery_area_size=10,
            num_restaurants=10,
            driver_speed=0.5
        ),
        'num_restaurants': 10
    },
    {
        'name': 'restaurants_15',
        'config': StructuralConfig(
            delivery_area_size=10,
            num_restaurants=15,
            driver_speed=0.5
        ),
        'num_restaurants': 15
    }
]

print(f"✓ Defined {len(infrastructure_configs)} restaurant count configurations")
for config in infrastructure_configs:
    struct_config = config['config']
    n_restaurants = config['num_restaurants']
    print(f"  • {config['name']}: {n_restaurants} restaurants, "
          f"{struct_config.delivery_area_size}×{struct_config.delivery_area_size}km area")

# %% CELL 5: Structural Seeds
"""
Infrastructure replication: Test robustness across random layout realizations.

Use same seeds (42, 100, 200) as previous studies for continuity and CRN.
"""

structural_seeds = [42, 100, 200]

print(f"\n✓ Structural seeds: {structural_seeds}")
print(f"✓ Testing restaurant count effects with {len(structural_seeds)} random layouts per count")
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
            'num_restaurants': infra_config['num_restaurants'],
            'seed': structural_seed
        })
        
        print(f"  ✓ Infrastructure created and analyzed")
        print(f"    • Restaurant count: {infra_config['num_restaurants']}")
        print(f"    • Service area: 10×10 km")
        print(f"    • Typical distance: {analysis_results['typical_distance']:.3f}km")

print(f"\n{'='*50}")
print(f"✓ Created {len(infrastructure_instances)} infrastructure instance(s)")
print(f"✓ Breakdown: {len(infrastructure_configs)} restaurant counts × {len(structural_seeds)} seeds")
print(f"{'='*50}")

# Display infrastructure comparison by restaurant count
print("\n📊 Infrastructure Comparison by Restaurant Count:")
for n_restaurants in sorted(set(inst['num_restaurants'] for inst in infrastructure_instances)):
    print(f"\n{n_restaurants} Restaurants:")
    count_instances = [inst for inst in infrastructure_instances if inst['num_restaurants'] == n_restaurants]
    for inst in count_instances:
        print(f"  Seed {inst['seed']}: typical_distance={inst['analysis']['typical_distance']:.3f}km")

# %% CELL 6.5: Visualize Restaurant Layouts
"""
OPTIONAL: Visualize infrastructure instances to understand spatial coverage.

For restaurant count study, this shows how spatial coverage changes with count.
"""

print("\n" + "="*50)
print("INFRASTRUCTURE LAYOUT VISUALIZATION")
print("="*50)

import matplotlib.pyplot as plt

print(f"\nVisualizing {len(infrastructure_instances)} infrastructure instances...")
print("Compare spatial coverage across restaurant counts.\n")

for instance in infrastructure_instances:
    print(f"{'='*50}")
    print(f"Layout: {instance['name']}")
    print(f"Restaurants: {instance['num_restaurants']}")
    print(f"Typical Distance: {instance['analysis']['typical_distance']:.3f}km")
    print(f"{'='*50}")
    
    # Visualize using stored analyzer
    instance['analyzer'].visualize_infrastructure()
    
    # Add custom header
    fig = plt.gcf()
    seed = instance['seed']
    n_restaurants = instance['infrastructure'].structural_config.num_restaurants
    typical_dist = instance['analysis']['typical_distance']
    
    custom_title = (f"Infrastructure Layout: {n_restaurants} Restaurants (Seed {seed})\n"
                   f"Typical Distance: {typical_dist:.3f}km | Area: 10×10km")
    
    fig.suptitle(custom_title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ {instance['name']} visualized\n")

print(f"{'='*50}")
print("✓ All layouts visualized")
print("✓ Observe spatial coverage patterns:")
print("  - 5 restaurants: Sparse coverage, potentially longer distances")
print("  - 10 restaurants: Baseline coverage from previous studies")
print("  - 15 restaurants: Denser coverage")
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
- Restaurant count effects likely most pronounced (high load conditions)
- Pairing benefits are substantial (established in previous study)
- Interaction effects most interesting

For each ratio, create validation pair (baseline + 2× baseline) for both
pairing enabled and disabled conditions.
"""

operational_configs = []

# Define arrival interval ratios to test
ratios = [5.0, 7.0]

# Define interval scales for validation
scales = [
    {'name': '2x_baseline', 'order': 2.0, 'driver_multiplier': 2.0},
    {'name': 'baseline', 'order': 1.0, 'driver_multiplier': 1.0}
]

# Define pairing configurations
pairing_modes = [
    {'name': 'no_pairing', 'enabled': False},
    {'name': 'pairing', 'enabled': True}
]

for ratio in ratios:
    for scale in scales:
        for pairing in pairing_modes:
            
            mean_order = scale['order']
            mean_driver = scale['driver_multiplier'] * ratio
            
            config_name = f"ratio_{ratio}_{pairing['name']}_{scale['name']}"
            
            operational_configs.append({
                'name': config_name,
                'config': OperationalConfig(
                    mean_order_inter_arrival_time=mean_order,
                    mean_driver_inter_arrival_time=mean_driver,
                    pairing_enabled=pairing['enabled'],
                    restaurants_proximity_threshold=4.0,
                    customers_proximity_threshold=3.5,
                    mean_service_duration=100,
                    service_duration_std_dev=20,
                    min_service_duration=60,
                    max_service_duration=140
                )
            })

print(f"✓ Defined {len(operational_configs)} operational configurations")
print(f"✓ Ratios: {ratios}")
print(f"✓ Each ratio tested with: pairing × no_pairing × baseline × 2x_baseline")

# %% CELL 9: Create Design Points
"""
Combine infrastructure × operational × scoring configurations.

Design: 3 restaurant counts × 3 seeds × 2 ratios × 2 pairing × 2 scales = 72 design points
"""

design_points = {}

print("\n" + "="*50)
print("DESIGN POINTS CREATION")
print("="*50)

for infra_instance in infrastructure_instances:
    for op_config in operational_configs:
        for scoring_config_dict in scoring_configs:
            
            # Generate design point name including restaurant count and seed info
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

print(f"\n✅ RESTAURANT COUNT STUDY COMPLETE!")
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

# Group design points by restaurant count for organized display
count_groups = {}
for design_name in all_time_series_data.keys():
    # Extract restaurant count from design name (e.g., "restaurants_5_seed42_ratio_5.0_pairing_baseline")
    parts = design_name.split('_')
    count_str = parts[1]  # Get "5" from "restaurants_5"
    
    if count_str not in count_groups:
        count_groups[count_str] = []
    count_groups[count_str].append(design_name)

print(f"✓ Grouped {len(all_time_series_data)} design points by {len(count_groups)} restaurant counts")

# Create plots systematically by restaurant count
plot_count = 0
for count_str in sorted(count_groups.keys(), key=int):
    print(f"\n{count_str} Restaurants:")
    
    for design_name in sorted(count_groups[count_str]):
        plot_title = f"Warmup Analysis: {design_name}"
        time_series_data = all_time_series_data[design_name]
        fig = viz.create_warmup_analysis_plot(time_series_data, title=plot_title)
        
        plt.show()
        print(f"    ✓ {design_name} plot displayed")
        plot_count += 1

print(f"\n✓ Warmup analysis visualization complete")
print(f"✓ Created {plot_count} warmup analysis plots")
print(f"✓ Organized by {len(count_groups)} restaurant count levels")

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

def extract_count_seed_ratio_pairing_and_type(design_name):
    """Extract restaurant count, seed, ratio, pairing status, and interval type from design point name."""
    # Pattern: restaurants_5_seed42_ratio_5.0_pairing_baseline or restaurants_5_seed42_ratio_5.0_no_pairing_baseline
    match = re.match(r'restaurants_(\d+)_seed(\d+)_ratio_([\d.]+)_(pairing|no_pairing)_(baseline|2x_baseline)', design_name)
    if match:
        num_restaurants = int(match.group(1))
        seed = int(match.group(2))
        ratio = float(match.group(3))
        pairing_status = match.group(4)
        interval_type = match.group(5)
        return num_restaurants, seed, ratio, pairing_status, interval_type
    return None, None, None, None, None

# Extract comprehensive metrics
metrics_data = []

for design_name, analysis_result in design_analysis_results.items():
    num_restaurants, seed, ratio, pairing_status, interval_type = extract_count_seed_ratio_pairing_and_type(design_name)
    if num_restaurants is None:
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
        'num_restaurants': num_restaurants,
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

# Sort by: interval_type, pairing_status, ratio, seed, num_restaurants
metrics_data.sort(key=lambda x: (x['interval_type'], x['pairing_status'], x['ratio'], x['seed'], x['num_restaurants']))

# ========================================
# MAIN TABLE: Restaurant count effects
# ========================================
print("\n🎯 KEY PERFORMANCE METRICS: RESTAURANT COUNT × PAIRING × RATIO EFFECTS")
print("="*220)
print("  Num     Seed   Ratio   Pairing      Interval       Assignment Time       Completion Rate        Pairing Rate       Pair Formation Time      Mean Total Distance      Available Drivers")
print("Restaurants                Status           Type       (mean_of_means)         (with 95% CI)         (with 95% CI)         (mean_of_means)           (with 95% CI)          (mean_of_means)")
print("="*220)

for row in metrics_data:
    num_restaurants = row['num_restaurants']
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
    
    print(f"    {num_restaurants:2d}      {seed:4d}   {ratio:3.1f}  {pairing_display:10s}  {interval_display:12s}     {mom_str:>16s}       {comp_str:>15s}       {pairing_rate_str:>18s}       {formation_str:>16s}          {dist_str:>18s}        {avail_mom_str:>18s}")

print("="*220)
print("\n✓ Metric extraction complete with restaurant count × pairing × ratio analysis")
print("✓ Sorted by: interval_type → pairing_status → ratio → seed → num_restaurants")
print("✓ Compare restaurant count effects within same seed group")
print("\nSCOPE REMINDER: Results specific to 10×10km service area. ")
print("Effects of area size not tested in this study.")

# %%