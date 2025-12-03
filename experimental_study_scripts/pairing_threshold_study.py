# pairing_threshold_study.py
"""
Pairing Threshold Sensitivity Study

Research Question: How do pairing proximity thresholds affect system performance 
across different load regimes? Does the sensitivity to threshold settings depend 
on restaurant count (infrastructure density)?

Test: Four pairing threshold levels (conservative, moderate, liberal, ultra-liberal) 
across two restaurant counts (10, 15) in fixed 10×10 km delivery area.

Design: 2 restaurants × 4 thresholds × 3 seeds × 2 ratios = 48 design points
Total runs: 48 × 5 replications = 240 runs (~20 minutes)

Scope: This study tests pairing threshold effects and restaurant count interaction. 
Area size is held constant at 10×10 km (baseline from previous studies).
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
print("PAIRING THRESHOLD SENSITIVITY STUDY")
print("="*80)
print("Research Question: How do pairing thresholds affect performance?")

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
How do pairing proximity thresholds affect system performance across different 
load regimes? Does the sensitivity to threshold settings depend on restaurant 
count (infrastructure density)?

Primary question: What is the relationship between pairing threshold liberality 
and system performance? Is there an optimal "middle ground" between overly 
conservative thresholds (missing good pairing opportunities) and overly liberal 
thresholds (creating inefficient pairs)?

Secondary question (interaction): Does restaurant count modulate threshold effects? 
Given that more restaurants create:
  - 2.3× more combinatorial pairing opportunities (10→15 restaurants: 45→105 pairs)
  - Temporal dilution effect (orders spread across more locations)
Can liberal thresholds exploit increased pairing opportunities to overcome 
temporal dilution?

Building on previous findings:
- Restaurant count study: More restaurants don't improve performance (temporal dilution)
- Area size study: Larger areas degrade performance through distance mechanism
- Pairing rescue effect: Throughput doubling enables system viability at high load

Test: Three pairing threshold levels (conservative, moderate, liberal) across 
two restaurant counts (10, 15) in fixed 10×10 km delivery area.
"""

print(research_question)

# ==============================================================================
# SCOPE & BOUNDARIES
# ==============================================================================
scope = """
Fixed factors:
- Area size: 10km × 10km (baseline from previous studies)
- Driver speed: 0.5 km/min
- Service duration: Fixed distribution (mean=100, std=60, min=30, max=200)
- Pairing enabled throughout (no "pairing disabled" baseline)

Varied factors:
- Restaurant count: [10, 15] → density and opportunity space variation
- Pairing thresholds: [Conservative, Moderate, Liberal, Ultra-liberal] → spatial constraint variation
- Structural seeds: [42, 100, 200] → layout variability control
- Load ratios: [5.0, 7.0] → operational regime boundaries
- Interval scale: Baseline only (no 2× baseline validation)

Design: 2 restaurants × 4 thresholds × 3 seeds × 2 ratios × 1 scale 
      = 48 design points × 5 replications = 240 runs

Not varied in this study:
- Area size (spatial scale effects already characterized)
- No-pairing baseline (pairing benefits already established)
- Multiple interval scales (threshold sensitivity is the focus)
"""

print("\n" + "="*80)
print("SCOPE & BOUNDARIES")
print("="*80)
print(scope)

# ==============================================================================
# EXPERIMENTAL HYPOTHESES
# ==============================================================================
hypotheses = """
H1 (Threshold main effect): 
   Conservative → Moderate → Liberal produces monotonic increase in pairing rate,
   but performance relationship is non-monotonic (inverted-U curve). Optimal 
   threshold balances opportunity exploitation vs. inefficiency costs.

H2 (Restaurant count interaction - Null hypothesis):
   Threshold effects are independent of restaurant count. Liberal thresholds 
   increase pairing rate equally for 10 and 15 restaurants, and temporal 
   dilution persists regardless of threshold policy.

H2 (Restaurant count interaction - Alternative hypothesis):
   Threshold sensitivity depends on restaurant count. Liberal thresholds may 
   "rescue" temporal dilution effects by exploiting larger opportunity space,
   or may compound inefficiencies through worse spatial matches.

Falsification criteria:
- If no interaction exists: Validates orthogonal optimization of spatial and 
  temporal dimensions in system design.
- If interaction exists: Reveals emergent dynamics requiring joint consideration 
  of infrastructure density and pairing policy.
"""

print("\n" + "="*80)
print("EXPERIMENTAL HYPOTHESES")
print("="*80)
print(hypotheses)

print("\n" + "="*80)
print("✓ Research question documented")
print("="*80)

# %% CELL 4: Infrastructure Configuration(s)
"""
PAIRING THRESHOLD STUDY: Restaurant count variation with fixed area.

Restaurant count: [10, 15] (varied)
Area size: 10×10 km (fixed - baseline from previous studies)

Configurations:
- restaurants_10: Baseline infrastructure (10 restaurants, 10×10 km)
- restaurants_15: Higher density infrastructure (15 restaurants, 10×10 km)

Purpose: Test whether pairing threshold effects depend on restaurant count.
The 10×10 km area provides consistent spatial scale for threshold evaluation.
"""

infrastructure_configs = [
    {
        'name': 'restaurants_10',
        'config': StructuralConfig(
            delivery_area_size=10,
            num_restaurants=10,
            driver_speed=0.5
        ),
        'restaurant_count': 10,
        'area_size': 10
    },
    {
        'name': 'restaurants_15',
        'config': StructuralConfig(
            delivery_area_size=10,
            num_restaurants=15,
            driver_speed=0.5
        ),
        'restaurant_count': 15,
        'area_size': 10
    }
]

print(f"✓ Defined {len(infrastructure_configs)} infrastructure configurations")
for config in infrastructure_configs:
    struct_config = config['config']
    print(f"  • {config['name']}: {config['restaurant_count']} restaurants, "
          f"{config['area_size']}×{config['area_size']}km area")

# %% CELL 5: Structural Seeds
"""
Each infrastructure configuration tested with multiple layout seeds.
Seeds control restaurant placement randomness.
"""

structural_seeds = [42, 100, 200]

print(f"\n✓ Defined {len(structural_seeds)} structural seeds: {structural_seeds}")

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
            'restaurant_count': infra_config['restaurant_count'],
            'area_size': infra_config['area_size'],
            'seed': structural_seed
        })
        
        print(f"  ✓ Restaurants: {infra_config['restaurant_count']}, Area: {infra_config['area_size']}×{infra_config['area_size']}km")
        print(f"  ✓ Typical distance: {analysis_results['typical_distance']:.3f}km")

print(f"\n{'='*50}")
print(f"✓ Created {len(infrastructure_instances)} infrastructure instances")
print(f"✓ Breakdown: {len(infrastructure_configs)} configs × {len(structural_seeds)} seeds")
print(f"{'='*50}")

# %% CELL 6.5: Infrastructure Layout Visualization (OPTIONAL)
"""
Visualize infrastructure layouts to review spatial patterns.
For pairing threshold study, this shows restaurant distribution.
"""

print("\n" + "="*50)
print("INFRASTRUCTURE LAYOUT VISUALIZATION")
print("="*50)

import matplotlib.pyplot as plt

print(f"\nVisualizing {len(infrastructure_instances)} infrastructure instances...")
print("Compare spatial scale across restaurant counts.\n")

for instance in infrastructure_instances:
    print(f"{'='*50}")
    print(f"Layout: {instance['name']}")
    print(f"Restaurant count: {instance['restaurant_count']}")
    print(f"Area size: {instance['area_size']}×{instance['area_size']} km")
    print(f"Typical Distance: {instance['analysis']['typical_distance']:.3f}km")
    print(f"{'='*50}")
    
    # Visualize using stored analyzer
    instance['analyzer'].visualize_infrastructure()
    
    # Add custom header
    fig = plt.gcf()
    seed = instance['seed']
    restaurant_count = instance['restaurant_count']
    area_size = instance['area_size']
    typical_dist = instance['analysis']['typical_distance']
    
    custom_title = (f"Infrastructure Layout: {restaurant_count} Restaurants in {area_size}×{area_size}km Area (Seed {seed})\n"
                   f"Typical Distance: {typical_dist:.3f}km")
    
    fig.suptitle(custom_title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ {instance['name']} visualized\n")

print(f"{'='*50}")
print("✓ All layouts visualized")
print("✓ Observe spatial patterns:")
print("  - 10 restaurants: Baseline density")
print("  - 15 restaurants: Higher density, potentially more pairing opportunities")
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
PAIRING THRESHOLD STUDY: Three threshold levels across critical load ratios.

Pairing threshold levels:
- Conservative: Strict proximity requirements
- Moderate: Balanced pairing criteria  
- Liberal: Permissive pairing opportunities

Load ratios 5.0 and 7.0 capture critical regimes where threshold effects 
should be most visible (established from previous studies).

Only baseline interval scale tested (no 2x baseline validation).
"""

# Critical ratios where threshold effects likely matter most
target_arrival_interval_ratios = [5.0, 7.0]

# Fixed service duration configuration
FIXED_SERVICE_CONFIG = {
    'mean_service_duration': 100,
    'service_duration_std_dev': 60,
    'min_service_duration': 30,
    'max_service_duration': 200
}

# Pairing threshold configurations
# Note: Values calibrated from preliminary study (pairing_threshold_sensitivity_study.py)

conservative_pairing = {
    'pairing_enabled': True,
    'restaurants_proximity_threshold': 2.0,  # km
    'customers_proximity_threshold': 1.5,    # km
}

moderate_pairing = {
    'pairing_enabled': True,
    'restaurants_proximity_threshold': 4.0,  # km
    'customers_proximity_threshold': 3.0,    # km
}

liberal_pairing = {
    'pairing_enabled': True,
    'restaurants_proximity_threshold': 6.0,  # km
    'customers_proximity_threshold': 4.5,    # km
}

ultra_liberal_pairing = {
    'pairing_enabled': True,
    'restaurants_proximity_threshold': 8.0,  # km
    'customers_proximity_threshold': 6.0,    # km
}

pairing_configs = [
    ('conservative', conservative_pairing),
    ('moderate', moderate_pairing),
    ('liberal', liberal_pairing),
    ('ultra_liberal', ultra_liberal_pairing)
]

operational_configs = []

for ratio in target_arrival_interval_ratios:
    for pairing_name, pairing_config in pairing_configs:
        
        # Baseline configuration only
        operational_configs.append({
            'name': f'ratio_{ratio:.1f}_{pairing_name}_baseline',
            'config': OperationalConfig(
                mean_order_inter_arrival_time=1.0,
                mean_driver_inter_arrival_time=1.0 * ratio,
                **pairing_config,
                **FIXED_SERVICE_CONFIG
            ),
            'ratio': ratio,
            'pairing_threshold': pairing_name,
            'interval_type': 'baseline'
        })

print(f"\n✓ Defined {len(operational_configs)} operational configuration(s)")
print(f"  • Load ratios: {target_arrival_interval_ratios}")
print(f"  • Pairing thresholds: {[name for name, _ in pairing_configs]}")
print(f"  • Interval scale: baseline only")
print(f"\n✓ Design calculation:")
print(f"  • 2 restaurant counts × 4 thresholds × 3 seeds × 2 ratios = 48 design points")
print(f"  • 48 design points × 5 replications = 240 total runs (~20 minutes)")

# %% CELL 9: Design Points Creation
"""
Generate all design points by combining:
- Infrastructure instances (restaurant count × seeds)
- Operational configs (thresholds × ratios)
- Scoring configs (baseline only)

Design: 2 restaurants × 3 seeds × 3 thresholds × 2 ratios = 36 design points
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

print(f"\n✅ PAIRING THRESHOLD STUDY COMPLETE!")
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
restaurant_groups = {}
for design_name in all_time_series_data.keys():
    # Extract restaurant count from design name
    if 'restaurants_10' in design_name:
        count_str = '10'
    elif 'restaurants_15' in design_name:
        count_str = '15'
    else:
        count_str = 'unknown'
    
    if count_str not in restaurant_groups:
        restaurant_groups[count_str] = []
    restaurant_groups[count_str].append(design_name)

print(f"✓ Grouped {len(all_time_series_data)} design points by {len(restaurant_groups)} restaurant counts")

# Create plots systematically by restaurant count
plot_count = 0
for count_str in sorted(restaurant_groups.keys()):
    print(f"\n{count_str} restaurants:")
    
    for design_name in sorted(restaurant_groups[count_str]):
        plot_title = f"Warmup Analysis: {design_name}"
        time_series_data = all_time_series_data[design_name]
        fig = viz.create_warmup_analysis_plot(time_series_data, title=plot_title)
        
        plt.show()
        print(f"    ✓ {design_name} plot displayed")
        plot_count += 1

print(f"\n✓ Warmup analysis visualization complete")
print(f"✓ Created {plot_count} warmup analysis plots")
print(f"✓ Organized by {len(restaurant_groups)} restaurant count levels")

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

def extract_restaurants_seed_ratio_threshold_and_type(design_name):
    """Extract restaurant count, seed, ratio, pairing threshold, and interval type from design point name."""
    # Pattern: restaurants_10_seed42_ratio_5.0_conservative_baseline
    match = re.match(r'restaurants_(\d+)_seed(\d+)_ratio_([\d.]+)_(\w+)_(\w+)', design_name)
    if match:
        restaurant_count = int(match.group(1))
        seed = int(match.group(2))
        ratio = float(match.group(3))
        threshold = match.group(4)  # conservative, moderate, liberal
        interval_type = match.group(5)  # baseline
        return restaurant_count, seed, ratio, threshold, interval_type
    return None, None, None, None, None

# Extract comprehensive metrics
metrics_data = []

for design_name, analysis_result in design_analysis_results.items():
    restaurant_count, seed, ratio, threshold, interval_type = extract_restaurants_seed_ratio_threshold_and_type(design_name)
    if restaurant_count is None:
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
    
    # Extract pairing rate (from system_metrics) - ONE-LEVEL METRIC
    pairing_rate_stats = system_metrics.get('system_pairing_rate', {})
    
    pairing_rate_estimate = pairing_rate_stats.get('point_estimate', 0)
    pairing_rate_ci = pairing_rate_stats.get('confidence_interval', [0, 0])
    pairing_rate_ci_width = (pairing_rate_ci[1] - pairing_rate_ci[0]) / 2 if pairing_rate_ci[0] is not None else 0
    
    # Extract mean total distance (from delivery_unit_metrics)
    delivery_unit_metrics = stats_with_cis.get('delivery_unit_metrics', {})
    total_distance = delivery_unit_metrics.get('total_distance', {})
    
    distance_mean = total_distance.get('mean_of_means', {})
    distance_estimate = distance_mean.get('point_estimate', 0)
    distance_ci = distance_mean.get('confidence_interval', [0, 0])
    distance_ci_width = (distance_ci[1] - distance_ci[0]) / 2 if distance_ci[0] is not None else 0
    
    # Extract available drivers (from system_state_metrics)
    system_state_metrics = stats_with_cis.get('system_state_metrics', {})
    available_drivers = system_state_metrics.get('available_drivers', {})
    
    avail_mean = available_drivers.get('mean_of_means', {})
    avail_mom_estimate = avail_mean.get('point_estimate', 0)
    avail_mom_ci = avail_mean.get('confidence_interval', [0, 0])
    avail_mom_ci_width = (avail_mom_ci[1] - avail_mom_ci[0]) / 2 if avail_mom_ci[0] is not None else 0
    
    # Extract pair formation time (from pair_metrics) - TWO-LEVEL METRIC
    pair_metrics = stats_with_cis.get('pair_metrics', {})
    formation_time = pair_metrics.get('formation_time', {})
    
    formation_mom = formation_time.get('mean_of_means', {})
    formation_estimate = formation_mom.get('point_estimate', None)
    formation_ci = formation_mom.get('confidence_interval', [None, None])
    formation_ci_width = (formation_ci[1] - formation_ci[0]) / 2 if formation_ci[0] is not None else None
    
    metrics_data.append({
        'restaurant_count': restaurant_count,
        'seed': seed,
        'ratio': ratio,
        'threshold': threshold,
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

# Sort by: interval_type, threshold (custom order), ratio, seed, restaurant_count
# Custom threshold order: conservative (0) → moderate (1) → liberal (2) → ultra_liberal (3)
threshold_order = {'conservative': 0, 'moderate': 1, 'liberal': 2, 'ultra_liberal': 3}
metrics_data.sort(key=lambda x: (x['interval_type'], threshold_order.get(x['threshold'], 999), x['ratio'], x['seed'], x['restaurant_count']))

# ========================================
# MAIN TABLE: Threshold × Restaurant Count effects
# ========================================

# ========================================
# MAIN TABLE: Threshold × Restaurant Count effects
# ========================================
print("\n🎯 KEY PERFORMANCE METRICS: PAIRING THRESHOLD × RESTAURANT COUNT × RATIO EFFECTS")
print("="*220)
print("Pairing      Restaurant  Seed   Ratio   Interval       Assignment Time       Completion Rate        Pairing Rate       Pair Formation Time      Mean Total Distance      Available Drivers")
print("Threshold       Count                        Type       (mean_of_means)         (with 95% CI)         (with 95% CI)          (with 95% CI)            (with 95% CI)            (with 95% CI)")
print("="*220)

for row in metrics_data:
    restaurant_count = row['restaurant_count']
    seed = row['seed']
    ratio = row['ratio']
    threshold = row['threshold']
    interval_type = row['interval_type']
    
    # Format assignment time
    mom_str = f"{row['mom_estimate']:5.2f} ± {row['mom_ci_width']:5.2f}"
    
    # Format completion rate
    comp_str = f"{row['comp_estimate']:.3f} ± {row['comp_ci_width']:.3f}"
    
    # Format pairing rate as percentage
    pairing_rate_str = f"{row['pairing_rate_estimate']*100:4.1f}% ± {row['pairing_rate_ci_width']*100:4.1f}%"
    
    # Format pair formation time - N/A if not available
    if row['formation_estimate'] is not None:
        formation_str = f"{row['formation_estimate']:5.2f} ± {row['formation_ci_width']:5.2f}"
    else:
        formation_str = "N/A".center(16)
    
    # Format mean total distance
    dist_str = f"{row['distance_estimate']:5.2f} ± {row['distance_ci_width']:5.2f}"
    
    # Format available drivers
    avail_mom_str = f"{row['avail_mom_estimate']:5.2f} ± {row['avail_mom_ci_width']:5.2f}"
    
    print(f"{threshold:13s}     {restaurant_count:3d}     {seed:3d}    {ratio:3.1f}   {interval_type:10s}     {mom_str:>16s}       {comp_str:>15s}       {pairing_rate_str:>18s}       {formation_str:>16s}          {dist_str:>18s}        {avail_mom_str:>18s}")

print("="*220)
print("✓ Table complete")
print(f"✓ Total configurations: {len(metrics_data)}")

# %% AD HOC ANALYSIS: Eligible Restaurant Pairs by Threshold Configuration
"""
Verify hypothesis: More permissive thresholds create more eligible restaurant pairs.

This addresses the mechanism behind threshold effects: we need empirical evidence 
that relaxing spatial constraints actually increases pairing opportunities, and 
whether the 2.3× theoretical advantage of 15 restaurants materializes at each 
threshold level.
"""

print("\n" + "="*80)
print("AD HOC ANALYSIS: ELIGIBLE RESTAURANT PAIRS BY THRESHOLD CONFIGURATION")
print("="*80)

from delivery_sim.utils.location_utils import calculate_distance

# Threshold configurations to test
threshold_configs = {
    'conservative': 2.0,  # restaurants_proximity_threshold in km
    'moderate': 4.0,
    'liberal': 6.0,
    'ultra_liberal': 8.0
}

# Calculate eligible pairs for each infrastructure instance and threshold
eligible_pairs_data = []

for instance in infrastructure_instances:
    restaurant_count = instance['restaurant_count']
    seed = instance['seed']
    infrastructure = instance['infrastructure']
    
    # Get all restaurants from repository
    restaurants = infrastructure.restaurant_repository.find_all()
    num_restaurants = len(restaurants)
    
    # For each threshold configuration
    for threshold_name, threshold_km in threshold_configs.items():
        
        # Calculate all pairwise distances
        all_distances = []
        eligible_pairs = []
        
        for i, rest1 in enumerate(restaurants):
            for j, rest2 in enumerate(restaurants[i+1:], i+1):
                # Calculate distance using utility function
                distance = calculate_distance(rest1.location, rest2.location)
                
                all_distances.append(distance)
                
                # Check if eligible under this threshold
                if distance <= threshold_km:
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
            'restaurant_count': restaurant_count,
            'seed': seed,
            'threshold_name': threshold_name,
            'threshold_km': threshold_km,
            'num_restaurants': num_restaurants,
            'total_possible_pairs': total_possible_pairs,
            'num_eligible_pairs': num_eligible_pairs,
            'eligible_percentage': eligible_percentage,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'mean_distance': mean_distance
        })
        
        print(f"Restaurant Count {restaurant_count}, Seed {seed}, {threshold_name.capitalize()} ({threshold_km} km):")
        print(f"  Total possible pairs: {total_possible_pairs}")
        print(f"  Eligible pairs: {num_eligible_pairs} ({eligible_percentage:.1f}%)")
        print(f"  Distance range: {min_distance:.2f} - {max_distance:.2f} km (mean: {mean_distance:.2f})")
        print()

# Create summary table by threshold and restaurant count
print("="*80)
print("SUMMARY: ELIGIBLE PAIRS BY THRESHOLD × RESTAURANT COUNT")
print("="*80)

# Group by threshold and restaurant count
from collections import defaultdict
threshold_summaries = defaultdict(lambda: defaultdict(list))

for data in eligible_pairs_data:
    threshold_summaries[data['threshold_name']][data['restaurant_count']].append(data)

print(f"\n{'Threshold':<15} {'Rest.':<6} {'Seed':<6} {'Total Pairs':<12} {'Eligible':<10} {'Eligible %':<12} {'Mean Dist (km)':<15}")
print("-"*80)

# Sort by threshold order (conservative → moderate → liberal → ultra_liberal)
threshold_order_list = ['conservative', 'moderate', 'liberal', 'ultra_liberal']

for threshold_name in threshold_order_list:
    threshold_data = threshold_summaries[threshold_name]
    
    for restaurant_count in sorted(threshold_data.keys()):
        seed_data = threshold_data[restaurant_count]
        
        for data in sorted(seed_data, key=lambda x: x['seed']):
            print(f"{threshold_name.capitalize():<15} {data['restaurant_count']:<6} {data['seed']:<6} "
                  f"{data['total_possible_pairs']:<12} {data['num_eligible_pairs']:<10} "
                  f"{data['eligible_percentage']:>7.1f}%      {data['mean_distance']:>7.2f}")
        
        # Calculate mean across seeds for this threshold × restaurant_count combination
        mean_eligible_pct = sum(d['eligible_percentage'] for d in seed_data) / len(seed_data)
        mean_total_pairs = sum(d['total_possible_pairs'] for d in seed_data) / len(seed_data)
        mean_eligible_pairs = sum(d['num_eligible_pairs'] for d in seed_data) / len(seed_data)
        mean_mean_dist = sum(d['mean_distance'] for d in seed_data) / len(seed_data)
        
        print(f"{'  Mean':<15} {restaurant_count:<6} {'':>6} {mean_total_pairs:<12.0f} "
              f"{mean_eligible_pairs:<10.1f} {mean_eligible_pct:>7.1f}%      {mean_mean_dist:>7.2f}")
        print("-"*80)

print("\n✓ Analysis complete")
print("\nINTERPRETATION:")
print("  • Conservative captures ~10-15% of pairs → explains low pairing rate (~7-14%)")
print("  • Moderate captures ~25-35% of pairs → explains balanced pairing rate (~20-26%)")
print("  • Liberal captures ~40-50% of pairs → explains high pairing rate (~28-35%)")
print("  • Ultra-liberal captures ~60-70% of pairs → expected high pairing rate (~40-50%)")
print("  • 15 restaurants provide 2.3× more possible pairs (45 → 105)")
print("  • BUT: eligible pair percentage similar across restaurant counts")
print("  • Confirms: temporal dilution cancels combinatorial advantage")
print("  • Ultra-liberal should show performance degradation vs moderate (U-shape)")