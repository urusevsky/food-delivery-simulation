# experimental_study_template.py
"""
TEMPLATE: Flexible Experimental Study Design for Food Delivery Simulation

This template supports varying:
- Infrastructure parameters (delivery area size, restaurant count, layout seeds)
- Operational parameters (arrival rates, pairing configs, service durations, etc.)
- Scoring parameters (priority scoring weights)
- All combinations (interaction studies)

Design Philosophy:
- Single configuration = list with 1 element (no special cases)
- Pre-create infrastructure instances for reviewability
- Uniform nested loop structure for design point creation
- Clear cell-based workflow for reproducibility
- Generic placeholders - adapt for your specific research question

Usage:
1. Copy this template for your specific research question
2. Fill in configurations (Cells 4-5, 7-8)
3. Create infrastructure instances (Cell 6)
4. Review if needed (Cell 6.5)
5. Run uniform preprocessing steps (Cells 9-14)
6. Implement custom analysis (Cells 15+)

Study Type Examples:
- Infrastructure study: Multiple infrastructure configs × 1 operational × 1 scoring
- Operational study: 1 infrastructure config × Multiple operational × 1 scoring  
- Weight study: 1 infrastructure × 1 operational × Multiple scoring
- Interaction study: Multiple across any combination
"""

# %% CELL 1: Enable Autoreload (Development Convenience)
"""
Automatically reload modules when they change.
Useful during development, can be removed for production runs.
"""
%load_ext autoreload
%autoreload 2

# %% CELL 2: Setup and Imports
"""
Standard imports for food delivery simulation experiments.
"""
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
print("EXPERIMENTAL STUDY: [YOUR STUDY NAME]")
print("="*80)
print("Research Question: [STATE YOUR RESEARCH QUESTION]")
print("Hypothesis: [STATE YOUR HYPOTHESIS]")

# %% CELL 3: Logging Configuration
"""
Configure logging verbosity.
Suppress verbose component logs, keep progress indicators.
"""
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

# %% CELL 3.5: Research Question and Study Purpose
"""
REQUIRED: Articulate what you're investigating and why.

This isn't bureaucracy - it's your decision anchor for:
- Which parameters to vary (configurations)
- Which metrics to focus on (analysis)
- What patterns to look for (interpretation)

Be specific enough to guide decisions, flexible enough to evolve.
"""

print("\n" + "="*80)
print("RESEARCH QUESTION AND PURPOSE")
print("="*80)

research_question = """
What specific phenomenon or gradient are you investigating?
Example: How does the ratio of order arrival rate to driver arrival rate 
affect system stability and performance?
"""

purpose_and_rationale = """
Why does this question matter for understanding delivery systems?
Example: This reveals whether the system is supply-constrained or 
demand-constrained, and identifies critical operating regimes.
"""

expected_pattern = """
What do you expect to find? (Intuition, not formal hypothesis)
Example: Performance should degrade as load ratio increases, but 
the functional form (linear/threshold/exponential) is unknown.
"""

analysis_focus = """
Given this question, which metrics are most relevant?
Example: Primary = assignment_time, completion_rate
         Secondary = driver_utilization, order_waiting_time
         Context = time series patterns to identify regime transitions
"""

print(research_question)
print(purpose_and_rationale)
print(expected_pattern)
print(analysis_focus)

print("\n" + "="*80)
print("✓ Research question articulated - use this to guide all subsequent decisions")
print("="*80)

# %% CELL 4: Infrastructure Configuration(s)
"""
Define infrastructure configuration(s) to test.

Parameters:
- delivery_area_size: Side length of square delivery area (km)
- num_restaurants: Number of restaurants in the area
- driver_speed: Constant driver speed (km/min)

Use generic names (infra_A, infra_B) or descriptive names based on your study.
"""

# ==== EXAMPLE 1: Infrastructure Study (Generic Placeholders) ====
infrastructure_configs = [
    {
        'name': 'infra_A',
        'config': StructuralConfig(
            delivery_area_size=10,
            num_restaurants=10,
            driver_speed=0.5
        )
    },
    {
        'name': 'infra_B',
        'config': StructuralConfig(
            delivery_area_size=10,
            num_restaurants=15,
            driver_speed=0.5
        )
    },
    {
        'name': 'infra_C',
        'config': StructuralConfig(
            delivery_area_size=10,
            num_restaurants=20,
            driver_speed=0.5
        )
    }
]

# ==== EXAMPLE 2: Operational Study (Fixed Infrastructure) ====
# infrastructure_configs = [
#     {
#         'name': 'baseline',
#         'config': StructuralConfig(
#             delivery_area_size=10,
#             num_restaurants=10,
#             driver_speed=0.5
#         )
#     }
# ]

print(f"✓ Defined {len(infrastructure_configs)} infrastructure configuration(s)")
for config in infrastructure_configs:
    struct_config = config['config']
    density = struct_config.num_restaurants / (struct_config.delivery_area_size ** 2)
    print(f"  • {config['name']}: {struct_config.num_restaurants} restaurants, "
          f"area={struct_config.delivery_area_size}km, density={density:.4f}/km²")

# %% CELL 5: Structural Seeds (Layout Variation)
"""
Define structural seeds for testing layout robustness.

QUICK EXPLORATION: [42]
ROBUSTNESS CHECK: [42, 123, 999]
COMPREHENSIVE: [42, 123, 456, 789, 999]

Same seeds used across ALL infrastructures for fair comparison.
"""

# For quick exploration
structural_seeds = [42]

# For robustness testing
# structural_seeds = [42, 123, 999]

print(f"✓ Structural seeds: {structural_seeds} ({len(structural_seeds)} layout(s) per infrastructure)")

# %% CELL 6: Create Infrastructure Instances
"""
Create and analyze all infrastructure instances.

This allows reviewing infrastructures (e.g., visualizing layouts) 
before proceeding to design point creation.
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

# %% CELL 6.5: [OPTIONAL] Review Infrastructure Instances
"""
Optional cell for reviewing/visualizing infrastructures before proceeding.

Uncomment and run if you want to inspect specific instances.
Examples:
- Visualize restaurant layouts
- Review typical distances
- Check spatial distributions
"""

# Example: Visualize first infrastructure
# from delivery_sim.infrastructure.infrastructure_visualizer import visualize_infrastructure
# visualize_infrastructure(infrastructure_instances[0]['infrastructure'])

# Example: Review typical distances across all instances
# print("\n📊 Infrastructure Analysis Summary:")
# for instance in infrastructure_instances:
#     print(f"  {instance['name']:20s}: typical_distance={instance['analysis']['typical_distance']:.3f}km")

# Example: Visualize specific instance by name
# target_instance = next(i for i in infrastructure_instances if i['name'] == 'infra_A_seed42')
# visualize_infrastructure(target_instance['infrastructure'])

# %% CELL 7: Scoring Configuration(s)
"""
Define scoring configuration(s) for priority scoring system.

MOST STUDIES: Single baseline configuration
WEIGHT STRATEGY STUDY: Multiple configurations to compare

If studying weight strategies, define multiple configs here.
Otherwise, use single default config.
"""

# For most studies - single default config
scoring_configs = [
    {
        'name': 'baseline',
        'config': ScoringConfig()  # Uses defaults
    }
]

# For weight strategy studies - multiple configs
# scoring_configs = [
#     {
#         'name': 'distance_focused',
#         'config': ScoringConfig(
#             weight_distance=0.5,
#             weight_throughput=0.3,
#             weight_fairness=0.2
#         )
#     },
#     {
#         'name': 'throughput_focused',
#         'config': ScoringConfig(
#             weight_distance=0.3,
#             weight_throughput=0.5,
#             weight_fairness=0.2
#         )
#     },
# ]

print(f"✓ Defined {len(scoring_configs)} scoring configuration(s)")
for config in scoring_configs:
    print(f"  • {config['name']}")

# %% CELL 8: Operational Configuration(s)
"""
Define operational configuration(s) to test.

Parameters:
- Arrival rates: mean_order_inter_arrival_time, mean_driver_inter_arrival_time
- Pairing: pairing_enabled, restaurants_proximity_threshold, customers_proximity_threshold
- Service: mean_service_duration, service_duration_std_dev, min/max_service_duration

Use generic names (oper_A, oper_B) or descriptive names based on your study.
"""

# ==== EXAMPLE 1: Operational Study (Generic Placeholders) ====
# operational_configs = [
#     {
#         'name': 'oper_A',
#         'config': OperationalConfig(
#             mean_order_inter_arrival_time=1.0,
#             mean_driver_inter_arrival_time=0.5,
#             pairing_enabled=True,
#             restaurants_proximity_threshold=4.0,
#             customers_proximity_threshold=3.0,
#             mean_service_duration=120,
#             service_duration_std_dev=60,
#             min_service_duration=30,
#             max_service_duration=240
#         )
#     },
#     {
#         'name': 'oper_B',
#         'config': OperationalConfig(
#             mean_order_inter_arrival_time=1.0,
#             mean_driver_inter_arrival_time=0.8,
#             pairing_enabled=True,
#             restaurants_proximity_threshold=4.0,
#             customers_proximity_threshold=3.0,
#             mean_service_duration=120,
#             service_duration_std_dev=60,
#             min_service_duration=30,
#             max_service_duration=240
#         )
#     }
# ]

# ==== EXAMPLE 2: Infrastructure Study (Fixed Operational) ====
operational_configs = [
    {
        'name': 'baseline',
        'config': OperationalConfig(
            mean_order_inter_arrival_time=1.0,
            mean_driver_inter_arrival_time=0.8,
            pairing_enabled=True,
            restaurants_proximity_threshold=4.0,
            customers_proximity_threshold=3.0,
            mean_service_duration=120,
            service_duration_std_dev=60,
            min_service_duration=30,
            max_service_duration=240
        )
    }
]

print(f"✓ Defined {len(operational_configs)} operational configuration(s)")
for config in operational_configs:
    op_config = config['config']
    arrival_interval_ratio = op_config.mean_driver_inter_arrival_time / op_config.mean_order_inter_arrival_time
    print(f"  • {config['name']}: arrival_interval_ratio={arrival_interval_ratio:.2f}, "
          f"pairing={'enabled' if op_config.pairing_enabled else 'disabled'}")

# %% CELL 9: Design Point Creation (SIMPLIFIED)
"""
Create design points from all combinations of:
- Infrastructure instances (pre-created)
- Operational configurations
- Scoring configurations

The triple nested loop structure is universal across all study types.
For most studies, one or more lists have only 1 element.
"""

design_points = {}

print("\n" + "="*50)
print("DESIGN POINTS CREATION")
print("="*50)

for infra_instance in infrastructure_instances:
    for op_config in operational_configs:
        for scoring_config_dict in scoring_configs:
            
            # Generate unique design point name
            # For single scoring config studies:
            design_name = f"{infra_instance['name']}_{op_config['name']}"
            
            # For weight strategy studies, include scoring config in name:
            # design_name = f"{infra_instance['name']}_{op_config['name']}_{scoring_config_dict['name']}"
            
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
"""
Define experiment execution parameters.

Trade-offs:
- Longer duration → Better steady-state estimates, slower execution
- More replications → Tighter confidence intervals, slower execution
- Shorter interval → More granular time series, larger data
"""

experiment_config = ExperimentConfig(
    simulation_duration=2000,  # Adjust based on warmup needs
    num_replications=5,        # Adjust based on variance needs
    operational_master_seed=100,
    collection_interval=1.0    # 1 minute intervals for time series
)

total_runs = len(design_points) * experiment_config.num_replications
estimated_time = total_runs * 5  # Rough estimate: 5 seconds per run

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
Progress logged to console.
"""

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
print(f"✓ Results structure: dict[design_point_name] → list[replication_results]")

# %% CELL 12: Time Series Data Processing for Warmup Analysis
"""
UNIFORM STEP: Extract time series data for warmup detection.

This step is the same for all studies - we always use:
- active_drivers: Primary indicator (Little's Law convergence)
- unassigned_delivery_entities: Auxiliary indicator (queue behavior)
"""

print("\n" + "="*50)
print("TIME SERIES DATA PROCESSING FOR WARMUP ANALYSIS")
print("="*50)

from delivery_sim.warmup_analysis.time_series_processing import extract_warmup_time_series

print("Processing time series data for warmup detection...")

all_time_series_data = extract_warmup_time_series(
    study_results=study_results,
    design_points=design_points,
    metrics=['active_drivers', 'unassigned_delivery_entities'],
    moving_average_window=100  # Adjust window size based on simulation duration
)

print(f"✓ Time series processing complete for {len(all_time_series_data)} design points")
print(f"✓ Metrics extracted: active_drivers, unassigned_delivery_entities")
print(f"✓ Ready for warmup analysis visualization")

# %% CELL 13: Warmup Analysis Visualization
"""
UNIFORM STEP: Visualize time series to identify warmup period.

Plots show:
- Active drivers vs Little's Law theoretical value (warmup indicator)
- Unassigned entities (queue behavior indicator)
"""

print("\n" + "="*50)
print("WARMUP ANALYSIS VISUALIZATION")
print("="*50)

from delivery_sim.warmup_analysis.visualization import WelchMethodVisualization
import matplotlib.pyplot as plt

print("Creating warmup analysis plots...")

# Initialize visualization
viz = WelchMethodVisualization(figsize=(16, 10))

# Create plots for each design point (or subset for quick inspection)
# For large studies, you may want to plot only a representative subset
for design_name, time_series_data in all_time_series_data.items():
    print(f"\nPlotting: {design_name}")
    
    fig = viz.create_warmup_analysis_plot(
        time_series_data, 
        title=f"Warmup Analysis: {design_name}"
    )
    
    plt.show()
    print(f"  ✓ {design_name} displayed")

print(f"\n✓ Warmup analysis visualization complete")
print(f"✓ Inspect plots to determine warmup period")
print(f"  • Look for active_drivers stabilizing around Little's Law value")
print(f"  • Note any transient behavior in unassigned_delivery_entities")

# %% CELL 14: Warmup Period Determination
"""
MANUAL STEP: Set warmup period based on visual inspection.

Update the value below after inspecting Cell 13 plots.
"""

print("\n" + "="*50)
print("WARMUP PERIOD DETERMINATION")
print("="*50)

# ⚠️ UPDATE THIS VALUE based on visual inspection of Cell 13
uniform_warmup_period = 500  # minutes

print(f"✓ Warmup period set: {uniform_warmup_period} minutes")
print(f"✓ Analysis window: {experiment_config.simulation_duration - uniform_warmup_period} minutes of steady-state data")
print(f"⚠️  Remember to update this value based on your warmup plots!")

# %% CELL 15: Process Through Analysis Pipeline
"""
UNIFORM STEP: Run all design points through analysis pipeline.

This step extracts metrics from post-warmup period and computes statistics.
Results stored in design_analysis_results for subsequent analysis.
"""

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
print(f"✓ Available metrics: order_metrics (assignment time, etc.), system_metrics (completion rate, etc.)")

# %% CELL 16: Extract and Present Key Metrics
"""
CUSTOM STEP: Extract metrics relevant to your research question.

This section depends on what you're studying. Examples:
- Infrastructure study: Compare metrics across infrastructure configs
- Operational study: Compare metrics across operational parameters
- Interaction study: Analyze interaction effects

Structure shown here is a placeholder - customize for your needs.
"""

print("\n" + "="*80)
print("METRIC EXTRACTION AND PRESENTATION")
print("="*80)
print("\n⚠️  TODO: Implement custom metric extraction for your research question")
print("\nExample workflow:")
print("  1. Decide which metrics matter for your hypothesis")
print("  2. Extract from design_analysis_results")
print("  3. Group/aggregate as needed (e.g., by infrastructure, by arrival ratio)")
print("  4. Compute summary statistics (mean, std, confidence intervals)")
print("  5. Present in tables or prepare for visualization")

# Example structure (customize for your study):
# for design_name, analysis_result in design_analysis_results.items():
#     # Extract metrics from statistics_with_cis
#     stats_with_cis = analysis_result.get('statistics_with_cis', {})
#     
#     # Order metrics (nested under mean_of_means)
#     order_metrics = stats_with_cis.get('order_metrics', {})
#     assignment_time = order_metrics.get('assignment_time', {}).get('mean_of_means', {})
#     mean_time = assignment_time.get('point_estimate')
#     ci = assignment_time.get('confidence_interval', [None, None])
#     
#     # System metrics (flat structure)
#     system_metrics = stats_with_cis.get('system_metrics', {})
#     completion_rate = system_metrics.get('system_completion_rate', {})
#     mean_comp = completion_rate.get('point_estimate')
#     ci_comp = completion_rate.get('confidence_interval', [None, None])
#     
#     print(f"\n{design_name}:")
#     print(f"  Assignment Time: {mean_time:.2f} min [{ci[0]:.2f}, {ci[1]:.2f}]")
#     print(f"  Completion Rate: {mean_comp:.3f} [{ci_comp[0]:.3f}, {ci_comp[1]:.3f}]")

# %% CELL 17: Visualization and Results Presentation
"""
CUSTOM STEP: Create visualizations to communicate findings.

Visualization type depends on your study:
- Infrastructure study: Bar plots comparing across infrastructures
- Operational study: Line plots showing trends
- Interaction study: Heatmaps or faceted plots

Customize based on your research question.
"""

print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)
print("\n⚠️  TODO: Implement visualizations for your research question")
print("\nConsider:")
print("  • What comparisons best test your hypothesis?")
print("  • How to show uncertainty (error bars, confidence bands)?")
print("  • Whether to use subplots for multiple metrics")
print("  • Color schemes and labels for clarity")

# Example visualization structure:
# import matplotlib.pyplot as plt
# import numpy as np
#
# fig, ax = plt.subplots(figsize=(10, 6))
# 
# # Extract data for plotting
# # ... your extraction logic ...
# 
# # Create plot
# ax.bar(x_positions, means, yerr=confidence_intervals, capsize=5)
# ax.set_xlabel('Configuration')
# ax.set_ylabel('Metric Value')
# ax.set_title('Performance Comparison Across Configurations')
# plt.show()

# %% CELL 18: Statistical Analysis (Optional)
"""
CUSTOM STEP: Perform statistical tests if needed.

Common tests:
- ANOVA: Compare means across multiple groups
- t-tests: Pairwise comparisons
- Correlation/regression: Relationship between factors and outcomes
"""

print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)
print("\n⚠️  TODO: Implement statistical tests if needed for your research")
print("\nConsider:")
print("  • Do you need to test for significant differences?")
print("  • Are there interaction effects to investigate?")
print("  • What's the appropriate significance level?")

# Example:
# from scipy import stats
# 
# # Extract metrics for two conditions
# condition_A_data = [...]
# condition_B_data = [...]
# 
# # Perform t-test
# t_stat, p_value = stats.ttest_ind(condition_A_data, condition_B_data)
# print(f"t-test: t={t_stat:.3f}, p={p_value:.4f}")
# 
# if p_value < 0.05:
#     print("Significant difference detected (p < 0.05)")

# %% CELL 19: Save Results (Optional)
"""
CUSTOM STEP: Save processed results and figures.

Saves for reproducibility and thesis writing.
"""

print("\n" + "="*80)
print("SAVE RESULTS")
print("="*80)
print("\n⚠️  TODO: Implement result saving if needed")

# Example:
# import pickle
# import json
# 
# results_dir = f'results/{your_study_name}/'
# os.makedirs(results_dir, exist_ok=True)
# 
# # Save processed data
# with open(f'{results_dir}/analysis_results.pkl', 'wb') as f:
#     pickle.dump(design_analysis_results, f)
# 
# # Save figures
# fig.savefig(f'{results_dir}/main_comparison.pdf')
# 
# # Save metadata
# metadata = {
#     'num_design_points': len(design_points),
#     'warmup_period': uniform_warmup_period,
#     'simulation_duration': experiment_config.simulation_duration,
#     # ... other relevant info
# }
# with open(f'{results_dir}/metadata.json', 'w') as f:
#     json.dump(metadata, f, indent=2)

print("\n" + "="*80)
print("EXPERIMENTAL STUDY TEMPLATE COMPLETE")
print("="*80)
print("\nNext steps:")
print("  ✓ Cells 1-15: Uniform preprocessing - executed")
print("  ⚠️  Cells 16-19: Customize for your research question")
print("\nRemember:")
print("  • Warmup analysis is critical - don't skip it!")
print("  • Update warmup_period after visual inspection")
print("  • All subsequent analysis uses design_analysis_results")