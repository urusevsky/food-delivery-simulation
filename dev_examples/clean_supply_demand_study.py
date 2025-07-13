# dev_examples/clean_supply_demand_study.py
"""
Clean Supply-Demand Study Workflow

Research Question: "How do supply-demand conditions affect system performance?"

Simple, clean implementation following user's gradual approach:
- No feature creep
- Manual design points dictionary creation
- Clean DesignPoint and ExperimentalRunner
"""

# %% Import and Setup
"""
Cell 1: Clean imports
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

print("✓ Clean imports successful")

# %% Step 1: Logging Configuration
"""
Cell 2: Setup logging
"""
print("\n" + "="*60)
print("CLEAN SUPPLY-DEMAND STUDY WORKFLOW")
print("="*60)

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

# %% Step 2: Infrastructure Setup
"""
Cell 3: Create and analyze infrastructure
"""
print("\nStep 2: Infrastructure Setup")

structural_config = StructuralConfig(
    delivery_area_size=10,
    num_restaurants=10,
    driver_speed=0.5
)

master_seed = 42

infrastructure = Infrastructure(structural_config, master_seed)
analyzer = InfrastructureAnalyzer(infrastructure)
analysis_results = analyzer.analyze_complete_infrastructure()

print(f"✓ Infrastructure ready: typical_distance={analysis_results['typical_distance']:.3f}km")

# %% Step 3: Default ScoringConfig
"""
Cell 4: Define default scoring config
"""
print("\nStep 3: Default ScoringConfig")

scoring_config = ScoringConfig()  # Default values

print("✓ Default ScoringConfig defined")

# %% Step 4: Create Design Points Dictionary
"""
Cell 5: Manually create design points dictionary for supply-demand study
"""
print("\nStep 4: Create Design Points Dictionary")

# Base operational parameters
base_params = {
    'pairing_enabled': False,
    'restaurants_proximity_threshold': None,
    'customers_proximity_threshold': None,
    'immediate_assignment_threshold': 100,
    'mean_service_duration': 100,
    'service_duration_std_dev': 60,
    'min_service_duration': 30,
    'max_service_duration': 200,
    'periodic_interval': 3.0
}

# Manually create design points dictionary
design_points = {}

# Low demand conditions
design_points["low_demand_low_supply"] = DesignPoint(
    infrastructure=infrastructure,
    operational_config=OperationalConfig(
        mean_order_inter_arrival_time=3.0,
        mean_driver_inter_arrival_time=10.0,
        **base_params
    ),
    scoring_config=scoring_config,
    name="low_demand_low_supply"
)

design_points["low_demand_medium_supply"] = DesignPoint(
    infrastructure=infrastructure,
    operational_config=OperationalConfig(
        mean_order_inter_arrival_time=3.0,
        mean_driver_inter_arrival_time=8.0,
        **base_params
    ),
    scoring_config=scoring_config,
    name="low_demand_medium_supply"
)

design_points["low_demand_high_supply"] = DesignPoint(
    infrastructure=infrastructure,
    operational_config=OperationalConfig(
        mean_order_inter_arrival_time=3.0,
        mean_driver_inter_arrival_time=5.0,
        **base_params
    ),
    scoring_config=scoring_config,
    name="low_demand_high_supply"
)

# Medium demand conditions
design_points["medium_demand_low_supply"] = DesignPoint(
    infrastructure=infrastructure,
    operational_config=OperationalConfig(
        mean_order_inter_arrival_time=2.0,
        mean_driver_inter_arrival_time=10.0,
        **base_params
    ),
    scoring_config=scoring_config,
    name="medium_demand_low_supply"
)

design_points["medium_demand_medium_supply"] = DesignPoint(
    infrastructure=infrastructure,
    operational_config=OperationalConfig(
        mean_order_inter_arrival_time=2.0,
        mean_driver_inter_arrival_time=8.0,
        **base_params
    ),
    scoring_config=scoring_config,
    name="medium_demand_medium_supply"
)

design_points["medium_demand_high_supply"] = DesignPoint(
    infrastructure=infrastructure,
    operational_config=OperationalConfig(
        mean_order_inter_arrival_time=2.0,
        mean_driver_inter_arrival_time=5.0,
        **base_params
    ),
    scoring_config=scoring_config,
    name="medium_demand_high_supply"
)

# High demand conditions
design_points["high_demand_low_supply"] = DesignPoint(
    infrastructure=infrastructure,
    operational_config=OperationalConfig(
        mean_order_inter_arrival_time=1.0,
        mean_driver_inter_arrival_time=10.0,
        **base_params
    ),
    scoring_config=scoring_config,
    name="high_demand_low_supply"
)

design_points["high_demand_medium_supply"] = DesignPoint(
    infrastructure=infrastructure,
    operational_config=OperationalConfig(
        mean_order_inter_arrival_time=1.0,
        mean_driver_inter_arrival_time=8.0,
        **base_params
    ),
    scoring_config=scoring_config,
    name="high_demand_medium_supply"
)

design_points["high_demand_high_supply"] = DesignPoint(
    infrastructure=infrastructure,
    operational_config=OperationalConfig(
        mean_order_inter_arrival_time=1.0,
        mean_driver_inter_arrival_time=5.0,
        **base_params
    ),
    scoring_config=scoring_config,
    name="high_demand_high_supply"
)

print(f"✓ Created {len(design_points)} design points manually")
print(f"  • Design points: {list(design_points.keys())}")

# Show load ratios
print(f"\n📊 Load Ratios (driver_interval/order_interval):")
for name, dp in design_points.items():
    load_ratio = dp.operational_config.mean_driver_inter_arrival_time / dp.operational_config.mean_order_inter_arrival_time
    print(f"  • {name}: {load_ratio:.1f}")

# %% Step 5: ExperimentConfig
"""
Cell 6: Define experiment parameters
"""
print("\nStep 5: ExperimentConfig")

experiment_config = ExperimentConfig(
    simulation_duration=200,
    num_replications=5,
    master_seed=42
)

print(f"✓ Experiment config: {experiment_config.simulation_duration}min, {experiment_config.num_replications} reps")
print(f"  Total replications: {len(design_points)} × {experiment_config.num_replications} = {len(design_points) * experiment_config.num_replications}")

# %% Step 6: Execute Study
"""
Cell 7: Run the experimental study
"""
print("\nStep 6: Execute Experimental Study")

runner = ExperimentalRunner()
print("✓ ExperimentalRunner created")

print(f"\nExecuting study...")
study_results = runner.run_experimental_study(design_points, experiment_config)

print(f"\n✅ Study completed!")
print(f"  • Design points executed: {len(study_results)}")
print(f"  • Results available for warmup analysis")

# %% Step 7: Extract System Snapshots for Warmup Analysis
"""
Cell 8: Extract system_snapshots from study_results for warmup analysis
"""
print("\nStep 7: Extract System Snapshots for Warmup Analysis")

# Extract system snapshots for each design point
design_point_snapshots = {}

for design_name, design_result in study_results.items():
    replication_results = design_result['replication_results']
    
    # Extract snapshots from each replication
    design_snapshots = []
    for replication in replication_results:
        snapshots = replication['system_snapshots']
        if snapshots:
            design_snapshots.append(snapshots)
    
    design_point_snapshots[design_name] = design_snapshots
    print(f"  • {design_name}: {len(design_snapshots)} replications, {sum(len(rep) for rep in design_snapshots)} total snapshots")

print(f"\n✓ Extracted system snapshots for {len(design_point_snapshots)} design points")

# %% Step 8: Apply Welch's Method to Each Design Point
"""
Cell 9: Apply Welch's method to each design point for warmup detection
"""
print("\nStep 8: Apply Welch's Method to Each Design Point")

from delivery_sim.warmup_analysis.welch_analyzer import WelchAnalyzer
from delivery_sim.warmup_analysis.visualization import WarmupVisualization
import matplotlib.pyplot as plt

# Initialize Welch analyzer
analyzer = WelchAnalyzer()

# Warmup detection metrics
warmup_metrics = ['active_drivers', 'active_delivery_entities']

# Analyze each design point separately
design_point_welch_results = {}

for design_name, snapshots in design_point_snapshots.items():
    if not snapshots or not snapshots[0]:  # Skip if no data
        print(f"⚠️ No snapshot data for {design_name}")
        continue
    
    print(f"\n--- Analyzing {design_name} ---")
    
    # Apply Welch's method to this design point
    welch_results = analyzer.analyze_warmup_convergence(
        multi_replication_snapshots=snapshots,
        metrics=warmup_metrics,
        collection_interval=0.5  # Should match SystemDataCollector setting
    )
    
    design_point_welch_results[design_name] = welch_results
    
    # Show final values for this design point
    for metric_name, data in welch_results.items():
        if data['cumulative_average']:
            final_value = data['cumulative_average'][-1]
            print(f"  • {metric_name}: final stabilized value = {final_value:.1f}")

print(f"\n✓ Welch analysis completed for {len(design_point_welch_results)} design points")

# %% Step 9: Create Warmup Visualization Plots
"""
Cell 10: Create Welch plots for visual inspection of each design point
"""
print("\nStep 9: Create Warmup Visualization Plots")

# Initialize visualization
viz = WarmupVisualization(figsize=(12, 6))

# Create plots for each design point
for design_name, welch_results in design_point_welch_results.items():
    print(f"\nCreating Welch plots for {design_name}...")
    
    # Create individual plots for each metric
    for metric_name in warmup_metrics:
        if metric_name in welch_results:
            fig = viz.create_welch_plot(
                welch_results=welch_results,
                metric_name=metric_name,
                title=f'Warmup Analysis: {design_name} - {metric_name.replace("_", " ").title()}'
            )
            plt.show()
    
    # Create combined view for this design point
    if len(welch_results) > 1:
        print(f"Combined view for {design_name}:")
        fig = viz.create_multi_metric_plot(
            welch_results=welch_results,
            title=f"Warmup Analysis: {design_name} - Both Metrics"
        )
        plt.show()

print(f"\n✓ Visualization plots created for all design points")

# %% Step 10: Determine Universal Warmup Period
"""
Cell 11: Analyze convergence patterns and determine uniform warmup period
"""
print("\nStep 10: Determine Universal Warmup Period")

print(f"🔍 Visual Inspection Guidance:")
print(f"  • Look at the BLUE lines (cross-replication averages) in the plots above")
print(f"  • Identify where each design point reaches stable oscillation")
print(f"  • Note any differences in convergence time between design points")
print(f"  • High-load conditions may need longer warmup than low-load conditions")

# Analyze convergence characteristics across design points
print(f"\n📊 Convergence Analysis by Design Point:")

convergence_summary = {}
for design_name, welch_results in design_point_welch_results.items():
    summary = {}
    
    for metric_name, data in welch_results.items():
        if data['cumulative_average']:
            # Simple heuristic: look at final 25% of simulation to see stability
            final_quarter_start = int(len(data['cumulative_average']) * 0.75)
            final_quarter = data['cumulative_average'][final_quarter_start:]
            
            if final_quarter:
                final_mean = sum(final_quarter) / len(final_quarter)
                final_std = (sum((x - final_mean)**2 for x in final_quarter) / len(final_quarter))**0.5
                coefficient_of_variation = final_std / final_mean if final_mean > 0 else 0
                
                summary[metric_name] = {
                    'final_value': data['cumulative_average'][-1],
                    'final_quarter_cv': coefficient_of_variation,
                    'total_time_points': len(data['time_points'])
                }
    
    convergence_summary[design_name] = summary
    
    # Extract load ratio for analysis
    dp = design_points[design_name]
    load_ratio = dp.operational_config.mean_driver_inter_arrival_time / dp.operational_config.mean_order_inter_arrival_time
    
    print(f"  • {design_name} (load_ratio={load_ratio:.1f}):")
    for metric_name, metric_data in summary.items():
        cv = metric_data['final_quarter_cv']
        final_val = metric_data['final_value']
        print(f"    - {metric_name}: final_value={final_val:.1f}, stability_cv={cv:.3f}")

# %% Step 11: Warmup Period Decision Framework
"""
Cell 12: Framework for deciding uniform warmup period
"""
print("\nStep 11: Warmup Period Decision Framework")

print(f"🎯 Decision Framework for Uniform Warmup Period:")
decision_framework = [
    "1. Visual Inspection: Look at Welch plots to identify convergence points",
    "2. Conservative Approach: Choose warmup period that works for ALL design points",
    "3. Load Factor Consideration: High-load configs may need longer warmup",
    "4. Analysis Window: Ensure (duration - warmup) provides sufficient data",
    "5. Iteration: Adjust simulation_duration if analysis window too small"
]

for step in decision_framework:
    print(f"  {step}")

# Provide convergence time estimates based on simulation context
simulation_duration = experiment_config.simulation_duration
print(f"\n📋 Simulation Context:")
print(f"  • Simulation duration: {simulation_duration} minutes")
print(f"  • Collection interval: 0.5 minutes")
print(f"  • Total time points per replication: ~{int(simulation_duration / 0.5)}")

# Rough guidelines based on supply-demand study characteristics
print(f"\n💡 Supply-Demand Study Considerations:")
considerations = [
    "Low-load conditions (high load_ratio): May converge quickly",
    "High-load conditions (low load_ratio): May need longer warmup", 
    "Driver arrival patterns: Consider mean_service_duration (100 min)",
    "System saturation: Watch for diverging vs converging patterns"
]

for consideration in considerations:
    print(f"  • {consideration}")

print(f"\n🎯 Suggested Next Steps:")
next_steps = [
    "1. Examine plots above to visually identify convergence points",
    "2. Choose conservative warmup period (e.g., 50-80 minutes)", 
    "3. Validate analysis window: (200 - warmup) should be ≥ 100 minutes",
    "4. If analysis window too small, increase simulation_duration and re-run",
    "5. Apply chosen warmup period to post-simulation analysis"
]

for step in next_steps:
    print(f"  {step}")

# %% Step 12: Warmup Period Selection
"""
Cell 13: Select and validate uniform warmup period
"""
print("\nStep 12: Warmup Period Selection")

# ======================================================================
# MANUAL WARMUP PERIOD SELECTION
# ======================================================================
# Based on visual inspection of the Welch plots above, choose your warmup period:

# TODO: Update this value based on your visual analysis of the plots
uniform_warmup_period = 80  # minutes - ADJUST BASED ON VISUAL INSPECTION

# ======================================================================

print(f"📌 Selected uniform warmup period: {uniform_warmup_period} minutes")

# Validate the selection
analysis_window = simulation_duration - uniform_warmup_period
warmup_ratio = uniform_warmup_period / simulation_duration

print(f"\n📊 Warmup Period Validation:")
print(f"  • Simulation duration: {simulation_duration} minutes")
print(f"  • Warmup period: {uniform_warmup_period} minutes ({warmup_ratio*100:.1f}% of total)")
print(f"  • Analysis window: {analysis_window} minutes ({(1-warmup_ratio)*100:.1f}% of total)")

# Validation checks
print(f"\n✅ Validation Checks:")

if warmup_ratio > 0.5:
    print(f"⚠️  WARNING: Warmup period is {warmup_ratio*100:.1f}% of simulation duration (> 50%)")
    print(f"   Consider extending simulation_duration or reducing warmup_period")
elif warmup_ratio > 0.4:
    print(f"⚠️  CAUTION: Warmup period is {warmup_ratio*100:.1f}% of simulation duration (> 40%)")
    print(f"   Analysis window is somewhat limited but may be acceptable")
else:
    print(f"✓ Good ratio: Warmup period is {warmup_ratio*100:.1f}% of simulation duration")

if analysis_window < 50:
    print(f"⚠️  WARNING: Analysis window ({analysis_window} min) may be too short")
    print(f"   Consider extending simulation_duration for more robust results")
elif analysis_window < 100:
    print(f"⚠️  CAUTION: Analysis window ({analysis_window} min) is somewhat limited")
else:
    print(f"✓ Analysis window ({analysis_window} min) should provide adequate data")

print(f"\n🎯 Warmup Period Decision:")
if warmup_ratio <= 0.4 and analysis_window >= 50:
    print(f"✅ Warmup period appears appropriate for comparative analysis")
    print(f"✅ Ready to proceed with post-simulation analysis")
    print(f"✅ All design points will use uniform warmup period: {uniform_warmup_period} minutes")
else:
    print(f"⚠️  Consider adjusting parameters:")
    if warmup_ratio > 0.4:
        print(f"   • Reduce warmup_period if plots show earlier convergence")
    if analysis_window < 50:
        print(f"   • Increase simulation_duration to: {uniform_warmup_period + 100} minutes")
    print(f"   • Re-run experimental study if adjustments needed")

# %% Step 13: Ready for Post-Simulation Analysis
"""
Cell 14: Prepare for post-simulation analysis with determined warmup period
"""
print("\nStep 13: Ready for Post-Simulation Analysis")

print(f"🎉 Warmup Analysis Complete!")
print(f"  • Uniform warmup period determined: {uniform_warmup_period} minutes")
print(f"  • Applied to all {len(design_points)} design points")
print(f"  • Analysis window: {analysis_window} minutes per design point")

print(f"\n📋 Summary for Thesis Documentation:")
thesis_summary = [
    f"Experimental design: {len(design_points)} supply-demand conditions",
    f"Replications per condition: {experiment_config.num_replications}",
    f"Total replications: {len(design_points) * experiment_config.num_replications}",
    f"Warmup method: Welch's visual inspection method",
    f"Uniform warmup period: {uniform_warmup_period} minutes",
    f"Analysis window: {analysis_window} minutes",
    f"Infrastructure reuse: ✓ (same across all conditions)"
]

for item in thesis_summary:
    print(f"  • {item}")

print(f"\n🚀 Next Research Workflow Steps:")
next_research_steps = [
    "Apply uniform warmup period to filter simulation data",
    "Calculate performance metrics for each design point", 
    "Perform statistical analysis and comparison",
    "Generate thesis-quality results and visualizations",
    "Draw conclusions about supply-demand effects"
]

for i, step in enumerate(next_research_steps, 1):
    print(f"  {i}. {step}")

print(f"\n✨ Research Milestone Achieved:")
print(f"✓ Clean experimental architecture implemented")
print(f"✓ Supply-demand study executed successfully") 
print(f"✓ Warmup analysis completed with visual validation")
print(f"✓ Uniform warmup period determined for fair comparison")
print(f"✓ Ready for post-simulation analysis and thesis writing")

print("\n🎯 Experimental workflow complete - ready for analysis! 🎉")