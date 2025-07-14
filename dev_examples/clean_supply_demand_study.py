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
    simulation_duration=1000,
    num_replications=3,
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

# %% Step 8: Simplified Warmup Analysis - Time Series Extraction
"""
Cell 8: Extract cross-replication averaged time series from all design points

This cell applies the new simplified warmup analysis approach:
- Focus on cross-replication averaging (the core insight)
- No complex cumulative smoothing
- Prepare data for visual inspection
"""
print("\nStep 8: Simplified Warmup Analysis - Time Series Extraction")
print("=" * 60)

# Import the new simplified warmup analysis modules
from delivery_sim.warmup_analysis.time_series_preprocessing import extract_time_series_for_inspection
from delivery_sim.warmup_analysis.visualization import TimeSeriesVisualization
import matplotlib.pyplot as plt

print("✓ New simplified warmup analysis modules imported")

# Extract time series data from all design points
print(f"\n📊 Extracting time series data from {len(study_results)} design points...")

all_time_series_data = {}

for design_name, design_results in study_results.items():
    print(f"  Processing {design_name}...")
    
    # Extract system snapshots from this design point's replications
    replication_snapshots = []
    for replication_result in design_results['replication_results']:
        snapshots = replication_result['system_snapshots']
        if snapshots:
            replication_snapshots.append(snapshots)
    
    if len(replication_snapshots) < 2:
        print(f"    ⚠️  Warning: Only {len(replication_snapshots)} replications for {design_name}")
        continue
    
    # Extract cross-replication averages (the core operation!)
    time_series_data = extract_time_series_for_inspection(
        multi_replication_snapshots=replication_snapshots,
        metrics=['active_drivers', 'active_delivery_entities'],
        collection_interval=0.5  # Should match SystemDataCollector setting
    )
    
    all_time_series_data[design_name] = time_series_data
    print(f"    ✓ Extracted data for {len(time_series_data)} metrics, {len(replication_snapshots)} replications")

print(f"\n✅ Time series extraction complete!")
print(f"  • Design points processed: {len(all_time_series_data)}")
print(f"  • Metrics per design point: {len(list(all_time_series_data.values())[0]) if all_time_series_data else 0}")
print(f"  • Ready for visual inspection")

# %% Step 9: Visual Inspection - Combined Plots for All Design Points  
"""
Cell 9: Create combined time series plots for visual warmup inspection

Shows cross-replication averaged time series for each design point.
Focus on visual pattern recognition to identify warmup periods.
"""
print("\nStep 9: Visual Inspection - Combined Plots for All Design Points")
print("=" * 60)

# Create visualization instance
viz = TimeSeriesVisualization(figsize=(14, 8))

print(f"🔍 Creating combined inspection plots for visual warmup determination...")
print(f"  • Total plots to display: {len(all_time_series_data)}")
print(f"  • Metrics per plot: active_drivers, active_delivery_entities")

# Create combined plot for each design point
plot_count = 0
for design_name, time_series_data in all_time_series_data.items():
    plot_count += 1
    
    print(f"\n--- Plot {plot_count}/{len(all_time_series_data)}: {design_name} ---")
    
    # Get basic info about this design point
    first_metric_data = list(time_series_data.values())[0]
    total_duration = max(first_metric_data['time_points'])
    replication_count = first_metric_data['replication_count']
    
    print(f"  • Duration: {total_duration:.1f} minutes")
    print(f"  • Replications: {replication_count}")
    
    # Create combined plot for this design point
    fig = viz.create_combined_inspection_plot(
        time_series_data, 
        title=f'Warmup Inspection: {design_name.replace("_", " ").title()}'
    )
    
    # Show the plot
    plt.show()
    
    print(f"  ✓ Plot displayed for visual inspection")

print(f"\n🎯 All {plot_count} plots displayed for visual inspection!")

# %% Step 10: Visual Inspection Guidance and Warmup Determination
"""
Cell 10: Guidance for visual inspection and warmup period determination

Provides systematic guidance for determining uniform warmup period across
all design points based on visual inspection of the plots above.
"""
print("\nStep 10: Visual Inspection Guidance and Warmup Determination")
print("=" * 60)

print("🔍 VISUAL INSPECTION METHODOLOGY")
print("-" * 40)

print("\n📋 Step-by-step visual inspection process:")
print("  1. Look at each of the 9 plots displayed above")
print("  2. For each plot, identify the transition point where:")
print("     • Lines stop trending/changing (transient phase)")
print("     • Lines start stable oscillation around consistent levels (steady-state)")
print("  3. Note the time point where this transition occurs for each design point")
print("  4. Choose the LATEST transition time across all design points")
print("  5. Add a conservative safety margin (e.g., +20-30 minutes)")

print("\n🎯 PATTERN RECOGNITION GUIDE:")
print("  • Early Phase (Transient): Lines trending upward as system 'warms up'")
print("  • Transition Point: Where trending behavior stops")
print("  • Steady Phase: Lines oscillating around stable levels")
print("  • Conservative Choice: Use transition point + safety margin")

print("\n⚖️ UNIFORM WARMUP REQUIREMENT:")
print("  • Same warmup period MUST be used for ALL design points")
print("  • Based on the slowest-converging design point")
print("  • Better to be conservative than risk initialization bias")

# Show simulation context for reference
print(f"\n📊 Simulation Context for Reference:")
print(f"  • Total simulation duration: {experiment_config.simulation_duration} minutes")
print(f"  • Number of design points: {len(all_time_series_data)}")
print(f"  • Replications per design point: {experiment_config.num_replications}")

# Provide warmup ratio guidance
print(f"\n📏 Warmup Period Guidelines:")
print(f"  • Total duration: {experiment_config.simulation_duration} minutes")
print(f"  • Recommended warmup ratio: ≤30% of total duration")
print(f"  • Maximum acceptable warmup: {experiment_config.simulation_duration * 0.3:.0f} minutes (30%)")
print(f"  • Analysis window should be ≥70% of total duration")

print(f"\n💡 Next Steps:")
print(f"  1. Examine all {len(all_time_series_data)} plots above")
print(f"  2. Identify transition points visually")
print(f"  3. Choose conservative uniform warmup period")
print(f"  4. Proceed to Cell 11 for warmup validation")

# Prepare variables for next cell
print(f"\n🔧 Preparation for Next Cell:")
print(f"  • Variable 'all_time_series_data' contains extracted data")
print(f"  • Variable 'experiment_config' contains simulation parameters")
print(f"  • Ready for warmup period validation in next cell")

# %% Step 11: Warmup Period Validation
"""
Cell 11: Validate your visually determined warmup period

Set your warmup period based on visual inspection and validate it across all design points.
"""
print("\nStep 11: Warmup Period Validation")
print("=" * 60)

# ======================================================================
# SET YOUR WARMUP PERIOD HERE BASED ON VISUAL INSPECTION
# ======================================================================
print("📌 WARMUP PERIOD DETERMINATION:")
print("Based on visual inspection of the plots above, set your warmup period:")

# TODO: Update this value based on your visual inspection of the plots
proposed_warmup_period = 80  # Replace with your visually determined value

print(f"  • Proposed warmup period: {proposed_warmup_period} minutes")
# ======================================================================

print(f"\n🔍 Validating warmup period across all design points...")

# Validation summary
validation_results = []

for design_name, time_series_data in all_time_series_data.items():
    # Get time series info
    first_metric = list(time_series_data.keys())[0]
    max_time = max(time_series_data[first_metric]['time_points'])
    
    # Calculate analysis window
    analysis_window = max_time - proposed_warmup_period
    warmup_ratio = proposed_warmup_period / max_time
    
    # Store validation info
    validation_info = {
        'design_name': design_name,
        'total_duration': max_time,
        'warmup_period': proposed_warmup_period,
        'analysis_window': analysis_window,
        'warmup_ratio': warmup_ratio
    }
    validation_results.append(validation_info)
    
    print(f"\n  📊 {design_name}:")
    print(f"    • Total duration: {max_time:.1f} minutes")
    print(f"    • Warmup period: {proposed_warmup_period:.1f} minutes ({warmup_ratio*100:.1f}%)")
    print(f"    • Analysis window: {analysis_window:.1f} minutes ({(1-warmup_ratio)*100:.1f}%)")
    
    # Validation checks
    if warmup_ratio > 0.5:
        print(f"    ⚠️  WARNING: Warmup is {warmup_ratio*100:.1f}% of total (>50%)")
    elif warmup_ratio > 0.3:
        print(f"    ⚠️  CAUTION: Warmup is {warmup_ratio*100:.1f}% of total (>30%)")
    else:
        print(f"    ✅ Good ratio: {warmup_ratio*100:.1f}% warmup")
    
    if analysis_window < 30:
        print(f"    ⚠️  WARNING: Analysis window ({analysis_window:.1f} min) may be too short")
    else:
        print(f"    ✅ Analysis window ({analysis_window:.1f} min) adequate")

# Overall validation summary
all_ratios = [v['warmup_ratio'] for v in validation_results]
all_windows = [v['analysis_window'] for v in validation_results]

print(f"\n📋 OVERALL VALIDATION SUMMARY:")
print(f"  • Warmup ratios: {min(all_ratios)*100:.1f}% - {max(all_ratios)*100:.1f}%")
print(f"  • Analysis windows: {min(all_windows):.1f} - {max(all_windows):.1f} minutes")

# Final recommendation
if max(all_ratios) <= 0.3 and min(all_windows) >= 30:
    print(f"  ✅ RECOMMENDED: Warmup period appears appropriate for all design points")
else:
    print(f"  ⚠️  CONSIDER ADJUSTMENT: Review warmup period based on validation results")

print(f"\n🎯 UNIFORM WARMUP PERIOD DETERMINED:")
print(f"  • Selected warmup period: {proposed_warmup_period} minutes")
print(f"  • Applies to ALL {len(all_time_series_data)} design points")
print(f"  • Ready for post-simulation analysis in next workflow phase")

print(f"\n➡️  Next: Use this warmup period for performance analysis across design points")

# Store for next workflow phase
uniform_warmup_period = proposed_warmup_period
print(f"\n✓ Variable 'uniform_warmup_period' = {uniform_warmup_period} minutes stored for next phase")