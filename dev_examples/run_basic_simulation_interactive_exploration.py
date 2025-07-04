
# %% Import and Setup
"""
Cell 1: Basic setup and imports only

IMPORTANT: If you modify any files in delivery_sim/, you MUST restart and re-run 
this cell to pick up the changes. Python caches imported modules and won't 
automatically reload them when source files change.
"""
# Add project root to Python path so we can import delivery_sim
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary modules
from delivery_sim.simulation.configuration import (
    StructuralConfig, OperationalConfig, ExperimentConfig, 
    LoggingConfig, ScoringConfig, SimulationConfig
)
from delivery_sim.simulation.simulation_runner import SimulationRunner
from delivery_sim.utils.logging_system import configure_logging, configure_component_level
import logging

print("✓ Imports successful")

# %% Logging Configuration  
"""
Cell 2: Configure logging for clean interactive output
Suppress noisy components, show only what we care about
"""

logging_config = LoggingConfig(
    console_level="DEBUG",  
    file_level="DEBUG",
    log_to_file=False,
    component_levels={
        # Step 1: Broad suppression (sets the default)
        "services": "ERROR",
        "entities": "ERROR", 
        "repositories": "ERROR",
        "utils": "ERROR",
        
        # Step 2: Surgical enablement (overrides the default)
        "simulation.runner": "INFO",
        #"entities.order": "SIMULATION",
        #"entities.pair": "SIMULATION",
        #"entities.driver": "SIMULATION",
        #"entities.delivery_unit": "SIMULATION",
        #"services.order_arrival": "SIMULATION",
        #"services.driver_arrival": "SIMULATION",
        #"services.pairing": "SIMULATION",
        #"services.assignment": "SIMULATION",
        #"services.delivery": "SIMULATION",
        #"services.driver_scheduling": "INFO"

    }
)

print("✓ Logging configuration defined - will be applied when simulation starts")

# %% Infrastructure Configuration  
"""
Cell 3: Define structural/geographical parameters
These represent the physical delivery environment
"""
structural_config = StructuralConfig(
    delivery_area_size=10,  # 10km x 10km area - modify for different geographies
    num_restaurants=10,     # Restaurant density - try 5, 10, 20
    driver_speed=0.5        # 0.5 km/min (30 km/h) - try 0.3, 0.5, 0.8
)

print(f"Structural config: {structural_config}")
print(f"Average restaurant spacing: ~{(structural_config.delivery_area_size**2 / structural_config.num_restaurants)**0.5:.1f} km")

# %% Operational Parameters
"""
Cell 4: Define business logic and operational rules
Focus on arrival patterns, pairing rules, and driver service behavior
Note: Assignment logic now uses priority scoring system instead of adjusted cost
"""
operational_config = OperationalConfig(
    # Arrival patterns - experiment with different system loads
    mean_order_inter_arrival_time=1.0,    # _ minutes between orders - try 1.0, 2.0, 4.0 
    mean_driver_inter_arrival_time=3.0,   # 3 minutes between drivers - try 2.0, 3.0, 5.0
    
    # Pairing strategy - experiment with pairing effectiveness
    pairing_enabled=False,
    restaurants_proximity_threshold=2.0,   # 2km for restaurant clustering - try 1.0, 2.0, 4.0
    customers_proximity_threshold=2.5,     # 2.5km for customer clustering - try 1.5, 2.5, 4.0
    
    # Driver service patterns
    mean_service_duration=120,      # 2 hours average service time
    service_duration_std_dev=60,    # 1 hour standard deviation
    min_service_duration=30,        # minimum 30 minutes
    max_service_duration=240,       # maximum 4 hours
    
    # Assignment strategy parameters (priority scoring system)
    immediate_assignment_threshold=100,    # Priority score threshold 
    periodic_interval=3.0                   # 3 minutes between global optimizations - try 2.0, 3.0, 5.0
)

print(f"Operational config: {operational_config}")

# %% Priority Scoring Configuration
"""
Cell 5: Configure the priority scoring system
This replaces the old adjusted cost framework with a principled multi-criteria approach
"""
scoring_config = ScoringConfig(
    # Business policy parameters (universal standards)
    max_distance_ratio_multiplier=2.0,     # Beyond 2x typical distance is unacceptable
    max_acceptable_delay=20,               # 20 minutes maximum wait time
    max_orders_per_trip=2,                  # Maximum orders per delivery trip
    
    # Strategic weights (business preferences) - must sum to 1.0
    weight_distance=0.333,                  # Weight for distance efficiency
    weight_throughput=0.333,                # Weight for throughput optimization  
    weight_fairness=0.334,                  # Weight for fairness (wait time)
    
    # Infrastructure analysis settings
    typical_distance_samples=1000           # Monte Carlo samples for typical distance calculation
)

print(f"Scoring config: {scoring_config}")
print("Weights sum:", scoring_config.weight_distance + scoring_config.weight_throughput + scoring_config.weight_fairness)

# %% Experiment Configuration
"""
Cell 6: Define experimental parameters
Single replication for basic model verification
"""
experiment_config = ExperimentConfig(
    simulation_duration=300,    # __ minutes for quick testing - adjust as needed
    num_replications=3,         # __ replication for basic testing
    master_seed=42             # Consistent seed for reproducibility
)

print(f"Experiment config: {experiment_config}")

# %% Complete Configuration Assembly
"""
Cell 7: Combine all configurations into complete simulation config
"""
simulation_config = SimulationConfig(
    structural_config=structural_config,
    operational_config=operational_config,
    experiment_config=experiment_config,
    logging_config=logging_config,
    scoring_config=scoring_config
)

print("Complete simulation configuration assembled:")
print(f"  Infrastructure: {structural_config.delivery_area_size}x{structural_config.delivery_area_size}km, {structural_config.num_restaurants} restaurants")
print(f"  Load: Orders every {operational_config.mean_order_inter_arrival_time}min, drivers every {operational_config.mean_driver_inter_arrival_time}min")
print(f"  Assignment: Priority threshold {operational_config.immediate_assignment_threshold}, periodic every {operational_config.periodic_interval}min")
print(f"  Duration: {experiment_config.simulation_duration} minutes, {experiment_config.num_replications} replication(s)")

# %% Run Simulation
"""
Cell 8: Execute the simulation
This cell runs the complete experiment using the new architecture
"""
print("\n" + "="*60)
print("STARTING SIMULATION")
print("="*60)

# Create and run simulation
runner = SimulationRunner()
results = runner.run_experiment(simulation_config)

print("\n" + "="*60)
print("SIMULATION COMPLETED")
print("="*60)

# %% Warmup Analysis Visualization
"""
Cell 9: Warmup Analysis Visualization
Create Welch plots for visual warmup period determination.

This cell applies Welch's method and creates visualization plots.
Use these plots to visually assess convergence patterns.

Next step: Run Cell 10 to set warmup_period based on visual inspection.
"""

print("\n" + "="*50)
print("WARMUP ANALYSIS - WELCH'S METHOD VISUALIZATION")
print("="*50)

# Import warmup analysis tools
from delivery_sim.warmup_analysis.welch_analyzer import WelchAnalyzer
from delivery_sim.warmup_analysis.visualization import WarmupVisualization
import matplotlib.pyplot as plt

# Step 1: Extract system snapshots from simulation results
print("Step 1: Extracting system snapshots...")

replication_snapshots = []
for i, replication_result in enumerate(results['replication_results']):
    snapshots = replication_result['system_snapshots']
    if snapshots:
        replication_snapshots.append(snapshots)
        print(f"  Replication {i+1}: {len(snapshots)} snapshots")

print(f"✓ Extracted data from {len(replication_snapshots)} replications")

# Check if we have sufficient data
if len(replication_snapshots) < 3:
    print("⚠️  WARNING: Welch analysis works best with ≥3 replications")
    print("   Consider increasing num_replications for more reliable results")

# Step 2: Apply Welch's method
print("\nStep 2: Applying Welch's method...")

analyzer = WelchAnalyzer()
warmup_metrics = ['active_drivers', 'active_delivery_entities']

welch_results = analyzer.analyze_warmup_convergence(
    multi_replication_snapshots=replication_snapshots,
    metrics=warmup_metrics,
    collection_interval=0.5  # Should match SystemDataCollector setting
)

print("✓ Cross-replication averaging and cumulative smoothing completed")

# Step 3: Create Welch plots for visual inspection
print("\nStep 3: Creating Welch plots for visual inspection...")

viz = WarmupVisualization(figsize=(12, 6))

# Create plots for each metric
for metric_name in warmup_metrics:
    if metric_name in welch_results:
        fig = viz.create_welch_plot(
            welch_results=welch_results,
            metric_name=metric_name,
            title=f'Warmup Detection: {metric_name.replace("_", " ").title()}'
        )
        plt.show()

# Create combined view
if len(welch_results) > 1:
    print("\nCombined view of all metrics:")
    fig = viz.create_multi_metric_plot(
        welch_results=welch_results,
        title="Warmup Analysis - Both Metrics"
    )
    plt.show()

# Step 4: Provide visual inspection guidance
print("\n" + "="*50)
print("VISUAL INSPECTION GUIDANCE")
print("="*50)

print("🔍 How to read the Welch plots:")
print("  • BLUE line = Cross-replication averages (primary indicator)")
print("  • RED line = Cumulative average (conservative reference)")
print("  • Human pattern recognition works well with blue line directly")
print()
print("📊 Look for convergence indicators:")
print("  • Blue line reaches stable oscillation around consistent level")
print("  • System transitions from trending to operational behavior")
print("  • Variability becomes consistent rather than decreasing")
print()
print("⏰ Choose warmup period:")
print("  • Identify where BLUE line reaches stable pattern")
print("  • Choose warmup period AT OR AFTER this stabilization point")
print("  • Conservative: If stabilization at 100, consider warmup = 120-150")
print("  • RED line provides conservative validation (naturally delayed)")
print()
print("💡 Pattern recognition tip:")
print("  • Blue line alone often provides sufficient visual clarity")
print("  • Smoothing helps but isn't required for good assessment")

# Display final values and simulation context
print(f"\n📈 Final stabilized values:")
for metric_name, data in welch_results.items():
    if data['cumulative_average']:
        final_value = data['cumulative_average'][-1]
        print(f"  • {metric_name.replace('_', ' ').title()}: {final_value:.1f}")

print(f"\n📋 Simulation context:")
print(f"  • Total duration: {experiment_config.simulation_duration} minutes")
print(f"  • Order arrival: every {operational_config.mean_order_inter_arrival_time} minutes")
print(f"  • Driver arrival: every {operational_config.mean_driver_inter_arrival_time} minutes")
print(f"  • System load ratio: {operational_config.mean_driver_inter_arrival_time / operational_config.mean_order_inter_arrival_time:.2f}")

print("\n" + "="*50)
print("PLOTS READY FOR VISUAL INSPECTION")
print("="*50)
print("➡️  Next: Run Cell 10 to set warmup_period based on your visual analysis")

# %% Warmup Period Determination
"""
Cell 10: Warmup Period Determination
Set the warmup period based on visual inspection of Welch plots from Cell 9.

Instructions:
1. Look at the Welch plots from the previous cell
2. Identify where the RED lines (cumulative averages) stabilize
3. Choose a conservative warmup period BEFORE stabilization
4. Update the warmup_period value below
"""

print("\n" + "="*50)
print("WARMUP PERIOD DETERMINATION")
print("="*50)

# ======================================================================
# SET YOUR WARMUP PERIOD HERE BASED ON VISUAL INSPECTION
# ======================================================================

# Update this value based on your visual analysis of the Welch plots above
warmup_period = 30  # TODO: Replace with your visually determined value

# ======================================================================

print(f"📌 Warmup period set to: {warmup_period} minutes")

# Validation and analysis window assessment
analysis_window = experiment_config.simulation_duration - warmup_period
warmup_ratio = warmup_period / experiment_config.simulation_duration

print(f"\n📊 Warmup assessment:")
print(f"  • Simulation duration: {experiment_config.simulation_duration} minutes")
print(f"  • Warmup period: {warmup_period} minutes ({warmup_ratio*100:.1f}% of total)")
print(f"  • Analysis window: {analysis_window} minutes ({(1-warmup_ratio)*100:.1f}% of total)")

# Validation checks
print(f"\n✅ Validation checks:")

if warmup_ratio > 0.5:
    print(f"⚠️  WARNING: Warmup period is {warmup_ratio*100:.1f}% of simulation duration (> 50%)")
    print("   Consider extending simulation_duration or reducing warmup_period")
    print("   Analysis window may be too small for reliable results")
elif warmup_ratio > 0.3:
    print(f"⚠️  CAUTION: Warmup period is {warmup_ratio*100:.1f}% of simulation duration (> 30%)")
    print("   Analysis window is somewhat limited but acceptable")
else:
    print(f"✓ Good ratio: Warmup period is {warmup_ratio*100:.1f}% of simulation duration")

if analysis_window < 30:
    print(f"⚠️  WARNING: Analysis window ({analysis_window} min) may be too short")
    print("   Consider extending simulation_duration for more robust results")
else:
    print(f"✓ Analysis window ({analysis_window} min) should provide adequate data")

# Guidance for next steps
print(f"\n🎯 Recommendation:")
if warmup_ratio <= 0.3 and analysis_window >= 30:
    print("✓ Warmup period appears appropriate for analysis")
    print("✓ Proceed to Cell 11 for post-simulation performance analysis")
else:
    print("⚠️  Consider adjusting warmup_period or simulation_duration")
    print("   You can re-run this cell with different warmup_period values")

print("\n" + "="*50)
print("WARMUP PERIOD DETERMINATION COMPLETE")
print("="*50)
print(f"Selected warmup_period: {warmup_period} minutes")
print("➡️  Next: Run Cell 11 for performance analysis using this warmup period")

# %% Metrics Analysis Pipeline: Post-Simulation Analysis - Using Determined Warmup Period
"""
Cell 11: Analyze simulation results using the new metrics pipeline
This demonstrates the complete pipeline from raw data to analyzable results

This cell uses the warmup_period set in Cell 10 based on visual inspection
of Welch plots, ensuring analysis is based on steady-state data only.
"""
# NEW: Use the analysis pipeline
from delivery_sim.analysis_pipeline.pipeline_coordinator import analyze_single_configuration, quick_summary

print("\n" + "="*60)
print("POST-SIMULATION ANALYSIS WITH DETERMINED WARMUP PERIOD")
print("="*60)

print(f"Using warmup_period = {warmup_period} minutes (determined through Welch analysis)")

# Run the complete analysis pipeline with the visually determined warmup period
experiment_summary = analyze_single_configuration(results, warmup_period)

# Get quick summary for key metrics
quick_results = quick_summary(experiment_summary, 
                            metrics_of_interest=['system_completion_rate', 'assignment_time', 'total_distance'])

print("\nQuick Results with Determined Warmup Period:")
for metric_name, data in quick_results.items():
    print(f"  {metric_name}: {data['formatted']}")

print(f"\nAnalysis Details:")
print(f"  Entity metrics: {list(experiment_summary['entity_metrics'].keys())}")
print(f"  System metrics: {list(experiment_summary['system_metrics'].keys())}")
print(f"  Replications: {experiment_summary['num_replications']}")
print(f"  Confidence level: {experiment_summary['confidence_level']*100}%")
print(f"  Warmup period: {experiment_summary['warmup_period']} minutes")
print(f"  Analysis window: {experiment_config.simulation_duration - warmup_period} minutes")

print("\n" + "="*60)
print("ANALYSIS COMPLETE - RESULTS BASED ON STEADY-STATE DATA")
print("="*60)
# %%