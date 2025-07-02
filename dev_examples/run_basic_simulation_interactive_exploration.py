
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
    mean_order_inter_arrival_time=2.0,    # 2 minutes between orders - try 1.0, 2.0, 4.0 
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
    simulation_duration=100,    # __ minutes for quick testing - adjust as needed
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

# %% Metrics Analysis Pipeline
"""
Cell 9: Analyze simulation results using the new metrics pipeline
This demonstrates the complete pipeline from raw data to analyzable results
"""
# NEW: Use the analysis pipeline
from delivery_sim.analysis_pipeline.pipeline_coordinator import analyze_single_configuration, quick_summary

print("\n" + "="*60)
print("NEW ANALYSIS PIPELINE TEST")
print("="*60)

# Run the complete pipeline
warmup_period = 30  # Same as your current analysis
experiment_summary = analyze_single_configuration(results, warmup_period)

# Get quick summary for key metrics
quick_results = quick_summary(experiment_summary, 
                            metrics_of_interest=['system_completion_rate', 'assignment_time', 'total_distance'])

print("\nQuick Results:")
for metric_name, data in quick_results.items():
    print(f"  {metric_name}: {data['formatted']}")

print(f"\nFull Results Available:")
print(f"  Entity metrics: {list(experiment_summary['entity_metrics'].keys())}")
print(f"  System metrics: {list(experiment_summary['system_metrics'].keys())}")
print(f"  Replications: {experiment_summary['num_replications']}")
print(f"  Confidence level: {experiment_summary['confidence_level']*100}%")
# %%
