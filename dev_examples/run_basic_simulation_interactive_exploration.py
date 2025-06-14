# %% [markdown]
# # Food Delivery Simulation - Basic Runner for Priority Scoring System
# Adapted to work with the latest model codebase using priority scoring system
# Designed for VS Code Python Interactive Window

# %% Import and Setup
"""
Cell 1: Import necessary modules and set up logging

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

from delivery_sim.simulation.configuration import (
    StructuralConfig, OperationalConfig, ExperimentConfig, 
    LoggingConfig, ScoringConfig, SimulationConfig
)
from delivery_sim.simulation.simulation_runner import SimulationRunner

# Configure logging for clean interactive output
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("All modules imported successfully!")
print("Ready to configure and run simulation with priority scoring system")

# %% Infrastructure Configuration  
"""
Cell 2: Define structural/geographical parameters
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
Cell 3: Define business logic and operational rules
Focus on arrival patterns, pairing rules, and driver service behavior
Note: Assignment logic now uses priority scoring system instead of adjusted cost
"""
operational_config = OperationalConfig(
    # Arrival patterns - experiment with different system loads
    mean_order_inter_arrival_time=2.0,    # 2 minutes between orders - try 1.0, 2.0, 4.0 
    mean_driver_inter_arrival_time=3.0,   # 3 minutes between drivers - try 2.0, 3.0, 5.0
    
    # Pairing strategy - experiment with pairing effectiveness
    pairing_enabled=True,
    restaurants_proximity_threshold=2.0,   # 2km for restaurant clustering - try 1.0, 2.0, 4.0
    customers_proximity_threshold=2.5,     # 2.5km for customer clustering - try 1.5, 2.5, 4.0
    
    # Driver service patterns
    mean_service_duration=120,      # 2 hours average service time
    service_duration_std_dev=60,    # 1 hour standard deviation
    min_service_duration=30,        # minimum 30 minutes
    max_service_duration=240,       # maximum 4 hours
    
    # Assignment strategy parameters (priority scoring system)
    immediate_assignment_threshold=75.0,    # Priority score threshold - try 70, 75, 80
    periodic_interval=3.0                   # 3 minutes between global optimizations - try 2.0, 3.0, 5.0
)

print(f"Operational config: {operational_config}")

# %% Priority Scoring Configuration
"""
Cell 4: Configure the priority scoring system
This replaces the old adjusted cost framework with a principled multi-criteria approach
"""
scoring_config = ScoringConfig(
    # Business policy parameters (universal standards)
    max_distance_ratio_multiplier=2.0,     # Beyond 2x typical distance is unacceptable
    max_acceptable_wait=20,               # 20 minutes maximum wait time
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
Cell 5: Define experimental parameters
Single replication for basic model verification
"""
experiment_config = ExperimentConfig(
    simulation_duration=100,    # 100 minutes for quick testing - adjust as needed
    num_replications=1,         # Single replication for basic testing
    master_seed=42             # Consistent seed for reproducibility
)

print(f"Experiment config: {experiment_config}")

# %% Logging Configuration
"""
Cell 6: Configure logging for clear output
Adjust logging levels to see more or less detail
"""
logging_config = LoggingConfig(
    console_level="INFO",       # Console output level
    file_level="DEBUG",         # File output level (if enabled)
    log_to_file=False,         # Set True to create log files
    component_levels={
        "simulation.runner": "INFO",
        "service.assignment": "INFO",
        "service.pairing": "INFO",
        "utils.priority_scoring": "DEBUG"  # More detail on scoring system
    }
)

print(f"Logging config: Console level = {logging_config.console_level}")

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

# %% Results Analysis
"""
Cell 9: Analyze basic simulation results
Extract and display key metrics from the completed simulation
"""
print("\n=== BASIC RESULTS ANALYSIS ===")

# Extract results from the first (and only) replication
replication_result = results['replication_results'][0]
repos = replication_result['repositories']

# Get repository data
orders = repos['order'].find_all()
drivers = repos['driver'].find_all()
pairs = repos['pair'].find_all()
delivery_units = repos['delivery_unit'].find_all()

print(f"\nEntity Counts:")
print(f"  Orders created: {len(orders)}")
print(f"  Drivers logged in: {len(drivers)}")
print(f"  Pairs formed: {len(pairs)}")
print(f"  Delivery units created: {len(delivery_units)}")

# Infrastructure characteristics
infra_chars = results['infrastructure_characteristics']
print(f"\nInfrastructure Characteristics:")
print(f"  Typical distance: {infra_chars['typical_distance']:.3f} km")
if 'restaurant_density' in infra_chars:
    print(f"  Restaurant density: {infra_chars['restaurant_density']:.3f} restaurants/km²")

# Basic efficiency metrics
if len(orders) > 0:
    print(f"\nBasic Efficiency Metrics:")
    
    # Calculate pairing efficiency
    paired_orders = sum(1 for pair in pairs for _ in pair.orders)
    pairing_rate = paired_orders / len(orders) if len(orders) > 0 else 0
    print(f"  Pairing rate: {pairing_rate:.1%} ({paired_orders}/{len(orders)} orders paired)")
    
    # Calculate delivery unit efficiency
    total_orders_in_units = sum(len(unit.orders) for unit in delivery_units)
    unit_efficiency = total_orders_in_units / len(delivery_units) if len(delivery_units) > 0 else 0
    print(f"  Avg orders per delivery unit: {unit_efficiency:.2f}")

print(f"\nSimulation completed successfully!")
print(f"Results structure contains: {list(results.keys())}")

# %% Priority Scoring Analysis
"""
Cell 10: Analyze priority scoring system behavior
Examine how the scoring system performed during the simulation
"""
print("\n=== PRIORITY SCORING SYSTEM ANALYSIS ===")

# Look for assignment events to analyze scoring behavior
events = replication_result.get('events', [])
assignment_events = [e for e in events if e.get('event_type') == 'driver_assigned']

if assignment_events:
    print(f"\nAssignment Events Analysis:")
    print(f"  Total assignments: {len(assignment_events)}")
    
    # Look for priority scores in assignment events
    scored_assignments = [e for e in assignment_events if 'priority_score' in e]
    if scored_assignments:
        scores = [e['priority_score'] for e in scored_assignments]
        print(f"  Priority scores recorded: {len(scores)}")
        print(f"  Score range: {min(scores):.1f} - {max(scores):.1f}")
        print(f"  Average score: {sum(scores)/len(scores):.1f}")
        
        # Count immediate vs periodic assignments
        immediate_count = sum(1 for score in scores if score >= operational_config.immediate_assignment_threshold)
        print(f"  Immediate assignments (≥{operational_config.immediate_assignment_threshold}): {immediate_count}")
        print(f"  Periodic assignments: {len(scores) - immediate_count}")
    else:
        print("  No priority scores found in assignment events")
else:
    print("No assignment events found in results")

print(f"\nPriority scoring system configured with:")
print(f"  Distance weight: {scoring_config.weight_distance:.3f}")
print(f"  Throughput weight: {scoring_config.weight_throughput:.3f}")
print(f"  Fairness weight: {scoring_config.weight_fairness:.3f}")
print(f"  Max distance ratio: {scoring_config.max_distance_ratio_multiplier}")
print(f"  Max acceptable wait: {scoring_config.max_acceptable_wait} minutes")

# %% Configuration Summary for Future Reference
"""
Cell 11: Summary of the configuration used
Useful for documenting what parameters were tested
"""
print("\n=== CONFIGURATION SUMMARY FOR REFERENCE ===")
print(f"""
Configuration Used:
  Geography: {structural_config.delivery_area_size}×{structural_config.delivery_area_size}km area
  Restaurants: {structural_config.num_restaurants} restaurants
  Driver speed: {structural_config.driver_speed} km/min
  
  Order arrival: Every {operational_config.mean_order_inter_arrival_time} min (avg)
  Driver arrival: Every {operational_config.mean_driver_inter_arrival_time} min (avg)
  Pairing: {'Enabled' if operational_config.pairing_enabled else 'Disabled'}
  
  Priority Scoring Weights:
    Distance: {scoring_config.weight_distance:.3f}
    Throughput: {scoring_config.weight_throughput:.3f}  
    Fairness: {scoring_config.weight_fairness:.3f}
  
  Assignment threshold: {operational_config.immediate_assignment_threshold}
  Simulation duration: {experiment_config.simulation_duration} minutes
  Seed: {experiment_config.master_seed}
""")

print("Simulation run completed successfully!")
print("Modify configuration parameters in earlier cells and re-run to explore different scenarios.")