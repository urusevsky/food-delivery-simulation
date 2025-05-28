# %% [markdown]
# # Food Delivery Simulation - Interactive Exploration
# This notebook allows iterative parameter tuning and result analysis

# %% Import and Setup
"""
Cell 1: Import all necessary modules and set up logging
This cell typically runs once per session unless you modify the code
"""
from delivery_sim.simulation.configuration import (
    StructuralConfig, OperationalConfig, ExperimentConfig, SimulationConfig
)
from delivery_sim.simulation.simulation_runner import SimulationRunner

# Configure logging for clean output in interactive mode
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("All modules imported successfully!")

# %% Infrastructure Configuration  
"""
Cell 2: Define the structural/geographical parameters
Modify these parameters to explore different delivery environments
"""
# This represents your "geography" - modify to test different scenarios
structural_config = StructuralConfig(
    delivery_area_size=10,  # Try: 5, 10, 15 for different area sizes
    num_restaurants=10,     # Try: 5, 10, 20 for different restaurant densities  
    driver_speed=0.5        # Try: 0.3, 0.5, 0.8 for different driver speeds
)

print(f"Structural config: {structural_config}")
print(f"Average restaurant spacing: ~{(structural_config.delivery_area_size**2 / structural_config.num_restaurants)**0.5:.1f} km")

# %% Operational Parameters
"""
Cell 3: Define business logic and operational rules
This is where you'll do most of your parameter exploration
"""
operational_config = OperationalConfig(
    # Arrival patterns - experiment with different system loads
    mean_order_inter_arrival_time=2.0,    # Try: 1.0, 2.0, 4.0 
    mean_driver_inter_arrival_time=3.0,   # Try: 2.0, 3.0, 5.0
    
    # Pairing strategy - experiment with pairing effectiveness
    pairing_enabled=True,
    restaurants_proximity_threshold=2.0,   # Try: 1.0, 2.0, 4.0
    customers_proximity_threshold=2.5,     # Try: 1.5, 2.5, 4.0
    
    # Driver service patterns
    mean_service_duration=120,
    service_duration_std_dev=60,
    min_service_duration=30,
    max_service_duration=240,
    
    # Assignment strategy - critical parameters for your research
    throughput_factor=1.5,                 # Try: 1.0, 1.5, 2.0
    age_factor=0.1,                        # Try: 0.05, 0.1, 0.2
    immediate_assignment_threshold=8.0,    # Try: 5.0, 8.0, 12.0
    periodic_interval=3.0                  # Try: 2.0, 3.0, 5.0
)

print(f"Operational config: {operational_config}")

# %% Experiment Configuration
"""
Cell 4: Define experimental parameters
Adjust simulation duration and data collection for your exploration needs
"""
experiment_config = ExperimentConfig(
    simulation_duration=100,    # Start small for quick iteration
    warmup_period=0,           # Skip warmup during exploration
    num_replications=1,        # Single run for parameter exploration
    master_seed=42,           # Keep consistent for parameter comparison
    metrics_collection_interval=5,
    event_recording_enabled=True
)

print(f"Experiment config: {experiment_config}")

# %% Simulation Setup
"""
Cell 5: Create and initialize the simulation runner
Execute this cell after modifying any configuration above
"""
# Combine all configurations
simulation_config = SimulationConfig(
    structural_config=structural_config,
    operational_config=operational_config,
    experiment_config=experiment_config
)

# Create and initialize the simulation runner
runner = SimulationRunner(simulation_config)
runner.initialize()

print("Simulation runner initialized successfully!")
print(f"Created {len(runner.restaurant_repository.find_all())} restaurants")

# %% Run Simulation
"""
Cell 6: Execute the simulation and capture results
Re-run this cell to test the same configuration multiple times
"""
print("Starting simulation...")
repositories = runner.run()

# %% Results Analysis
"""
Cell 7: Analyze and display simulation results
Modify this cell to explore different aspects of your results
"""
# Extract basic statistics
order_repo = repositories['order']
driver_repo = repositories['driver'] 
pair_repo = repositories['pair']
delivery_unit_repo = repositories['delivery_unit']

orders = order_repo.find_all()
drivers = driver_repo.find_all()
pairs = pair_repo.find_all()
delivery_units = delivery_unit_repo.find_all()

# Calculate key metrics
total_orders = len(orders)
completed_orders = len([o for o in orders if o.state == 'delivered'])
completion_rate = completed_orders / total_orders if total_orders > 0 else 0

total_drivers = len(drivers)
pairs_formed = len(pairs)
pair_rate = pairs_formed / total_orders if total_orders > 0 else 0

# Display comprehensive results
print("\n" + "="*50)
print("SIMULATION RESULTS SUMMARY")
print("="*50)
print(f"Orders: {completed_orders}/{total_orders} completed ({completion_rate:.1%})")
print(f"Drivers: {total_drivers} total")
print(f"Pairs: {pairs_formed} formed ({pair_rate:.1%} of orders)")
print(f"Delivery Units: {len(delivery_units)} created")

# %% Detailed Analysis (Optional)
"""
Cell 8: Deep dive into specific metrics when needed
Expand this section as you develop more sophisticated analysis
"""
# Calculate delivery time statistics
completed_orders_objects = [o for o in orders if o.state == 'delivered']
if completed_orders_objects:
    delivery_times = [o.delivery_time - o.arrival_time for o in completed_orders_objects if o.delivery_time]
    avg_delivery_time = sum(delivery_times) / len(delivery_times)
    print(f"\nAverage delivery time: {avg_delivery_time:.1f} minutes")

# Analyze assignment patterns
immediate_assignments = len([du for du in delivery_units if du.assignment_path == 'immediate'])
periodic_assignments = len([du for du in delivery_units if du.assignment_path == 'periodic'])
print(f"Assignment breakdown: {immediate_assignments} immediate, {periodic_assignments} periodic")