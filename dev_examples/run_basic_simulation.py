"""
Basic example demonstrating how to configure and run a single simulation.

This script shows the most straightforward usage of the SimulationRunner:
1. Create configuration objects
2. Initialize a simulation runner
3. Run the simulation
4. Display basic results
"""

from delivery_sim.simulation.configuration import (
    StructuralConfig, OperationalConfig, ExperimentConfig, SimulationConfig
)
from delivery_sim.simulation.simulation_runner import SimulationRunner

def main():
    # Create structural configuration
    structural_config = StructuralConfig(
        delivery_area_size=10,  # 10 km x 10 km area
        num_restaurants=10,
        driver_speed=0.5  # 0.5 km per minute (30 km/h)
    )

    # Create operational configuration
    operational_config = OperationalConfig(
        # Arrival rates
        mean_order_inter_arrival_time=2.0,  # 2 minutes between orders on average
        mean_driver_inter_arrival_time=3.0,  # 3 minutes between drivers on average
        
        # Pairing configuration
        pairing_enabled=True,
        restaurants_proximity_threshold=1.0,  # 1 km
        customers_proximity_threshold=1.5,   # 1.5 km
        
        # Driver service configuration
        mean_service_duration=120,  # 2 hours average
        service_duration_std_dev=60,  # 1 hour standard deviation
        min_service_duration=30,     # minimum 30 minutes
        max_service_duration=240,    # maximum 4 hours
        
        # Assignment parameters
        throughput_factor=1.5,  # value of delivering one more order
        age_factor=0.1,         # value per minute of waiting time
        immediate_assignment_threshold=1.5,  # km in adjusted cost
        periodic_interval=3.0   # 3 minutes between global optimizations
    )

    # Create experiment configuration
    experiment_config = ExperimentConfig(
        simulation_duration=100,  # 8 hours
        warmup_period=0,        # 2 hour warm-up
        num_replications=1,       # single replication for now
        master_seed=42,           # seed for reproducibility
        metrics_collection_interval=5,  # collect metrics every 5 minutes
        event_recording_enabled=True    # record all events for analysis
    )

    # Create combined configuration
    simulation_config = SimulationConfig(
        structural_config=structural_config,
        operational_config=operational_config,
        experiment_config=experiment_config
    )

    # Create and initialize the simulation runner
    runner = SimulationRunner(simulation_config)
    
    # For clarity, we use separate method calls rather than chaining
    runner.initialize()
    results = runner.run()

    # Report basic statistics
    order_repo = results['order']
    driver_repo = results['driver']
    pair_repo = results['pair']
    delivery_unit_repo = results['delivery_unit']

    print("\nSimulation Results Summary:")
    print(f"  {len(order_repo.find_all())} orders created")
    print(f"  {len(driver_repo.find_all())} drivers logged in")
    print(f"  {len(pair_repo.find_all())} pairs formed")
    print(f"  {len(delivery_unit_repo.find_all())} delivery units created")

if __name__ == "__main__":
    main()