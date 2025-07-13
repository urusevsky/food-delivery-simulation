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

# %% Step 7: Ready for Warmup Analysis
"""
Cell 8: Prepare for next research workflow step
"""
print("\nStep 7: Ready for Warmup Analysis")

print(f"🔬 Raw Data Available:")
for design_name in study_results.keys():
    replication_count = study_results[design_name]['num_replications']
    print(f"  • {design_name}: {replication_count} replications")

print(f"\n🎯 Next Research Workflow Steps:")
next_steps = [
    "Extract system_snapshots from study_results for warmup analysis",
    "Apply Welch's method to determine uniform warmup period",
    "Create time series plots for visual inspection", 
    "Validate warmup period works for all design points",
    "Proceed to post-simulation analysis and comparison"
]

for i, step in enumerate(next_steps, 1):
    print(f"  {i}. {step}")

print(f"\n✨ Clean Architecture Benefits:")
print(f"  • Simple DesignPoint class - no feature creep")
print(f"  • Simple ExperimentalRunner - just execution")
print(f"  • Manual design points dictionary - clear and explicit")
print(f"  • Infrastructure reused across all design points")
print(f"  • Ready for gradual feature addition as needs emerge")

print("\n🎉 Clean supply-demand study workflow complete!")