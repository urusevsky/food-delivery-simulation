# delivery_sim/experimental/experimental_runner.py
"""
ExperimentalRunner: Simple Multi-Configuration Orchestrator

Clean, simple runner that takes a dict of DesignPoint instances and executes them.
No feature creep - just execution orchestration.
"""

from delivery_sim.simulation.simulation_runner import SimulationRunner
from delivery_sim.utils.logging_system import get_logger


class ExperimentalRunner:
    """
    Simple orchestrator for multi-configuration experimental execution.
    
    Takes dict of DesignPoint instances and executes them.
    """
    
    def __init__(self):
        """Initialize ExperimentalRunner."""
        self.logger = get_logger("experimental.runner")
        self.logger.info("ExperimentalRunner initialized")
    
    def run_experimental_study(self, design_points, experiment_config):
        """
        Execute experimental study with DesignPoint instances.
        
        REFACTORED: SimulationRunner.run_experiment() now returns direct replication_results list.
        
        Args:
            design_points: Dict mapping names to DesignPoint instances
            experiment_config: ExperimentConfig (duration, replications, seed, collection_interval)
            
        Returns:
            dict: Experimental results organized by design point name
        """
        self.logger.info(f"Starting experimental study with {len(design_points)} design points")
        
        # Initialize results container
        study_results = {}
        
        # Execute each design point 
        for design_name, design_point in design_points.items():
            self.logger.info(f"--- Executing design point: {design_name} ---")
            
            # Create SimulationRunner with this design point's infrastructure
            runner = SimulationRunner(design_point.infrastructure)
            
            # Run this design point
            replication_results = runner.run_experiment(
                operational_config=design_point.operational_config,
                experiment_config=experiment_config,
                scoring_config=design_point.scoring_config
            )
            
            # ✅ REFACTORED: Consistent naming - replication_results throughout codebase
            study_results[design_name] = replication_results
            self.logger.info(f"Completed {design_name}: {len(replication_results)} replications")
        
        self.logger.info(f"Experimental study completed: {len(design_points)} design points")
        return study_results