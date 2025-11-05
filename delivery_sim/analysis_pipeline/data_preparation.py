# delivery_sim/analysis_pipeline/data_preparation.py
"""
Revamped data preparation module for analysis pipeline.

This module provides centralized population creation for all analytical needs.
Each population is clearly defined based on the specific research question it serves.
All filtering logic lives here - metric calculation functions only do calculations.
"""

from delivery_sim.utils.entity_type_utils import EntityType


class AnalyticalPopulations:
    """
    Centralized factory for creating analytical populations.
    
    This class serves as the single source of truth for all data filtering logic.
    Each method creates a specific population tailored to particular metric needs.
    """
    
    def __init__(self, repositories, warmup_period, system_snapshots=None):
        """
        Initialize the population factory.
        
        Args:
            repositories: Dict containing all entity repositories from simulation
            warmup_period: Duration to exclude from analysis (warmup bias elimination)
            system_snapshots: List of system state snapshots (optional, for state metrics)
        """
        self._repositories = repositories
        self._warmup_period = warmup_period
        self._system_snapshots = system_snapshots or []
    
    def get_cohort_orders(self):
        """
        Get all cohort orders (arrived post-warmup).
        
        This population defines the complete set of orders the system was 
        responsible for during the representative period. Used as denominator 
        for system-wide effectiveness metrics like completion rate.
        
        Research Question: "What was the total demand placed on the system 
        during its normal operational state?"
        
        Returns:
            list: All orders with arrival_time >= warmup_period, regardless of completion status
        """
        if 'order' not in self._repositories:
            return []
            
        all_orders = self._repositories['order'].find_all()
        return [
            order for order in all_orders 
            if order.arrival_time >= self._warmup_period
        ]
    
    def get_cohort_completed_orders(self):
        """
        Get all cohort orders that were completed (arrived post-warmup and delivered).
        
        This population serves dual purposes:
        1. order completion numerator (how many cohort orders were successfully delivered)
        2. Unbiased performance sample (for calculating averages of completed orders)
        
        Since orders are atomic entities, completed cohort orders represent 
        both successful outcomes and an unbiased performance sample.
        
        Research Questions: 
        - "Of all cohort orders, how many were delivered?"
        - "For completed cohort orders, what was their typical performance?"
        
        Returns:
            list: Cohort orders that were delivered
        """
        cohort = self.get_cohort_orders()
        return [
            order for order in cohort 
            if order.delivery_time is not None
        ]
    
    def get_cohort_completed_delivery_units(self):
        """
        Get completed cohort delivery units for calculating unbiased average performance metrics.
        
        This population applies strict filtering to ensure no contamination 
        bias from warmup period. For delivery units, ALL constituent orders must 
        have arrived post-warmup, explicitly excluding "hybrid" pairs.
        
        Only completed delivery units are included to ensure complete lifecycle data.
        
        Research Question: "For delivery units representing pure steady-state 
        operations that were completed, what is their expected performance?"
        
        Returns:
            list: Completed delivery units where all constituent orders arrived post-warmup
        """
        if 'delivery_unit' not in self._repositories:
            return []
            
        all_units = self._repositories['delivery_unit'].find_all()
        unbiased_units = []
        
        for unit in all_units:
            if unit.completion_time is not None:  # positive pattern ← Consistent with other methods
                entity = unit.delivery_entity
                
                if entity.entity_type == EntityType.ORDER:
                # Single order: order must have arrived after warmup
                    if entity.arrival_time >= self._warmup_period:
                        unbiased_units.append(unit)
                        
                elif entity.entity_type == EntityType.PAIR:
                    # Paired orders: BOTH orders must have arrived after warmup (strict filtering)
                    if (entity.order1.arrival_time >= self._warmup_period and 
                        entity.order2.arrival_time >= self._warmup_period):
                        unbiased_units.append(unit)
            
        return unbiased_units
    
    def get_cohort_paired_orders(self):
        """
        Get cohort orders that were paired (regardless of completion).
        
        This captures the pairing decision for cohort orders regardless of whether 
        the pair was subsequently assigned to a driver. Used for calculating pairing 
        effectiveness and related system metrics.
        
        Research Question: "Of all cohort orders, how many were successfully 
        paired with another order?"
        
        Returns:
            list: Cohort orders that were paired
        """
        cohort = self.get_cohort_orders()
        return [
            order for order in cohort 
            if order.pair is not None
        ]
    
    def get_post_warmup_snapshots(self):
        """
        Get system snapshots after warmup period.
        
        Filters snapshots to only include those captured during the representative
        operational period, excluding the warmup phase where system state is still
        stabilizing.
        
        Research Question: "What was the typical system state during 
        the representative operational period?"
        
        Returns:
            list: Snapshots with timestamp >= warmup_period
        """
        return [
            snapshot for snapshot in self._system_snapshots
            if snapshot['timestamp'] >= self._warmup_period
        ]


class AnalysisData:
    """
    Container object holding all analytical populations for a replication.
    
    This object serves as the single source of truth passed to all metric 
    calculation functions, providing clean and consistent interfaces.
    """
    
    def __init__(self, populations):
        """
        Initialize with pre-computed populations.
        
        Args:
            populations: AnalyticalPopulations instance
        """
        # Create all populations upfront for this replication
        self.cohort_orders = populations.get_cohort_orders()
        self.cohort_completed_orders = populations.get_cohort_completed_orders()
        self.cohort_completed_delivery_units = populations.get_cohort_completed_delivery_units()
        self.cohort_paired_orders = populations.get_cohort_paired_orders()
        self.post_warmup_snapshots = populations.get_post_warmup_snapshots()  # NEW


def prepare_analysis_data(repositories, warmup_period, system_snapshots=None):
    """
    Main entry point for creating analysis-ready data populations.
    
    This function creates all analytical populations needed for metrics calculation
    and packages them in a clean container object.
    
    Args:
        repositories: Dict containing all entity repositories from simulation
        warmup_period: Duration to exclude from analysis
        system_snapshots: List of system state snapshots (optional)
        
    Returns:
        AnalysisData: Container with all populations ready for metric calculations
    """
    populations = AnalyticalPopulations(repositories, warmup_period, system_snapshots)
    return AnalysisData(populations)


def get_analysis_time_window(simulation_duration, warmup_period):
    """
    Calculate effective analysis time window after warmup exclusion.
    
    Args:
        simulation_duration: Total simulation duration
        warmup_period: Duration to exclude from start
        
    Returns:
        tuple: (analysis_start_time, analysis_end_time)
    """
    analysis_start_time = warmup_period
    analysis_end_time = simulation_duration
    
    if analysis_start_time >= analysis_end_time:
        raise ValueError(
            f"Warmup period ({warmup_period}) must be less than "
            f"simulation duration ({simulation_duration})"
        )
    
    return analysis_start_time, analysis_end_time