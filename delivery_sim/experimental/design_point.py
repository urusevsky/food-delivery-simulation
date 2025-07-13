# delivery_sim/experimental/design_point.py
"""
DesignPoint: Clean Specification of Experimental Condition

Simple container for Infrastructure + OperationalConfig + ScoringConfig.
No feature creep - just the essentials.
"""

class DesignPoint:
    """
    Clean specification of an experimental condition.
    
    Simply stores Infrastructure, OperationalConfig, and ScoringConfig
    that together define how the delivery system will behave.
    """
    
    def __init__(self, infrastructure, operational_config, scoring_config, name=None):
        """
        Initialize design point with system configuration.
        
        Args:
            infrastructure: Infrastructure instance
            operational_config: OperationalConfig instance
            scoring_config: ScoringConfig instance
            name: Optional name for this design point
        """
        # Validate infrastructure is analyzed
        if not infrastructure.has_analysis_results():
            raise ValueError("Infrastructure must be pre-analyzed before use in DesignPoint")
        
        # Store configuration
        self.infrastructure = infrastructure
        self.operational_config = operational_config
        self.scoring_config = scoring_config
        self.name = name
    
    def __str__(self):
        """String representation."""
        if self.name:
            return f"DesignPoint(name='{self.name}')"
        else:
            return f"DesignPoint(order={self.operational_config.mean_order_inter_arrival_time}, driver={self.operational_config.mean_driver_inter_arrival_time})"