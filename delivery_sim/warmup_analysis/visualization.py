# delivery_sim/warmup_analysis/visualization.py
"""
Streamlined Warmup Analysis Visualization

Clean, focused visualization for:
1. Warmup detection using active_drivers as primary indicator
2. Regime classification using unassigned_delivery_entities as auxiliary indicator  
3. Load ratio hypothesis testing via baseline vs 2x_baseline comparisons
"""

import matplotlib.pyplot as plt
from delivery_sim.utils.logging_system import get_logger


class WelchMethodVisualization:
    """
    Streamlined visualization for warmup analysis and regime classification.
    
    Focuses on dual-purpose metrics:
    - active_drivers: Primary indicator for warmup detection (driver capacity steady state)
    - unassigned_delivery_entities: Auxiliary indicator for regime classification
    """
    
    def __init__(self, figsize=(16, 10)):
        self.figsize = figsize
        self.logger = get_logger("warmup_analysis.visualization")
    
    def create_warmup_analysis_plot(self, time_series_data, title=None):
        """
        Create warmup analysis plot with three metrics.
        
        Shows:
        - Active drivers: Total capacity (Primary warmup indicator)
        - Available drivers: Idle capacity (System throughput indicator)
        - Unassigned entities: Queue pressure (Regime classifier)
        
        Args:
            time_series_data: Data for ONE design point (contains all three metrics)
            title: Optional plot title
            
        Returns:
            matplotlib.figure.Figure
        """
        if 'active_drivers' not in time_series_data:
            raise ValueError("active_drivers metric required for warmup analysis")
        if 'available_drivers' not in time_series_data:
            raise ValueError("available_drivers metric required for capacity analysis")
        if 'unassigned_delivery_entities' not in time_series_data:
            raise ValueError("unassigned_delivery_entities metric required for regime analysis")
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # ========================================================================
        # PANEL 1: Active Drivers (Capacity Ceiling - Warmup Detection)
        # ========================================================================
        active_data = time_series_data['active_drivers']
        
        # Background: Cross-replication averages (raw data)
        ax1.plot(active_data['time_points'], active_data['cross_rep_averages'], 
                color='lightblue', alpha=0.3, linewidth=0.5, label='Raw average')
        
        # Foreground: Moving average (Welch method)
        ax1.plot(active_data['time_points'], active_data['moving_averages'],
                color='blue', linewidth=2, label=f"Welch MA-{active_data['moving_average_window']}")
        
        # Little's Law reference
        if 'theoretical_value' in active_data:
            littles_law = active_data['theoretical_value']
            ax1.axhline(littles_law, color='red', linestyle='--', linewidth=1.5,
                    label=f"Little's Law: {littles_law:.1f}")
        
        ax1.set_ylabel('Active Drivers', fontsize=11, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Active Drivers', fontsize=10, loc='left')
        
        # ========================================================================
        # PANEL 2: Available Drivers (Idle Capacity - NEW!)
        # ========================================================================
        available_data = time_series_data['available_drivers']
        
        # Background: Raw data
        ax2.plot(available_data['time_points'], available_data['cross_rep_averages'],
                color='lightgreen', alpha=0.3, linewidth=0.5, label='Raw average')
        
        # Foreground: Moving average
        ax2.plot(available_data['time_points'], available_data['moving_averages'],
                color='green', linewidth=2, label=f"Welch MA-{available_data['moving_average_window']}")
        
        ax2.set_ylabel('Available Drivers', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Available Drivers (Idle Capacity)', fontsize=10, loc='left')
        
        # ========================================================================
        # PANEL 3: Unassigned Delivery Entities (Queue Pressure)
        # ========================================================================
        unassigned_data = time_series_data['unassigned_delivery_entities']
        
        # Background: Raw data
        ax3.plot(unassigned_data['time_points'], unassigned_data['cross_rep_averages'],
                color='lavender', alpha=0.3, linewidth=0.5, label='Raw average')
        
        # Foreground: Moving average
        ax3.plot(unassigned_data['time_points'], unassigned_data['moving_averages'],
                color='purple', linewidth=2, label=f"Welch MA-{unassigned_data['moving_average_window']}")
        
        ax3.set_xlabel('Simulation Time (minutes)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Unassigned Delivery Entities', fontsize=11, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Unassigned Delivery Entities', fontsize=10, loc='left')
        
        plt.tight_layout()
        
        return fig