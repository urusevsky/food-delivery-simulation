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
        Create clean warmup analysis plot with dual-purpose metrics.
        
        Shows both metrics in context:
        - Active drivers: Primary signal for warmup timing
        - Unassigned entities: Contextual signal for regime classification
        
        Args:
            time_series_data: Data for ONE design point (contains both metrics)
            title: Optional plot title
            
        Returns:
            matplotlib.figure.Figure
        """
        if 'active_drivers' not in time_series_data:
            raise ValueError("active_drivers metric required for warmup analysis")
        if 'unassigned_delivery_entities' not in time_series_data:
            raise ValueError("unassigned_delivery_entities metric required for regime analysis")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # TOP: Active Drivers (Primary - warmup detection)
        active_data = time_series_data['active_drivers']
        
        # Background: Cross-replication averages (noisy data)
        ax1.plot(active_data['time_points'], active_data['cross_rep_averages'], 
                'lightblue', alpha=0.6, linewidth=1, 
                label=f"{active_data['replication_count']} reps")
        
        # Foreground: Welch's method smoothed trend (primary signal)
        ax1.plot(active_data['time_points'], active_data['moving_averages'], 
                'blue', linewidth=3, 
                label=f'Welch MA-{active_data["moving_average_window"]}')
        
        # Little's Law reference (theoretical convergence target)
        if 'theoretical_value' in active_data:
            theoretical = active_data['theoretical_value']
            ax1.axhline(y=theoretical, color='red', linestyle='--', linewidth=2, 
                       label=f"Little's Law: {theoretical:.1f}")
        
        ax1.set_ylabel('Active Drivers', fontsize=12)
        ax1.set_title('Active Drivers', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # BOTTOM: Unassigned Entities (Auxiliary - regime classification)  
        unassigned_data = time_series_data['unassigned_delivery_entities']
        
        # Background: Cross-replication averages
        ax2.plot(unassigned_data['time_points'], unassigned_data['cross_rep_averages'],
                'lightblue', alpha=0.6, linewidth=1, 
                label=f"{unassigned_data['replication_count']} reps")
        
        # Foreground: Welch's method smoothed trend
        ax2.plot(unassigned_data['time_points'], unassigned_data['moving_averages'],
                'blue', linewidth=3, 
                label=f'Welch MA-{unassigned_data["moving_average_window"]}')
        
        # Final value annotation (helps with regime assessment)
        final_value = unassigned_data['moving_averages'][-1] if unassigned_data['moving_averages'] else 0
        ax2.text(0.98, 0.95, f'Final: {final_value:.1f}',
                transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7), fontsize=9)
        
        ax2.set_ylabel('Unassigned Delivery Entities', fontsize=12)
        ax2.set_xlabel('Simulation Time (minutes)', fontsize=12)
        ax2.set_title('Unassigned Delivery Entities', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        self.logger.info(f"Created warmup analysis plot: {title or 'Unnamed'}")
        return fig