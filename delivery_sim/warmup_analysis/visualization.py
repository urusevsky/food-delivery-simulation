# delivery_sim/warmup_analysis/visualization.py
"""
Simplified Time Series Visualization for Warmup Analysis

Clean, focused visualization for visual inspection of cross-replication averaged
time series data. Emphasizes clarity and pattern recognition over complex smoothing.

The approach: Show cross-replication averages clearly and let human pattern
recognition identify the transition from transient to steady-state behavior.
"""

import matplotlib.pyplot as plt
from delivery_sim.utils.logging_system import get_logger


class TimeSeriesVisualization:
    """
    Simple visualization for time series warmup analysis.
    
    Focuses on clear, readable plots of cross-replication averages for
    visual inspection and warmup period determination.
    """
    
    def __init__(self, figsize=(12, 6)):
        self.figsize = figsize
        self.logger = get_logger("warmup_analysis.visualization")
    
    def create_time_series_plot(self, time_series_data, metric_name, title=None):
        """
        Create clean time series plot for visual warmup inspection.
        
        Shows cross-replication averages with clear visual guidance for
        identifying warmup periods through pattern recognition.
        
        Args:
            time_series_data: Results from TimeSeriesPreprocessor
            metric_name: Name of metric to plot
            title: Optional custom title
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if metric_name not in time_series_data:
            raise ValueError(f"Metric {metric_name} not found in data")
        
        data = time_series_data[metric_name]
        time_points = data['time_points']
        cross_rep_averages = data['cross_rep_averages']
        replication_count = data['replication_count']
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Plot cross-replication averages with emphasis
        ax.plot(time_points, cross_rep_averages, 
               'b-', linewidth=2, alpha=0.8,
               label=f'Cross-Replication Average ({replication_count} reps)')
        
        # Add subtle grid for easier reading
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Clean formatting
        ax.set_xlabel('Simulation Time (minutes)', fontsize=12)
        ax.set_ylabel(f'{metric_name.replace("_", " ").title()}', fontsize=12)
        ax.set_title(title or f'Time Series Inspection: {metric_name.replace("_", " ").title()}', fontsize=14)
        ax.legend(fontsize=11)
        
        # Add visual inspection guidance
        guidance_text = (
            'Visual Inspection Guide:\n'
            '• Look for transition from trending to stable oscillation\n' 
            '• Choose warmup period AFTER stabilization begins\n'
            '• Conservative: Add safety margin to identified point'
        )
        
        ax.text(0.02, 0.98, guidance_text,
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
               fontsize=9)
        
        # Add final value annotation for context
        if cross_rep_averages:
            final_value = cross_rep_averages[-1]
            ax.text(0.98, 0.02, f'Final Value: {final_value:.1f}',
                   transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                   fontsize=10)
        
        plt.tight_layout()
        
        self.logger.info(f"Created time series plot for {metric_name}")
        return fig
    
    def create_combined_inspection_plot(self, time_series_data, title=None):
        """
        Create combined plot showing all metrics for comprehensive warmup inspection.
        
        Shows multiple metrics in subplots for comparing warmup patterns across
        different system measurements.
        
        Args:
            time_series_data: Results from TimeSeriesPreprocessor
            title: Optional overall title
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        metrics = list(time_series_data.keys())
        if not metrics:
            raise ValueError("No metrics found in time_series_data")
        
        # Create subplots (vertical stack for easy comparison)
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(self.figsize[0], self.figsize[1] * n_metrics * 0.7))
        
        # Handle single metric case
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(metrics):
            data = time_series_data[metric_name]
            time_points = data['time_points']
            cross_rep_averages = data['cross_rep_averages']
            replication_count = data['replication_count']
            
            ax = axes[i]
            
            # Plot with emphasis on clarity
            ax.plot(time_points, cross_rep_averages, 
                   'b-', linewidth=2, alpha=0.8,
                   label=f'{replication_count} replications')
            
            # Clean formatting
            ax.set_ylabel(f'{metric_name.replace("_", " ").title()}', fontsize=11)
            ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Add final value for context
            if cross_rep_averages:
                final_value = cross_rep_averages[-1]
                ax.text(0.98, 0.95, f'Final: {final_value:.1f}',
                       transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                       fontsize=9)
        
        # Only add x-label to bottom plot
        axes[-1].set_xlabel('Simulation Time (minutes)', fontsize=12)
        
        # Overall title and guidance
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle('Combined Time Series Inspection for Warmup Analysis', fontsize=16)
        
        # Add overall guidance at the top
        fig.text(0.02, 0.95, 
                'Look for consistent warmup patterns across metrics. Choose conservative warmup period that works for all.',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        self.logger.info(f"Created combined inspection plot for {len(metrics)} metrics")
        return fig
    
    def create_warmup_comparison_plot(self, time_series_data, warmup_candidates, title=None):
        """
        Create plot showing different warmup period candidates for comparison.
        
        Helps visualize how different warmup choices would affect the analysis window.
        
        Args:
            time_series_data: Results from TimeSeriesPreprocessor
            warmup_candidates: List of potential warmup periods to visualize
            title: Optional title
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Use first metric for demonstration
        first_metric = list(time_series_data.keys())[0]
        data = time_series_data[first_metric]
        time_points = data['time_points']
        cross_rep_averages = data['cross_rep_averages']
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot the time series
        ax.plot(time_points, cross_rep_averages, 'b-', linewidth=2, alpha=0.8,
               label='Cross-Replication Average')
        
        # Add vertical lines for warmup candidates
        colors = ['red', 'orange', 'green', 'purple', 'brown']
        for i, warmup_period in enumerate(warmup_candidates):
            if warmup_period <= max(time_points):
                color = colors[i % len(colors)]
                ax.axvline(x=warmup_period, color=color, linestyle='--', linewidth=2, alpha=0.7,
                          label=f'Warmup = {warmup_period}')
        
        # Formatting
        ax.set_xlabel('Simulation Time (minutes)', fontsize=12)
        ax.set_ylabel(f'{first_metric.replace("_", " ").title()}', fontsize=12)
        ax.set_title(title or 'Warmup Period Comparison', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add guidance
        ax.text(0.02, 0.98, 
               'Compare warmup candidates:\n• Conservative choice preserves more steady-state data\n• Analysis window = Total duration - Warmup period',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
               fontsize=9)
        
        plt.tight_layout()
        
        self.logger.info(f"Created warmup comparison plot with {len(warmup_candidates)} candidates")
        return fig


def create_inspection_plots(time_series_data, show_combined=True, show_individual=True):
    """
    Convenience function to create standard inspection plots.
    
    Args:
        time_series_data: Results from TimeSeriesPreprocessor
        show_combined: Whether to create combined plot
        show_individual: Whether to create individual metric plots
        
    Returns:
        list: List of created figures
    """
    viz = TimeSeriesVisualization()
    figures = []
    
    # Individual plots
    if show_individual:
        for metric_name in time_series_data.keys():
            fig = viz.create_time_series_plot(time_series_data, metric_name)
            figures.append(fig)
    
    # Combined plot
    if show_combined and len(time_series_data) > 1:
        fig = viz.create_combined_inspection_plot(time_series_data)
        figures.append(fig)
    
    return figures