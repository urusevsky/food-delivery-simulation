# delivery_sim/warmup_analysis/visualization.py
"""
Enhanced Warmup Analysis Visualization with Welch's Method

Implements Welch's graphical procedure with Little's Law theoretical validation
for principled yet intuitive warmup period determination.
"""

import matplotlib.pyplot as plt
from delivery_sim.utils.logging_system import get_logger


class WelchMethodVisualization:
    """
    Enhanced visualization implementing Welch's graphical method for warmup analysis.
    
    Combines cross-replication averaging, moving average smoothing, and Little's Law
    theoretical validation for robust visual warmup detection.
    """
    
    def __init__(self, figsize=(12, 6)):
        self.figsize = figsize
        self.logger = get_logger("warmup_analysis.visualization")
    
    def create_welch_method_plot(self, time_series_data, metric_name, title=None):
        """
        Create Welch's method plot with theoretical validation for warmup inspection.
        
        Shows cross-replication averages, moving average trend, and Little's Law
        theoretical reference for principled warmup determination.
        
        Args:
            time_series_data: Enhanced results from TimeSeriesPreprocessor
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
        moving_averages = data['moving_averages']
        replication_count = data['replication_count']
        window_size = data['moving_average_window']
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Plot cross-replication averages (background data)
        ax.plot(time_points, cross_rep_averages, 
               'lightblue', linewidth=1, alpha=0.6,
               label=f'Cross-Rep Averages ({replication_count} reps)')
        
        # Plot Welch's method moving averages (primary trend)
        ax.plot(time_points, moving_averages, 
               'blue', linewidth=3, alpha=0.9,
               label=f'Welch\'s Method (MA-{window_size})')
        
        # Add Little's Law theoretical reference if available for active drivers
        if metric_name == 'active_drivers' and 'theoretical_value' in data:
            theoretical_value = data['theoretical_value']
            ax.axhline(y=theoretical_value, color='red', linestyle='--', linewidth=2,
                      label=f'Little\'s Law Prediction: {theoretical_value:.1f}')
            
            # Add convergence assessment text
            final_ma_value = moving_averages[-1] if moving_averages else 0
            convergence_error = abs(final_ma_value - theoretical_value) / theoretical_value * 100
            
            ax.text(0.98, 0.95, 
                   f'Final MA: {final_ma_value:.1f}\n'
                   f'Theoretical: {theoretical_value:.1f}\n'
                   f'Error: {convergence_error:.1f}%',
                   transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9),
                   fontsize=10)
        
        # Clean formatting
        ax.set_xlabel('Simulation Time (minutes)', fontsize=12)
        ax.set_ylabel(f'{metric_name.replace("_", " ").title()}', fontsize=12)
        ax.set_title(title or f'Welch\'s Method Warmup Analysis: {metric_name.replace("_", " ").title()}', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add Welch's method guidance
        guidance_text = (
            'Welch\'s Method Interpretation:\n'
            '• Blue line shows smoothed trend\n'
            '• Look for convergence to theoretical value\n'
            '• Choose warmup after trend stabilizes'
        )
        
        ax.text(0.02, 0.98, guidance_text,
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
               fontsize=9)
        
        plt.tight_layout()
        
        self.logger.info(f"Created Welch's method plot for {metric_name}")
        return fig
    
    def create_combined_welch_inspection_plot(self, time_series_data, title=None):
        """
        Create combined Welch's method plot for comprehensive warmup inspection.
        
        Shows multiple metrics with moving averages and theoretical references
        for systematic warmup period determination.
        
        Args:
            time_series_data: Enhanced results from TimeSeriesPreprocessor
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
            moving_averages = data['moving_averages']
            replication_count = data['replication_count']
            window_size = data['moving_average_window']
            
            ax = axes[i]
            
            # Plot cross-replication averages (background)
            ax.plot(time_points, cross_rep_averages, 
                   'lightblue', linewidth=1, alpha=0.6,
                   label=f'{replication_count} reps')
            
            # Plot Welch's method moving averages (primary)
            ax.plot(time_points, moving_averages, 
                   'blue', linewidth=2.5, alpha=0.9,
                   label=f'Welch MA-{window_size}')
            
            # Add Little's Law reference for active drivers
            if metric_name == 'active_drivers' and 'theoretical_value' in data:
                theoretical_value = data['theoretical_value']
                ax.axhline(y=theoretical_value, color='red', linestyle='--', linewidth=2,
                          label=f'Little\'s Law: {theoretical_value:.1f}')
                
                # Convergence info
                final_ma_value = moving_averages[-1] if moving_averages else 0
                ax.text(0.98, 0.95, f'Final: {final_ma_value:.1f}',
                       transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7),
                       fontsize=9)
            else:
                # For non-driver metrics, show final value
                final_value = moving_averages[-1] if moving_averages else 0
                ax.text(0.98, 0.95, f'Final: {final_value:.1f}',
                       transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                       fontsize=9)
            
            # Clean formatting
            ax.set_ylabel(f'{metric_name.replace("_", " ").title()}', fontsize=11)
            ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Only add x-label to bottom plot
        axes[-1].set_xlabel('Simulation Time (minutes)', fontsize=12)
        
        # Overall title and guidance
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle('Welch\'s Method Combined Warmup Analysis', fontsize=16)
        
        # Add methodology guidance at the top
        fig.text(0.02, 0.95, 
                'Welch\'s Method: Focus on blue smoothed lines. For active drivers, look for convergence to red theoretical line.',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        self.logger.info(f"Created combined Welch's method plot for {len(metrics)} metrics")
        return fig
    
    def create_warmup_candidate_comparison_plot(self, time_series_data, warmup_candidates, 
                                              metric_name='active_drivers', title=None):
        """
        Create plot comparing different warmup period candidates using Welch's method.
        
        Shows how different warmup choices affect the analysis window relative to
        the Welch's method convergence pattern.
        
        Args:
            time_series_data: Enhanced time series data
            warmup_candidates: List of potential warmup periods to visualize
            metric_name: Metric to use for comparison (default: active_drivers)
            title: Optional title
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if metric_name not in time_series_data:
            raise ValueError(f"Metric {metric_name} not found in data")
        
        data = time_series_data[metric_name]
        time_points = data['time_points']
        moving_averages = data['moving_averages']
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot Welch's method trend
        ax.plot(time_points, moving_averages, 'blue', linewidth=3, alpha=0.8,
               label='Welch\'s Method Trend')
        
        # Add Little's Law reference if available
        if 'theoretical_value' in data:
            theoretical_value = data['theoretical_value']
            ax.axhline(y=theoretical_value, color='red', linestyle='--', linewidth=2,
                      label=f'Little\'s Law Target: {theoretical_value:.1f}')
        
        # Add vertical lines for warmup candidates
        colors = ['orange', 'green', 'purple', 'brown', 'pink']
        for i, warmup_period in enumerate(warmup_candidates):
            if warmup_period <= max(time_points):
                color = colors[i % len(colors)]
                ax.axvline(x=warmup_period, color=color, linestyle=':', linewidth=2, alpha=0.8,
                          label=f'Candidate: {warmup_period} min')
        
        # Formatting
        ax.set_xlabel('Simulation Time (minutes)', fontsize=12)
        ax.set_ylabel(f'{metric_name.replace("_", " ").title()}', fontsize=12)
        ax.set_title(title or f'Warmup Candidate Comparison: {metric_name.replace("_", " ").title()}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add guidance
        ax.text(0.02, 0.98, 
               'Warmup Comparison Guide:\n• Choose warmup after Welch trend stabilizes\n• Conservative choice ensures validity',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
               fontsize=9)
        
        plt.tight_layout()
        
        self.logger.info(f"Created warmup comparison plot with {len(warmup_candidates)} candidates")
        return fig


def create_welch_inspection_plots(time_series_data, show_combined=True, show_individual=True):
    """
    Convenience function to create standard Welch's method inspection plots.
    
    Args:
        time_series_data: Enhanced results from TimeSeriesPreprocessor
        show_combined: Whether to create combined plot
        show_individual: Whether to create individual metric plots
        
    Returns:
        list: List of created figures
    """
    viz = WelchMethodVisualization()
    figures = []
    
    # Individual plots
    if show_individual:
        for metric_name in time_series_data.keys():
            fig = viz.create_welch_method_plot(time_series_data, metric_name)
            figures.append(fig)
    
    # Combined plot
    if show_combined and len(time_series_data) > 1:
        fig = viz.create_combined_welch_inspection_plot(time_series_data)
        figures.append(fig)
    
    return figures


# Maintain backward compatibility with old naming
TimeSeriesVisualization = WelchMethodVisualization
create_inspection_plots = create_welch_inspection_plots