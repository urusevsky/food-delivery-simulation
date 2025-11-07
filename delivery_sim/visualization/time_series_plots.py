"""
Flexible time series visualization for exploratory analysis.

Unlike warmup_analysis.visualization (specialized for warmup detection),
this module provides general-purpose plotting capabilities for understanding
system dynamics, testing hypotheses, and exploring mechanisms.

Key capabilities:
- Single replication plots (no averaging)
- Multiple replications overlaid
- Cross-replication statistics (mean, bands)
- Flexible metric selection
- Customizable layouts
"""

import matplotlib.pyplot as plt
import numpy as np
from delivery_sim.utils.logging_system import get_logger

class TimeSeriesVisualizer:
    """
    General-purpose time series visualization for system state metrics.
    
    Design philosophy: Flexibility over standardization.
    Provides building blocks for various visualization needs.
    """
    
    def __init__(self, figsize=(16, 10)):
        self.figsize = figsize
        self.logger = get_logger("visualization.time_series")
    
    def plot_single_replication(self, snapshots, metrics, title=None, 
                               apply_smoothing=False, window=100):
        """
        Plot time series for a single replication.
        
        Args:
            snapshots: List of snapshot dicts from single replication
            metrics: List of metric names to plot (e.g., ['available_drivers', 'unassigned_delivery_entities'])
            title: Optional plot title
            apply_smoothing: Whether to apply moving average
            window: Window size for moving average if apply_smoothing=True
            
        Returns:
            fig, axes
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=self.figsize, sharex=True)
        if n_metrics == 1:
            axes = [axes]
        
        timestamps = [s['timestamp'] for s in snapshots]
        
        for idx, metric_name in enumerate(metrics):
            values = [s[metric_name] for s in snapshots]
            
            axes[idx].plot(timestamps, values, alpha=0.5, label='Raw', linewidth=1)
            
            if apply_smoothing:
                smoothed = self._moving_average(values, window)
                axes[idx].plot(timestamps, smoothed, label=f'MA-{window}', linewidth=2)
            
            axes[idx].set_ylabel(metric_name.replace('_', ' ').title())
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Simulation Time (minutes)')
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig, axes
    
    def plot_multiple_replications(self, replication_results, metrics, 
                                   design_name=None, show_average=True,
                                   apply_smoothing=False, window=100):
        """
        Plot multiple replications overlaid to see between-replication variability.
        
        Args:
            replication_results: List of replication result dicts with 'system_snapshots'
            metrics: List of metric names to plot
            design_name: Optional design point name for title
            show_average: Whether to overlay cross-replication average
            apply_smoothing: Whether to apply moving average to individual traces
            window: Window size for moving average
            
        Returns:
            fig, axes
        """
        n_metrics = len(metrics)
        n_reps = len(replication_results)
        
        fig, axes = plt.subplots(n_metrics, 1, figsize=self.figsize, sharex=True)
        if n_metrics == 1:
            axes = [axes]
        
        colors = plt.cm.tab10(range(n_reps))
        
        for idx, metric_name in enumerate(metrics):
            all_rep_data = []
            
            for rep_idx, rep_result in enumerate(replication_results):
                snapshots = rep_result['system_snapshots']
                timestamps = [s['timestamp'] for s in snapshots]
                values = [s[metric_name] for s in snapshots]
                
                if apply_smoothing:
                    values = self._moving_average(values, window)
                
                axes[idx].plot(timestamps, values, alpha=0.4, color=colors[rep_idx],
                             label=f'Rep {rep_idx+1}', linewidth=1)
                
                all_rep_data.append(values)
            
            if show_average:
                # Cross-replication average
                avg_values = np.mean(all_rep_data, axis=0)
                axes[idx].plot(timestamps, avg_values, color='black', 
                             linewidth=2.5, label='Average', zorder=10)
            
            axes[idx].set_ylabel(metric_name.replace('_', ' ').title())
            axes[idx].legend(loc='upper right', fontsize=8, ncol=2)
            axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Simulation Time (minutes)')
        title = f"Multiple Replications: {design_name}" if design_name else "Multiple Replications"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig, axes
    
    def plot_complementary_analysis(self, snapshots, metric_pairs, title=None):
        """
        Plot pairs of metrics on same axes to examine complementary relationships.
        
        Useful for testing hypothesis: "available_drivers and unassigned_orders 
        should be complementary in individual replications"
        
        Args:
            snapshots: Single replication snapshots
            metric_pairs: List of (metric1, metric2) tuples to plot together
            title: Optional title
            
        Returns:
            fig, axes
        """
        n_pairs = len(metric_pairs)
        fig, axes = plt.subplots(n_pairs, 1, figsize=self.figsize, sharex=True)
        if n_pairs == 1:
            axes = [axes]
        
        timestamps = [s['timestamp'] for s in snapshots]
        
        for idx, (metric1, metric2) in enumerate(metric_pairs):
            ax = axes[idx]
            ax2 = ax.twinx()  # Secondary y-axis
            
            values1 = [s[metric1] for s in snapshots]
            values2 = [s[metric2] for s in snapshots]
            
            line1 = ax.plot(timestamps, values1, color='blue', label=metric1.replace('_', ' ').title())
            line2 = ax2.plot(timestamps, values2, color='red', label=metric2.replace('_', ' ').title())
            
            ax.set_ylabel(metric1.replace('_', ' ').title(), color='blue')
            ax2.set_ylabel(metric2.replace('_', ' ').title(), color='red')
            
            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right')
            
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Simulation Time (minutes)')
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig, axes
    
    def _moving_average(self, values, window):
        """Calculate moving average with window size."""
        if len(values) < window:
            return values
        return np.convolve(values, np.ones(window)/window, mode='same')