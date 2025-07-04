import matplotlib.pyplot as plt
from delivery_sim.utils.logging_system import get_logger

class WarmupVisualization:
    """
    Simple visualization tools for Welch plot warmup detection.
    
    Focuses on the core Welch plot: cross-replication averages 
    with cumulative average overlay for visual inspection.
    """
    
    def __init__(self, figsize=(12, 6)):
        self.figsize = figsize
        self.logger = get_logger("warmup_analysis.visualization")
    
    def create_welch_plot(self, welch_results, metric_name, title=None):
        """
        Create the classic Welch plot as described in simulation textbook.
        
        Shows:
        - Cross-replication averages (blue line) 
        - Cumulative average smoothing (red line)
        
        Args:
            welch_results: Results from WelchAnalyzer.analyze_warmup_convergence()
            metric_name: Name of metric to plot
            title: Optional custom title
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if metric_name not in welch_results:
            raise ValueError(f"Metric {metric_name} not found in results")
        
        results = welch_results[metric_name]
        time_points = results['time_points']
        cross_rep_averages = results['cross_rep_averages']
        cumulative_average = results['cumulative_average']
        replication_count = results['replication_count']
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Plot cross-replication averages (Ȳ.j)
        ax.plot(time_points, cross_rep_averages, 
               'b-', linewidth=1, alpha=0.7,
               label=f'Cross-Replication Average ({replication_count} reps)')
        
        # Plot cumulative average smoothing (key for convergence assessment)
        ax.plot(time_points, cumulative_average, 
               'r-', linewidth=2,
               label='Cumulative Average (Smoothed)')
        
        # Formatting
        ax.set_xlabel('Simulation Time (minutes)')
        ax.set_ylabel(f'{metric_name.replace("_", " ").title()}')
        ax.set_title(title or f'Welch Plot: {metric_name.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add convergence assessment guidance
        ax.text(0.02, 0.98, 
               'Visual Inspection: Look for red line to stabilize\n' +
               'Warmup period should end BEFORE stabilization', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
               fontsize=9)
        
        plt.tight_layout()
        
        self.logger.info(f"Created Welch plot for {metric_name}")
        return fig
    
    def create_multi_metric_plot(self, welch_results, title=None):
        """
        Show Welch plots for multiple metrics in subplots.
        
        Args:
            welch_results: Results from WelchAnalyzer.analyze_warmup_convergence()
            title: Optional overall title
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        metrics = list(welch_results.keys())
        if not metrics:
            raise ValueError("No metrics found in welch_results")
        
        # Create subplots (vertical stack)
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(self.figsize[0], self.figsize[1] * n_metrics))
        
        # Handle single metric case
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(metrics):
            results = welch_results[metric_name]
            time_points = results['time_points']
            cross_rep_averages = results['cross_rep_averages']
            cumulative_average = results['cumulative_average']
            replication_count = results['replication_count']
            
            ax = axes[i]
            
            # Plot data
            ax.plot(time_points, cross_rep_averages, 
                   'b-', linewidth=1, alpha=0.7,
                   label=f'Cross-Rep Avg ({replication_count} reps)')
            ax.plot(time_points, cumulative_average, 
                   'r-', linewidth=2,
                   label='Cumulative Average')
            
            # Formatting
            ax.set_ylabel(f'{metric_name.replace("_", " ").title()}')
            ax.set_title(f'{metric_name.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Only add x-label to bottom plot
        axes[-1].set_xlabel('Simulation Time (minutes)')
        
        # Overall title
        if title:
            fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        
        self.logger.info(f"Created multi-metric plot for {len(metrics)} metrics")
        return fig