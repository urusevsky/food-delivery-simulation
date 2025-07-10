# delivery_sim/infrastructure/infrastructure_analyzer.py
"""
InfrastructureAnalyzer: Comprehensive Analysis for Infrastructure Instances

This module provides comprehensive analysis capabilities for Infrastructure instances,
supporting both computational efficiency (reusable analysis) and analytical necessity
(informed parameter design for experimental studies).

Key Analysis Capabilities:
- Basic infrastructure characteristics (typical distance, density patterns)
- Restaurant spatial pattern analysis (for pairing parameter design)
- Customer distance pattern analysis (for pairing parameter design)
- Infrastructure visualization (for experimental design validation)
- Comprehensive analysis orchestration
"""

import numpy as np
import matplotlib.pyplot as plt
from delivery_sim.utils.location_utils import calculate_distance
from delivery_sim.utils.logging_system import get_logger


class InfrastructureAnalyzer:
    """
    Comprehensive analyzer for Infrastructure instances.
    
    Provides all analysis capabilities needed for infrastructure characterization
    and experimental parameter design. Results can be cached in Infrastructure
    instances for reuse across multiple experimental configurations.
    """
    
    def __init__(self, infrastructure):
        """
        Initialize analyzer with an Infrastructure instance.
        
        Args:
            infrastructure: Infrastructure instance to analyze
        """
        self.infrastructure = infrastructure
        self.logger = get_logger("infrastructure.analyzer")
        
        # Quick reference to infrastructure components
        self.restaurant_repository = infrastructure.get_restaurant_repository()
        self.structural_rng = infrastructure.get_structural_rng()
        self.structural_config = infrastructure.structural_config
        
        self.logger.debug(f"InfrastructureAnalyzer initialized for: {infrastructure}")
    
    def analyze_complete_infrastructure(self, scoring_config=None):
        """
        Orchestrate comprehensive infrastructure analysis.
        
        This method performs all analysis types and caches results in the
        Infrastructure instance for reuse across experimental configurations.
        
        Args:
            scoring_config: Optional ScoringConfig for analysis parameters
            
        Returns:
            dict: Complete analysis results with all analysis types
        """
        self.logger.info("Starting comprehensive infrastructure analysis...")
        
        # Get basic characteristics (fast, no computation needed)
        basic_characteristics = self.infrastructure.get_basic_characteristics()
        
        # Calculate typical distance (Monte Carlo sampling - expensive)
        sample_size = getattr(scoring_config, 'typical_distance_samples', 1000) if scoring_config else 1000
        typical_distance = self.calculate_typical_distance(sample_size)
        
        # Analyze restaurant spatial patterns (for pairing parameter design)
        restaurant_analysis = self.analyze_restaurant_spatial_patterns()
        
        # Analyze customer distance patterns (for pairing parameter design)
        customer_analysis = self.analyze_customer_distance_patterns()
        
        # Assemble complete analysis results
        complete_analysis = {
            # Basic characteristics
            **basic_characteristics,
            'typical_distance': typical_distance,
            
            # Spatial pattern analysis  
            'restaurant_spatial_analysis': restaurant_analysis,
            'customer_distance_analysis': customer_analysis,
            
            # Analysis metadata
            'analysis_sample_size': sample_size,
            'infrastructure_signature': self.infrastructure.get_infrastructure_signature()
        }
        
        # Cache results in Infrastructure instance
        self.infrastructure.set_analysis_results(complete_analysis)
        
        self.logger.info(f"Infrastructure analysis complete: typical_distance={typical_distance:.3f}km, "
                        f"restaurant_patterns=analyzed, customer_patterns=analyzed")
        
        return complete_analysis
    
    def calculate_typical_distance(self, sample_size=1000):
        """
        Calculate characteristic single-order delivery distance for scoring normalization.
        
        This is the original method used for priority scoring system normalization.
        Uses Monte Carlo sampling to determine typical distance in this infrastructure.
        
        Args:
            sample_size: Number of samples for Monte Carlo estimation
            
        Returns:
            float: Typical distance for this geographic configuration
        """
        samples = []
        restaurants = self.restaurant_repository.find_all()
        area_size = self.structural_config.delivery_area_size
        
        if not restaurants:
            raise ValueError("No restaurants found in repository for distance calculation")
        
        self.logger.debug(f"Calculating typical distance with {sample_size} samples")
        self.logger.debug(f"Area: {area_size}x{area_size}km, Restaurants: {len(restaurants)}")
        
        for _ in range(sample_size):
            # Sample driver location (uniform in delivery area)
            driver_loc = self.structural_rng.rng.uniform(0, area_size, size=2).tolist()
            
            # Sample restaurant (uniform selection)
            restaurant = self.structural_rng.rng.choice(restaurants)
            
            # Sample customer location (uniform in delivery area)
            customer_loc = self.structural_rng.rng.uniform(0, area_size, size=2).tolist()
            
            # Calculate full delivery distance
            distance = (
                calculate_distance(driver_loc, restaurant.location) +
                calculate_distance(restaurant.location, customer_loc)
            )
            samples.append(distance)
        
        # Use median for robustness to outliers
        typical_distance = np.median(samples)
        
        self.logger.debug(f"Typical distance calculated: {typical_distance:.3f}km")
        self.logger.debug(f"Distance distribution: min={np.min(samples):.3f}, "
                         f"mean={np.mean(samples):.3f}, max={np.max(samples):.3f}")
        
        return typical_distance
    
    def analyze_restaurant_spatial_patterns(self):
        """
        Analyze all pairwise restaurant distances for pairing parameter design.
        
        This analysis helps determine meaningful ranges for restaurants_proximity_threshold
        by characterizing the actual distance distribution between restaurants.
        
        Returns:
            dict: Contains pairwise distances and statistical characterization
        """
        restaurants = self.restaurant_repository.find_all()
        
        if len(restaurants) < 2:
            self.logger.warning("Need at least 2 restaurants for spatial analysis")
            return {'error': 'Insufficient restaurants for spatial analysis'}
        
        self.logger.debug(f"Analyzing spatial patterns for {len(restaurants)} restaurants")
        
        # Calculate all pairwise distances
        pairwise_distances = []
        for i, rest1 in enumerate(restaurants):
            for j, rest2 in enumerate(restaurants[i+1:], i+1):
                distance = calculate_distance(rest1.location, rest2.location)
                pairwise_distances.append({
                    'restaurant_1': rest1.restaurant_id,
                    'restaurant_2': rest2.restaurant_id,
                    'distance': distance
                })
        
        # Statistical characterization
        distances = [pair['distance'] for pair in pairwise_distances]
        
        distance_statistics = {
            'count': len(distances),
            'min': np.min(distances),
            'p10': np.percentile(distances, 10),
            'p25': np.percentile(distances, 25),
            'p50': np.percentile(distances, 50),
            'p75': np.percentile(distances, 75),
            'p90': np.percentile(distances, 90),
            'max': np.max(distances),
            'mean': np.mean(distances),
            'std': np.std(distances)
        }
        
        # Calculate pairing feasibility at different thresholds
        max_distance = np.max(distances)
        thresholds = np.arange(1.0, max_distance + 1, 0.5)
        pairing_feasibility = []
        
        for threshold in thresholds:
            eligible_pairs = sum(1 for d in distances if d <= threshold)
            percentage = eligible_pairs / len(distances)
            pairing_feasibility.append({
                'threshold': threshold, 
                'eligible_pairs': eligible_pairs,
                'eligible_percentage': percentage
            })
        
        self.logger.debug(f"Restaurant spatial analysis complete: {len(distances)} pairwise distances")
        self.logger.debug(f"Distance range: {distance_statistics['min']:.2f} - {distance_statistics['max']:.2f} km")
        
        return {
            'pairwise_distances': pairwise_distances,
            'distance_statistics': distance_statistics,
            'pairing_feasibility_curve': pairing_feasibility
        }
    
    def analyze_customer_distance_patterns(self, sample_size=10000):
        """
        Analyze distances between uniformly distributed customers for pairing parameter design.
        
        This helps determine meaningful ranges for customers_proximity_threshold
        by characterizing typical distances between random customer locations.
        
        Args:
            sample_size: Number of customer pairs to sample
            
        Returns:
            dict: Statistical characterization of customer-to-customer distances
        """
        area_size = self.structural_config.delivery_area_size
        
        self.logger.debug(f"Analyzing customer distance patterns with {sample_size} samples")
        self.logger.debug(f"Area: {area_size}x{area_size}km")
        
        distances = []
        for _ in range(sample_size):
            # Two random customer locations
            customer1 = self.structural_rng.rng.uniform(0, area_size, size=2).tolist()
            customer2 = self.structural_rng.rng.uniform(0, area_size, size=2).tolist()
            distance = calculate_distance(customer1, customer2)
            distances.append(distance)
        
        # Statistical characterization
        distance_statistics = {
            'count': len(distances),
            'min': np.min(distances),
            'p10': np.percentile(distances, 10),
            'p25': np.percentile(distances, 25),
            'p50': np.percentile(distances, 50),
            'p75': np.percentile(distances, 75),
            'p90': np.percentile(distances, 90),
            'max': np.max(distances),
            'mean': np.mean(distances),
            'std': np.std(distances)
        }
        
        self.logger.debug(f"Customer distance analysis complete")
        self.logger.debug(f"Distance range: {distance_statistics['min']:.2f} - {distance_statistics['max']:.2f} km")
        self.logger.debug(f"Median customer distance: {distance_statistics['p50']:.2f} km")
        
        return {
            'distance_statistics': distance_statistics,
            'area_size': area_size,
            'sample_size': sample_size
        }
    
    def visualize_infrastructure(self, show_closest_pairs=3, figsize=(10, 10)):
        """
        Create visualization of delivery area with restaurant locations.
        
        Shows restaurant distribution and highlights closest restaurant pairs
        to aid in pairing parameter selection.
        
        Args:
            show_closest_pairs: Number of closest restaurant pairs to highlight
            figsize: Figure size for matplotlib
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        restaurants = self.restaurant_repository.find_all()
        area_size = self.structural_config.delivery_area_size
        
        if not restaurants:
            raise ValueError("No restaurants found for visualization")
        
        # Extract coordinates
        x_coords = [r.location[0] for r in restaurants]
        y_coords = [r.location[1] for r in restaurants]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot restaurants
        scatter = ax.scatter(x_coords, y_coords, s=200, c='red', marker='s', 
                           alpha=0.8, edgecolors='black', linewidth=1,
                           label=f'Restaurants (n={len(restaurants)})')
        
        # Add restaurant IDs
        for restaurant in restaurants:
            ax.annotate(restaurant.restaurant_id, 
                       (restaurant.location[0], restaurant.location[1]),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, fontweight='bold')
        
        # Highlight closest pairs if requested and possible
        if show_closest_pairs > 0 and len(restaurants) >= 2:
            spatial_analysis = self.analyze_restaurant_spatial_patterns()
            
            if 'pairwise_distances' in spatial_analysis:
                closest_pairs = sorted(spatial_analysis['pairwise_distances'], 
                                     key=lambda x: x['distance'])[:show_closest_pairs]
                
                colors = ['blue', 'green', 'orange', 'purple', 'brown']
                
                for i, pair in enumerate(closest_pairs):
                    # Find restaurant objects
                    rest1 = self.restaurant_repository.find_by_id(pair['restaurant_1'])
                    rest2 = self.restaurant_repository.find_by_id(pair['restaurant_2'])
                    
                    if rest1 and rest2:
                        # Draw line
                        color = colors[i % len(colors)]
                        ax.plot([rest1.location[0], rest2.location[0]], 
                               [rest1.location[1], rest2.location[1]], 
                               color=color, linestyle='--', linewidth=2, alpha=0.7,
                               label=f'#{i+1} closest: {pair["distance"]:.1f}km')
        
        # Formatting
        ax.set_xlim(-0.5, area_size + 0.5)
        ax.set_ylim(-0.5, area_size + 0.5)
        ax.set_xlabel('X coordinate (km)', fontsize=12)
        ax.set_ylabel('Y coordinate (km)', fontsize=12)
        ax.set_title(f'Infrastructure Layout: {area_size}×{area_size}km Delivery Area', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add area info as text
        basic_chars = self.infrastructure.get_basic_characteristics()
        ax.text(0.02, 0.98, 
               f'Area: {area_size}×{area_size} km\n'
               f'Restaurants: {len(restaurants)}\n'
               f'Density: {basic_chars["restaurant_density"]:.3f} rest/km²',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
               fontsize=10)
        
        plt.tight_layout()
        
        self.logger.debug(f"Infrastructure visualization created: {len(restaurants)} restaurants, "
                         f"{show_closest_pairs} closest pairs highlighted")
        
        return fig
    
    def generate_parameter_design_report(self):
        """
        Generate comprehensive parameter design report for experimental design.
        
        This method provides guidance for setting pairing thresholds and other
        operational parameters based on infrastructure characteristics.
        
        Returns:
            dict: Parameter design recommendations based on infrastructure analysis
        """
        # Ensure infrastructure has been analyzed
        if not self.infrastructure.has_analysis_results():
            self.logger.info("Infrastructure not analyzed yet - running analysis...")
            self.analyze_complete_infrastructure()
        
        analysis = self.infrastructure.get_analysis_results()
        
        # Restaurant threshold recommendations
        restaurant_stats = analysis['restaurant_spatial_analysis']['distance_statistics']
        restaurant_recommendations = {
            'conservative': restaurant_stats['p25'],  # 25th percentile - pairs closest restaurants
            'moderate': restaurant_stats['p50'],      # 50th percentile - pairs about half
            'aggressive': restaurant_stats['p75']     # 75th percentile - pairs most restaurants
        }
        
        # Customer threshold recommendations  
        customer_stats = analysis['customer_distance_analysis']['distance_statistics']
        customer_recommendations = {
            'conservative': customer_stats['p25'],   # 25th percentile - nearby customers only
            'moderate': customer_stats['p50'],       # 50th percentile - moderate distance customers
            'aggressive': customer_stats['p75']      # 75th percentile - wider customer range
        }
        
        # Scoring system characteristics
        scoring_characteristics = {
            'typical_distance': analysis['typical_distance'],
            'distance_efficiency_range': f"0km (perfect) to {analysis['typical_distance'] * 2:.1f}km (unacceptable)",
            'geographic_context': f"{analysis['area_size']}x{analysis['area_size']}km area"
        }
        
        self.logger.info("Parameter design report generated based on infrastructure analysis")
        
        return {
            'infrastructure_summary': {
                'area_size': analysis['area_size'],
                'restaurant_count': analysis['restaurant_count'],
                'typical_distance': analysis['typical_distance']
            },
            'pairing_parameter_recommendations': {
                'restaurant_threshold_options': restaurant_recommendations,
                'customer_threshold_options': customer_recommendations,
                'pairing_feasibility_note': "Check analysis['restaurant_spatial_analysis']['pairing_feasibility_curve'] for detailed threshold impacts"
            },
            'scoring_system_characteristics': scoring_characteristics
        }


def analyze_infrastructure_for_experiment(infrastructure, scoring_config=None):
    """
    Convenience function for comprehensive infrastructure analysis.
    
    Creates InfrastructureAnalyzer and performs complete analysis in one call.
    Results are automatically cached in the Infrastructure instance.
    
    Args:
        infrastructure: Infrastructure instance to analyze
        scoring_config: Optional ScoringConfig for analysis parameters
        
    Returns:
        dict: Complete analysis results
    """
    analyzer = InfrastructureAnalyzer(infrastructure)
    return analyzer.analyze_complete_infrastructure(scoring_config)