# dev_examples/infrastructure_independence_demo.py
"""
Infrastructure Independence Interactive Demo

Demonstrates the new Infrastructure-first workflow and analysis capabilities.
Shows how Infrastructure can be created once and reused across multiple configurations.

IMPORTANT: If you modify any files in delivery_sim/infrastructure/, you MUST restart 
and re-run the import cell to pick up the changes.
"""

# %% Import and Setup
"""
Cell 1: Basic setup and imports

IMPORTANT: If you modify any files in delivery_sim/infrastructure/, you MUST restart 
and re-run this cell to pick up the changes. Python caches imported modules.
"""
# Add project root to Python path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from delivery_sim.simulation.configuration import LoggingConfig, StructuralConfig, ScoringConfig
from delivery_sim.infrastructure.infrastructure import Infrastructure
from delivery_sim.infrastructure.infrastructure_analyzer import InfrastructureAnalyzer
from delivery_sim.utils.logging_system import configure_logging
import matplotlib.pyplot as plt

print("✓ Infrastructure module imports successful")

# %% Logging Configuration
"""
Cell 2: Configure logging for clean interactive output
"""
logging_config = LoggingConfig(
    console_level="INFO",
    component_levels={
        # Step 1: Broad suppression (sets the default)
        "services": "ERROR",
        "entities": "ERROR", 
        "repositories": "ERROR",
        "utils": "ERROR",
        "system_data": "ERROR",
        
        # Step 2: Surgical enablement (overrides the default)
        "simulation.runner": "INFO",
        "infrastructure": "INFO",  # Show infrastructure activities
    }
)
configure_logging(logging_config)

print("="*60)
print("INFRASTRUCTURE INDEPENDENCE DEMONSTRATION")
print("="*60)
print("✓ Logging configuration applied")

# %% Infrastructure Creation
"""
Cell 3: Create Infrastructure as First-Class Entity

This demonstrates the new Infrastructure-first approach where infrastructure
becomes an independent, reusable component separate from operational simulation.
"""
print("\n📋 Phase 1: Creating Infrastructure...")

# Define structural configuration (physical environment)
structural_config = StructuralConfig(
    delivery_area_size=10,  # 10x10 km area - modify to test different geographies
    num_restaurants=5,     # 15 restaurants - try 10, 15, 20
    driver_speed=0.5        # 0.5 km/min - doesn't affect spatial patterns
)

master_seed = 42  # Change to see different infrastructure layouts

# Create Infrastructure as first-class entity
infrastructure = Infrastructure(structural_config, master_seed)

print(f"✓ Infrastructure created: {infrastructure}")
print(f"  • Area: {structural_config.delivery_area_size}x{structural_config.delivery_area_size} km")
print(f"  • Restaurants: {structural_config.num_restaurants}")
print(f"  • Deterministic seed: {master_seed}")

# Quick validation
basic_chars = infrastructure.get_basic_characteristics()
print(f"  • Restaurant density: {basic_chars['restaurant_density']:.3f} restaurants/km²")
print(f"  • Average spacing: {basic_chars['average_restaurant_spacing']:.3f} km")

# %% Infrastructure Analysis Setup
"""
Cell 4: Configure Analysis Parameters

Optional scoring config for analysis parameters. Higher sample sizes give
more accurate typical distance calculations but take longer to compute.
"""
print("\n📊 Phase 2: Setting Up Infrastructure Analysis...")

# Optional scoring config for analysis parameters
scoring_config = ScoringConfig(
    typical_distance_samples=1000  # More samples for better accuracy - try 1000, 2000, 5000
)

print(f"✓ Analysis configuration:")
print(f"  • Monte Carlo samples: {scoring_config.typical_distance_samples}")
print(f"  • This will analyze restaurant patterns, customer patterns, and typical distances")

# %% Comprehensive Infrastructure Analysis
"""
Cell 5: Perform Comprehensive Infrastructure Analysis

This is the expensive computation that gets cached in the Infrastructure instance
for reuse across multiple experimental configurations.
"""
print("\n🔬 Phase 3: Running Comprehensive Analysis...")

# Create analyzer and perform comprehensive analysis
analyzer = InfrastructureAnalyzer(infrastructure)
analysis_results = analyzer.analyze_complete_infrastructure(scoring_config)

print(f"✓ Analysis complete - results cached in infrastructure instance")
print(f"  • Typical distance: {analysis_results['typical_distance']:.3f} km")
print(f"  • Restaurant density: {analysis_results['restaurant_density']:.3f} restaurants/km²")
print(f"  • Average spacing: {analysis_results['average_restaurant_spacing']:.3f} km")
print(f"  • Analysis cached for reuse: {infrastructure.has_analysis_results()}")

# %% Spatial Pattern Analysis
"""
Cell 6: Examine Spatial Patterns for Parameter Design

This analysis provides the foundation for informed parameter selection
in experimental designs, particularly for pairing thresholds.
"""
print("\n🗺️ Phase 4: Spatial Pattern Analysis for Parameter Design...")

# Restaurant spatial patterns (for restaurants_proximity_threshold design)
restaurant_analysis = analysis_results['restaurant_spatial_analysis']
if 'distance_statistics' in restaurant_analysis:
    rest_stats = restaurant_analysis['distance_statistics']
    print(f"📍 Restaurant Distance Patterns:")
    print(f"    • Min distance: {rest_stats['min']:.2f} km")
    print(f"    • 25th percentile: {rest_stats['p25']:.2f} km")
    print(f"    • Median: {rest_stats['p50']:.2f} km")
    print(f"    • 75th percentile: {rest_stats['p75']:.2f} km")
    print(f"    • Max distance: {rest_stats['max']:.2f} km")
    print(f"    • Total pairs: {rest_stats['count']}")

# Customer distance patterns (for customers_proximity_threshold design)
customer_analysis = analysis_results['customer_distance_analysis']
if 'distance_statistics' in customer_analysis:
    cust_stats = customer_analysis['distance_statistics']
    print(f"\n👥 Customer Distance Patterns:")
    print(f"    • 25th percentile: {cust_stats['p25']:.2f} km")
    print(f"    • Median distance: {cust_stats['p50']:.2f} km")
    print(f"    • 75th percentile: {cust_stats['p75']:.2f} km")
    print(f"    • 90th percentile: {cust_stats['p90']:.2f} km")

# %% Parameter Design Recommendations
"""
Cell 7: Generate Parameter Design Guidance

This shows how infrastructure analysis directly informs experimental parameter selection,
replacing guesswork with data-driven parameter design.
"""
print("\n🎯 Phase 5: Parameter Design Recommendations...")

param_report = analyzer.generate_parameter_design_report()

print("📋 Pairing Parameter Recommendations:")
rest_recs = param_report['pairing_parameter_recommendations']['restaurant_threshold_options']
print(f"  Restaurant Proximity Thresholds:")
print(f"    • Conservative (25th percentile): {rest_recs['conservative']:.2f} km")
print(f"    • Moderate (50th percentile): {rest_recs['moderate']:.2f} km") 
print(f"    • Aggressive (75th percentile): {rest_recs['aggressive']:.2f} km")

cust_recs = param_report['pairing_parameter_recommendations']['customer_threshold_options']
print(f"  Customer Proximity Thresholds:")
print(f"    • Conservative (25th percentile): {cust_recs['conservative']:.2f} km")
print(f"    • Moderate (50th percentile): {cust_recs['moderate']:.2f} km")
print(f"    • Aggressive (75th percentile): {cust_recs['aggressive']:.2f} km")

# Scoring system characteristics
scoring_chars = param_report['scoring_system_characteristics']
print(f"\n⚖️ Priority Scoring System Characteristics:")
print(f"    • Typical distance: {scoring_chars['typical_distance']:.3f} km")
print(f"    • Distance efficiency range: {scoring_chars['distance_efficiency_range']}")
print(f"    • Geographic context: {scoring_chars['geographic_context']}")

# %% Infrastructure Visualization
"""
Cell 8: Create Infrastructure Visualization

Visual representation of the infrastructure layout with closest restaurant pairs
highlighted to aid in parameter selection and experimental design validation.
"""
print("\n📈 Phase 6: Infrastructure Visualization...")

# Create visualization showing restaurant layout and closest pairs
fig = analyzer.visualize_infrastructure(show_closest_pairs=3, figsize=(12, 10))
plt.show()

print("✓ Infrastructure visualization displayed")
print("  • Red squares: Restaurant locations with IDs")
print("  • Colored dashed lines: Closest restaurant pairs (helpful for pairing threshold selection)")
print("  • Yellow info box: Basic infrastructure characteristics")

# %% Reusability Demonstration
"""
Cell 9: Demonstrate Infrastructure Reusability

This shows how the same infrastructure can be reused across multiple
experimental configurations, achieving O(1) instead of O(m) setup time.
"""
print("\n♻️ Phase 7: Infrastructure Reusability Demonstration...")

print("✅ Infrastructure is now ready for experimental design reuse:")
print(f"  • Infrastructure signature: {infrastructure.get_infrastructure_signature()}")
print(f"  • Analysis results cached: {infrastructure.has_analysis_results()}")
print(f"  • Typical distance available: {infrastructure.get_analysis_results()['typical_distance']:.3f} km")
print(f"  • Spatial patterns analyzed: ✓")
print(f"  • Parameter recommendations available: ✓")

# Show how different experimental configurations can reuse the same infrastructure
print(f"\n🧪 Example experimental configurations that could reuse this infrastructure:")
example_configs = [
    "Low demand: order_interval=2.0, driver_interval=3.0",
    "High demand: order_interval=0.5, driver_interval=3.0", 
    "Pairing enabled vs disabled comparison",
    "Assignment threshold study: thresholds=[50, 70, 90]",
    "Periodic interval study: intervals=[2.0, 3.0, 5.0]",
    "Weight sensitivity: distance vs throughput vs fairness"
]

for i, config in enumerate(example_configs, 1):
    print(f"  {i}. {config}")

print(f"\n🎯 All configurations would share:")
print(f"  • Same {basic_chars['restaurant_count']} restaurants at same locations")
print(f"  • Same typical_distance ({analysis_results['typical_distance']:.3f} km) for scoring normalization")
print(f"  • Same spatial patterns for parameter validation")
print(f"  • Zero redundant infrastructure setup time")
print(f"  • Consistent geographical context for fair comparison")

# %% Summary and Next Steps
"""
Cell 10: Summary and Next Steps

Summary of achievements and preparation for next architectural phases.
"""
print("\n" + "="*60)
print("INFRASTRUCTURE INDEPENDENCE DEMONSTRATION COMPLETE")
print("="*60)

achievements = [
    "✓ Infrastructure created as first-class entity",
    "✓ Comprehensive analysis performed and cached",
    "✓ Parameter design guidance generated",
    "✓ Infrastructure visualization created",
    "✓ Reusability across configurations demonstrated"
]

for achievement in achievements:
    print(achievement)

print(f"\n📊 Key Results Summary:")
print(f"  • Infrastructure: {infrastructure}")
print(f"  • Typical distance: {analysis_results['typical_distance']:.3f} km")
print(f"  • Recommended restaurant threshold (moderate): {rest_recs['moderate']:.2f} km")
print(f"  • Recommended customer threshold (moderate): {cust_recs['moderate']:.2f} km")

print(f"\n🚀 Ready for Next Architectural Phases:")
print(f"  1. Runner Redesign: Modify SimulationRunner to accept Infrastructure input")
print(f"  2. ExperimentalRunner: Create multi-configuration experimental capability") 
print(f"  3. Integration: Connect with existing analysis pipeline")
print(f"  4. Testing: Validate efficiency gains and analytical benefits")

print(f"\n💡 Architecture Benefits Achieved:")
print(f"  • Computational Efficiency: Infrastructure setup O(1) vs O(m)")
print(f"  • Analytical Rigor: Data-driven parameter design")
print(f"  • Architectural Clarity: Clean Infrastructure/Operations separation")
print(f"  • Research Capability: Foundation for systematic experimental design")

print("\nInfrastructure Independence implementation complete! 🎉")