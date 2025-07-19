# complete_supply_demand_study.py
"""
Complete Supply-Demand Study: 9 Design Points Validation

Research Hypothesis: Operational regimes are determined by load ratio 
(driver_interval/order_interval), not absolute arrival rates.

This script tests all 9 combinations of:
- Order intervals: [1.0, 2.0, 3.0] minutes  
- Driver intervals: [5.0, 8.0, 10.0] minutes
- Load ratios: [1.67 to 10.0] spanning all operational regimes
"""

# %% Step 1: Setup and Imports
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from delivery_sim.simulation.configuration import (
    StructuralConfig, OperationalConfig, ExperimentConfig, 
    LoggingConfig, ScoringConfig
)
from delivery_sim.infrastructure.infrastructure import Infrastructure
from delivery_sim.infrastructure.infrastructure_analyzer import InfrastructureAnalyzer
from delivery_sim.experimental.design_point import DesignPoint
from delivery_sim.experimental.experimental_runner import ExperimentalRunner
from delivery_sim.utils.logging_system import configure_logging

print("="*80)
print("COMPLETE SUPPLY-DEMAND STUDY: 9 DESIGN POINTS")
print("="*80)
print("Hypothesis: Load ratio determines operational regime, not absolute rates")

# %% Step 2: Logging Configuration
logging_config = LoggingConfig(
    console_level="INFO",
    component_levels={
        "services": "ERROR", "entities": "ERROR", "repositories": "ERROR",
        "utils": "ERROR", "system_data": "ERROR",
        "simulation.runner": "INFO", "infrastructure": "INFO", 
        "experimental.runner": "INFO",
    }
)
configure_logging(logging_config)
print("✓ Clean logging configured")

# %% Step 3: Infrastructure Setup (Reusable)
print("\n" + "="*50)
print("INFRASTRUCTURE SETUP")
print("="*50)

structural_config = StructuralConfig(
    delivery_area_size=10,
    num_restaurants=10,
    driver_speed=0.5
)

master_seed = 42
infrastructure = Infrastructure(structural_config, master_seed)
analyzer = InfrastructureAnalyzer(infrastructure)
analysis_results = analyzer.analyze_complete_infrastructure()

print(f"✓ Infrastructure: {infrastructure}")
print(f"✓ Typical distance: {analysis_results['typical_distance']:.3f}km")

# %% Step 4: Complete Design Points Matrix (3×3 = 9 points)
print("\n" + "="*50)
print("DESIGN POINTS MATRIX CREATION")
print("="*50)

scoring_config = ScoringConfig()

# Base operational parameters (consistent across all design points)
base_params = {
    'pairing_enabled': False,
    'restaurants_proximity_threshold': None,
    'customers_proximity_threshold': None,
    'immediate_assignment_threshold': 100,  # Always immediate assignment
    'mean_service_duration': 100,
    'service_duration_std_dev': 60,
    'min_service_duration': 30,
    'max_service_duration': 200,
    'periodic_interval': 3.0
}

# Create complete 3×3 matrix
design_points = {}

# Order intervals and driver intervals to test
order_intervals = [1.0, 2.0, 3.0]  # High, Medium, Low demand
driver_intervals = [5.0, 8.0, 10.0]  # High, Medium, Low supply

print("Creating 3×3 design points matrix:")
print("Order Intervals: [1.0, 2.0, 3.0] minutes")
print("Driver Intervals: [5.0, 8.0, 10.0] minutes")

# Generate all combinations systematically
for order_interval in order_intervals:
    for driver_interval in driver_intervals:
        # Create descriptive name
        demand_level = "high" if order_interval == 1.0 else "medium" if order_interval == 2.0 else "low"
        supply_level = "high" if driver_interval == 5.0 else "medium" if driver_interval == 8.0 else "low"
        
        design_name = f"{demand_level}_demand_{supply_level}_supply"
        load_ratio = driver_interval / order_interval
        
        design_points[design_name] = DesignPoint(
            infrastructure=infrastructure,
            operational_config=OperationalConfig(
                mean_order_inter_arrival_time=order_interval,
                mean_driver_inter_arrival_time=driver_interval,
                **base_params
            ),
            scoring_config=scoring_config,
            name=design_name
        )
        
        print(f"  ✓ {design_name}: Load Ratio = {load_ratio:.2f}")

print(f"\n✓ Created {len(design_points)} design points")

# Display load ratio distribution for hypothesis validation
print(f"\n📊 Load Ratio Distribution (for regime validation):")
load_ratios = []
for name, dp in design_points.items():
    ratio = dp.operational_config.mean_driver_inter_arrival_time / dp.operational_config.mean_order_inter_arrival_time
    load_ratios.append(ratio)
    print(f"  {name}: {ratio:.2f}")

print(f"Load ratio range: {min(load_ratios):.2f} - {max(load_ratios):.2f} (wide range for regime discovery)")

# %% Step 5: Experiment Configuration
print("\n" + "="*50)
print("EXPERIMENT CONFIGURATION")
print("="*50)

experiment_config = ExperimentConfig(
    simulation_duration=1000,  # Sufficient duration for regime observation
    num_replications=3,        # Multiple replications for statistical validity
    master_seed=42
)

print(f"✓ Duration: {experiment_config.simulation_duration} minutes")
print(f"✓ Replications: {experiment_config.num_replications}")
print(f"✓ Total simulation runs: {len(design_points)} × {experiment_config.num_replications} = {len(design_points) * experiment_config.num_replications}")

# %% Step 6: Execute Complete Experimental Study
print("\n" + "="*50)
print("EXPERIMENTAL EXECUTION")
print("="*50)

runner = ExperimentalRunner()
print("✓ ExperimentalRunner initialized")

print(f"\nExecuting complete study with {len(design_points)} design points...")
study_results = runner.run_experimental_study(design_points, experiment_config)

print(f"\n✅ EXPERIMENTAL STUDY COMPLETE!")
print(f"✓ Design points executed: {len(study_results)}")
print(f"✓ Ready for time series analysis and regime validation")

# %% Step 7: Time Series Data Extraction for Regime Analysis
print("\n" + "="*50)
print("TIME SERIES ANALYSIS FOR REGIME VALIDATION")
print("="*50)

from delivery_sim.warmup_analysis.time_series_preprocessing import TimeSeriesPreprocessor
from delivery_sim.warmup_analysis.visualization import WelchMethodVisualization
import matplotlib.pyplot as plt

print("Extracting time series data for all design points...")

preprocessor = TimeSeriesPreprocessor()
viz = WelchMethodVisualization(figsize=(14, 8))

all_time_series_data = {}

for design_name, design_results in study_results.items():
    print(f"  Processing {design_name}...")
    
    # Extract system snapshots from replications
    replication_snapshots = []
    for replication_result in design_results['replication_results']:
        snapshots = replication_result['system_snapshots']
        if snapshots:
            replication_snapshots.append(snapshots)
    
    if len(replication_snapshots) < 2:
        print(f"    ⚠️  Warning: Only {len(replication_snapshots)} replications")
        continue
    
    # Extract time series with Welch's method
    basic_data = preprocessor.extract_cross_replication_averages(
        multi_replication_snapshots=replication_snapshots,
        metrics=['active_drivers', 'active_delivery_entities'],
        collection_interval=0.5,
        moving_average_window=50
    )
    
    # Add Little's Law validation
    single_design_dict = {design_name: design_points[design_name]}
    enhanced_data = preprocessor.add_little_law_theoretical_values(
        time_series_data={design_name: basic_data},
        design_points_dict=single_design_dict
    )
    
    all_time_series_data[design_name] = enhanced_data[design_name]
    
    # Show load ratio and Little's Law prediction
    load_ratio = design_points[design_name].operational_config.mean_driver_inter_arrival_time / design_points[design_name].operational_config.mean_order_inter_arrival_time
    if 'active_drivers' in enhanced_data[design_name] and 'theoretical_value' in enhanced_data[design_name]['active_drivers']:
        theoretical_value = enhanced_data[design_name]['active_drivers']['theoretical_value']
        print(f"    ✓ Load Ratio: {load_ratio:.2f}, Little's Law: {theoretical_value:.1f} drivers")

print(f"\n✓ Time series extraction complete for {len(all_time_series_data)} design points")

# %% Step 8: Systematic Time Series Visualization (Load Ratio Ordered)
print("\n" + "="*50)
print("REGIME VALIDATION: TIME SERIES PLOTS BY LOAD RATIO")
print("="*50)

print("Creating time series plots ordered by load ratio...")
print("Hypothesis test: Similar load ratios should show similar regime patterns")

# Sort design points by load ratio for systematic analysis
design_load_ratios = []
for design_name in all_time_series_data.keys():
    load_ratio = design_points[design_name].operational_config.mean_driver_inter_arrival_time / design_points[design_name].operational_config.mean_order_inter_arrival_time
    design_load_ratios.append((load_ratio, design_name))

design_load_ratios.sort()  # Sort by load ratio

print(f"\nPlot sequence (by increasing load ratio):")
for ratio, name in design_load_ratios:
    print(f"  {ratio:.2f}: {name}")

# Create plots in load ratio order
plot_count = 0
for load_ratio, design_name in design_load_ratios:
    plot_count += 1
    enhanced_data = all_time_series_data[design_name]
    
    print(f"\n--- Plot {plot_count}/{len(design_load_ratios)}: {design_name} (Load Ratio: {load_ratio:.2f}) ---")
    
    # Get simulation info
    if 'active_drivers' in enhanced_data:
        first_metric_data = enhanced_data['active_drivers']
        total_duration = max(first_metric_data['time_points'])
        replication_count = first_metric_data['replication_count']
        
        # Little's Law validation
        if 'theoretical_value' in first_metric_data:
            theoretical_value = first_metric_data['theoretical_value']
            moving_averages = first_metric_data['moving_averages']
            final_ma_value = moving_averages[-1] if moving_averages else 0
            convergence_error = abs(final_ma_value - theoretical_value) / theoretical_value * 100
            print(f"  • Little's Law: {theoretical_value:.1f} vs Final: {final_ma_value:.1f} (Error: {convergence_error:.1f}%)")
    
    # Create clean plot without regime assumptions
    fig = viz.create_combined_welch_inspection_plot(
        enhanced_data, 
        title=f'Load Ratio {load_ratio:.2f}: {design_name.replace("_", " ").title()}'
    )
    
    # Add hypothesis testing annotation
    fig.text(0.5, 0.01, 
             f'Hypothesis Test: Observe Active Delivery Entities pattern for Load Ratio {load_ratio:.2f}',
             ha='center', fontsize=11, weight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.show()
    print(f"  ✓ Plot displayed")

print(f"\n🎯 All {plot_count} time series plots displayed!")
print(f"\n📋 HYPOTHESIS VALIDATION GUIDANCE:")
print(f"Look for patterns in Active Delivery Entities (bottom panels):")
print(f"• Do similar load ratios show similar entity count patterns?")
print(f"• Are there natural regime transitions at certain load ratio values?") 
print(f"• Do absolute arrival rates matter, or just the ratio?")
print(f"• Active Drivers (top panels): Should consistently converge to Little's Law predictions")

# %% Step 9: Warmup Period Determination
print("\n" + "="*50)
print("WARMUP PERIOD DETERMINATION")
print("="*50)

print("📌 VISUAL INSPECTION GUIDANCE:")
print("Based on the plots above, determine uniform warmup period:")
print("1. Focus on Active Drivers convergence (top panels)")
print("2. Find latest convergence point across ALL design points")
print("3. Add conservative margin for methodological rigor")

# Set your warmup period based on visual inspection
proposed_warmup_period = 300  # UPDATE THIS based on visual inspection

print(f"\n⚙️  Proposed warmup period: {proposed_warmup_period} minutes")

# Quick validation
validation_summary = []
for design_name in all_time_series_data.keys():
    first_metric = list(all_time_series_data[design_name].keys())[0]
    max_time = max(all_time_series_data[design_name][first_metric]['time_points'])
    analysis_window = max_time - proposed_warmup_period
    warmup_ratio = proposed_warmup_period / max_time
    
    validation_summary.append({
        'design_name': design_name,
        'warmup_ratio': warmup_ratio,
        'analysis_window': analysis_window
    })

avg_warmup_ratio = sum(v['warmup_ratio'] for v in validation_summary) / len(validation_summary)
min_analysis_window = min(v['analysis_window'] for v in validation_summary)

print(f"✓ Average warmup ratio: {avg_warmup_ratio*100:.1f}%")
print(f"✓ Minimum analysis window: {min_analysis_window:.1f} minutes")

if avg_warmup_ratio <= 0.4 and min_analysis_window >= 30:
    print("✅ Warmup period appears appropriate")
else:
    print("⚠️  Consider adjusting warmup period")

uniform_warmup_period = proposed_warmup_period

# %% Step 10: Performance Metrics Analysis
print("\n" + "="*50)
print("PERFORMANCE METRICS CALCULATION")
print("="*50)

from delivery_sim.analysis_pipeline.pipeline_coordinator import analyze_single_configuration

print(f"Calculating performance metrics for all {len(study_results)} design points...")
print(f"Using uniform warmup period: {uniform_warmup_period} minutes")

metrics_results = {}

for design_name, design_results in study_results.items():
    print(f"  Processing {design_name}...")
    
    try:
        analysis_result = analyze_single_configuration(
            simulation_results=design_results,
            warmup_period=uniform_warmup_period,
            confidence_level=0.95
        )
        
        metrics_results[design_name] = {
            'analysis': analysis_result,
            'status': 'success'
        }
        print(f"    ✓ Success")
        
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")
        metrics_results[design_name] = {
            'analysis': None,
            'status': 'error',
            'error': str(e)
        }

print(f"\n✓ Metrics calculation complete")

# %% Step 11: Evidence Table for Hypothesis Validation
print("\n" + "="*50)
print("HYPOTHESIS VALIDATION: EVIDENCE TABLE")
print("="*50)

import pandas as pd

print("Creating evidence table ordered by load ratio...")

# Extract performance metrics
table_data = []

for design_name, result in metrics_results.items():
    if result['status'] != 'success':
        continue
    
    analysis = result['analysis']
    
    try:
        # Extract assignment time
        entity_metrics = analysis.get('entity_metrics', {})
        orders_metrics = entity_metrics.get('orders', {})
        assignment_time_data = orders_metrics.get('assignment_time', {})
        
        assignment_time_mean = None
        if assignment_time_data and 'mean' in assignment_time_data:
            assignment_time_mean = assignment_time_data['mean'].get('point_estimate')
            assignment_time_ci = assignment_time_data['mean'].get('confidence_interval', [None, None])
            if assignment_time_mean and assignment_time_ci[0] is not None:
                ci_width = (assignment_time_ci[1] - assignment_time_ci[0]) / 2
                assignment_formatted = f"{assignment_time_mean:.1f}±{ci_width:.1f}"
            else:
                assignment_formatted = f"{assignment_time_mean:.1f}" if assignment_time_mean else "N/A"
        else:
            assignment_formatted = "N/A"
        
        # Extract completion rate
        system_metrics = analysis.get('system_metrics', {})
        completion_rate_data = system_metrics.get('system_completion_rate', {})
        completion_rate = completion_rate_data.get('point_estimate') if completion_rate_data else None
        completion_formatted = f"{completion_rate:.1%}" if completion_rate else "N/A"
        
        # Get load ratio
        design_point = design_points[design_name]
        load_ratio = (design_point.operational_config.mean_driver_inter_arrival_time / 
                     design_point.operational_config.mean_order_inter_arrival_time)
        
        table_data.append({
            'Design Point': design_name.replace('_', ' ').title(),
            'Load Ratio': f"{load_ratio:.2f}",
            'Assignment Time (min)': assignment_formatted,
            'Completion Rate': completion_formatted,
            'Load Ratio Value': load_ratio,  # For sorting
            'Assignment Value': assignment_time_mean if assignment_time_mean else 999,
            'Completion Value': completion_rate if completion_rate else 0
        })
        
    except Exception as e:
        print(f"  ⚠️  Error extracting metrics for {design_name}: {str(e)}")

# Create and display table
if table_data:
    df = pd.DataFrame(table_data)
    df_display = df.sort_values('Load Ratio Value')[['Design Point', 'Load Ratio', 'Assignment Time (min)', 'Completion Rate']]
    
    print("\n🎯 PERFORMANCE METRICS BY LOAD RATIO")
    print("="*80)
    print(df_display.to_string(index=False))
    
    print(f"\n📊 HYPOTHESIS VALIDATION ANALYSIS:")
    print(f"Examine if similar load ratios produce similar performance regardless of absolute values:")
    
    print(f"\n🔬 Raw Data by Load Ratio (NO PRECONCEIVED BOUNDARIES):")
    print("Observe patterns and determine boundaries empirically:")
    
    for _, row in df_display.iterrows():
        print(f"  Load Ratio {row['Load Ratio']}: {row['Design Point']} → Assignment: {row['Assignment Time (min)']}, Completion: {row['Completion Rate']}")
    
    print(f"\n📋 BOUNDARY DETERMINATION GUIDANCE:")
    print(f"Look for natural breaks in performance metrics:")
    print(f"• Where do completion rates drop significantly?")
    print(f"• Where do assignment times increase dramatically?") 
    print(f"• Do similar load ratios cluster in performance?")
    print(f"• What load ratio thresholds emerge from the data?")
    
    print(f"\n✅ EMPIRICAL EVIDENCE PRESENTED!")
    print(f"Review both time series patterns AND performance metrics to validate:")
    print(f"'Load ratio determines operational regime, not absolute arrival rates'")

else:
    print("⚠️  No valid data available for evidence table")

print(f"\n📚 COMPLETE STUDY FINISHED!")
print(f"✓ Time series plots: Visual regime patterns by load ratio")
print(f"✓ Performance table: Quantitative validation of hypothesis")
print(f"✓ Ready for thesis analysis and conclusions")
# %%