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
print("FOCUSED TEMPORAL DYNAMICS STUDY: 2 DESIGN POINTS")
print("="*80)
print("Research Focus: Temporal stability with identical load ratios")

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

# %% Step 4: Focused Design Points for Temporal Dynamics Investigation
print("\n" + "="*50)
print("FOCUSED INVESTIGATION: IDENTICAL LOAD RATIO, DIFFERENT DYNAMICS")
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

# Create focused design points for temporal analysis
design_points = {}

print("Creating 2 design points with identical Load Ratio 5.00:")
print("Focus: Understanding temporal dynamics with same load ratio")

# High Demand High Supply (1.0 order interval, 5.0 driver interval)
design_points["high_demand_high_supply"] = DesignPoint(
    infrastructure=infrastructure,
    operational_config=OperationalConfig(
        mean_order_inter_arrival_time=1.0,
        mean_driver_inter_arrival_time=5.0,
        **base_params
    ),
    scoring_config=scoring_config,
    name="high_demand_high_supply"
)

# Medium Demand Low Supply (2.0 order interval, 10.0 driver interval)  
design_points["medium_demand_low_supply"] = DesignPoint(
    infrastructure=infrastructure,
    operational_config=OperationalConfig(
        mean_order_inter_arrival_time=2.0,
        mean_driver_inter_arrival_time=10.0,
        **base_params
    ),
    scoring_config=scoring_config,
    name="medium_demand_low_supply"
)

print(f"\n✓ Created {len(design_points)} design points for temporal investigation")

# Display detailed comparison
print(f"\n📊 Design Points Comparison (Both Load Ratio 5.00):")
for name, dp in design_points.items():
    order_interval = dp.operational_config.mean_order_inter_arrival_time
    driver_interval = dp.operational_config.mean_driver_inter_arrival_time
    load_ratio = driver_interval / order_interval
    order_rate = 1.0 / order_interval
    driver_rate = 1.0 / driver_interval
    
    print(f"  {name}:")
    print(f"    • Order interval: {order_interval:.1f} min (rate: {order_rate:.2f}/min)")
    print(f"    • Driver interval: {driver_interval:.1f} min (rate: {driver_rate:.2f}/min)")
    print(f"    • Load ratio: {load_ratio:.2f}")
    print(f"    • Absolute demand intensity: {order_rate:.2f} orders/min")

print(f"\n🎯 Research Focus:")
print(f"  • Same load ratio, different absolute arrival rates")
print(f"  • Extended simulation duration for temporal pattern analysis")
print(f"  • Increased replications for statistical robustness")
print(f"  • Hypothesis: Temporal stability differs despite identical load ratios")

# %% Step 5: Extended Experiment Configuration for Temporal Analysis
print("\n" + "="*50)
print("EXTENDED EXPERIMENT CONFIGURATION")
print("="*50)

experiment_config = ExperimentConfig(
    simulation_duration=2000,  # Extended duration for temporal pattern analysis
    num_replications=5,        # Increased replications for statistical robustness
    master_seed=42
)

print(f"✓ Extended duration: {experiment_config.simulation_duration} minutes")
print(f"✓ Increased replications: {experiment_config.num_replications}")
print(f"✓ Total simulation runs: {len(design_points)} × {experiment_config.num_replications} = {len(design_points) * experiment_config.num_replications}")

print(f"\n📊 Extended Analysis Benefits:")
print(f"  • 2x longer duration reveals long-term temporal patterns")
print(f"  • 5 replications improve confidence intervals")
print(f"  • Better capture of regime switching dynamics")
print(f"  • More robust warmup period determination")

# %% Step 6: Execute Focused Temporal Dynamics Study
print("\n" + "="*50)
print("TEMPORAL DYNAMICS EXPERIMENTAL EXECUTION")
print("="*50)

runner = ExperimentalRunner()
print("✓ ExperimentalRunner initialized")

print(f"\nExecuting focused study with {len(design_points)} design points...")
print("Focus: Understanding temporal patterns with identical load ratios")
study_results = runner.run_experimental_study(design_points, experiment_config)

print(f"\n✅ TEMPORAL DYNAMICS STUDY COMPLETE!")
print(f"✓ Design points executed: {len(study_results)}")
print(f"✓ Ready for extended time series analysis and temporal pattern investigation")

# %% Step 7: Extended Time Series Analysis for Temporal Dynamics
print("\n" + "="*50)
print("EXTENDED TIME SERIES ANALYSIS FOR TEMPORAL DYNAMICS")
print("="*50)

from delivery_sim.warmup_analysis.time_series_preprocessing import TimeSeriesPreprocessor
from delivery_sim.warmup_analysis.visualization import WelchMethodVisualization
import matplotlib.pyplot as plt

print("Extracting extended time series data for temporal dynamics analysis...")

preprocessor = TimeSeriesPreprocessor()
viz = WelchMethodVisualization(figsize=(16, 10))  # Larger plots for extended data

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
    
    # Extract time series with Welch's method (larger window for longer data)
    basic_data = preprocessor.extract_cross_replication_averages(
        multi_replication_snapshots=replication_snapshots,
        metrics=['active_drivers', 'active_delivery_entities'],
        collection_interval=0.5,
        moving_average_window=100  # Larger window for 2000-minute simulation
    )
    
    # Add Little's Law validation
    single_design_dict = {design_name: design_points[design_name]}
    enhanced_data = preprocessor.add_little_law_theoretical_values(
        time_series_data={design_name: basic_data},
        design_points_dict=single_design_dict
    )
    
    all_time_series_data[design_name] = enhanced_data[design_name]
    
    # Show detailed analysis info
    load_ratio = design_points[design_name].operational_config.mean_driver_inter_arrival_time / design_points[design_name].operational_config.mean_order_inter_arrival_time
    order_rate = 1.0 / design_points[design_name].operational_config.mean_order_inter_arrival_time
    
    if 'active_drivers' in enhanced_data[design_name] and 'theoretical_value' in enhanced_data[design_name]['active_drivers']:
        theoretical_value = enhanced_data[design_name]['active_drivers']['theoretical_value']
        print(f"    ✓ Load Ratio: {load_ratio:.2f}, Order Rate: {order_rate:.2f}/min, Little's Law: {theoretical_value:.1f} drivers")

print(f"\n✓ Extended time series extraction complete for {len(all_time_series_data)} design points")
print(f"✓ Extended duration provides {experiment_config.simulation_duration} minutes of temporal data")

# %% Step 8: Temporal Dynamics Visualization (Extended Duration)
print("\n" + "="*50)
print("TEMPORAL DYNAMICS: EXTENDED TIME SERIES PLOTS")
print("="*50)

print("Creating extended time series plots for temporal dynamics investigation...")
print("Focus: Understanding regime switching and temporal stability patterns")

# Sort design points by absolute order rate for systematic analysis
design_order_rates = []
for design_name in all_time_series_data.keys():
    order_rate = 1.0 / design_points[design_name].operational_config.mean_order_inter_arrival_time
    load_ratio = design_points[design_name].operational_config.mean_driver_inter_arrival_time / design_points[design_name].operational_config.mean_order_inter_arrival_time
    design_order_rates.append((order_rate, load_ratio, design_name))

design_order_rates.sort()  # Sort by order rate

print(f"\nPlot sequence (by increasing order rate, both Load Ratio 5.00):")
for order_rate, load_ratio, name in design_order_rates:
    print(f"  {order_rate:.2f} orders/min (Load Ratio {load_ratio:.2f}): {name}")

# Create extended plots for temporal analysis
plot_count = 0
for order_rate, load_ratio, design_name in design_order_rates:
    plot_count += 1
    enhanced_data = all_time_series_data[design_name]
    
    print(f"\n--- Plot {plot_count}/{len(design_order_rates)}: {design_name} ---")
    print(f"    Order Rate: {order_rate:.2f}/min, Load Ratio: {load_ratio:.2f}")
    
    # Get extended simulation info
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
            print(f"    Little's Law: {theoretical_value:.1f} vs Final: {final_ma_value:.1f} (Error: {convergence_error:.1f}%)")
            print(f"    Extended duration: {total_duration:.1f} minutes ({replication_count} replications)")
    
    # Create extended plot for temporal dynamics
    fig = viz.create_combined_welch_inspection_plot(
        enhanced_data, 
        title=f'Temporal Dynamics: {design_name.replace("_", " ").title()} (Order Rate: {order_rate:.2f}/min, Load Ratio: {load_ratio:.2f})'
    )
    
    # Add temporal dynamics annotation
    fig.text(0.5, 0.01, 
             f'Temporal Analysis: Observe regime switching patterns over {experiment_config.simulation_duration} minutes',
             ha='center', fontsize=11, weight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.show()
    print(f"  ✓ Extended temporal plot displayed")

print(f"\n🎯 Extended temporal analysis complete!")
print(f"\n📋 TEMPORAL DYNAMICS INVESTIGATION:")
print(f"Compare the two plots for temporal stability patterns:")
print(f"• Regime consistency: Do patterns remain stable over 2000 minutes?")
print(f"• Switching frequency: How often do regimes change?")
print(f"• Recovery patterns: How does system recover from high-stress periods?")
print(f"• Absolute rate impact: How does order intensity affect temporal stability?")

# %% Step 9: Extended Warmup Period Determination
print("\n" + "="*50)
print("EXTENDED WARMUP PERIOD DETERMINATION")
print("="*50)

print("📌 EXTENDED VISUAL INSPECTION GUIDANCE:")
print("Based on the extended plots above, determine uniform warmup period:")
print("1. Focus on Active Drivers convergence (top panels)")
print("2. Find latest convergence point across BOTH design points")
print("3. Account for extended 2000-minute duration")
print("4. Add conservative margin for methodological rigor")

# Set your warmup period based on visual inspection of extended data
proposed_warmup_period = 600  # UPDATE THIS based on extended visual inspection

print(f"\n⚙️  Proposed warmup period for extended simulation: {proposed_warmup_period} minutes")

# Extended validation for longer simulation
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
print(f"✓ Extended data provides {min_analysis_window:.0f} minutes of post-warmup analysis")

if avg_warmup_ratio <= 0.4 and min_analysis_window >= 60:
    print("✅ Extended warmup period appears appropriate")
else:
    print("⚠️  Consider adjusting warmup period for extended simulation")

uniform_warmup_period = proposed_warmup_period

# %% Step 10: Extended Performance Metrics Analysis
print("\n" + "="*50)
print("EXTENDED PERFORMANCE METRICS CALCULATION")
print("="*50)

from delivery_sim.analysis_pipeline.pipeline_coordinator import analyze_single_configuration

print(f"Calculating extended performance metrics for {len(study_results)} design points...")
print(f"Using uniform warmup period: {uniform_warmup_period} minutes")
print(f"Extended analysis window: {experiment_config.simulation_duration - uniform_warmup_period} minutes")

metrics_results = {}

for design_name, design_results in study_results.items():
    print(f"  Processing {design_name} with extended data...")
    
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
        print(f"    ✓ Extended analysis success")
        
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")
        metrics_results[design_name] = {
            'analysis': None,
            'status': 'error',
            'error': str(e)
        }

print(f"\n✓ Extended metrics calculation complete")

# %% Step 11: Enhanced Evidence Table with Within-Replication Variability
print("\n" + "="*50)
print("ENHANCED HYPOTHESIS VALIDATION: EVIDENCE TABLE")
print("="*50)

import pandas as pd

print("Creating enhanced evidence table with within-replication variability...")

# Extract performance metrics including within-replication std
table_data = []

for design_name, result in metrics_results.items():
    if result['status'] != 'success':
        continue
    
    analysis = result['analysis']
    
    try:
        # Extract assignment time metrics
        entity_metrics = analysis.get('entity_metrics', {})
        orders_metrics = entity_metrics.get('orders', {})
        assignment_time_data = orders_metrics.get('assignment_time', {})
        
        # Mean assignment time
        assignment_time_mean = None
        assignment_time_mean_ci = None
        if assignment_time_data and 'mean' in assignment_time_data:
            assignment_time_mean = assignment_time_data['mean'].get('point_estimate')
            assignment_time_ci = assignment_time_data['mean'].get('confidence_interval', [None, None])
            if assignment_time_mean and assignment_time_ci[0] is not None:
                ci_width = (assignment_time_ci[1] - assignment_time_ci[0]) / 2
                assignment_time_mean_ci = f"{assignment_time_mean:.1f}±{ci_width:.1f}"
            else:
                assignment_time_mean_ci = f"{assignment_time_mean:.1f}" if assignment_time_mean else "N/A"
        
        # Within-replication standard deviation (NEW!)
        assignment_time_std_mean = None
        assignment_time_std_ci = None
        if assignment_time_data and 'std' in assignment_time_data:
            assignment_time_std_mean = assignment_time_data['std'].get('point_estimate')
            assignment_time_std_ci_raw = assignment_time_data['std'].get('confidence_interval', [None, None])
            if assignment_time_std_mean and assignment_time_std_ci_raw[0] is not None:
                std_ci_width = (assignment_time_std_ci_raw[1] - assignment_time_std_ci_raw[0]) / 2
                assignment_time_std_ci = f"{assignment_time_std_mean:.1f}±{std_ci_width:.1f}"
            else:
                assignment_time_std_ci = f"{assignment_time_std_mean:.1f}" if assignment_time_std_mean else "N/A"
        
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
            'Mean Assignment Time': assignment_time_mean_ci,
            'Within-Rep Std': assignment_time_std_ci,  # NEW COLUMN!
            'Completion Rate': completion_formatted,
            'Load Ratio Value': load_ratio,  # For sorting
            'Mean Value': assignment_time_mean if assignment_time_mean else 999,
            'Std Value': assignment_time_std_mean if assignment_time_std_mean else 0,  # For comparison
        })
        
    except Exception as e:
        print(f"  ⚠️  Error extracting metrics for {design_name}: {str(e)}")

# Create and display enhanced table
if table_data:
    df = pd.DataFrame(table_data)
    df_display = df.sort_values('Load Ratio Value')[['Design Point', 'Load Ratio', 'Mean Assignment Time', 'Within-Rep Std', 'Completion Rate']]
    
    print("\n🎯 ENHANCED PERFORMANCE METRICS BY LOAD RATIO")
    print("="*100)
    print(df_display.to_string(index=False))
    
    print(f"\n📊 ROOM FOR ERROR HYPOTHESIS TEST:")
    print(f"Testing: Medium Demand Low Supply should have higher within-replication variability")
    
    # Extract within-replication std values for hypothesis test
    high_demand_high_supply_std = None
    medium_demand_low_supply_std = None
    
    for _, row in df.iterrows():
        if 'High Demand High Supply' in row['Design Point']:
            high_demand_high_supply_std = row['Std Value']
        elif 'Medium Demand Low Supply' in row['Design Point']:
            medium_demand_low_supply_std = row['Std Value']
    
    print(f"\n🔬 Within-Replication Variability Comparison:")
    if high_demand_high_supply_std is not None and medium_demand_low_supply_std is not None:
        print(f"  • High Demand High Supply (20 avg drivers): σ = {high_demand_high_supply_std:.1f} min")
        print(f"  • Medium Demand Low Supply (10 avg drivers): σ = {medium_demand_low_supply_std:.1f} min")
        
        if medium_demand_low_supply_std > high_demand_high_supply_std:
            ratio = medium_demand_low_supply_std / high_demand_high_supply_std
            print(f"  ✅ HYPOTHESIS CONFIRMED: Low Supply has {ratio:.1f}× higher variability!")
            print(f"      → Less buffer capacity = more sensitivity to fluctuations")
        else:
            print(f"  ❌ HYPOTHESIS NOT CONFIRMED: Expected higher variability in Low Supply")
    else:
        print(f"  ⚠️  Cannot compare - missing std data")
    
    print(f"\n💡 Key Insights:")
    print(f"  • 'Mean Assignment Time': Across-replication average (central tendency)")
    print(f"  • 'Within-Rep Std': System variability characterization (your hypothesis target)")
    print(f"  • Higher within-rep std = more 'wild swings' in performance")
    print(f"  • This tests 'room for error' theory independent of statistical precision")

else:
    print("⚠️  No valid data available for enhanced evidence table")

print(f"\n📚 METHODOLOGY VALIDATION COMPLETE!")
print(f"✓ Within-replication vs across-replication variance distinguished")
print(f"✓ 'Room for error' hypothesis quantitatively tested")
print(f"✓ System characterization separated from statistical precision")
# %%