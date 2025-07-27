# load_ratio_driven_supply_demand_study.py
"""
Load Ratio-Driven Supply-Demand Study: Systematic Validation Design

Research Hypothesis: Load ratio determines operational regime characteristics,
with systematic validation pairs testing robustness across different absolute arrival rates.

Design Pattern: For each load ratio R:
- Baseline Interval: (1.0, R) → "Higher intensity" (1.0 orders/min, 1/R drivers/min)
- 2x Baseline: (2.0, 2R) → "Half intensity" (0.5 orders/min, 1/2R drivers/min)
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
print("LOAD RATIO-DRIVEN SUPPLY-DEMAND STUDY: SYSTEMATIC VALIDATION DESIGN")
print("="*80)
print("Research Focus: Load ratio determines regime with systematic validation pairs")

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

# %% Step 4: Systematic Load Ratio-Driven Design Points Creation
print("\n" + "="*50)
print("SYSTEMATIC LOAD RATIO-DRIVEN DESIGN CREATION")
print("="*50)

scoring_config = ScoringConfig()

# Base operational parameters (consistent across all design points)
base_params = {
    'pairing_enabled': False,
    'restaurants_proximity_threshold': None,
    'customers_proximity_threshold': None,
    'mean_service_duration': 100,
    'service_duration_std_dev': 60,
    'min_service_duration': 30,
    'max_service_duration': 200,
}

# Define target load ratios for systematic exploration
target_load_ratios = [2.5, 3.5, 4.0, 5.0, 5.5, 6.0, 7.0]

print("Creating systematic load ratio-driven design points:")
print("Pattern: Baseline Interval (1.0, R) + 2x Baseline (2.0, 2R)")
print(f"Target load ratios: {target_load_ratios}")

# Create design points systematically
design_points = {}

for load_ratio in target_load_ratios:
    # Baseline interval design point: (1.0, load_ratio)
    baseline_name = f"load_ratio_{load_ratio:.1f}_baseline"
    design_points[baseline_name] = DesignPoint(
        infrastructure=infrastructure,
        operational_config=OperationalConfig(
            mean_order_inter_arrival_time=1.0,
            mean_driver_inter_arrival_time=load_ratio,
            **base_params
        ),
        scoring_config=scoring_config,
        name=baseline_name
    )
    
    # 2x baseline design point: (2.0, 2*load_ratio)
    double_baseline_name = f"load_ratio_{load_ratio:.1f}_2x_baseline"
    design_points[double_baseline_name] = DesignPoint(
        infrastructure=infrastructure,
        operational_config=OperationalConfig(
            mean_order_inter_arrival_time=2.0,
            mean_driver_inter_arrival_time=2.0 * load_ratio,
            **base_params
        ),
        scoring_config=scoring_config,
        name=double_baseline_name
    )
    
    print(f"  ✓ Load Ratio {load_ratio:.1f}: Baseline (1.0, {load_ratio:.1f}) + 2x Baseline (2.0, {2.0*load_ratio:.1f})")

print(f"\n✓ Created {len(design_points)} design points systematically")

# Display systematic validation pairs
print(f"\n📊 Systematic Validation Pairs Overview:")
for load_ratio in target_load_ratios:
    baseline_order_rate = 1.0 / 1.0
    double_baseline_order_rate = 1.0 / 2.0
    baseline_driver_rate = 1.0 / load_ratio
    double_baseline_driver_rate = 1.0 / (2.0 * load_ratio)
    
    print(f"  Load Ratio {load_ratio:.1f}:")
    print(f"    • Baseline: {baseline_order_rate:.2f} orders/min, {baseline_driver_rate:.3f} drivers/min")
    print(f"    • 2x Baseline: {double_baseline_order_rate:.2f} orders/min, {double_baseline_driver_rate:.3f} drivers/min")

print(f"\n🎯 Research Hypothesis:")
print(f"  • Load ratio determines operational regime characteristics")
print(f"  • Validation pairs (baseline vs 2x baseline) should show:")
print(f"    - Same regime behavior (time series patterns)")
print(f"    - Same CV structure (variability patterns)")
print(f"    - Different absolute performance (scale effects)")

# %% Step 5: Extended Experiment Configuration for Regime Analysis
print("\n" + "="*50)
print("EXPERIMENT CONFIGURATION")
print("="*50)

experiment_config = ExperimentConfig(
    simulation_duration=2000,  # Extended duration for regime pattern analysis
    num_replications=5,        # Multiple replications for statistical robustness
    master_seed=42
)

print(f"✓ Extended duration: {experiment_config.simulation_duration} minutes")
print(f"✓ Replications: {experiment_config.num_replications}")
print(f"✓ Total simulation runs: {len(design_points)} × {experiment_config.num_replications} = {len(design_points) * experiment_config.num_replications}")
print(f"✓ Load ratio coverage: {min(target_load_ratios):.1f} - {max(target_load_ratios):.1f} with systematic validation")

# %% Step 6: Execute Load Ratio-Driven Study
print("\n" + "="*50)
print("LOAD RATIO-DRIVEN EXPERIMENTAL EXECUTION")
print("="*50)

runner = ExperimentalRunner()
print("✓ ExperimentalRunner initialized")

print(f"\nExecuting systematic load ratio study with {len(target_load_ratios)} load ratios...")
print("Focus: Systematic validation of load ratio→regime hypothesis")
study_results = runner.run_experimental_study(design_points, experiment_config)

print(f"\n✅ LOAD RATIO-DRIVEN STUDY COMPLETE!")
print(f"✓ Load ratios tested: {len(target_load_ratios)}")
print(f"✓ Validation pairs: {len(target_load_ratios)}")
print(f"✓ Total design points executed: {len(study_results)}")
print(f"✓ Ready for systematic regime validation analysis")

# %% Step 7: Time Series Analysis for Load Ratio Validation
print("\n" + "="*50)
print("TIME SERIES ANALYSIS FOR LOAD RATIO VALIDATION")
print("="*50)

from delivery_sim.warmup_analysis.time_series_preprocessing import TimeSeriesPreprocessor
from delivery_sim.warmup_analysis.visualization import WelchMethodVisualization
import matplotlib.pyplot as plt

print("Extracting time series data for systematic load ratio validation...")

preprocessor = TimeSeriesPreprocessor()
viz = WelchMethodVisualization(figsize=(16, 10))  # Larger plots for detailed analysis

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
        moving_average_window=100  # Larger window for 2000-minute simulation
    )
    
    # Add Little's Law validation
    single_design_dict = {design_name: design_points[design_name]}
    enhanced_data = preprocessor.add_little_law_theoretical_values(
        time_series_data={design_name: basic_data},
        design_points_dict=single_design_dict
    )
    
    all_time_series_data[design_name] = enhanced_data[design_name]
    
    # Extract load ratio from design name for analysis
    load_ratio_str = design_name.split('_')[2]  # Extract from "load_ratio_X.X_..."
    load_ratio = float(load_ratio_str)
    
    if 'active_drivers' in enhanced_data[design_name] and 'theoretical_value' in enhanced_data[design_name]['active_drivers']:
        theoretical_value = enhanced_data[design_name]['active_drivers']['theoretical_value']
        print(f"    ✓ Load Ratio: {load_ratio:.1f}, Little's Law: {theoretical_value:.1f} drivers")

print(f"\n✓ Time series extraction complete for {len(all_time_series_data)} design points")

# %% Step 8: Systematic Load Ratio Visualization (Validation Pairs)
print("\n" + "="*50)
print("SYSTEMATIC LOAD RATIO VALIDATION: PAIRED COMPARISONS")
print("="*50)

print("Creating systematic validation plots by load ratio...")
print("Hypothesis test: Validation pairs should show same regime patterns")

# Sort and group design points by load ratio for systematic comparison
load_ratio_groups = {}
for design_name in all_time_series_data.keys():
    load_ratio_str = design_name.split('_')[2]  # Extract from "load_ratio_X.X_..."
    load_ratio = float(load_ratio_str)
    
    if load_ratio not in load_ratio_groups:
        load_ratio_groups[load_ratio] = []
    load_ratio_groups[load_ratio].append(design_name)

# Create paired plots for each load ratio
plot_count = 0
for load_ratio in sorted(load_ratio_groups.keys()):
    design_names = load_ratio_groups[load_ratio]
    
    if len(design_names) != 2:
        print(f"⚠️  Load ratio {load_ratio} has {len(design_names)} design points, expected 2")
        continue
    
    # Sort by baseline vs 2x baseline
    baseline_design = [name for name in design_names if 'baseline' in name and '2x' not in name][0]
    double_baseline_design = [name for name in design_names if '2x_baseline' in name][0]
    
    print(f"\n--- Load Ratio {load_ratio:.1f} Validation Pair ---")
    print(f"  Baseline: {baseline_design}")
    print(f"  2x Baseline: {double_baseline_design}")
    
    # Create side-by-side comparison plots
    for i, (design_name, label) in enumerate([(baseline_design, 'Baseline'), (double_baseline_design, '2x Baseline')]):
        plot_count += 1
        enhanced_data = all_time_series_data[design_name]
        
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
                print(f"    {label}: Little's Law {theoretical_value:.1f} vs Final {final_ma_value:.1f} (Error: {convergence_error:.1f}%)")
        
        # Create validation plot
        fig = viz.create_combined_welch_inspection_plot(
            enhanced_data, 
            title=f'Load Ratio {load_ratio:.1f}: {label} Interval Pattern'
        )
        
        # Add validation annotation
        fig.text(0.5, 0.01, 
                 f'Validation Test: Compare regime patterns between Baseline vs 2x Baseline for Load Ratio {load_ratio:.1f}',
                 ha='center', fontsize=11, weight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.show()
        print(f"    ✓ {label} plot displayed")

print(f"\n🎯 Systematic validation plots complete!")
print(f"\n📋 LOAD RATIO VALIDATION GUIDANCE:")
print(f"For each load ratio, compare Baseline vs 2x Baseline patterns:")
print(f"• Active Drivers (top): Should both converge to Little's Law predictions")
print(f"• Active Delivery Entities (bottom): Should show SAME regime pattern")
print(f"• If hypothesis holds: Same qualitative behavior, different quantitative performance")

# %% Step 9: Warmup Period Determination for Extended Data
print("\n" + "="*50)
print("WARMUP PERIOD DETERMINATION FOR EXTENDED SIMULATION")
print("="*50)

print("📌 EXTENDED SIMULATION WARMUP GUIDANCE:")
print("Based on the systematic validation plots above:")
print("1. Focus on Active Drivers convergence across ALL design points")
print("2. Find latest convergence point across all load ratios and validation pairs")
print("3. Account for extended 2000-minute duration")
print("4. Add conservative margin for methodological rigor")

# Set your warmup period based on visual inspection of systematic validation plots
proposed_warmup_period = 500  # UPDATE THIS based on systematic visual inspection

print(f"\n⚙️  Proposed warmup period for systematic study: {proposed_warmup_period} minutes")

# Extended validation for systematic design
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
print(f"✓ Systematic data provides {min_analysis_window:.0f} minutes of post-warmup analysis")

if avg_warmup_ratio <= 0.4 and min_analysis_window >= 60:
    print("✅ Warmup period appears appropriate for systematic study")
else:
    print("⚠️  Consider adjusting warmup period for extended systematic simulation")

uniform_warmup_period = proposed_warmup_period

# %% Step 10: Systematic Performance Metrics Analysis
print("\n" + "="*50)
print("SYSTEMATIC PERFORMANCE METRICS CALCULATION")
print("="*50)

from delivery_sim.analysis_pipeline.pipeline_coordinator import analyze_single_configuration

print(f"Calculating systematic performance metrics for {len(study_results)} design points...")
print(f"Using uniform warmup period: {uniform_warmup_period} minutes")
print(f"Extended analysis focus: Load ratio validation across baseline vs 2x baseline pairs")

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

print(f"\n✓ Systematic metrics calculation complete")

# %% Step 11: Systematic Validation Evidence Table
print("\n" + "="*50)
print("SYSTEMATIC LOAD RATIO VALIDATION: EVIDENCE TABLE")
print("="*50)

import pandas as pd

print("Creating systematic validation evidence table...")

# Extract performance metrics with systematic organization
table_data = []

for design_name, result in metrics_results.items():
    if result['status'] != 'success':
        continue
    
    analysis = result['analysis']
    
    try:
        # Extract load ratio and interval type from design name
        name_parts = design_name.split('_')
        load_ratio = float(name_parts[2])
        interval_type = "2x Baseline" if "2x" in design_name else "Baseline"
        
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
        
        # Within-replication standard deviation
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
        
        table_data.append({
            'Load Ratio': f"{load_ratio:.1f}",
            'Interval Type': interval_type,
            'Design Point': f"LR{load_ratio:.1f}_{interval_type.replace(' ', '')}",
            'Mean Assignment Time': assignment_time_mean_ci,
            'Within-Rep Std': assignment_time_std_ci,
            'Completion Rate': completion_formatted,
            'Load Ratio Value': load_ratio,  # For sorting
            'Mean Value': assignment_time_mean if assignment_time_mean else 999,
            'Std Value': assignment_time_std_mean if assignment_time_std_mean else 0,
            'Completion Value': completion_rate if completion_rate else 0
        })
        
    except Exception as e:
        print(f"  ⚠️  Error extracting metrics for {design_name}: {str(e)}")

# Create and display systematic validation table
if table_data:
    df = pd.DataFrame(table_data)
    df_display = df.sort_values(['Load Ratio Value', 'Interval Type'])[['Load Ratio', 'Interval Type', 'Mean Assignment Time', 'Within-Rep Std', 'Completion Rate']]
    
    print("\n🎯 SYSTEMATIC LOAD RATIO VALIDATION: EVIDENCE TABLE")
    print("="*120)
    print(df_display.to_string(index=False))
    

# %%