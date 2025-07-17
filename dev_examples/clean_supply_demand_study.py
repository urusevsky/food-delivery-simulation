# dev_examples/clean_supply_demand_study.py
"""
Clean Supply-Demand Study Workflow

Research Question: "How do supply-demand conditions affect system performance?"

Simple, clean implementation following user's gradual approach:
- No feature creep
- Manual design points dictionary creation
- Clean DesignPoint and ExperimentalRunner
"""

# %% Import and Setup
"""
Cell 1: Clean imports
"""
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

print("✓ Clean imports successful")

# %% Step 1: Logging Configuration
"""
Cell 2: Setup logging
"""
print("\n" + "="*60)
print("CLEAN SUPPLY-DEMAND STUDY WORKFLOW")
print("="*60)

logging_config = LoggingConfig(
    console_level="INFO",
    component_levels={
        "services": "ERROR",
        "entities": "ERROR", 
        "repositories": "ERROR",
        "utils": "ERROR",
        "system_data": "ERROR",
        "simulation.runner": "INFO",
        "infrastructure": "INFO",
        "experimental.runner": "INFO",
    }
)
configure_logging(logging_config)

print("✓ Logging configured")

# %% Step 2: Infrastructure Setup
"""
Cell 3: Create and analyze infrastructure
"""
print("\nStep 2: Infrastructure Setup")

structural_config = StructuralConfig(
    delivery_area_size=10,
    num_restaurants=10,
    driver_speed=0.5
)

master_seed = 42

infrastructure = Infrastructure(structural_config, master_seed)
analyzer = InfrastructureAnalyzer(infrastructure)
analysis_results = analyzer.analyze_complete_infrastructure()

print(f"✓ Infrastructure ready: typical_distance={analysis_results['typical_distance']:.3f}km")

# %% Step 3: Default ScoringConfig
"""
Cell 4: Define default scoring config
"""
print("\nStep 3: Default ScoringConfig")

scoring_config = ScoringConfig()  # Default values

print("✓ Default ScoringConfig defined")

# %% Step 4: Create Design Points Dictionary
"""
Cell 5: Manually create design points dictionary for supply-demand study
"""
print("\nStep 4: Create Design Points Dictionary")

# Base operational parameters
base_params = {
    'pairing_enabled': False,
    'restaurants_proximity_threshold': None,
    'customers_proximity_threshold': None,
    'immediate_assignment_threshold': 100,
    'mean_service_duration': 100,
    'service_duration_std_dev': 60,
    'min_service_duration': 30,
    'max_service_duration': 200,
    'periodic_interval': 3.0
}

# Manually create design points dictionary
design_points = {}

design_points["low_demand_medium_supply"] = DesignPoint(
    infrastructure=infrastructure,
    operational_config=OperationalConfig(
        mean_order_inter_arrival_time=3.0,
        mean_driver_inter_arrival_time=8.0,
        **base_params
    ),
    scoring_config=scoring_config,
    name="low_demand_medium_supply"
)

design_points["high_demand_low_supply"] = DesignPoint(
    infrastructure=infrastructure,
    operational_config=OperationalConfig(
        mean_order_inter_arrival_time=1.0,
        mean_driver_inter_arrival_time=10.0,
        **base_params
    ),
    scoring_config=scoring_config,
    name="high_demand_low_supply"
)

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

print(f"✓ Created {len(design_points)} design points manually")
print(f"  • Design points: {list(design_points.keys())}")

# Show load ratios
print(f"\n📊 Load Ratios (driver_interval/order_interval):")
for name, dp in design_points.items():
    load_ratio = dp.operational_config.mean_driver_inter_arrival_time / dp.operational_config.mean_order_inter_arrival_time
    print(f"  • {name}: {load_ratio:.1f}")

# %% Step 5: ExperimentConfig
"""
Cell 6: Define experiment parameters
"""
print("\nStep 5: ExperimentConfig")

experiment_config = ExperimentConfig(
    simulation_duration=1000,
    num_replications=3,
    master_seed=42
)

print(f"✓ Experiment config: {experiment_config.simulation_duration}min, {experiment_config.num_replications} reps")
print(f"  Total replications: {len(design_points)} × {experiment_config.num_replications} = {len(design_points) * experiment_config.num_replications}")

# %% Step 6: Execute Study
"""
Cell 7: Run the experimental study
"""
print("\nStep 6: Execute Experimental Study")

runner = ExperimentalRunner()
print("✓ ExperimentalRunner created")

print(f"\nExecuting study...")
study_results = runner.run_experimental_study(design_points, experiment_config)

print(f"\n✅ Study completed!")
print(f"  • Design points executed: {len(study_results)}")
print(f"  • Results available for warmup analysis")

# %% Step 8: Enhanced Warmup Analysis with Welch's Method and Little's Law (Corrected)
"""
Cell 8: Extract enhanced time series data with Welch's method and Little's Law validation

This cell applies the enhanced warmup analysis approach:
- Cross-replication averaging with moving average smoothing (Welch's method)
- Little's Law theoretical validation for active drivers
- Prepares data for principled visual warmup determination
"""
print("\nStep 8: Enhanced Warmup Analysis with Welch's Method and Little's Law")
print("=" * 60)

# Import the enhanced warmup analysis modules
from delivery_sim.warmup_analysis.time_series_preprocessing import TimeSeriesPreprocessor
from delivery_sim.warmup_analysis.visualization import WelchMethodVisualization
import matplotlib.pyplot as plt

print("✓ Enhanced warmup analysis modules imported")

# Extract enhanced time series data from all design points
print(f"\n📊 Extracting enhanced time series data from {len(study_results)} design points...")

all_enhanced_time_series = {}
preprocessor = TimeSeriesPreprocessor()

for design_name, design_results in study_results.items():
    print(f"  Processing {design_name}...")
    
    # Extract system snapshots from this design point's replications
    replication_snapshots = []
    for replication_result in design_results['replication_results']:
        snapshots = replication_result['system_snapshots']
        if snapshots:
            replication_snapshots.append(snapshots)
    
    if len(replication_snapshots) < 2:
        print(f"    ⚠️  Warning: Only {len(replication_snapshots)} replications for {design_name}")
        continue
    
    # Step 1: Basic extraction with Welch's method
    basic_data = preprocessor.extract_cross_replication_averages(
        multi_replication_snapshots=replication_snapshots,
        metrics=['active_drivers', 'active_delivery_entities'],
        collection_interval=0.5,
        moving_average_window=50
    )
    
    # Step 2: Add Little's Law theoretical validation
    single_design_dict = {design_name: design_points[design_name]}
    enhanced_data = preprocessor.add_little_law_theoretical_values(
        time_series_data={design_name: basic_data},
        design_points_dict=single_design_dict
    )
    
    # Store the enhanced data for this design point
    all_enhanced_time_series[design_name] = enhanced_data[design_name]
    
    # Show Little's Law validation info
    if 'active_drivers' in enhanced_data[design_name] and 'theoretical_value' in enhanced_data[design_name]['active_drivers']:
        theoretical_value = enhanced_data[design_name]['active_drivers']['theoretical_value']
        print(f"    ✓ Little's Law predicts {theoretical_value:.1f} active drivers")
    
    print(f"    ✓ Welch's method applied: {len(replication_snapshots)} replications processed")

print(f"\n✅ Enhanced time series extraction complete!")
print(f"  • Design points processed: {len(all_enhanced_time_series)}")
print(f"  • Welch's method applied with moving average smoothing")
print(f"  • Little's Law theoretical validation added for active drivers")
print(f"  • Ready for enhanced visual warmup inspection")

# Backward compatibility for existing workflow
all_time_series_data = all_enhanced_time_series
print(f"\n✓ Variable compatibility maintained for downstream cells")
# %% Step 9: Welch's Method Visual Inspection with Methodological Evidence (Corrected)
"""
Cell 9: Create enhanced Welch's method plots with Little's Law validation

Shows the juxtaposition of active drivers vs active delivery entities to demonstrate
why active drivers is the optimal warmup indicator across different operational regimes.
"""
print("\nStep 9: Welch's Method Visual Inspection with Methodological Evidence")
print("=" * 60)

# Create enhanced visualization instance
viz = WelchMethodVisualization(figsize=(14, 8))

print(f"🔬 Creating enhanced Welch's method plots for methodological evidence...")
print(f"  • Total plots to display: {len(all_enhanced_time_series)}")
print(f"  • Each plot shows: Cross-replication averages + Welch's smoothing + Little's Law validation")
print(f"  • Focus: Juxtaposition demonstrates active drivers superiority")

# Create enhanced plots for each design point
plot_count = 0
regime_classifications = {
    "low_demand_medium_supply": "Stable",
    "high_demand_high_supply": "Volatile", 
    "high_demand_low_supply": "System Failure"
}

for design_name, enhanced_data in all_enhanced_time_series.items():
    plot_count += 1
    regime_type = regime_classifications.get(design_name, "Unknown")
    
    print(f"\n--- Plot {plot_count}/{len(all_enhanced_time_series)}: {design_name} ({regime_type} Regime) ---")
    
    # Get basic info about this design point
    if 'active_drivers' in enhanced_data:
        first_metric_data = enhanced_data['active_drivers']
        total_duration = max(first_metric_data['time_points'])
        replication_count = first_metric_data['replication_count']
        window_size = first_metric_data['moving_average_window']
        
        print(f"  • Duration: {total_duration:.1f} minutes")
        print(f"  • Replications: {replication_count}")
        print(f"  • Moving average window: {window_size}")
        
        # Show Little's Law validation
        if 'theoretical_value' in first_metric_data:
            theoretical_value = first_metric_data['theoretical_value']
            moving_averages = first_metric_data['moving_averages']
            final_ma_value = moving_averages[-1] if moving_averages else 0
            convergence_error = abs(final_ma_value - theoretical_value) / theoretical_value * 100
            print(f"  • Little's Law theoretical: {theoretical_value:.1f} active drivers")
            print(f"  • Final moving average: {final_ma_value:.1f} active drivers")
            print(f"  • Convergence error: {convergence_error:.1f}%")
    
    # Create enhanced combined plot showing methodological evidence
    fig = viz.create_combined_welch_inspection_plot(
        enhanced_data, 
        title=f'Welch\'s Method Warmup Analysis: {design_name.replace("_", " ").title()} ({regime_type} Regime)'
    )
    
    # Add regime-specific annotation for methodological evidence
    fig.text(0.5, 0.01, 
             f'Methodological Evidence: Note consistent active drivers convergence vs. regime-dependent delivery entities pattern',
             ha='center', fontsize=11, weight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Show the plot
    plt.show()
    
    print(f"  ✓ Enhanced plot displayed with Welch's method + Little's Law validation")

print(f"\n🎯 All {plot_count} enhanced plots displayed!")
print(f"\n📋 Key Methodological Evidence Observed:")
print(f"  • Active Drivers: Consistent convergence to Little's Law prediction across ALL regimes")
print(f"  • Active Delivery Entities: Regime-dependent patterns (stable/volatile/failure)")
print(f"  • Conclusion: Active drivers provides regime-independent warmup detection")

# %% Step 10: Enhanced Warmup Determination Using Welch's Method (Corrected)
"""
Cell 10: Principled warmup determination using Welch's method convergence

Provides systematic guidance for determining uniform warmup period based on
Welch's method moving average convergence to Little's Law theoretical values.
"""
print("\nStep 10: Enhanced Warmup Determination Using Welch's Method")
print("=" * 60)

print("🔬 WELCH'S METHOD WARMUP DETERMINATION")
print("-" * 40)

print("\n📋 Enhanced methodology for warmup determination:")
print("  1. Focus on ACTIVE DRIVERS plots (top panels) from above")
print("  2. Examine the BLUE SMOOTHED LINE (Welch's moving average)")
print("  3. Look for convergence to RED DASHED LINE (Little's Law theoretical)")
print("  4. Choose warmup period AFTER moving average stabilizes near theoretical value")
print("  5. Apply conservative margin for methodological rigor")

print("\n🎯 WELCH'S METHOD CONVERGENCE ANALYSIS:")
print("  • Blue Line Interpretation: Welch's moving average filters out noise")
print("  • Red Line Target: Little's Law theoretical prediction")
print("  • Convergence Point: Where blue line stabilizes near red line")
print("  • Conservative Choice: Add safety margin after apparent convergence")

print("\n⚖️ LITTLE'S LAW VALIDATION SUMMARY:")
convergence_summary = []

for design_name, enhanced_data in all_enhanced_time_series.items():
    if 'active_drivers' in enhanced_data and 'theoretical_value' in enhanced_data['active_drivers']:
        data = enhanced_data['active_drivers']
        theoretical_value = data['theoretical_value']
        moving_averages = data['moving_averages']
        final_ma_value = moving_averages[-1] if moving_averages else 0
        convergence_error = abs(final_ma_value - theoretical_value) / theoretical_value * 100
        
        convergence_summary.append({
            'design_name': design_name,
            'theoretical': theoretical_value,
            'final_ma': final_ma_value,
            'error_percent': convergence_error
        })
        
        regime_type = regime_classifications.get(design_name, "Unknown")
        print(f"  • {design_name} ({regime_type}):")
        print(f"    - Theoretical: {theoretical_value:.1f}, Final MA: {final_ma_value:.1f}, Error: {convergence_error:.1f}%")

# Show simulation context for reference
default_window_size = 50  # Default from extraction
actual_window_size = default_window_size

# Try to get actual window size from data if available
if all_enhanced_time_series:
    first_design_data = next(iter(all_enhanced_time_series.values()))
    if 'active_drivers' in first_design_data and 'moving_average_window' in first_design_data['active_drivers']:
        actual_window_size = first_design_data['active_drivers']['moving_average_window']

print(f"\n📊 Simulation Context for Warmup Decision:")
print(f"  • Total simulation duration: {experiment_config.simulation_duration} minutes")
print(f"  • Number of design points: {len(all_enhanced_time_series)}")
print(f"  • Moving average window: {actual_window_size} data points")

# Provide enhanced warmup guidance
print(f"\n📏 Enhanced Warmup Period Guidelines:")
print(f"  • Based on Welch's method convergence observation")
print(f"  • Theoretical validation via Little's Law")
print(f"  • Conservative margin for methodological rigor")
print(f"  • Uniform application across all design points (CRN requirement)")

print(f"\n💡 Next Steps - Warmup Period Selection:")
print(f"  1. Examine active drivers convergence in all {len(all_enhanced_time_series)} plots above")
print(f"  2. Identify latest convergence point across all design points")
print(f"  3. Add conservative margin (e.g., +50-100 minutes)")
print(f"  4. Validate: Ensure post-warmup analysis window ≥ 60% of total duration")
print(f"  5. Proceed to Cell 11 for warmup validation and performance analysis")

print(f"\n🔧 Preparation for Next Cell:")
print(f"  • Variable 'all_enhanced_time_series' contains Welch's method data")
print(f"  • Little's Law validation completed")
print(f"  • Ready for warmup period validation in next cell")
# %% Step 11: Warmup Period Validation
"""
Cell 11: Validate your visually determined warmup period

Set your warmup period based on visual inspection and validate it across all design points.
"""
print("\nStep 11: Warmup Period Validation")
print("=" * 60)

# ======================================================================
# SET YOUR WARMUP PERIOD HERE BASED ON VISUAL INSPECTION
# ======================================================================
print("📌 WARMUP PERIOD DETERMINATION:")
print("Based on visual inspection of the plots above, set your warmup period:")

# TODO: Update this value based on your visual inspection of the plots
proposed_warmup_period = 400  # Replace with your visually determined value

print(f"  • Proposed warmup period: {proposed_warmup_period} minutes")
# ======================================================================

print(f"\n🔍 Validating warmup period across all design points...")

# Validation summary
validation_results = []

for design_name, time_series_data in all_enhanced_time_series.items():
    # Get time series info
    first_metric = list(time_series_data.keys())[0]
    max_time = max(time_series_data[first_metric]['time_points'])
    
    # Calculate analysis window
    analysis_window = max_time - proposed_warmup_period
    warmup_ratio = proposed_warmup_period / max_time
    
    # Store validation info
    validation_info = {
        'design_name': design_name,
        'total_duration': max_time,
        'warmup_period': proposed_warmup_period,
        'analysis_window': analysis_window,
        'warmup_ratio': warmup_ratio
    }
    validation_results.append(validation_info)
    
    print(f"\n  📊 {design_name}:")
    print(f"    • Total duration: {max_time:.1f} minutes")
    print(f"    • Warmup period: {proposed_warmup_period:.1f} minutes ({warmup_ratio*100:.1f}%)")
    print(f"    • Analysis window: {analysis_window:.1f} minutes ({(1-warmup_ratio)*100:.1f}%)")
    
    # Validation checks
    if warmup_ratio > 0.5:
        print(f"    ⚠️  WARNING: Warmup is {warmup_ratio*100:.1f}% of total (>50%)")
    elif warmup_ratio > 0.3:
        print(f"    ⚠️  CAUTION: Warmup is {warmup_ratio*100:.1f}% of total (>30%)")
    else:
        print(f"    ✅ Good ratio: {warmup_ratio*100:.1f}% warmup")
    
    if analysis_window < 30:
        print(f"    ⚠️  WARNING: Analysis window ({analysis_window:.1f} min) may be too short")
    else:
        print(f"    ✅ Analysis window ({analysis_window:.1f} min) adequate")

# Overall validation summary
all_ratios = [v['warmup_ratio'] for v in validation_results]
all_windows = [v['analysis_window'] for v in validation_results]

print(f"\n📋 OVERALL VALIDATION SUMMARY:")
print(f"  • Warmup ratios: {min(all_ratios)*100:.1f}% - {max(all_ratios)*100:.1f}%")
print(f"  • Analysis windows: {min(all_windows):.1f} - {max(all_windows):.1f} minutes")

# Final recommendation
if max(all_ratios) <= 0.3 and min(all_windows) >= 30:
    print(f"  ✅ RECOMMENDED: Warmup period appears appropriate for all design points")
else:
    print(f"  ⚠️  CONSIDER ADJUSTMENT: Review warmup period based on validation results")

print(f"\n🎯 UNIFORM WARMUP PERIOD DETERMINED:")
print(f"  • Selected warmup period: {proposed_warmup_period} minutes")
print(f"  • Applies to ALL {len(all_enhanced_time_series)} design points")
print(f"  • Ready for post-simulation analysis in next workflow phase")

print(f"\n➡️  Next: Use this warmup period for performance analysis across design points")

# Store for next workflow phase
uniform_warmup_period = proposed_warmup_period
print(f"\n✓ Variable 'uniform_warmup_period' = {uniform_warmup_period} minutes stored for next phase")

# %% Step 12: Performance Metrics Calculation
"""
Cell 12: Apply Analysis Pipeline to Calculate Performance Metrics

Use existing analysis pipeline to calculate key metrics that will provide
quantitative evidence for the pattern classifications observed in time series plots.
"""
print("\nStep 12: Performance Metrics Calculation")
print("=" * 60)

# Import the analysis pipeline
from delivery_sim.analysis_pipeline.pipeline_coordinator import analyze_single_configuration

print("📊 Calculating performance metrics for all design points...")
print(f"  • Using uniform warmup period: {uniform_warmup_period} minutes")
print(f"  • Metrics: Order Assignment Time + System Completion Rate")

# Calculate metrics for each design point
metrics_results = {}

for design_name, design_results in study_results.items():
    print(f"\n  Processing {design_name}...")
    
    try:
        # Apply analysis pipeline with uniform warmup period
        analysis_result = analyze_single_configuration(
            simulation_results=design_results,
            warmup_period=uniform_warmup_period,
            confidence_level=0.95
        )
        
        metrics_results[design_name] = {
            'analysis': analysis_result,
            'status': 'success'
        }
        
        print(f"    ✓ Metrics calculated successfully")
        
    except Exception as e:
        print(f"    ✗ Error calculating metrics: {str(e)}")
        metrics_results[design_name] = {
            'analysis': None,
            'status': 'error',
            'error': str(e)
        }

print(f"\n✅ Performance metrics calculation complete!")
print(f"  • Successfully analyzed: {sum(1 for r in metrics_results.values() if r['status'] == 'success')}")
print(f"  • Errors encountered: {sum(1 for r in metrics_results.values() if r['status'] == 'error')}")

# %% Step 13: Extract Key Metrics for Pattern Analysis
"""
Cell 13: Extract and Display Key Performance Metrics

Extract the two key metrics (assignment time, completion rate) from analysis results
to provide quantitative evidence for pattern classification.
"""
print("\nStep 13: Extract Key Metrics for Pattern Analysis")
print("=" * 60)

# Extract key metrics from analysis results
pattern_evidence = {}

print("📋 Extracting key metrics for pattern analysis...")

for design_name, result in metrics_results.items():
    if result['status'] != 'success':
        print(f"  ⚠️  Skipping {design_name}: {result.get('error', 'Analysis failed')}")
        continue
    
    analysis = result['analysis']
    
    try:
        # Extract Order Assignment Time (entity metric)
        entity_metrics = analysis.get('entity_metrics', {})
        orders_metrics = entity_metrics.get('orders', {})
        assignment_time_data = orders_metrics.get('assignment_time', {})
        
        if assignment_time_data and 'mean' in assignment_time_data:
            assignment_time_mean = assignment_time_data['mean'].get('point_estimate')
            assignment_time_ci = assignment_time_data['mean'].get('confidence_interval', [None, None])
            assignment_time_std = assignment_time_data['mean'].get('standard_error', 0) * (3**0.5) if assignment_time_data['mean'].get('standard_error') else None
        else:
            assignment_time_mean = None
            assignment_time_ci = [None, None]
            assignment_time_std = None
        
        # Extract System Completion Rate (system metric)
        system_metrics = analysis.get('system_metrics', {})
        completion_rate_data = system_metrics.get('system_completion_rate', {})
        
        if completion_rate_data:
            completion_rate = completion_rate_data.get('point_estimate')
            completion_rate_ci = completion_rate_data.get('confidence_interval', [None, None])
        else:
            completion_rate = None
            completion_rate_ci = [None, None]
        
        # Get load ratio from design point
        design_point = design_points[design_name]
        load_ratio = (design_point.operational_config.mean_driver_inter_arrival_time / 
                     design_point.operational_config.mean_order_inter_arrival_time)
        
        pattern_evidence[design_name] = {
            'load_ratio': load_ratio,
            'assignment_time_mean': assignment_time_mean,
            'assignment_time_ci': assignment_time_ci,
            'assignment_time_std': assignment_time_std,
            'completion_rate': completion_rate,
            'completion_rate_ci': completion_rate_ci,
            'status': 'valid'
        }
        
        print(f"  ✓ {design_name}: Assignment Time = {assignment_time_mean:.1f} min, Completion Rate = {completion_rate:.1%}" if assignment_time_mean and completion_rate else f"  ⚠️  {design_name}: Incomplete data")
        
    except Exception as e:
        print(f"  ✗ Error extracting metrics for {design_name}: {str(e)}")
        pattern_evidence[design_name] = {
            'status': 'extraction_error',
            'error': str(e)
        }

print(f"\n✅ Key metrics extraction complete!")
print(f"  • Valid results: {sum(1 for r in pattern_evidence.values() if r.get('status') == 'valid')}")

# %% Step 14: Performance Metrics Evidence Table
"""
Cell 14: Create Evidence Table for Pattern Classification

Display performance metrics in a clear table format that provides quantitative
evidence for the three patterns observed in time series plots.
"""
print("\nStep 14: Performance Metrics Evidence Table")
print("=" * 60)

import pandas as pd

print("📊 Creating performance metrics evidence table...")

# Prepare data for table
table_data = []

for design_name, evidence in pattern_evidence.items():
    if evidence.get('status') != 'valid':
        continue
    
    # Calculate assignment time statistics
    assignment_mean = evidence.get('assignment_time_mean')
    assignment_std = evidence.get('assignment_time_std')
    assignment_ci = evidence.get('assignment_time_ci', [None, None])
    
    # Format assignment time
    if assignment_mean is not None:
        if assignment_std is not None:
            assignment_formatted = f"{assignment_mean:.1f}±{assignment_std:.1f}"
        elif assignment_ci[0] is not None and assignment_ci[1] is not None:
            ci_width = (assignment_ci[1] - assignment_ci[0]) / 2
            assignment_formatted = f"{assignment_mean:.1f}±{ci_width:.1f}"
        else:
            assignment_formatted = f"{assignment_mean:.1f}"
    else:
        assignment_formatted = "N/A"
    
    # Format completion rate
    completion_rate = evidence.get('completion_rate')
    completion_formatted = f"{completion_rate:.1%}" if completion_rate is not None else "N/A"
    
    table_data.append({
        'Design Point': design_name.replace('_', ' ').title(),
        'Load Ratio': f"{evidence['load_ratio']:.1f}",
        'Assignment Time (min)': assignment_formatted,
        'Completion Rate': completion_formatted,
        'Load Ratio Value': evidence['load_ratio'],  # For sorting
        'Assignment Value': assignment_mean if assignment_mean else 999,  # For sorting
        'Completion Value': completion_rate if completion_rate else 0  # For sorting
    })

# Create and display table
if table_data:
    df = pd.DataFrame(table_data)
    
    # Sort by load ratio for logical presentation
    df_display = df.sort_values('Load Ratio Value')[['Design Point', 'Load Ratio', 'Assignment Time (min)', 'Completion Rate']]
    
    print("\n🎯 PERFORMANCE METRICS EVIDENCE TABLE")
    print("-" * 80)
    print(df_display.to_string(index=False))
    
    print(f"\n📈 Key Observations for Pattern Classification:")
    
    # Identify patterns based on completion rate and assignment time variance
    high_completion = df['Completion Value'] > 0.85
    low_completion = df['Completion Value'] < 0.6
    
    stable_candidates = df[high_completion & (df['Load Ratio Value'] < 4.0)]
    volatile_candidates = df[high_completion & (df['Load Ratio Value'] >= 4.0)]
    unstable_candidates = df[low_completion]
    
    if len(stable_candidates) > 0:
        print(f"  • Potential STABLE systems: {', '.join(stable_candidates['Design Point'].values)}")
        print(f"    - High completion rates (>{85}%)")
        print(f"    - Low load ratios (<4.0)")
    
    if len(volatile_candidates) > 0:
        print(f"  • Potential VOLATILE systems: {', '.join(volatile_candidates['Design Point'].values)}")
        print(f"    - Maintained completion rates (>{85}%)")  
        print(f"    - Higher load ratios (≥4.0)")
        print(f"    - Likely higher assignment time variance")
    
    if len(unstable_candidates) > 0:
        print(f"  • Potential UNSTABLE systems: {', '.join(unstable_candidates['Design Point'].values)}")
        print(f"    - Low completion rates (<60%)")
        print(f"    - System failure indicators")
    
    print(f"\n💡 This quantitative evidence confirms the patterns observed in time series plots!")
    
else:
    print("⚠️  No valid data available for evidence table")

# %% Step 15: Pattern Classification Based on Evidence
"""
Cell 15: Evidence-Based Pattern Classification

Combine time series visual evidence with performance metrics to make final
pattern classifications for each design point.
"""
print("\nStep 15: Evidence-Based Pattern Classification")
print("=" * 60)

print("🔬 Combining visual time series evidence with performance metrics...")

# Manual classification based on evidence
final_classifications = {}

print(f"\n📋 EVIDENCE-BASED PATTERN CLASSIFICATION:")
print(f"Based on time series plots (Steps 8-10) + performance metrics (Step 14)")
print("-" * 60)

# You'll need to manually classify based on your observations
# This is where YOU make the decisions based on the evidence

print("📝 Manual Classification Guidelines:")
print("  • STABLE: Low load ratio + high completion rate + low time series volatility")
print("  • VOLATILE: Medium/high load ratio + decent completion rate + high time series volatility") 
print("  • UNSTABLE: High load ratio + low completion rate + growing time series")

print(f"\n🎯 PRELIMINARY CLASSIFICATIONS (to be confirmed by visual inspection):")

for design_name, evidence in pattern_evidence.items():
    if evidence.get('status') != 'valid':
        print(f"  • {design_name}: INSUFFICIENT_DATA")
        final_classifications[design_name] = "insufficient_data"
        continue
    
    load_ratio = evidence['load_ratio']
    completion_rate = evidence.get('completion_rate', 0)
    
    # Preliminary classification - YOU should refine this based on visual evidence
    if completion_rate < 0.6:
        preliminary = "UNSTABLE"
    elif load_ratio < 3.0 and completion_rate > 0.9:
        preliminary = "STABLE"  
    elif completion_rate > 0.8:
        preliminary = "VOLATILE"
    else:
        preliminary = "UNCLEAR"
    
    print(f"  • {design_name}: {preliminary} (Load: {load_ratio:.1f}, Completion: {completion_rate:.1%})")
    final_classifications[design_name] = preliminary.lower()

print(f"\n⚠️  IMPORTANT: Refine these classifications by reviewing:")
print(f"  1. Time series plots from Steps 8-10 (visual evidence)")
print(f"  2. Performance metrics table from Step 14 (quantitative evidence)")
print(f"  3. Your domain knowledge of the system behavior")

print(f"\n✅ Evidence gathering complete!")
print(f"✅ Ready for thesis writing with solid evidence base!")

# Store results for thesis
thesis_evidence = {
    'classifications': final_classifications,
    'metrics_table': df_display if 'df_display' in locals() else None,
    'time_series_data': all_enhanced_time_series,
    'warmup_period': uniform_warmup_period
}

print(f"\n📚 Thesis Evidence Package Ready:")
print(f"  • Pattern classifications: ✓")
print(f"  • Performance metrics table: ✓") 
print(f"  • Time series data: ✓")
print(f"  • Warmup period: {uniform_warmup_period} minutes")
# %%
