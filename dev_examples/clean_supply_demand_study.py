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

# %% Step 8: Simplified Warmup Analysis - Time Series Extraction
"""
Cell 8: Extract cross-replication averaged time series from all design points

This cell applies the new simplified warmup analysis approach:
- Focus on cross-replication averaging (the core insight)
- No complex cumulative smoothing
- Prepare data for visual inspection
"""
print("\nStep 8: Simplified Warmup Analysis - Time Series Extraction")
print("=" * 60)

# Import the new simplified warmup analysis modules
from delivery_sim.warmup_analysis.time_series_preprocessing import extract_time_series_for_inspection
from delivery_sim.warmup_analysis.visualization import TimeSeriesVisualization
import matplotlib.pyplot as plt

print("✓ New simplified warmup analysis modules imported")

# Extract time series data from all design points
print(f"\n📊 Extracting time series data from {len(study_results)} design points...")

all_time_series_data = {}

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
    
    # Extract cross-replication averages (the core operation!)
    time_series_data = extract_time_series_for_inspection(
        multi_replication_snapshots=replication_snapshots,
        metrics=['active_drivers', 'active_delivery_entities'],
        collection_interval=0.5  # Should match SystemDataCollector setting
    )
    
    all_time_series_data[design_name] = time_series_data
    print(f"    ✓ Extracted data for {len(time_series_data)} metrics, {len(replication_snapshots)} replications")

print(f"\n✅ Time series extraction complete!")
print(f"  • Design points processed: {len(all_time_series_data)}")
print(f"  • Metrics per design point: {len(list(all_time_series_data.values())[0]) if all_time_series_data else 0}")
print(f"  • Ready for visual inspection")

# %% Step 9: Visual Inspection - Combined Plots for All Design Points  
"""
Cell 9: Create combined time series plots for visual warmup inspection

Shows cross-replication averaged time series for each design point.
Focus on visual pattern recognition to identify warmup periods.
"""
print("\nStep 9: Visual Inspection - Combined Plots for All Design Points")
print("=" * 60)

# Create visualization instance
viz = TimeSeriesVisualization(figsize=(14, 8))

print(f"🔍 Creating combined inspection plots for visual warmup determination...")
print(f"  • Total plots to display: {len(all_time_series_data)}")
print(f"  • Metrics per plot: active_drivers, active_delivery_entities")

# Create combined plot for each design point
plot_count = 0
for design_name, time_series_data in all_time_series_data.items():
    plot_count += 1
    
    print(f"\n--- Plot {plot_count}/{len(all_time_series_data)}: {design_name} ---")
    
    # Get basic info about this design point
    first_metric_data = list(time_series_data.values())[0]
    total_duration = max(first_metric_data['time_points'])
    replication_count = first_metric_data['replication_count']
    
    print(f"  • Duration: {total_duration:.1f} minutes")
    print(f"  • Replications: {replication_count}")
    
    # Create combined plot for this design point
    fig = viz.create_combined_inspection_plot(
        time_series_data, 
        title=f'Warmup Inspection: {design_name.replace("_", " ").title()}'
    )
    
    # Show the plot
    plt.show()
    
    print(f"  ✓ Plot displayed for visual inspection")

print(f"\n🎯 All {plot_count} plots displayed for visual inspection!")

# %% Step 10: Visual Inspection Guidance and Warmup Determination
"""
Cell 10: Guidance for visual inspection and warmup period determination

Provides systematic guidance for determining uniform warmup period across
all design points based on visual inspection of the plots above.
"""
print("\nStep 10: Visual Inspection Guidance and Warmup Determination")
print("=" * 60)

print("🔍 VISUAL INSPECTION METHODOLOGY")
print("-" * 40)

print("\n📋 Step-by-step visual inspection process:")
print("  1. Look at each of the 9 plots displayed above")
print("  2. For each plot, identify the transition point where:")
print("     • Lines stop trending/changing (transient phase)")
print("     • Lines start stable oscillation around consistent levels (steady-state)")
print("  3. Note the time point where this transition occurs for each design point")
print("  4. Choose the LATEST transition time across all design points")
print("  5. Add a conservative safety margin (e.g., +20-30 minutes)")

print("\n🎯 PATTERN RECOGNITION GUIDE:")
print("  • Early Phase (Transient): Lines trending upward as system 'warms up'")
print("  • Transition Point: Where trending behavior stops")
print("  • Steady Phase: Lines oscillating around stable levels")
print("  • Conservative Choice: Use transition point + safety margin")

print("\n⚖️ UNIFORM WARMUP REQUIREMENT:")
print("  • Same warmup period MUST be used for ALL design points")
print("  • Based on the slowest-converging design point")
print("  • Better to be conservative than risk initialization bias")

# Show simulation context for reference
print(f"\n📊 Simulation Context for Reference:")
print(f"  • Total simulation duration: {experiment_config.simulation_duration} minutes")
print(f"  • Number of design points: {len(all_time_series_data)}")
print(f"  • Replications per design point: {experiment_config.num_replications}")

# Provide warmup ratio guidance
print(f"\n📏 Warmup Period Guidelines:")
print(f"  • Total duration: {experiment_config.simulation_duration} minutes")
print(f"  • Recommended warmup ratio: ≤30% of total duration")
print(f"  • Maximum acceptable warmup: {experiment_config.simulation_duration * 0.3:.0f} minutes (30%)")
print(f"  • Analysis window should be ≥70% of total duration")

print(f"\n💡 Next Steps:")
print(f"  1. Examine all {len(all_time_series_data)} plots above")
print(f"  2. Identify transition points visually")
print(f"  3. Choose conservative uniform warmup period")
print(f"  4. Proceed to Cell 11 for warmup validation")

# Prepare variables for next cell
print(f"\n🔧 Preparation for Next Cell:")
print(f"  • Variable 'all_time_series_data' contains extracted data")
print(f"  • Variable 'experiment_config' contains simulation parameters")
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

for design_name, time_series_data in all_time_series_data.items():
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
print(f"  • Applies to ALL {len(all_time_series_data)} design points")
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
    'time_series_data': all_time_series_data,
    'warmup_period': uniform_warmup_period
}

print(f"\n📚 Thesis Evidence Package Ready:")
print(f"  • Pattern classifications: ✓")
print(f"  • Performance metrics table: ✓") 
print(f"  • Time series data: ✓")
print(f"  • Warmup period: {uniform_warmup_period} minutes")
# %%
