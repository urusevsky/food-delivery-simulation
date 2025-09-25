# load_ratio_driven_supply_demand_study.py
"""
Load Ratio-Driven Supply-Demand Study: Systematic Validation Design

Research Hypothesis: Load ratio determines operational regime characteristics,
with systematic validation pairs testing robustness across different absolute arrival rates.

Design Pattern: For each load ratio R:
- Baseline Interval: (1.0, R) → "Higher intensity" (1.0 orders/min, 1/R drivers/min)
- 2x Baseline: (2.0, 2R) → "Half intensity" (0.5 orders/min, 1/2R drivers/min)

*Refined Hypothesis after experimentation:
"Operational regime characteristics are determined by the interaction between load ratio and absolute operational intensity,
with driver capacity serving as the primary system bottleneck."
"""
# %% Enable Autoreload (run once per session)
%load_ext autoreload 
%autoreload 2 
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
        "simulation.runner": "INFO", "utils.infrastructure_analyzer": "INFO", 
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
target_load_ratios = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]

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
    master_seed=42,
    collection_interval=1.0
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

# %% Step 7: Time Series Data Processing for Warmup Analysis
print("\n" + "="*50)
print("TIME SERIES DATA PROCESSING FOR WARMUP ANALYSIS")
print("="*50)

from delivery_sim.warmup_analysis.time_series_processing import extract_warmup_time_series
from delivery_sim.warmup_analysis.visualization import WelchMethodVisualization

print("Processing time series data for warmup detection...")

# Simple, direct extraction - no complex data restructuring
all_time_series_data = extract_warmup_time_series(
    study_results=study_results,
    design_points=design_points,
    metrics=['active_drivers', 'unassigned_delivery_entities'],
    moving_average_window=100  # Larger window for 2000-minute simulation
)

print(f"✓ Time series processing complete for {len(all_time_series_data)} design points")
print(f"✓ Each design point has {len(next(iter(all_time_series_data.values())))} metrics")
print(f"✓ Ready for warmup analysis visualization")

# %% Step 8: Simplified Warmup Analysis Visualization
print("\n" + "="*50)
print("WARMUP ANALYSIS VISUALIZATION: SIMPLIFIED APPROACH")
print("="*50)

from delivery_sim.warmup_analysis.visualization import WelchMethodVisualization
import matplotlib.pyplot as plt

print("Creating warmup analysis plots using simplified visualization...")

# Initialize visualization object
viz = WelchMethodVisualization(figsize=(16, 10))

# Group design points by load ratio for organized display
load_ratio_groups = {}
for design_name in all_time_series_data.keys():
    # Extract load ratio from design name (e.g., "load_ratio_3.0_baseline")
    load_ratio_str = design_name.split('_')[2]  # "3.0"
    load_ratio = float(load_ratio_str)
    
    if load_ratio not in load_ratio_groups:
        load_ratio_groups[load_ratio] = []
    load_ratio_groups[load_ratio].append(design_name)

print(f"✓ Grouped {len(all_time_series_data)} design points by {len(load_ratio_groups)} load ratios")

# Create plots systematically by load ratio
plot_count = 0
for load_ratio in sorted(load_ratio_groups.keys()):
    design_names = load_ratio_groups[load_ratio]
    
    print(f"\n--- Load Ratio {load_ratio:.1f} ---")
    
    # Sort design names for consistent ordering (baseline first, then 2x baseline)
    design_names_sorted = sorted(design_names, key=lambda x: (load_ratio, '2x' in x))
    
    for design_name in design_names_sorted:
        plot_count += 1
        
        # Determine interval type for title
        interval_type = "2x Baseline" if "2x_baseline" in design_name else "Baseline"
        
        # Extract intervals directly from configuration (no conversion needed)
        order_interval = 1.0 if interval_type == "Baseline" else 2.0
        driver_interval = load_ratio if interval_type == "Baseline" else 2.0 * load_ratio
        
        # Create plot title with actual configuration intervals
        plot_title = (f'Load Ratio {load_ratio:.1f} - {interval_type} Interval\n'
                     f'(Order: {order_interval:.1f}min, Driver: {driver_interval:.1f}min)')
        
        print(f"  Creating plot {plot_count}: {interval_type}")
        
        # Use simplified visualization - one function call!
        time_series_data = all_time_series_data[design_name]
        fig = viz.create_warmup_analysis_plot(time_series_data, title=plot_title)
        
        plt.show()
        print(f"    ✓ {design_name} plot displayed")

print(f"\n🎯 WARMUP ANALYSIS COMPLETE!")
print(f"✓ Created {plot_count} warmup analysis plots")
print(f"✓ Organized by {len(load_ratio_groups)} load ratios")
print(f"✓ Each plot shows active drivers (warmup signal) + unassigned entities (regime signal)")

# %% Step 9: Warmup Period Determination
print("\n" + "="*50)
print("WARMUP PERIOD DETERMINATION")
print("="*50)

# Set warmup period based on visual inspection of Step 8 plots
uniform_warmup_period = 500  # UPDATE THIS based on visual inspection

print(f"✓ Warmup period set: {uniform_warmup_period} minutes")
print(f"✓ Based on visual inspection of active drivers oscillation around Little's Law values")
print(f"✓ Analysis window: {experiment_config.simulation_duration - uniform_warmup_period} minutes of post-warmup data")

# %%
# ==================================================================================
# STEP 10: EXPERIMENTAL ANALYSIS USING NEW REDESIGNED PIPELINE
# ==================================================================================

print(f"\n{'='*80}")
print("STEP 10: EXPERIMENTAL ANALYSIS USING NEW REDESIGNED PIPELINE")
print(f"{'='*80}\n")

# Import the new redesigned pipeline
from delivery_sim.analysis_pipeline_redesigned.pipeline_coordinator import ExperimentAnalysisPipeline

# Initialize pipeline focused on order metrics only
pipeline = ExperimentAnalysisPipeline(
    warmup_period=uniform_warmup_period,  # 500 minutes from Step 9
    enabled_metric_types=['order_metrics'],  # Focus on order metrics only
    confidence_level=0.95
)

# Process each design point through the new pipeline
design_analysis_results = {}

print(f"Processing {len(study_results)} design points through redesigned analysis pipeline...")
print(f"Warmup period: {uniform_warmup_period} minutes")
print(f"Confidence level: 95%")
print(f"Focus metric: Order Assignment Time\n")

for i, (design_name, replication_results) in enumerate(study_results.items(), 1):
    print(f"[{i:2d}/{len(study_results)}] Analyzing {design_name}...")
    
    # Simple, direct approach - let it fail with full context
    analysis_result = pipeline.analyze_experiment(replication_results)
    design_analysis_results[design_name] = analysis_result
    
    # Simple success confirmation
    print(f"    ✓ Processed {analysis_result['num_replications']} replications")

print(f"\n✓ Completed analysis for all {len(design_analysis_results)} design points")
print("Analysis results stored in 'design_analysis_results'")

# %%
# ==================================================================================
# STEP 11: EXTRACT AND PRESENT FOCUSED ORDER ASSIGNMENT TIME METRICS
# ==================================================================================

print(f"\n{'='*80}")
print("STEP 11: ORDER ASSIGNMENT TIME STATISTICS EXTRACTION AND PRESENTATION")
print(f"{'='*80}\n")

import re

def extract_load_ratio_and_type(design_name):
    """Extract load ratio and interval type from design point name."""
    pattern = r"load_ratio_(\d+\.?\d*)_(.+)"
    match = re.match(pattern, design_name)
    
    if match:
        load_ratio = float(match.group(1))
        interval_suffix = match.group(2)
        
        if interval_suffix == "baseline":
            interval_type = "Baseline"
        elif interval_suffix == "2x_baseline":
            interval_type = "2x Baseline"
        else:
            interval_type = interval_suffix.replace("_", " ").title()
        
        return load_ratio, interval_type
    else:
        return None, design_name

def format_metric_value(value, decimal_places=2):
    """Format metric value for display."""
    if value is None:
        return "N/A"
    return f"{value:.{decimal_places}f}"

def format_ci_value(point_estimate, ci_bounds, decimal_places=2):
    """Format value with confidence interval."""
    if point_estimate is None:
        return "N/A"
    
    if ci_bounds and ci_bounds[0] is not None and ci_bounds[1] is not None:
        lower, upper = ci_bounds
        margin = (upper - lower) / 2
        return f"{point_estimate:.{decimal_places}f} ± {margin:.{decimal_places}f}"
    else:
        return f"{point_estimate:.{decimal_places}f}"

# Extract metrics from analysis results
assignment_time_results = []

for design_name, analysis_result in design_analysis_results.items():
    try:
        # Navigate to assignment time metrics
        order_metrics = analysis_result['results']['order_metrics']['assignment_time']  # ✅ Removed 'order' layer
        
        # Extract the three statistics of interest (updated names)
        mean_of_means_data = order_metrics['mean_of_means']
        std_of_means_data = order_metrics['std_of_means']  # Changed from variance_of_means
        mean_of_stds_data = order_metrics['mean_of_stds']  # Changed from mean_of_variances
        
        # Extract values
        mean_of_means = mean_of_means_data['point_estimate']
        mean_of_means_ci = mean_of_means_data['confidence_interval']
        std_of_means = std_of_means_data['point_estimate']  # Now directly std, no conversion
        mean_of_stds = mean_of_stds_data['point_estimate']  # Now directly mean of stds
        
        # Parse design point information
        load_ratio, interval_type = extract_load_ratio_and_type(design_name)
        
        # Store results
        assignment_time_results.append({
            'design_name': design_name,
            'load_ratio': load_ratio,
            'interval_type': interval_type,
            'mean_of_means': mean_of_means,
            'mean_of_means_ci': mean_of_means_ci,
            'std_of_means': std_of_means,  # Updated variable name
            'mean_of_stds': mean_of_stds   # Updated variable name
        })
        
    except KeyError as e:
        print(f"⚠ Warning: Could not extract assignment time metrics from {design_name}: {e}")
        # Debug: Show actual structure
        try:
            if 'order_metrics' in analysis_result['results']:
                metric_keys = list(analysis_result['results']['order_metrics'].keys())
                print(f"   Available metrics in order_metrics: {metric_keys}")  # ✅ Updated message
                if 'assignment_time' in metric_keys:
                    stat_keys = list(analysis_result['results']['order_metrics']['assignment_time'].keys())
                    print(f"   Available statistics in 'assignment_time': {stat_keys}")
        except:
            print(f"   Could not inspect structure further")
    except Exception as e:
        print(f"✗ Error processing {design_name}: {e}")

# Sort and display results table
assignment_time_results.sort(key=lambda x: (x['load_ratio'], x['interval_type']))

print("🎯 ORDER ASSIGNMENT TIME: STATISTICS OF STATISTICS ANALYSIS")
print("=" * 95)
print(f"{'Load Ratio':>10} {'Interval Type':>12} {'Mean of Means':>20} {'Std of Means':>15} {'Mean of Stds':>15}")
print("=" * 95)

for result in assignment_time_results:
    load_ratio = format_metric_value(result['load_ratio'], 1) if result['load_ratio'] else "N/A"
    interval_type = result['interval_type'][:12]
    
    mean_of_means_formatted = format_ci_value(
        result['mean_of_means'], 
        result['mean_of_means_ci'], 
        decimal_places=2
    )
    
    std_of_means_formatted = format_metric_value(result['std_of_means'], 2)  # Updated
    mean_of_stds_formatted = format_metric_value(result['mean_of_stds'], 2)  # Updated
    
    print(f"{load_ratio:>10} {interval_type:>12} {mean_of_means_formatted:>20} "
          f"{std_of_means_formatted:>15} {mean_of_stds_formatted:>15}")

print("=" * 95)
print(f"✓ Extracted and displayed metrics from {len(assignment_time_results)} design points")
print("Results stored in 'assignment_time_results' for further analysis")
print("\nColumn Interpretations:")
print("• Mean of Means: Average assignment time across replications")  
print("• Std of Means: System consistency between replications (lower = more consistent)")
print("• Mean of Stds: Average volatility within replications (service predictability)")

# %%
