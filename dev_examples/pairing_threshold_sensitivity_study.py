# pairing_threshold_sensitivity_study.py
"""
Threshold Sensitivity Study: Validation Pairs Across Pairing Threshold Configurations

Research Question: How do different pairing threshold configurations affect 
pairing effectiveness and system performance across various load ratios and absolute scales?

Threshold Sensitivity Design:
- Threshold Configurations: Conservative (2.0, 1.5), Moderate (4.0, 3.0), Liberal (6.0, 4.5)
- Load Ratios: [2.0, 3.5, 5.0, 7.0] with validation pairs
- Baseline: (1.0, LR) vs 2x Baseline: (2.0, 2×LR) for each threshold × load ratio combination
- Pairing Only: Focus on pairing-enabled configurations to understand threshold sensitivity

This reveals:
1. How threshold liberality affects pairing effectiveness across regimes
2. Whether threshold effects are consistent across different load ratios  
3. Scale dependency of threshold effectiveness
4. Optimal threshold selection for different operational contexts
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
print("THRESHOLD SENSITIVITY STUDY: PAIRING THRESHOLD CONFIGURATIONS")
print("="*80)
print("Research Focus: Threshold × Load Ratio × Absolute Scale Interaction Effects")

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

# %% Step 3: Infrastructure Setup (Consistent with Previous Studies)
print("\n" + "="*50)
print("INFRASTRUCTURE SETUP")
print("="*50)

# Same infrastructure as previous studies for consistency
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
print(f"✓ Consistent with previous load ratio and exploratory studies")

# %% Step 4: Threshold Sensitivity Design Points with Validation Pairs
print("\n" + "="*50)
print("THRESHOLD SENSITIVITY DESIGN POINTS: VALIDATION PAIRS ACROSS THRESHOLDS")
print("="*50)

scoring_config = ScoringConfig()

# Threshold configurations for sensitivity analysis
threshold_configurations = {
    'conservative': {
        'restaurants_proximity_threshold': 2.0,  # km
        'customers_proximity_threshold': 1.5     # km
    },
    'moderate': {
        'restaurants_proximity_threshold': 4.0,  # km (baseline from previous studies)
        'customers_proximity_threshold': 3.0     # km
    },
    'liberal': {
        'restaurants_proximity_threshold': 6.0,  # km
        'customers_proximity_threshold': 4.5     # km
    }
}

print(f"Threshold Configurations for Sensitivity Analysis:")
for config_name, thresholds in threshold_configurations.items():
    print(f"  • {config_name.capitalize()}: restaurants={thresholds['restaurants_proximity_threshold']}km, customers={thresholds['customers_proximity_threshold']}km")

# Base operational parameters
base_params = {
    'mean_service_duration': 100,
    'service_duration_std_dev': 60,
    'min_service_duration': 30,
    'max_service_duration': 200,
    'immediate_assignment_threshold': 100,  # All periodic assignment
    'periodic_interval': 3.0
}

# Target load ratios for sensitivity analysis
target_load_ratios = [2.0, 3.5, 5.0, 7.0]

print(f"\nThreshold Sensitivity Design Pattern:")
print(f"Load Ratios: {target_load_ratios}")
print(f"For each threshold config × load ratio: Baseline + 2x Baseline (pairing enabled)")
print(f"Total design points: {len(threshold_configurations)} thresholds × {len(target_load_ratios)} load ratios × 2 validation pairs = {len(threshold_configurations) * len(target_load_ratios) * 2}")

design_points = {}

for config_name, thresholds in threshold_configurations.items():
    print(f"\n  Creating design points for {config_name.capitalize()} thresholds:")
    
    # Pairing configuration for this threshold set
    pairing_params = {
        'pairing_enabled': True,
        'restaurants_proximity_threshold': thresholds['restaurants_proximity_threshold'],
        'customers_proximity_threshold': thresholds['customers_proximity_threshold']
    }
    
    for load_ratio in target_load_ratios:
        
        # === Baseline Interval Design Points ===
        
        # Baseline + Pairing
        baseline_pairing_name = f"threshold_{config_name}_load_ratio_{load_ratio:.1f}_baseline"
        design_points[baseline_pairing_name] = DesignPoint(
            infrastructure=infrastructure,
            operational_config=OperationalConfig(
                mean_order_inter_arrival_time=1.0,
                mean_driver_inter_arrival_time=load_ratio,
                **base_params,
                **pairing_params
            ),
            scoring_config=scoring_config,
            name=baseline_pairing_name
        )
        
        # === 2x Baseline Design Points ===
        
        # 2x Baseline + Pairing
        double_baseline_pairing_name = f"threshold_{config_name}_load_ratio_{load_ratio:.1f}_2x_baseline"
        design_points[double_baseline_pairing_name] = DesignPoint(
            infrastructure=infrastructure,
            operational_config=OperationalConfig(
                mean_order_inter_arrival_time=2.0,
                mean_driver_inter_arrival_time=2.0 * load_ratio,
                **base_params,
                **pairing_params
            ),
            scoring_config=scoring_config,
            name=double_baseline_pairing_name
        )
        
        print(f"    ✓ Load Ratio {load_ratio:.1f}: Baseline + 2x Baseline")

print(f"\n✓ Created {len(design_points)} threshold sensitivity design points")
print(f"✓ Design enables analysis of: Threshold Configuration × Load Ratio × Absolute Scale interactions")

# %% Step 5: Experiment Configuration (Consistent with Previous Studies)
print("\n" + "="*50)
print("EXPERIMENT CONFIGURATION")
print("="*50)

experiment_config = ExperimentConfig(
    simulation_duration=2000,  # Consistent with previous studies
    num_replications=5,        # Consistent with previous studies
    master_seed=42
)

print(f"✓ Duration: {experiment_config.simulation_duration} minutes (consistent)")
print(f"✓ Replications: {experiment_config.num_replications} (consistent)")
print(f"✓ Total simulation runs: {len(design_points)} × {experiment_config.num_replications} = {len(design_points) * experiment_config.num_replications}")

# %% Step 6: Execute Threshold Sensitivity Study
print("\n" + "="*50)
print("THRESHOLD SENSITIVITY EXECUTION")
print("="*50)

runner = ExperimentalRunner()
print("✓ ExperimentalRunner initialized")

print(f"\nExecuting threshold sensitivity study with validation pairs...")
print("Focus: How do different pairing thresholds affect effectiveness across load ratios and scales?")
study_results = runner.run_experimental_study(design_points, experiment_config)

print(f"\n✅ THRESHOLD SENSITIVITY STUDY COMPLETE!")
print(f"✓ All {len(design_points)} design points executed")
print(f"✓ Results contain threshold × load ratio × scale interaction data")

# %% Step 7: Warmup Period (From Previous Study)
print("\n" + "="*50)
print("WARMUP PERIOD (FROM PREVIOUS STUDY)")
print("="*50)

# Use verified warmup from previous load ratio study
uniform_warmup_period = 500  # Verified by visual inspection in previous study

print(f"✓ Using verified warmup period: {uniform_warmup_period} minutes")
print(f"✓ Based on visual inspection from load_ratio_driven_supply_demand_study.py")
print(f"✓ Streamlined approach - no warmup detection needed")

# %% Step 8: Threshold Sensitivity Performance Metrics Analysis
print("\n" + "="*50)
print("THRESHOLD SENSITIVITY PERFORMANCE METRICS WITH SERVICE RELIABILITY")
print("="*50)

from delivery_sim.analysis_pipeline.pipeline_coordinator import analyze_single_configuration

print(f"Calculating threshold sensitivity metrics including within-replication variability...")
print(f"Focus: How pairing threshold configurations affect effectiveness across load ratios and scales")

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

print(f"\n✓ Threshold sensitivity metrics calculation complete")

# %% Step 9: Threshold Sensitivity Evidence Table with Within-Replication Variability
print("\n" + "="*50)
print("THRESHOLD SENSITIVITY HYPOTHESIS VALIDATION: EVIDENCE TABLE")
print("="*50)

import pandas as pd

print("Creating threshold sensitivity evidence table with within-replication variability...")

# Extract performance metrics including within-replication std
table_data = []

for design_name, result in metrics_results.items():
    if result['status'] != 'success':
        continue
    
    analysis = result['analysis']
    
    try:
        # Parse threshold configuration, load ratio, and interval type from design name
        parts = design_name.split('_')
        threshold_config = parts[1]  # conservative, moderate, liberal
        load_ratio = float(parts[4])  # extract load ratio value
        interval_type = "Baseline" if "2x" not in design_name else "2x Baseline"
        
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
        
        # Within-replication standard deviation (SERVICE RELIABILITY)
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
        
        # Extract pairing effectiveness
        pairing_effectiveness_data = system_metrics.get('pairing_effectiveness', {})
        pairing_effectiveness = pairing_effectiveness_data.get('point_estimate') if pairing_effectiveness_data else None
        pairing_formatted = f"{pairing_effectiveness:.1%}" if pairing_effectiveness else "0.0%"
        
        table_data.append({
            'Threshold Config': threshold_config.capitalize(),
            'Load Ratio': f"{load_ratio:.1f}",
            'Interval Type': interval_type,
            'Pairing Effectiveness': pairing_formatted,
            'Mean Assignment Time': assignment_time_mean_ci,
            'Service Reliability (Std)': assignment_time_std_ci,
            'Completion Rate': completion_formatted,
            'Threshold Config Value': threshold_config,
            'Load Ratio Value': load_ratio,
            'Assignment Time Value': assignment_time_mean if assignment_time_mean else 999,
            'Service Reliability Value': assignment_time_std_mean if assignment_time_std_mean else 0,
            'Pairing Effectiveness Value': pairing_effectiveness if pairing_effectiveness else 0.0
        })
        
    except Exception as e:
        print(f"  ⚠️  Error extracting metrics for {design_name}: {str(e)}")

# Create and display threshold sensitivity evidence table
if table_data:
    df = pd.DataFrame(table_data)
    df_display = df.sort_values(['Load Ratio Value', 'Threshold Config Value', 'Interval Type'])[
        ['Threshold Config', 'Load Ratio', 'Interval Type', 'Pairing Effectiveness', 'Mean Assignment Time', 'Service Reliability (Std)', 'Completion Rate']
    ]
    
    print("\n🎯 THRESHOLD SENSITIVITY EVIDENCE TABLE")
    print("="*140)
    print(df_display.to_string(index=False))
    
    print(f"\n📊 THRESHOLD SENSITIVITY ANALYSIS:")
    print(f"Research Questions:")
    print(f"1. How does threshold liberality affect pairing effectiveness across regimes?")
    print(f"2. Are threshold effects consistent across different load ratios?")
    print(f"3. Do validation pairs show robust threshold sensitivity patterns?")
    
    # Enhanced analysis by load ratio and threshold configuration
    print(f"\n🔬 Load Ratio × Threshold Configuration Analysis:")
    
    for load_ratio in sorted(df['Load Ratio Value'].unique()):
        print(f"\n  Load Ratio {load_ratio:.1f}:")
        
        load_ratio_subset = df[df['Load Ratio Value'] == load_ratio]
        
        for threshold_config in ['conservative', 'moderate', 'liberal']:
            threshold_subset = load_ratio_subset[load_ratio_subset['Threshold Config Value'] == threshold_config]
            
            if len(threshold_subset) == 2:  # Should have both baseline and 2x baseline
                baseline_row = threshold_subset[threshold_subset['Interval Type'] == 'Baseline'].iloc[0]
                double_baseline_row = threshold_subset[threshold_subset['Interval Type'] == '2x Baseline'].iloc[0]
                
                baseline_effectiveness = baseline_row['Pairing Effectiveness Value']
                double_baseline_effectiveness = double_baseline_row['Pairing Effectiveness Value']
                
                baseline_assignment_time = baseline_row['Assignment Time Value']
                double_baseline_assignment_time = double_baseline_row['Assignment Time Value']
                
                print(f"    {threshold_config.capitalize()}:")
                print(f"      • Baseline Effectiveness: {baseline_effectiveness:.1%}, Assignment Time: {baseline_assignment_time:.1f}min")
                print(f"      • 2x Baseline Effectiveness: {double_baseline_effectiveness:.1%}, Assignment Time: {double_baseline_assignment_time:.1f}min")
    
    print(f"\n📋 KEY THRESHOLD SENSITIVITY INSIGHTS:")
    print(f"• Do liberal thresholds consistently improve pairing effectiveness?")
    print(f"• Are threshold effects regime-dependent (different across load ratios)?")
    print(f"• Do validation pairs show consistent threshold sensitivity?")
    print(f"• Which threshold configuration shows optimal performance balance?")
    
    print(f"\n✅ THRESHOLD SENSITIVITY ANALYSIS COMPLETE!")
    print(f"✓ Threshold configurations tested across load ratios")
    print(f"✓ Validation pairs analyzed for robustness")
    print(f"✓ Threshold × regime interaction effects revealed")
    print(f"✓ Foundation for optimal threshold selection insights")

else:
    print("⚠️  No valid data available for threshold sensitivity table")

print(f"\n" + "="*80)
print("THRESHOLD SENSITIVITY STUDY COMPLETE")
print("="*80)
print("✓ Research Questions: Threshold Configuration × Load Ratio × Absolute Scale interactions")
print("✓ Method: Multiple threshold configs + Validation pairs + Pairing-only focus")
print("✓ Evidence: Threshold-dependent pairing effectiveness + Performance optimization")
print("✓ Next: Transition zone high-resolution mapping based on optimal thresholds")
# %%