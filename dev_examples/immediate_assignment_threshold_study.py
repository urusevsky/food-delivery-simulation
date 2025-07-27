# immediate_assignment_threshold_study.py
"""
Immediate Assignment Threshold Study: Validation Pairs Across Assignment Strategies

Research Question: How do different immediate assignment threshold values affect 
system performance and operational dynamics across various load ratios and absolute scales?

Immediate Assignment Threshold Design:
- Threshold Values: [0, 65, 100] (Pure Greedy, Hybrid, Pure Optimization)
- Load Ratios: [3.5, 5.0, 7.0] with validation pairs
- Baseline: (1.0, LR) vs 2x Baseline: (2.0, 2×LR) for each threshold × load ratio combination
- Pairing Enabled: Fixed moderate pairing thresholds (restaurants=4.0km, customers=3.0km)

This reveals:
1. How immediate vs periodic assignment affects system performance across regimes
2. Whether optimal assignment strategy varies with load ratio
3. Scale dependency of assignment threshold effectiveness
4. Interaction between assignment strategy and pairing effectiveness
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
print("IMMEDIATE ASSIGNMENT THRESHOLD STUDY: ASSIGNMENT STRATEGY INVESTIGATION")
print("="*80)
print("Research Focus: Assignment Threshold × Load Ratio × Absolute Scale Interaction Effects")

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
    delivery_area_size=20,
    num_restaurants=10,
    driver_speed=0.5
)

master_seed = 42
infrastructure = Infrastructure(structural_config, master_seed)
analyzer = InfrastructureAnalyzer(infrastructure)
analysis_results = analyzer.analyze_complete_infrastructure()

print(f"✓ Infrastructure: {infrastructure}")
print(f"✓ Typical distance: {analysis_results['typical_distance']:.3f}km")
print(f"✓ Consistent with previous load ratio and pairing studies")

# %% Step 4: Assignment Threshold Design Points with Validation Pairs
print("\n" + "="*50)
print("ASSIGNMENT THRESHOLD DESIGN POINTS: VALIDATION PAIRS ACROSS THRESHOLDS")
print("="*50)

scoring_config = ScoringConfig()
print(scoring_config)

# Immediate assignment threshold values for strategy analysis
immediate_assignment_thresholds = [0, 65, 100]

print(f"Immediate Assignment Threshold Values for Strategy Analysis:")
for threshold in immediate_assignment_thresholds:
    if threshold == 0:
        strategy_description = "Pure Greedy (all immediate assignment)"
    elif threshold == 100:
        strategy_description = "Pure Optimization (all periodic assignment)"
    else:
        strategy_description = f"Hybrid Strategy (immediate if score > {threshold})"
    print(f"  • {threshold}: {strategy_description}")

# Fixed pairing configuration (moderate thresholds from previous studies)
pairing_params = {
    'pairing_enabled': False,
    'restaurants_proximity_threshold': 4.0,  # km (moderate)
    'customers_proximity_threshold': 3.0     # km (moderate)
}

print(f"\nFixed Pairing Configuration (Moderate Thresholds):")
print(f"  • Restaurants proximity: {pairing_params['restaurants_proximity_threshold']}km")
print(f"  • Customers proximity: {pairing_params['customers_proximity_threshold']}km")

# Base operational parameters (excluding immediate_assignment_threshold which varies)
base_params = {
    'mean_service_duration': 100,
    'service_duration_std_dev': 60,
    'min_service_duration': 30,
    'max_service_duration': 200,
    'periodic_interval': 3.0
}

# Target load ratios for assignment strategy analysis
target_load_ratios = [5.0, 7.0, 8.0]

print(f"\nAssignment Threshold Design Pattern:")
print(f"Load Ratios: {target_load_ratios}")
print(f"For each threshold × load ratio: Baseline + 2x Baseline (pairing enabled)")
print(f"Total design points: {len(immediate_assignment_thresholds)} thresholds × {len(target_load_ratios)} load ratios × 2 validation pairs = {len(immediate_assignment_thresholds) * len(target_load_ratios) * 2}")

design_points = {}

for threshold in immediate_assignment_thresholds:
    print(f"\n  Creating design points for immediate assignment threshold {threshold}:")
    
    for load_ratio in target_load_ratios:
        
        # === Baseline Interval Design Points ===
        
        # Baseline + Assignment Threshold
        baseline_name = f"threshold_{threshold}_load_ratio_{load_ratio:.1f}_baseline"
        design_points[baseline_name] = DesignPoint(
            infrastructure=infrastructure,
            operational_config=OperationalConfig(
                mean_order_inter_arrival_time=1.0,
                mean_driver_inter_arrival_time=load_ratio,
                **base_params,
                **pairing_params,
                immediate_assignment_threshold=threshold
            ),
            scoring_config=scoring_config,
            name=baseline_name
        )
        
        # === 2x Baseline Design Points ===
        
        # 2x Baseline + Assignment Threshold
        double_baseline_name = f"threshold_{threshold}_load_ratio_{load_ratio:.1f}_2x_baseline"
        design_points[double_baseline_name] = DesignPoint(
            infrastructure=infrastructure,
            operational_config=OperationalConfig(
                mean_order_inter_arrival_time=2.0,
                mean_driver_inter_arrival_time=2.0 * load_ratio,
                **base_params,
                **pairing_params,
                immediate_assignment_threshold=threshold
            ),
            scoring_config=scoring_config,
            name=double_baseline_name
        )
        
        print(f"    ✓ Load Ratio {load_ratio:.1f}: Baseline + 2x Baseline")

print(f"\n✓ Created {len(design_points)} assignment threshold design points")
print(f"✓ Design enables analysis of: Assignment Threshold × Load Ratio × Absolute Scale interactions")

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

# %% Step 6: Execute Assignment Threshold Study
print("\n" + "="*50)
print("ASSIGNMENT THRESHOLD EXECUTION")
print("="*50)

runner = ExperimentalRunner()
print("✓ ExperimentalRunner initialized")

print(f"\nExecuting assignment threshold study with validation pairs...")
print("Focus: How do different immediate assignment thresholds affect performance across load ratios and scales?")
study_results = runner.run_experimental_study(design_points, experiment_config)

print(f"\n✅ ASSIGNMENT THRESHOLD STUDY COMPLETE!")
print(f"✓ All {len(design_points)} design points executed")
print(f"✓ Results contain assignment threshold × load ratio × scale interaction data")

# %% Step 7: Warmup Period (From Previous Study)
print("\n" + "="*50)
print("WARMUP PERIOD (FROM PREVIOUS STUDY)")
print("="*50)

# Use verified warmup from previous load ratio study
uniform_warmup_period = 500  # Verified by visual inspection in previous study

print(f"✓ Using verified warmup period: {uniform_warmup_period} minutes")
print(f"✓ Based on visual inspection from load_ratio_driven_supply_demand_study.py")
print(f"✓ Streamlined approach - no warmup detection needed")

# %% Step 8: Assignment Threshold Performance Metrics Analysis
print("\n" + "="*50)
print("ASSIGNMENT THRESHOLD PERFORMANCE METRICS WITH SERVICE RELIABILITY")
print("="*50)

from delivery_sim.analysis_pipeline.pipeline_coordinator import analyze_single_configuration

print(f"Calculating assignment threshold metrics including within-replication variability...")
print(f"Focus: How immediate assignment thresholds affect performance across load ratios and scales")

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
            'error_message': str(e)
        }

print(f"\n✅ ASSIGNMENT THRESHOLD ANALYSIS COMPLETE!")
print(f"✓ Processed {sum(1 for r in metrics_results.values() if r['status'] == 'success')} successful configurations")

# %% Step 9: Assignment Threshold Performance Summary Table
print("\n" + "="*50)
print("ASSIGNMENT THRESHOLD PERFORMANCE SUMMARY TABLE")
print("="*50)

import pandas as pd
import re

def extract_threshold_from_name(name):
    """Extract immediate assignment threshold value from design point name."""
    match = re.search(r'threshold_(\d+)_', name)
    return int(match.group(1)) if match else None

def extract_load_ratio_from_name(name):
    """Extract load ratio from design point name."""
    match = re.search(r'load_ratio_(\d+\.?\d*)_', name)
    return float(match.group(1)) if match else None

def extract_interval_type_from_name(name):
    """Extract interval type from design point name."""
    if name.endswith('_2x_baseline'):
        return '2x Baseline'
    elif name.endswith('_baseline'):
        return 'Baseline'
    else:
        return None

# Create comprehensive assignment threshold performance table
if metrics_results:
    table_data = []
    
    for design_name, result in metrics_results.items():
        if result['status'] == 'success':
            analysis = result['analysis']
            
            # Extract design characteristics
            threshold = extract_threshold_from_name(design_name)
            load_ratio = extract_load_ratio_from_name(design_name)
            interval_type = extract_interval_type_from_name(design_name)
            
            if threshold is not None and load_ratio is not None and interval_type is not None:
                
                # Extract assignment time metrics (following reference script pattern)
                entity_metrics = analysis.get('entity_metrics', {})
                orders_metrics = entity_metrics.get('orders', {})
                
                assignment_time_data = orders_metrics.get('assignment_time', {})
                assignment_time_mean = None
                if assignment_time_data and 'mean' in assignment_time_data:
                    assignment_time_mean = assignment_time_data['mean'].get('point_estimate')
                
                # Extract travel time metrics  
                travel_time_data = orders_metrics.get('travel_time', {})
                travel_time_mean = None
                if travel_time_data and 'mean' in travel_time_data:
                    travel_time_mean = travel_time_data['mean'].get('point_estimate')
                
                # Extract fulfillment time metrics
                fulfillment_time_data = orders_metrics.get('fulfillment_time', {})
                fulfillment_time_mean = None
                if fulfillment_time_data and 'mean' in fulfillment_time_data:
                    fulfillment_time_mean = fulfillment_time_data['mean'].get('point_estimate')
                
                # Extract completion rate (following reference script pattern)
                system_metrics = analysis.get('system_metrics', {})
                completion_rate_data = system_metrics.get('system_completion_rate', {})
                completion_rate = completion_rate_data.get('point_estimate') if completion_rate_data else None
                
                # Extract pairing effectiveness (following reference script pattern)
                pairing_effectiveness_data = system_metrics.get('pairing_effectiveness', {})
                pairing_effectiveness = pairing_effectiveness_data.get('point_estimate') if pairing_effectiveness_data else None
                
                table_data.append({
                    'Threshold Value': threshold,
                    'Threshold': str(threshold),
                    'Load Ratio Value': load_ratio,
                    'Load Ratio': f"{load_ratio:.1f}",
                    'Interval Type': interval_type,
                    'Assignment Time Value': assignment_time_mean if assignment_time_mean else 999,
                    'Assignment Time': f"{assignment_time_mean:.1f}min" if assignment_time_mean else "N/A",
                    'Travel Time Value': travel_time_mean if travel_time_mean else 999,
                    'Travel Time': f"{travel_time_mean:.1f}min" if travel_time_mean else "N/A",
                    'Fulfillment Time Value': fulfillment_time_mean if fulfillment_time_mean else 999,
                    'Fulfillment Time': f"{fulfillment_time_mean:.1f}min" if fulfillment_time_mean else "N/A",
                    'Completion Rate Value': completion_rate if completion_rate else 0,
                    'Completion Rate': f"{completion_rate:.1%}" if completion_rate else "N/A",
                    'Pairing Effectiveness Value': pairing_effectiveness if pairing_effectiveness else 0.0,
                    'Pairing Effectiveness': f"{pairing_effectiveness:.1%}" if pairing_effectiveness else "0.0%",
                    'Design Name': design_name
                })
    
    if table_data:
        df = pd.DataFrame(table_data)
        
        # Sort by threshold, then load ratio, then interval type
        df = df.sort_values(['Threshold Value', 'Load Ratio Value', 'Interval Type'])
        
        # Display assignment threshold performance table
        display_columns = ['Threshold', 'Load Ratio', 'Interval Type', 'Assignment Time', 'Travel Time', 'Fulfillment Time', 'Completion Rate', 'Pairing Effectiveness']
        print(df[display_columns].to_string(index=False))
        
        print(f"\n🔬 Assignment Threshold Research Questions:")
        print(f"1. How does immediate vs periodic assignment affect performance across regimes?")
        print(f"2. Are assignment strategy effects consistent across different load ratios?")
        print(f"3. Do validation pairs show robust assignment threshold patterns?")
        
        # Enhanced analysis by load ratio and assignment threshold
        print(f"\n🔬 Load Ratio × Assignment Threshold Analysis:")
        
        for load_ratio in sorted(df['Load Ratio Value'].unique()):
            print(f"\n  Load Ratio {load_ratio:.1f}:")
            
            load_ratio_subset = df[df['Load Ratio Value'] == load_ratio]
            
            # Use the order for assignment thresholds
            for threshold in [0, 65, 100]:
                threshold_subset = load_ratio_subset[load_ratio_subset['Threshold Value'] == threshold]
                
                if len(threshold_subset) == 2:  # Should have both baseline and 2x baseline
                    baseline_row = threshold_subset[threshold_subset['Interval Type'] == 'Baseline'].iloc[0]
                    double_baseline_row = threshold_subset[threshold_subset['Interval Type'] == '2x Baseline'].iloc[0]
                    
                    baseline_assignment_time = baseline_row['Assignment Time Value']
                    double_baseline_assignment_time = double_baseline_row['Assignment Time Value']
                    
                    baseline_travel_time = baseline_row['Travel Time Value']
                    double_baseline_travel_time = double_baseline_row['Travel Time Value']
                    
                    baseline_fulfillment_time = baseline_row['Fulfillment Time Value']
                    double_baseline_fulfillment_time = double_baseline_row['Fulfillment Time Value']
                    
                    baseline_completion_rate = baseline_row['Completion Rate Value']
                    double_baseline_completion_rate = double_baseline_row['Completion Rate Value']
                    
                    baseline_pairing_effectiveness = baseline_row['Pairing Effectiveness Value']
                    double_baseline_pairing_effectiveness = double_baseline_row['Pairing Effectiveness Value']
                    
                    print(f"    Threshold {threshold}:")
                    print(f"      • Baseline: Assign {baseline_assignment_time:.1f}min, Travel {baseline_travel_time:.1f}min, Fulfill {baseline_fulfillment_time:.1f}min, Complete {baseline_completion_rate:.1%}")
                    print(f"      • 2x Baseline: Assign {double_baseline_assignment_time:.1f}min, Travel {double_baseline_travel_time:.1f}min, Fulfill {double_baseline_fulfillment_time:.1f}min, Complete {double_baseline_completion_rate:.1%}")
        
        print(f"\n📋 KEY ASSIGNMENT THRESHOLD INSIGHTS:")
        print(f"• Does immediate assignment (threshold 0) improve assignment speed at the cost of travel efficiency?")
        print(f"• Which strategy provides better overall fulfillment time for customers?")
        print(f"• Are assignment strategy effects regime-dependent (different across load ratios)?")
        print(f"• Do validation pairs show consistent assignment threshold patterns?")
        print(f"• How do the trade-offs between assignment speed and route optimization balance out?")
        
        print(f"\n✅ ASSIGNMENT THRESHOLD ANALYSIS COMPLETE!")
        print(f"✓ Assignment speed vs route optimization trade-offs analyzed across load ratios")
        print(f"✓ Travel time and fulfillment time metrics included for comprehensive view")
        print(f"✓ Validation pairs analyzed for robustness")
        print(f"✓ Assignment strategy × regime interaction effects revealed")
        print(f"✓ Foundation for optimal assignment strategy insights considering full delivery lifecycle")

    else:
        print("⚠️  No valid data available for assignment threshold table")

print(f"\n" + "="*80)
print("ASSIGNMENT THRESHOLD STUDY COMPLETE")
print("="*80)
print("✓ Research Questions: Assignment speed vs route optimization trade-offs across load ratios")
print("✓ Method: Multiple assignment thresholds + Validation pairs + Fixed pairing enabled")
print("✓ Evidence: Assignment vs travel vs fulfillment time optimization across regimes")
print("✓ Next: Analysis of immediate vs periodic assignment lifecycle trade-offs")
# %%


