# pairing_threshold_sensitivity_study.py
"""
Priority Scoring Weight Strategy Study: Multi-Objective Optimization Across Load Ratios

Research Question: How do different priority scoring weight strategies affect 
system performance across various load ratios and absolute scales?

Weight Strategy Design:
- Weight Configurations: Distance-focused (0.7,0.2,0.1), Throughput-focused (0.2,0.7,0.1), 
  Fairness-focused (0.2,0.2,0.6), Balanced (0.33,0.33,0.34), Efficiency-only (0.5,0.5,0.0), 
  Distance-throughput (0.6,0.4,0.0)
- Load Ratios: [3.0, 5.0, 7.0] covering optimal, efficient, and stressed regimes
- Baseline: (1.0, LR) vs 2x Baseline: (2.0, 2×LR) for each weight strategy × load ratio combination
- Fixed Pairing Thresholds: Moderate (4.0km restaurants, 3.0km customers) to isolate weight effects

This reveals:
1. Load ratio-dependent optimal weight combinations for multi-objective optimization
2. Whether weight strategies show consistent performance across different absolute scales
3. Trade-offs between distance efficiency, throughput optimization, and fairness considerations
4. Context-sensitive priority scoring strategies for different operational regimes
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

# %% Step 4: Define experimental design points
print("🔧 DEFINING EXPERIMENTAL DESIGN POINTS...")

# Fixed pairing thresholds (moderate configuration to isolate weight effects)
FIXED_PAIRING_CONFIG = {
    'restaurants_proximity_threshold': 4.0,  # km
    'customers_proximity_threshold': 3.0     # km
}

# PRIMARY EXPERIMENTAL FACTOR: Weight Strategy Configurations
weight_strategy_configurations = {
    'distance_focused': {
        'weight_distance': 0.7,
        'weight_throughput': 0.2, 
        'weight_fairness': 0.1
    },
    'throughput_focused': {
        'weight_distance': 0.2,
        'weight_throughput': 0.7,
        'weight_fairness': 0.1
    },
    'fairness_focused': {
        'weight_distance': 0.2,
        'weight_throughput': 0.2,
        'weight_fairness': 0.6
    },
    'balanced': {
        'weight_distance': 0.33,
        'weight_throughput': 0.33,
        'weight_fairness': 0.34
    },
    'efficiency_only': {
        'weight_distance': 0.5,
        'weight_throughput': 0.5,
        'weight_fairness': 0.0
    },

}

# Load ratios to test (efficient subset for multi-objective analysis)
target_load_ratios = [3.0, 5.0, 7.0]  # Optimal, Efficient, Stressed regimes

# Create design points
design_points = {}
design_point_counter = 1

for weight_strategy_name, weight_config in weight_strategy_configurations.items():
    for load_ratio in target_load_ratios:
        for interval_type in ['baseline', '2x_baseline']:
            
            # Create operational config with fixed pairing thresholds
            operational_config = OperationalConfig(
                # Interval configuration (same as before)
                mean_order_inter_arrival_time=1.0 if interval_type == 'baseline' else 2.0,
                mean_driver_inter_arrival_time=load_ratio if interval_type == 'baseline' else 2.0 * load_ratio,
                
                # FIXED pairing configuration (moderate thresholds)
                pairing_enabled=True,
                restaurants_proximity_threshold=FIXED_PAIRING_CONFIG['restaurants_proximity_threshold'],
                customers_proximity_threshold=FIXED_PAIRING_CONFIG['customers_proximity_threshold'],
                
                # Driver service configuration (same as before)
                mean_service_duration=120,
                service_duration_std_dev=30,
                min_service_duration=60,
                max_service_duration=240
            )
            
            # Create scoring config with experimental weight strategy
            scoring_config = ScoringConfig(
                # Weight configuration (PRIMARY EXPERIMENTAL FACTOR)
                weight_distance=weight_config['weight_distance'],
                weight_throughput=weight_config['weight_throughput'], 
                weight_fairness=weight_config['weight_fairness'],
                
                # Fixed scoring parameters
                max_distance_ratio_multiplier=2.0,
                max_acceptable_delay=30.0,
                max_orders_per_trip=2,
                typical_distance_samples=1000
            )
            
            # Create design point
            design_point_name = f"{weight_strategy_name}_LR{load_ratio}_{interval_type}"
            design_points[design_point_name] = DesignPoint(
                infrastructure=infrastructure,  # Reuse same infrastructure
                operational_config=operational_config,
                scoring_config=scoring_config,
                name=design_point_name
            )
            
            print(f"  {design_point_counter:2d}. {design_point_name}")
            design_point_counter += 1

print(f"\n📊 EXPERIMENTAL DESIGN SUMMARY:")
print(f"   Weight Strategies: {len(weight_strategy_configurations)} configurations")
print(f"   Load Ratios: {len(target_load_ratios)} ratios {target_load_ratios}")
print(f"   Validation Pairs: 2 per load ratio (baseline + 2x baseline)")
print(f"   Total Design Points: {len(design_points)}")
print(f"   Pairing Thresholds: FIXED at moderate ({FIXED_PAIRING_CONFIG})")
print(f"   Primary Factor: WEIGHT STRATEGY CONFIGURATIONS")

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

# %% Step 9: Weight Strategy Evidence Table with Multi-Objective Performance Analysis
print("\n" + "="*50)
print("WEIGHT STRATEGY OPTIMIZATION HYPOTHESIS VALIDATION: EVIDENCE TABLE")
print("="*50)

import pandas as pd

print("Creating weight strategy evidence table with multi-objective performance analysis...")

# Extract performance metrics including within-replication std
table_data = []

for design_name, result in metrics_results.items():
    if result['status'] != 'success':
        continue
    
    analysis = result['analysis']
    
    try:
        # Parse weight strategy, load ratio, and interval type from design name
        # Expected format: {weight_strategy}_LR{load_ratio}_{interval_type}
        parts = design_name.split('_LR')
        weight_strategy = parts[0]  # distance_focused, throughput_focused, etc.
        
        remainder = parts[1].split('_')
        load_ratio = float(remainder[0])  # extract load ratio value
        interval_type = "Baseline" if "baseline" in remainder[1] and "2x" not in remainder[1] else "2x Baseline"
        
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
            'Weight Strategy': weight_strategy.replace('_', ' ').title(),
            'Load Ratio': f"{load_ratio:.1f}",
            'Interval Type': interval_type,
            'Pairing Effectiveness': pairing_formatted,
            'Mean Assignment Time': assignment_time_mean_ci,
            'Service Reliability (Std)': assignment_time_std_ci,
            'Completion Rate': completion_formatted,
            'Weight Strategy Value': weight_strategy,
            'Load Ratio Value': load_ratio,
            'Assignment Time Value': assignment_time_mean if assignment_time_mean else 999,
            'Service Reliability Value': assignment_time_std_mean if assignment_time_std_mean else 0,
            'Pairing Effectiveness Value': pairing_effectiveness if pairing_effectiveness else 0.0
        })
        
    except Exception as e:
        print(f"  ⚠️  Error extracting metrics for {design_name}: {str(e)}")

# Create and display weight strategy evidence table
if table_data:
    df = pd.DataFrame(table_data)
    
    # Define custom order for weight strategies
    weight_order = {
        'distance_focused': 0, 
        'throughput_focused': 1, 
        'fairness_focused': 2, 
        'balanced': 3, 
        'efficiency_only': 4, 
        'distance_throughput': 5
    }
    df['Weight Strategy Sort Order'] = df['Weight Strategy Value'].map(weight_order)
    
    # Sort with custom weight strategy order
    df_display = df.sort_values(['Load Ratio Value', 'Weight Strategy Sort Order', 'Interval Type'])[
        ['Weight Strategy', 'Load Ratio', 'Interval Type', 'Pairing Effectiveness', 'Mean Assignment Time', 'Service Reliability (Std)', 'Completion Rate']
    ]
    
    print("\n🎯 WEIGHT STRATEGY OPTIMIZATION EVIDENCE TABLE")
    print("="*140)
    print(df_display.to_string(index=False))
    
    print(f"\n📊 WEIGHT STRATEGY ANALYSIS:")
    print(f"Research Questions:")
    print(f"1. How do different priority scoring weights affect system performance across regimes?")
    print(f"2. Are optimal weight strategies consistent across different load ratios?")
    print(f"3. Do validation pairs show robust weight strategy performance patterns?")
    print(f"4. Which multi-objective trade-offs provide optimal system performance?")
    
    # Enhanced analysis by load ratio and weight strategy
    print(f"\n🔬 Load Ratio × Weight Strategy Analysis:")
    
    for load_ratio in sorted(df['Load Ratio Value'].unique()):
        print(f"\n  Load Ratio {load_ratio:.1f}:")
        
        load_ratio_subset = df[df['Load Ratio Value'] == load_ratio]
        
        # Find optimal weight strategy for this load ratio (baseline only for clarity)
        baseline_subset = load_ratio_subset[load_ratio_subset['Interval Type'] == 'Baseline']
        
        if len(baseline_subset) > 0:
            # Sort by assignment time (lower is better)
            optimal_baseline = baseline_subset.loc[baseline_subset['Assignment Time Value'].idxmin()]
            
            print(f"    Optimal Strategy (Baseline): {optimal_baseline['Weight Strategy']}")
            print(f"      • Assignment Time: {optimal_baseline['Assignment Time Value']:.1f}min")
            print(f"      • Pairing Effectiveness: {optimal_baseline['Pairing Effectiveness Value']:.1%}")
            print(f"      • Completion Rate: {optimal_baseline['Completion Rate']}")
            
            # Show all strategies for comparison
            print(f"    All Strategies (Baseline):")
            for _, row in baseline_subset.sort_values('Assignment Time Value').iterrows():
                print(f"      • {row['Weight Strategy']}: {row['Assignment Time Value']:.1f}min "
                      f"(Effectiveness: {row['Pairing Effectiveness Value']:.1%})")
    
    print(f"\n📋 KEY WEIGHT STRATEGY INSIGHTS:")
    print(f"• Which weight strategies perform optimally across different load ratios?")
    print(f"• Are distance-focused strategies better in high-stress regimes?")
    print(f"• Do throughput-focused strategies provide capacity benefits?")
    print(f"• How does fairness consideration affect overall system performance?")
    print(f"• Are balanced strategies consistently competitive across regimes?")
    
    print(f"\n✅ WEIGHT STRATEGY ANALYSIS COMPLETE!")
    print(f"✓ Weight strategy configurations tested across load ratios")
    print(f"✓ Validation pairs analyzed for robustness")
    print(f"✓ Weight strategy × regime interaction effects revealed")
    print(f"✓ Multi-objective optimization insights for priority scoring")

else:
    print("⚠️  No valid data available for weight strategy table")

print(f"\n" + "="*80)
print("WEIGHT STRATEGY OPTIMIZATION STUDY COMPLETE")
print("="*80)
print("✓ Research Questions: Weight Strategy × Load Ratio × Absolute Scale interactions")
print("✓ Method: Multiple weight configs + Validation pairs + Fixed moderate pairing thresholds")
print("✓ Evidence: Multi-objective trade-offs + Context-dependent optimization strategies")
print("✓ Next: Dynamic weight adjustment algorithms based on optimal strategies")
# %%