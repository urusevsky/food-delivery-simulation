# pairing_enabled_study.py
"""
Pairing Enabled Study

Research Question: How does enabling pairing affects system performance across different load ratios and absolute scales(validation pairs)
versus no pairing ?
Experimental Design:
- target_load_ratios = [2.0, 3.5, 5.0, 7.0] with validation pairs
- Baseline: (1.0, LR) vs 2x Baseline: (2.0, 2×LR) 
- Pairing Thresholds: restaurants=4.0km, customers=3.0km

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
print("ENHANCED PAIRING THRESHOLDS STUDY: VALIDATION PAIRS + SERVICE RELIABILITY")
print("="*80)
print("Research Focus: Pairing × Load Ratio × Absolute Scale Interaction Effects")

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

# %% Step 4: Enhanced Design Points with Validation Pairs
print("\n" + "="*50)
print("ENHANCED DESIGN POINTS: VALIDATION PAIRS + PAIRING")
print("="*50)

scoring_config = ScoringConfig()

# Pairing thresholds from exploratory study
restaurants_threshold = 4.0  # km
customers_threshold = 3.0    # km

print(f"Pairing Thresholds (from exploratory study):")
print(f"  • restaurants_proximity_threshold: {restaurants_threshold} km")
print(f"  • customers_proximity_threshold: {customers_threshold} km")

# Base operational parameters
base_params = {
    'mean_service_duration': 100,
    'service_duration_std_dev': 60,
    'min_service_duration': 30,
    'max_service_duration': 200,
    'immediate_assignment_threshold': 100,  # All periodic assignment
    'periodic_interval': 3.0
}

# Pairing configuration
pairing_params = {
    'pairing_enabled': True,
    'restaurants_proximity_threshold': restaurants_threshold,
    'customers_proximity_threshold': customers_threshold
}

# No pairing configuration
no_pairing_params = {
    'pairing_enabled': False,
    'restaurants_proximity_threshold': None,
    'customers_proximity_threshold': None
}

# Enhanced design with validation pairs
target_load_ratios = [2.0, 3.5, 5.0, 7.0]

print(f"\nEnhanced Design Pattern:")
print(f"Load Ratios: {target_load_ratios}")
print(f"For each load ratio: Baseline + 2x Baseline × (Pairing + No Pairing)")

design_points = {}

for load_ratio in target_load_ratios:
    
    # === Baseline Interval Design Points ===
    
    # Baseline + Pairing
    baseline_pairing_name = f"load_ratio_{load_ratio:.1f}_baseline_pairing"
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
    
    # Baseline + No Pairing
    baseline_no_pairing_name = f"load_ratio_{load_ratio:.1f}_baseline_no_pairing"
    design_points[baseline_no_pairing_name] = DesignPoint(
        infrastructure=infrastructure,
        operational_config=OperationalConfig(
            mean_order_inter_arrival_time=1.0,
            mean_driver_inter_arrival_time=load_ratio,
            **base_params,
            **no_pairing_params
        ),
        scoring_config=scoring_config,
        name=baseline_no_pairing_name
    )
    
    # === 2x Baseline Design Points ===
    
    # 2x Baseline + Pairing
    double_baseline_pairing_name = f"load_ratio_{load_ratio:.1f}_2x_baseline_pairing"
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
    
    # 2x Baseline + No Pairing
    double_baseline_no_pairing_name = f"load_ratio_{load_ratio:.1f}_2x_baseline_no_pairing"
    design_points[double_baseline_no_pairing_name] = DesignPoint(
        infrastructure=infrastructure,
        operational_config=OperationalConfig(
            mean_order_inter_arrival_time=2.0,
            mean_driver_inter_arrival_time=2.0 * load_ratio,
            **base_params,
            **no_pairing_params
        ),
        scoring_config=scoring_config,
        name=double_baseline_no_pairing_name
    )
    
    print(f"  ✓ Load Ratio {load_ratio:.1f}:")
    print(f"    • Baseline (1.0, {load_ratio:.1f}): Pairing + No Pairing")
    print(f"    • 2x Baseline (2.0, {2.0*load_ratio:.1f}): Pairing + No Pairing")

print(f"\n✓ Created {len(design_points)} enhanced design points")
print(f"✓ Design enables analysis of: Load Ratio × Absolute Scale × Pairing interactions")

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

# %% Step 6: Execute Enhanced Study
print("\n" + "="*50)
print("ENHANCED EXECUTION")
print("="*50)

runner = ExperimentalRunner()
print("✓ ExperimentalRunner initialized")

print(f"\nExecuting enhanced pairing study with validation pairs...")
print("Focus: How does pairing interact with load ratio AND absolute scale?")
study_results = runner.run_experimental_study(design_points, experiment_config)

print(f"\n✅ ENHANCED PAIRING STUDY COMPLETE!")
print(f"✓ Design points executed: {len(study_results)}")
print(f"✓ Ready for enhanced analysis with service reliability metrics")

# %% Step 7: Warmup Period (From Previous Study)
print("\n" + "="*50)
print("WARMUP PERIOD (FROM PREVIOUS STUDY)")
print("="*50)

# Use verified warmup from previous load ratio study
uniform_warmup_period = 500  # Verified by visual inspection in previous study

print(f"✓ Using verified warmup period: {uniform_warmup_period} minutes")
print(f"✓ Based on visual inspection from load_ratio_driven_supply_demand_study.py")
print(f"✓ Streamlined approach - no warmup detection needed")

# %% Step 8: Enhanced Performance Metrics Analysis
print("\n" + "="*50)
print("ENHANCED PERFORMANCE METRICS WITH SERVICE RELIABILITY")
print("="*50)

from delivery_sim.analysis_pipeline.pipeline_coordinator import analyze_single_configuration

print(f"Calculating enhanced metrics including within-replication variability...")
print(f"Focus: Pairing effects on mean performance AND service reliability")

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

print(f"\n✓ Enhanced metrics calculation complete")

# %% Step 9: Enhanced Evidence Table with Service Reliability
print("\n" + "="*50)
print("ENHANCED EVIDENCE TABLE: PAIRING + SERVICE RELIABILITY")
print("="*50)

import pandas as pd

print("Creating enhanced evidence table with service reliability analysis...")

# Extract enhanced performance metrics
table_data = []

for design_name, result in metrics_results.items():
    if result['status'] != 'success':
        continue
    
    analysis = result['analysis']
    
    try:
        # Parse design name for load ratio, interval type, and pairing status
        name_parts = design_name.split('_')
        load_ratio = float(name_parts[2])
        
        interval_type = "2x Baseline" if "2x" in design_name else "Baseline"
        pairing_enabled = "pairing" in design_name and "no_pairing" not in design_name
        pairing_status = "Pairing" if pairing_enabled else "No Pairing"
        
        # Extract assignment time metrics (mean and std)
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
        pairing_formatted = f"{pairing_effectiveness:.1%}" if pairing_effectiveness else "0.0%" if not pairing_enabled else "N/A"
        
        table_data.append({
            'Load Ratio': f"{load_ratio:.1f}",
            'Interval Type': interval_type,
            'Pairing Status': pairing_status,
            'Pairing Effectiveness': pairing_formatted,
            'Mean Assignment Time': assignment_time_mean_ci,
            'Service Reliability (Std)': assignment_time_std_ci,
            'Completion Rate': completion_formatted,
            'Load Ratio Value': load_ratio,
            'Pairing Enabled': pairing_enabled,
            'Assignment Time Value': assignment_time_mean if assignment_time_mean else 999,
            'Service Reliability Value': assignment_time_std_mean if assignment_time_std_mean else 0,
            'Pairing Effectiveness Value': pairing_effectiveness if pairing_effectiveness else 0.0
        })
        
    except Exception as e:
        print(f"  ⚠️  Error extracting metrics for {design_name}: {str(e)}")

# Create and display enhanced evidence table
if table_data:
    df = pd.DataFrame(table_data)
    df_display = df.sort_values(['Load Ratio Value', 'Interval Type', 'Pairing Enabled'])[
        ['Load Ratio', 'Interval Type', 'Pairing Status', 'Pairing Effectiveness', 'Mean Assignment Time', 'Service Reliability (Std)', 'Completion Rate']
    ]
    
    print("\n🎯 ENHANCED PAIRING EFFECTIVENESS + SERVICE RELIABILITY EVIDENCE TABLE")
    print("="*140)
    print(df_display.to_string(index=False))
    
    print(f"\n📊 ENHANCED INTERACTION ANALYSIS:")
    print(f"Research Questions:")
    print(f"1. How does absolute scale interact with pairing effectiveness?")
    print(f"2. Are pairing benefits robust across operational intensities?")
    
    # Enhanced analysis by load ratio and interval type
    print(f"\n🔬 Enhanced Load Ratio × Scale × Pairing Analysis:")
    
    for load_ratio in sorted(df['Load Ratio Value'].unique()):
        print(f"\n  Load Ratio {load_ratio:.1f}:")
        
        load_ratio_subset = df[df['Load Ratio Value'] == load_ratio]
        
        for interval_type in ['Baseline', '2x Baseline']:
            interval_subset = load_ratio_subset[load_ratio_subset['Interval Type'] == interval_type]
            
            if len(interval_subset) == 2:  # Should have both pairing and no pairing
                no_pairing_row = interval_subset[~interval_subset['Pairing Enabled']].iloc[0]
                pairing_row = interval_subset[interval_subset['Pairing Enabled']].iloc[0]
                
                pairing_effectiveness = pairing_row['Pairing Effectiveness Value']
                
                # Performance comparison
                assignment_time_no_pairing = no_pairing_row['Assignment Time Value']
                assignment_time_pairing = pairing_row['Assignment Time Value']
                assignment_time_change = assignment_time_pairing - assignment_time_no_pairing
                
                # Service reliability comparison
                reliability_no_pairing = no_pairing_row['Service Reliability Value']
                reliability_pairing = pairing_row['Service Reliability Value']
                reliability_change = reliability_pairing - reliability_no_pairing
                
                print(f"    {interval_type}:")
                print(f"      • Pairing Effectiveness: {pairing_effectiveness:.1%}")
                print(f"      • Assignment Time Impact: {assignment_time_change:+.1f} min")
                print(f"      • Service Reliability Impact: {reliability_change:+.1f} std")
    
    print(f"\n📋 KEY ENHANCED INSIGHTS:")
    print(f"• Do validation pairs show consistent pairing effectiveness?")
    print(f"• Are there scale-dependent pairing benefits?")
    print(f"• Which combination shows most promising results?")
    

else:
    print("⚠️  No valid data available for enhanced evidence table")

print(f"\n" + "="*80)
print("ENHANCED PAIRING STUDY COMPLETE")
print("="*80)
print("✓ Research Questions: Pairing × Load Ratio × Absolute Scale interactions")
print("✓ Method: Validation pairs")
print("✓ Metrics: Pairing effectiveness + Mean performance + Service variability")
print("✓ Next: Threshold sensitivity analysis")
# %%