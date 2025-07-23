# pairing_thresholds_exploratory_study.py
"""
Pairing Thresholds Exploratory Study: Load Ratio × Pairing Interaction Effects

Research Question: How do pairing thresholds interact with supply-demand conditions 
to affect both pairing effectiveness and overall system performance?

Exploratory Design:
- Load Ratios: [3.0, 5.0] (stable vs stressed regimes)
- Pairing Thresholds: Single starting pair (restaurants=4.0km, customers=3.0km)
- Validation: Compare pairing enabled vs disabled for each load ratio

This reveals:
1. How pairing effectiveness emerges under different system stress levels
2. Whether pairing helps or hurts performance under stress
3. Baseline understanding before broader threshold exploration
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
print("PAIRING THRESHOLDS EXPLORATORY STUDY")
print("="*80)
print("Research Focus: Load Ratio × Pairing Thresholds Interaction Effects")

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

# %% Step 3: Infrastructure Setup (Reuse from Load Ratio Study)
print("\n" + "="*50)
print("INFRASTRUCTURE SETUP")
print("="*50)

# Same infrastructure as load ratio study for consistency
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

# Enhanced infrastructure analysis for pairing parameter selection
restaurant_analysis = analysis_results.get('restaurant_spatial_analysis', {})
customer_analysis = analysis_results.get('customer_distance_analysis', {})

print(f"\n📊 Enhanced Infrastructure Analysis for Pairing:")
if 'distance_statistics' in restaurant_analysis:
    rest_stats = restaurant_analysis['distance_statistics']
    print(f"  Restaurant Distance Patterns:")
    print(f"    • Min distance: {rest_stats['min']:.2f} km")
    print(f"    • 25th percentile: {rest_stats['p25']:.2f} km")
    print(f"    • Median: {rest_stats['p50']:.2f} km")
    print(f"    • 75th percentile: {rest_stats['p75']:.2f} km")

if 'distance_statistics' in customer_analysis:
    cust_stats = customer_analysis['distance_statistics']
    print(f"  Customer Distance Patterns:")
    print(f"    • 25th percentile: {cust_stats['p25']:.2f} km")
    print(f"    • Median: {cust_stats['p50']:.2f} km")
    print(f"    • 75th percentile: {cust_stats['p75']:.2f} km")

# %% Step 4: Exploratory Design Points Creation
print("\n" + "="*50)
print("EXPLORATORY PAIRING DESIGN POINTS")
print("="*50)

scoring_config = ScoringConfig()

# Starting pairing thresholds for exploration
starting_restaurants_threshold = 4.0  # km - moderate threshold
starting_customers_threshold = 3.0    # km - moderate threshold

print(f"Starting Pairing Thresholds:")
print(f"  • restaurants_proximity_threshold: {starting_restaurants_threshold} km")
print(f"  • customers_proximity_threshold: {starting_customers_threshold} km")

# Base operational parameters
base_params = {
    'mean_service_duration': 100,
    'service_duration_std_dev': 60,
    'min_service_duration': 30,
    'max_service_duration': 200,
    'immediate_assignment_threshold': 100,  # All immediate assignment
    'periodic_interval': 3.0
}

# Pairing configuration
pairing_params = {
    'pairing_enabled': True,
    'restaurants_proximity_threshold': starting_restaurants_threshold,
    'customers_proximity_threshold': starting_customers_threshold
}

# No pairing configuration (baseline)
no_pairing_params = {
    'pairing_enabled': False,
    'restaurants_proximity_threshold': None,
    'customers_proximity_threshold': None
}

# Select subset of load ratios for focused exploration
exploratory_load_ratios = [3.0, 5.0]  # Stable vs Stressed regimes

print(f"\nExploratory Load Ratios: {exploratory_load_ratios}")
print(f"Design: Each load ratio tested with and without pairing for comparison")

# Create design points systematically
design_points = {}

for load_ratio in exploratory_load_ratios:
    # Pairing enabled design point
    pairing_name = f"load_ratio_{load_ratio:.1f}_pairing_enabled"
    design_points[pairing_name] = DesignPoint(
        infrastructure=infrastructure,
        operational_config=OperationalConfig(
            mean_order_inter_arrival_time=1.0,
            mean_driver_inter_arrival_time=load_ratio,
            **base_params,
            **pairing_params
        ),
        scoring_config=scoring_config,
        name=pairing_name
    )
    
    # Pairing disabled design point (baseline comparison)
    no_pairing_name = f"load_ratio_{load_ratio:.1f}_no_pairing"
    design_points[no_pairing_name] = DesignPoint(
        infrastructure=infrastructure,
        operational_config=OperationalConfig(
            mean_order_inter_arrival_time=1.0,
            mean_driver_inter_arrival_time=load_ratio,
            **base_params,
            **no_pairing_params
        ),
        scoring_config=scoring_config,
        name=no_pairing_name
    )
    
    print(f"  ✓ Load Ratio {load_ratio:.1f}: Pairing Enabled + Pairing Disabled")

print(f"\n✓ Created {len(design_points)} exploratory design points")

# %% Step 5: Experiment Configuration
print("\n" + "="*50)
print("EXPLORATORY EXPERIMENT CONFIGURATION")
print("="*50)

experiment_config = ExperimentConfig(
    simulation_duration=2000,  # Shorter for exploratory - can extend later
    num_replications=5,        # Fewer replications for exploratory speed
    master_seed=42
)

print(f"✓ Exploratory duration: {experiment_config.simulation_duration} minutes")
print(f"✓ Replications: {experiment_config.num_replications}")
print(f"✓ Total simulation runs: {len(design_points)} × {experiment_config.num_replications} = {len(design_points) * experiment_config.num_replications}")
print(f"✓ Focus: Understanding pairing effectiveness emergence under different load ratios")

# %% Step 6: Execute Exploratory Study
print("\n" + "="*50)
print("EXPLORATORY EXECUTION")
print("="*50)

runner = ExperimentalRunner()
print("✓ ExperimentalRunner initialized")

print(f"\nExecuting exploratory pairing study...")
print("Focus: How does pairing interact with load ratio to affect system performance?")
study_results = runner.run_experimental_study(design_points, experiment_config)

print(f"\n✅ EXPLORATORY PAIRING STUDY COMPLETE!")
print(f"✓ Design points executed: {len(study_results)}")
print(f"✓ Ready for pairing effectiveness analysis")

# %% Step 7: Warmup Period Determination
print("\n" + "="*50)
print("WARMUP PERIOD DETERMINATION")
print("="*50)

# Use conservative warmup based on previous load ratio study experience
proposed_warmup_period = 500  # Conservative for exploratory

print(f"⚙️  Proposed warmup period: {proposed_warmup_period} minutes")
print(f"✓ Based on previous load ratio study experience")
print(f"✓ Conservative approach for exploratory reliability")

uniform_warmup_period = proposed_warmup_period

# %% Step 8: Enhanced Performance Metrics with Pairing Effectiveness
print("\n" + "="*50)
print("ENHANCED PAIRING EFFECTIVENESS ANALYSIS")
print("="*50)

from delivery_sim.analysis_pipeline.pipeline_coordinator import analyze_single_configuration

print(f"Calculating enhanced metrics with pairing effectiveness...")
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

print(f"\n✓ Enhanced metrics calculation complete")

# %% Step 9: Pairing Effectiveness Evidence Table
print("\n" + "="*50)
print("PAIRING EFFECTIVENESS EVIDENCE TABLE")
print("="*50)

import pandas as pd

print("Creating enhanced evidence table with pairing effectiveness...")

# Extract performance metrics with pairing focus
table_data = []

for design_name, result in metrics_results.items():
    if result['status'] != 'success':
        continue
    
    analysis = result['analysis']
    
    try:
        # Extract load ratio and pairing status from design name
        name_parts = design_name.split('_')
        load_ratio = float(name_parts[2])
        pairing_enabled = "pairing_enabled" in design_name
        pairing_status = "Pairing Enabled" if pairing_enabled else "No Pairing"
        
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
        
        # Extract completion rate
        system_metrics = analysis.get('system_metrics', {})
        completion_rate_data = system_metrics.get('system_completion_rate', {})
        completion_rate = completion_rate_data.get('point_estimate') if completion_rate_data else None
        completion_formatted = f"{completion_rate:.1%}" if completion_rate else "N/A"
        
        # Extract pairing effectiveness (NEW!)
        pairing_effectiveness_data = system_metrics.get('pairing_effectiveness', {})
        pairing_effectiveness = pairing_effectiveness_data.get('point_estimate') if pairing_effectiveness_data else None
        pairing_formatted = f"{pairing_effectiveness:.1%}" if pairing_effectiveness else "0.0%" if not pairing_enabled else "N/A"
        
        table_data.append({
            'Load Ratio': f"{load_ratio:.1f}",
            'Pairing Status': pairing_status,
            'Pairing Effectiveness': pairing_formatted,
            'Mean Assignment Time': assignment_time_mean_ci,
            'Completion Rate': completion_formatted,
            'Load Ratio Value': load_ratio,  # For sorting
            'Pairing Enabled': pairing_enabled,  # For comparison
            'Assignment Time Value': assignment_time_mean if assignment_time_mean else 999,
            'Pairing Effectiveness Value': pairing_effectiveness if pairing_effectiveness else 0.0
        })
        
    except Exception as e:
        print(f"  ⚠️  Error extracting metrics for {design_name}: {str(e)}")

# Create and display pairing effectiveness table
if table_data:
    df = pd.DataFrame(table_data)
    df_display = df.sort_values(['Load Ratio Value', 'Pairing Enabled'])[['Load Ratio', 'Pairing Status', 'Pairing Effectiveness', 'Mean Assignment Time', 'Completion Rate']]
    
    print("\n🎯 PAIRING EFFECTIVENESS EVIDENCE TABLE")
    print("="*100)
    print(df_display.to_string(index=False))
    
    print(f"\n📊 PAIRING INTERACTION ANALYSIS:")
    print(f"Research Question: How does pairing interact with load ratio?")
    
    # Compare pairing enabled vs disabled for each load ratio
    print(f"\n🔬 Load Ratio × Pairing Interaction Effects:")
    
    for load_ratio in sorted(df['Load Ratio Value'].unique()):
        load_ratio_subset = df[df['Load Ratio Value'] == load_ratio]
        
        if len(load_ratio_subset) == 2:  # Should have both pairing enabled and disabled
            no_pairing_row = load_ratio_subset[~load_ratio_subset['Pairing Enabled']].iloc[0]
            pairing_row = load_ratio_subset[load_ratio_subset['Pairing Enabled']].iloc[0]
            
            pairing_effectiveness = pairing_row['Pairing Effectiveness Value']
            assignment_time_no_pairing = no_pairing_row['Assignment Time Value']
            assignment_time_pairing = pairing_row['Assignment Time Value']
            
            if assignment_time_no_pairing and assignment_time_pairing:
                assignment_time_change = assignment_time_pairing - assignment_time_no_pairing
                assignment_time_change_pct = (assignment_time_change / assignment_time_no_pairing) * 100
                
                print(f"  Load Ratio {load_ratio:.1f}:")
                print(f"    • Pairing Effectiveness: {pairing_effectiveness:.1%}")
                print(f"    • Assignment Time Impact: {assignment_time_change:+.1f} min ({assignment_time_change_pct:+.1f}%)")
                
                if assignment_time_change < 0:
                    print(f"    ✅ Pairing IMPROVES performance")
                elif assignment_time_change > 0:
                    print(f"    ⚠️  Pairing WORSENS performance")
                else:
                    print(f"    ➡️  Pairing has NEUTRAL performance impact")
    
    print(f"\n📋 KEY EXPLORATORY INSIGHTS:")
    print(f"• How does pairing effectiveness emerge under different load ratios?")
    print(f"• Does pairing help or hurt performance under system stress?")
    print(f"• Are the starting thresholds (restaurants={starting_restaurants_threshold}km, customers={starting_customers_threshold}km) reasonable?")
    print(f"• What load ratio shows most promising pairing benefits?")
    
    print(f"\n✅ EXPLORATORY PAIRING ANALYSIS COMPLETE!")
    print(f"✓ Pairing effectiveness measured and analyzed")
    print(f"✓ Load ratio × pairing interaction effects revealed")
    print(f"✓ Foundation established for broader threshold exploration")

else:
    print("⚠️  No valid data available for pairing effectiveness table")

print(f"\n" + "="*80)
print("EXPLORATORY PAIRING STUDY COMPLETE")
print("="*80)
print("✓ Research Question: How does pairing interact with supply-demand conditions?")
print("✓ Method: Load ratio × pairing enabled/disabled comparison")
print("✓ Evidence: Pairing effectiveness + performance impact analysis")
print("✓ Next: Expand to multiple threshold combinations if promising results")
# %%