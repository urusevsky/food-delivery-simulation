"""
System state metrics from time series snapshots.

TIMING OF CALCULATION:
These metrics must be calculated DURING simulation runtime because they capture
transient system state that cannot be reconstructed post-simulation. For example,
"how many drivers were available at t=501" requires querying driver states at that
exact moment - this information is not preserved in entity attributes.

CALCULATION DURING SIMULATION:
SystemDataCollector calls SystemDataDefinitions.create_snapshot_data() at regular
intervals (default 1.0 min) during simulation, which queries repositories to count
entities in specific states (e.g., drivers with state=AVAILABLE).

PROCESSING IN ANALYSIS PIPELINE:
Since metrics are already calculated and stored in snapshots, this module only
needs to extract them (remove timestamp identifier) for aggregation. The extraction
function plays the same structural role as calculate_all_order_metrics(), but
performs extraction instead of calculation.

CONTRAST WITH ENTITY-BASED METRICS:
Entity metrics (e.g., assignment_time) can be calculated post-simulation because
the necessary data (order.arrival_time, order.assignment_time) is preserved in
entity attributes. State metrics require real-time calculation because system state
is ephemeral.
"""

def extract_snapshot_metrics(snapshot):
    """
    Extract pre-calculated metrics from a snapshot.
    
    Snapshots already contain metric values calculated during simulation.
    This function simply filters out the timestamp identifier, returning
    a metrics dict suitable for aggregation.
    
    This plays the same structural role as calculate_all_order_metrics(entity),
    but performs extraction rather than calculation because metrics were already
    computed during simulation runtime.
    
    Args:
        snapshot: Dict with structure:
                  {'timestamp': 501.0,
                   'available_drivers': 5,
                   'active_drivers': 20,
                   'unassigned_orders': 3,
                   'unassigned_pairs': 0,
                   'delivering_drivers': 15,
                   'unassigned_delivery_entities': 3}
        
    Returns:
        dict: Metrics without timestamp identifier:
              {'available_drivers': 5,
               'active_drivers': 20,
               'unassigned_orders': 3,
               ...}
    
    Example usage in pipeline:
        >>> snapshots = analysis_data.post_warmup_snapshots
        >>> individual_metrics = [extract_snapshot_metrics(s) for s in snapshots]
        >>> # Then aggregate across snapshots to get statistics
    """
    return {k: v for k, v in snapshot.items() if k != 'timestamp'}