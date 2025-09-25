# analysis_pipeline_redesigned/core/statistics_engine.py
"""
Centralized statistics calculation engine.

This module provides the single source of truth for all statistical calculations
across the analysis pipeline. It extracts and centralizes the statistical logic
from the original entity_replication_summaries.py module.

Design philosophy for research context:
- Report issues via logging rather than silent handling
- Assume well-formed input data from controlled simulation
- Focus on core functionality and consistency
"""

import numpy as np
from delivery_sim.utils.logging_system import get_logger


class StatisticsEngine:
    """
    Centralized engine for all statistical calculations in the analysis pipeline.
    
    This class provides consistent statistical calculations across different
    metric types and aggregation levels.
    """
    
    def __init__(self):
        self.logger = get_logger("analysis_pipeline_redesigned.statistics_engine")
    
    def calculate_statistics(self, values):
        """
        Calculate comprehensive summary statistics for a collection of values.
        
        This is the core statistical function used throughout the analysis pipeline.
        Uses sample variance (ddof=1) for consistency with statistical best practices.
        
        Args:
            values: List of numeric values (None values will be filtered out)
            
        Returns:
            dict: Summary statistics containing count, mean, variance, std, 
                  min, max, and percentiles (p25, p50, p75, p95)
        """
        # Filter out None values
        valid_values = [v for v in values if v is not None]
        
        if not valid_values:
            self.logger.warning("calculate_statistics received empty data - check upstream data preparation")
            return self._empty_statistics()
        
        if len(valid_values) == 1:
            self.logger.info("calculate_statistics received single value - variance will be zero")
            return self._single_value_statistics(valid_values[0])
        
        # Convert to numpy array for calculation
        values_array = np.array(valid_values)
        
        # Calculate core statistics using sample variance (ddof=1)
        count = len(valid_values)
        mean = np.mean(values_array)
        variance = np.var(values_array, ddof=1)  # Sample variance
        std = np.sqrt(variance)
        
        # Calculate range and percentiles
        min_val = np.min(values_array)
        max_val = np.max(values_array)
        percentiles = np.percentile(values_array, [25, 50, 75, 95])
        
        return {
            'count': count,
            'mean': float(mean),
            'variance': float(variance),
            'std': float(std),
            'min': float(min_val),
            'max': float(max_val),
            'p25': float(percentiles[0]),
            'p50': float(percentiles[1]),  # median
            'p75': float(percentiles[2]),
            'p95': float(percentiles[3])
        }
    
    def calculate_basic_statistics(self, values):
        """
        Calculate basic summary statistics (count, mean, variance, std only).
        
        Lightweight version for cases where only central tendency and 
        dispersion are needed.
        """
        valid_values = [v for v in values if v is not None]
        
        if not valid_values:
            self.logger.warning("calculate_basic_statistics received empty data")
            return {'count': 0, 'mean': None, 'variance': None, 'std': None}
        
        if len(valid_values) == 1:
            return {
                'count': 1, 
                'mean': float(valid_values[0]), 
                'variance': 0.0, 
                'std': 0.0
            }
        
        values_array = np.array(valid_values)
        variance = np.var(values_array, ddof=1)
        
        return {
            'count': len(valid_values),
            'mean': float(np.mean(values_array)),
            'variance': float(variance),
            'std': float(np.sqrt(variance))
        }
    
    def extract_statistic_for_experiment_aggregation(self, replication_summaries, metric_name, statistic_type):
        """
        Extract a specific statistic across replication summaries for two-level aggregation.
        
        ✅ SIMPLIFIED: Removed entity_type parameter - no longer needed with flat structure.
        
        Args:
            replication_summaries: List of replication-level summary dictionaries
            metric_name: Metric name key (e.g., 'assignment_time', 'total_distance')
            statistic_type: Statistic to extract (e.g., 'mean', 'std', 'p95')
            
        Returns:
            List of statistic values across replications for experiment-level CI calculation
            
        Example:
            # Extract mean assignment times across all replications
            means = engine.extract_statistic_for_experiment_aggregation(
                replication_summaries, 'assignment_time', 'mean'  # ← No entity_type needed
            )
            # Result: [12.47, 13.12, 11.85] - one mean per replication
        """
        statistic_values = []
        
        for i, rep_summary in enumerate(replication_summaries):
            try:
                # ✅ SIMPLIFIED: Direct access to metric, no entity_type navigation
                metric_data = rep_summary[metric_name]
                stat_value = metric_data[statistic_type]
                statistic_values.append(float(stat_value))
            except KeyError as e:
                self.logger.error(f"Missing data structure in replication {i} summary: {e}")
                self.logger.error(f"Expected structure: replication[{metric_name}][{statistic_type}]")  # ← Updated error message
                raise
            except (TypeError, ValueError) as e:
                self.logger.error(f"Invalid data type in replication {i} summary: {e}")
                raise
        
        if not statistic_values:
            self.logger.warning(f"No values extracted for {metric_name}.{statistic_type}")
        else:
            self.logger.debug(f"Extracted {len(statistic_values)} values for {metric_name}.{statistic_type}")
        
        return statistic_values
    
    def _empty_statistics(self):
        """Create statistics dictionary for empty data."""
        return {
            'count': 0,
            'mean': None,
            'variance': None,
            'std': None,
            'min': None,
            'max': None,
            'p25': None,
            'p50': None,
            'p75': None,
            'p95': None
        }
    
    def _single_value_statistics(self, value):
        """Create statistics dictionary for single value."""
        float_value = float(value)
        return {
            'count': 1,
            'mean': float_value,
            'variance': 0.0,
            'std': 0.0,
            'min': float_value,
            'max': float_value,
            'p25': float_value,
            'p50': float_value,
            'p75': float_value,
            'p95': float_value
        }

