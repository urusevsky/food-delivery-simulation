# analysis_pipeline/statistics_engine.py
"""
Pure statistical calculation engine.

Single responsibility: Compute statistics from provided values.
NO data extraction - that's the extraction_engine's job.
"""

import numpy as np
from delivery_sim.utils.logging_system import get_logger


class StatisticsEngine:
    """
    Pure statistical computation engine.
    
    Only performs mathematical calculations on provided data.
    """
    
    def __init__(self):
        self.logger = get_logger("analysis_pipeline.statistics_engine")
    
    def calculate_statistics(self, values, include_percentiles=True):
        """
        Calculate summary statistics for a collection of values.
        
        Args:
            values: List of numeric values (None values filtered out)
            include_percentiles: If True, include min/max/percentiles (default: True)
            
        Returns:
            dict: Statistics dictionary with count, mean, variance, std
                  Plus min/max/percentiles if include_percentiles=True
        """
        # Filter and handle edge cases
        valid_values = [v for v in values if v is not None]
        
        # Handle empty data
        if not valid_values:
            base_stats = {'count': 0, 'mean': None, 'variance': None, 'std': None}
            if include_percentiles:
                base_stats.update({
                    'min': None, 'max': None, 
                    'p25': None, 'p50': None, 'p75': None, 'p95': None
                })
            return base_stats
        
        # Handle single value
        if len(valid_values) == 1:
            val = float(valid_values[0])
            base_stats = {'count': 1, 'mean': val, 'variance': 0.0, 'std': 0.0}
            if include_percentiles:
                base_stats.update({
                    'min': val, 'max': val, 
                    'p25': val, 'p50': val, 'p75': val, 'p95': val
                })
            return base_stats
        
        # Calculate statistics for multiple values
        values_array = np.array(valid_values)
        count = len(valid_values)
        mean = np.mean(values_array)
        variance = np.var(values_array, ddof=1)  # Sample variance
        std = np.sqrt(variance)
        
        # Build result dictionary
        result = {
            'count': count,
            'mean': float(mean),
            'variance': float(variance),
            'std': float(std)
        }
        
        # Add percentiles if requested
        if include_percentiles:
            min_val = np.min(values_array)
            max_val = np.max(values_array)
            percentiles = np.percentile(values_array, [25, 50, 75, 95])
            
            result.update({
                'min': float(min_val),
                'max': float(max_val),
                'p25': float(percentiles[0]),
                'p50': float(percentiles[1]),  # median
                'p75': float(percentiles[2]),
                'p95': float(percentiles[3])
            })
        
        return result