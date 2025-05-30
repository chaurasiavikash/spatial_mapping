
# ============================================================================
# FILE: src/data/validator.py (COMPLETE IMPLEMENTATION)
# ============================================================================
import numpy as np
import pandas as pd
import xarray as xr
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class TROPOMIValidator:
    """Validate TROPOMI data quality and processing results."""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def validate_dataset(self, ds: xr.Dataset) -> Dict:
        """Validate a TROPOMI dataset."""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Check for required variables
        required_vars = ['ch4', 'qa_value']
        for var in required_vars:
            if var not in ds.data_vars:
                validation_results['errors'].append(f"Missing required variable: {var}")
                validation_results['valid'] = False
        
        # Check data ranges
        if 'ch4' in ds.data_vars:
            ch4_data = ds.ch4.values.flatten()
            ch4_data = ch4_data[~np.isnan(ch4_data)]
            
            if len(ch4_data) == 0:
                validation_results['errors'].append("No valid CH4 data found")
                validation_results['valid'] = False
            else:
                # Check realistic ranges
                if np.min(ch4_data) < 1000:  # ppb
                    validation_results['warnings'].append("Some CH4 values below 1000 ppb (unrealistic)")
                if np.max(ch4_data) > 3000:  # ppb
                    validation_results['warnings'].append("Some CH4 values above 3000 ppb (check for outliers)")
                
                validation_results['statistics']['ch4_mean'] = float(np.mean(ch4_data))
                validation_results['statistics']['ch4_std'] = float(np.std(ch4_data))
                validation_results['statistics']['ch4_count'] = len(ch4_data)
        
        return validation_results
