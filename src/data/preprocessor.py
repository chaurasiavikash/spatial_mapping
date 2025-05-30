
# ============================================================================
# FILE: src/data/preprocessor.py
# ============================================================================
import numpy as np
import xarray as xr
import logging
from typing import Dict, Optional, Tuple
from scipy import ndimage
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class TROPOMIPreprocessor:
    """Preprocess TROPOMI methane data for hotspot detection."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def preprocess_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Complete preprocessing pipeline for TROPOMI data."""
        logger.info("Starting data preprocessing")
        
        # Apply spatial and temporal filters
        ds_filtered = self.apply_filters(ds)
        
        # Remove outliers
        ds_clean = self.remove_outliers(ds_filtered)
        
        # Calculate background concentrations
        ds_background = self.calculate_background(ds_clean)
        
        # Calculate enhancements
        ds_enhanced = self.calculate_enhancements(ds_background)
        
        # Apply smoothing
        ds_smooth = self.apply_smoothing(ds_enhanced)
        
        logger.info("Data preprocessing completed")
        return ds_smooth
    
    def apply_filters(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply basic quality and range filters."""
        logger.info("Applying quality filters")
        
        # Remove invalid values
        ds_filtered = ds.where(ds.ch4 > 0)
        ds_filtered = ds_filtered.where(ds_filtered.ch4 < 3000)  # Remove unrealistic values
        
        # Apply QA filter
        qa_threshold = self.config['tropomi']['quality_threshold']
        ds_filtered = ds_filtered.where(ds_filtered.qa_value >= qa_threshold)
        
        return ds_filtered
    
    def remove_outliers(self, ds: xr.Dataset, method: str = 'iqr') -> xr.Dataset:
        """Remove statistical outliers from the data."""
        logger.info(f"Removing outliers using {method} method")
        
        ch4_data = ds.ch4
        
        if method == 'iqr':
            q1 = ch4_data.quantile(0.25)
            q3 = ch4_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            mask = (ch4_data >= lower_bound) & (ch4_data <= upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((ch4_data - ch4_data.mean()) / ch4_data.std())
            mask = z_scores <= 3
            
        else:
            raise ValueError(f"Unknown outlier removal method: {method}")
        
        return ds.where(mask)
    
    def calculate_background(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate background methane concentrations."""
        logger.info("Calculating background concentrations")
        
        # Calculate temporal background (median over time)
        temporal_background = ds.ch4.median(dim='time')
        
        # Calculate spatial background using percentile
        percentile = self.config['detection']['background_percentile']
        spatial_background = ds.ch4.quantile(percentile / 100.0, dim=['lat', 'lon'])
        
        # Store backgrounds in dataset
        ds_with_bg = ds.copy()
        ds_with_bg['temporal_background'] = temporal_background
        ds_with_bg['spatial_background'] = spatial_background
        
        return ds_with_bg
    
    def calculate_enhancements(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate methane enhancements above background."""
        logger.info("Calculating methane enhancements")
        
        # Calculate enhancement relative to temporal background
        enhancement = ds.ch4 - ds.temporal_background
        
        # Store enhancement in dataset
        ds_enhanced = ds.copy()
        ds_enhanced['enhancement'] = enhancement
        
        # Calculate relative enhancement
        relative_enhancement = (ds.ch4 / ds.temporal_background - 1) * 100
        ds_enhanced['relative_enhancement'] = relative_enhancement
        
        return ds_enhanced
    
    def apply_smoothing(self, ds: xr.Dataset, sigma: float = 1.0) -> xr.Dataset:
        """Apply Gaussian smoothing to reduce noise."""
        logger.info(f"Applying Gaussian smoothing with sigma={sigma}")
        
        ds_smooth = ds.copy()
        
        # Apply smoothing to enhancement field
        if 'enhancement' in ds.data_vars:
            enhancement_smooth = xr.apply_ufunc(
                lambda x: ndimage.gaussian_filter(x, sigma=sigma, mode='nearest'),
                ds.enhancement,
                input_core_dims=[['lat', 'lon']],
                output_core_dims=[['lat', 'lon']],
                dask='parallelized'
            )
            ds_smooth['enhancement_smooth'] = enhancement_smooth
        
        return ds_smooth
    
    def calculate_statistics(self, ds: xr.Dataset) -> Dict:
        """Calculate summary statistics for the dataset."""
        logger.info("Calculating dataset statistics")
        
        stats = {}
        
        if 'ch4' in ds.data_vars:
            ch4_data = ds.ch4.values.flatten()
            ch4_data = ch4_data[~np.isnan(ch4_data)]
            
            stats['ch4'] = {
                'mean': float(np.mean(ch4_data)),
                'std': float(np.std(ch4_data)),
                'min': float(np.min(ch4_data)),
                'max': float(np.max(ch4_data)),
                'median': float(np.median(ch4_data)),
                'count': len(ch4_data)
            }
        
        if 'enhancement' in ds.data_vars:
            enh_data = ds.enhancement.values.flatten()
            enh_data = enh_data[~np.isnan(enh_data)]
            
            stats['enhancement'] = {
                'mean': float(np.mean(enh_data)),
                'std': float(np.std(enh_data)),
                'min': float(np.min(enh_data)),
                'max': float(np.max(enh_data)),
                'median': float(np.median(enh_data)),
                'count': len(enh_data)
            }
        
        return stats
