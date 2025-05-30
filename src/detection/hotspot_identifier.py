
# ============================================================================
# FILE: src/detection/hotspot_identifier.py (COMPLETE IMPLEMENTATION)
# ============================================================================
import numpy as np
import pandas as pd
import xarray as xr
import logging
from typing import Dict, List, Tuple, Optional
from scipy import ndimage

logger = logging.getLogger(__name__)

class HotspotIdentifier:
    """Identify and characterize methane hotspots."""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def identify_hotspots(self, ds: xr.Dataset) -> pd.DataFrame:
        """Identify hotspots from processed dataset."""
        
        if 'hotspot_labels' not in ds.data_vars:
            logger.warning("No hotspot labels found in dataset")
            return pd.DataFrame()
        
        hotspots = []
        
        for t, time_val in enumerate(ds.time.values):
            hotspot_2d = ds.hotspot_labels.isel(time=t).values
            enhancement_2d = ds.enhancement.isel(time=t).values
            
            unique_ids = np.unique(hotspot_2d)
            unique_ids = unique_ids[unique_ids > 0]  # Remove background (0)
            
            for hotspot_id in unique_ids:
                mask = hotspot_2d == hotspot_id
                
                # Calculate properties
                hotspot_enh = enhancement_2d[mask]
                valid_enh = hotspot_enh[~np.isnan(hotspot_enh)]
                
                if len(valid_enh) > 0:
                    # Get coordinates
                    lat_coords, lon_coords = np.where(mask)
                    lats = ds.lat.values[lat_coords]
                    lons = ds.lon.values[lon_coords]
                    
                    hotspot_info = {
                        'time': time_val,
                        'hotspot_id': int(hotspot_id),
                        'center_lat': float(np.mean(lats)),
                        'center_lon': float(np.mean(lons)),
                        'area_pixels': int(np.sum(mask)),
                        'max_enhancement': float(np.max(valid_enh)),
                        'mean_enhancement': float(np.mean(valid_enh)),
                        'total_enhancement': float(np.sum(valid_enh))
                    }
                    
                    hotspots.append(hotspot_info)
        
        return pd.DataFrame(hotspots)
