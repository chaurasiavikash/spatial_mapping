
# ============================================================================
# FILE: src/detection/anomaly_detector.py
# ============================================================================
import numpy as np
import xarray as xr
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class MethaneAnomalyDetector:
    """Detect methane anomalies and hotspots in TROPOMI data."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.detection_params = config['detection']
        
    def detect_hotspots(self, ds: xr.Dataset) -> xr.Dataset:
        """Main hotspot detection pipeline."""
        logger.info("Starting hotspot detection")
        
        # Statistical anomaly detection
        ds_anomalies = self.statistical_anomaly_detection(ds)
        
        # Spatial clustering of anomalies
        ds_clustered = self.cluster_anomalies(ds_anomalies)
        
        # Temporal persistence filtering
        ds_persistent = self.filter_temporal_persistence(ds_clustered)
        
        logger.info("Hotspot detection completed")
        return ds_persistent
    
    def statistical_anomaly_detection(self, ds: xr.Dataset) -> xr.Dataset:
        """Detect statistical anomalies using threshold-based method."""
        logger.info("Performing statistical anomaly detection")
        
        if 'enhancement' not in ds.data_vars:
            raise ValueError("Enhancement field not found in dataset")
        
        enhancement = ds.enhancement
        
        # Calculate local statistics
        window_size = self.detection_params['spatial_window']
        
        # Local mean and std using rolling window
        local_mean = enhancement.rolling(
            lat=window_size, lon=window_size, center=True, min_periods=1
        ).mean()
        
        local_std = enhancement.rolling(
            lat=window_size, lon=window_size, center=True, min_periods=1
        ).std()
        
        # Calculate z-scores
        z_scores = (enhancement - local_mean) / local_std
        
        # Identify anomalies
        threshold = self.detection_params['anomaly_threshold']
        min_enhancement = self.detection_params['min_enhancement']
        
        anomaly_mask = (
            (z_scores > threshold) & 
            (enhancement > min_enhancement)
        )
        
        # Store results
        ds_result = ds.copy()
        ds_result['z_scores'] = z_scores
        ds_result['local_mean'] = local_mean
        ds_result['local_std'] = local_std
        ds_result['anomaly_mask'] = anomaly_mask
        
        return ds_result
    
    def cluster_anomalies(self, ds: xr.Dataset) -> xr.Dataset:
        """Cluster spatially connected anomalies."""
        logger.info("Clustering spatial anomalies")
        
        if 'anomaly_mask' not in ds.data_vars:
            raise ValueError("Anomaly mask not found in dataset")
        
        ds_result = ds.copy()
        cluster_labels = []
        
        # Process each time step
        for t, time_val in enumerate(ds.time.values):
            anomaly_2d = ds.anomaly_mask.isel(time=t).values
            
            if not np.any(anomaly_2d):
                # No anomalies in this time step
                cluster_labels.append(np.zeros_like(anomaly_2d, dtype=int))
                continue
            
            # Find connected components
            labeled_array, num_features = ndimage.label(anomaly_2d)
            
            # Filter small clusters
            min_cluster_size = 4  # minimum pixels in a cluster
            for cluster_id in range(1, num_features + 1):
                cluster_size = np.sum(labeled_array == cluster_id)
                if cluster_size < min_cluster_size:
                    labeled_array[labeled_array == cluster_id] = 0
            
            cluster_labels.append(labeled_array)
        
        # Convert to xarray
        cluster_array = np.stack(cluster_labels, axis=0)
        ds_result['cluster_labels'] = (['time', 'lat', 'lon'], cluster_array)
        
        return ds_result
    
    def filter_temporal_persistence(self, ds: xr.Dataset) -> xr.Dataset:
        """Filter hotspots based on temporal persistence."""
        logger.info("Filtering for temporal persistence")
        
        if 'cluster_labels' not in ds.data_vars:
            raise ValueError("Cluster labels not found in dataset")
        
        temporal_window = self.detection_params['temporal_window']
        
        # Calculate persistence for each location
        anomaly_count = (ds.cluster_labels > 0).sum(dim='time')
        persistence_fraction = anomaly_count / len(ds.time)
        
        # Define persistent hotspots (appear in at least 20% of time steps)
        persistent_mask = persistence_fraction >= 0.2
        
        ds_result = ds.copy()
        ds_result['persistence_fraction'] = persistence_fraction
        ds_result['persistent_mask'] = persistent_mask
        
        # Create final hotspot mask
        final_hotspots = ds.cluster_labels.where(persistent_mask, 0)
        ds_result['hotspot_labels'] = final_hotspots
        
        return ds_result
    
    def extract_hotspot_features(self, ds: xr.Dataset) -> pd.DataFrame:
        """Extract features for each detected hotspot."""
        logger.info("Extracting hotspot features")
        
        if 'hotspot_labels' not in ds.data_vars:
            raise ValueError("Hotspot labels not found in dataset")
        
        features_list = []
        
        # Process each time step
        for t, time_val in enumerate(ds.time.values):
            hotspots_2d = ds.hotspot_labels.isel(time=t).values
            enhancement_2d = ds.enhancement.isel(time=t).values
            
            # Get unique hotspot IDs (excluding 0)
            hotspot_ids = np.unique(hotspots_2d)
            hotspot_ids = hotspot_ids[hotspot_ids > 0]
            
            for hotspot_id in hotspot_ids:
                mask = hotspots_2d == hotspot_id
                
                if not np.any(mask):
                    continue
                
                # Calculate features
                features = self._calculate_hotspot_features(
                    mask, enhancement_2d, ds, t, hotspot_id
                )
                features['time'] = time_val
                features['hotspot_id'] = int(hotspot_id)
                
                features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _calculate_hotspot_features(self, mask: np.ndarray, enhancement: np.ndarray, 
                                   ds: xr.Dataset, time_idx: int, hotspot_id: int) -> Dict:
        """Calculate features for a single hotspot."""
        
        # Spatial features
        hotspot_enh = enhancement[mask]
        valid_enh = hotspot_enh[~np.isnan(hotspot_enh)]
        
        if len(valid_enh) == 0:
            return {}
        
        # Get coordinates
        lat_coords, lon_coords = np.where(mask)
        lats = ds.lat.values[lat_coords]
        lons = ds.lon.values[lon_coords]
        
        features = {
            'area_pixels': int(np.sum(mask)),
            'max_enhancement': float(np.max(valid_enh)),
            'mean_enhancement': float(np.mean(valid_enh)),
            'total_enhancement': float(np.sum(valid_enh)),
            'center_lat': float(np.mean(lats)),
            'center_lon': float(np.mean(lons)),
            'lat_extent': float(np.max(lats) - np.min(lats)),
            'lon_extent': float(np.max(lons) - np.min(lons))
        }
        
        return features
