# ============================================================================
# FILE: src/detection/anomaly_detector.py (FIXED VERSION)
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
    """Detect methane anomalies and hotspots in TROPOMI data - FIXED VERSION."""
    
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
        """Detect statistical anomalies using FIXED threshold-based method."""
        logger.info("Performing statistical anomaly detection")
        
        if 'enhancement' not in ds.data_vars:
            raise ValueError("Enhancement field not found in dataset")
        
        enhancement = ds.enhancement
        
        # Log enhancement statistics for debugging
        logger.info(f"Enhancement statistics:")
        logger.info(f"  Mean: {enhancement.mean().values:.2f} ppb")
        logger.info(f"  Std:  {enhancement.std().values:.2f} ppb")
        logger.info(f"  Min:  {enhancement.min().values:.2f} ppb")
        logger.info(f"  Max:  {enhancement.max().values:.2f} ppb")
        
        # Get detection parameters
        anomaly_threshold = self.detection_params['anomaly_threshold']
        min_enhancement = self.detection_params['min_enhancement']
        
        logger.info(f"Detection thresholds:")
        logger.info(f"  Anomaly threshold: {anomaly_threshold} std devs")
        logger.info(f"  Min enhancement: {min_enhancement} ppb")
        
        # Method 1: Simple absolute threshold (most reliable)
        simple_mask = enhancement > min_enhancement
        n_simple = simple_mask.sum().values
        logger.info(f"Simple threshold method: {n_simple} pixels above {min_enhancement} ppb")
        
        # Method 2: Statistical threshold using global statistics
        global_mean = enhancement.mean()
        global_std = enhancement.std()
        
        statistical_threshold = global_mean + anomaly_threshold * global_std
        statistical_mask = enhancement > statistical_threshold
        n_statistical = statistical_mask.sum().values
        logger.info(f"Statistical method: {n_statistical} pixels above {statistical_threshold.values:.2f} ppb")
        
        # Method 3: Percentile-based method
        percentile_threshold = enhancement.quantile(0.95)  # Top 5%
        percentile_mask = enhancement > percentile_threshold
        n_percentile = percentile_mask.sum().values
        logger.info(f"Percentile method: {n_percentile} pixels above {percentile_threshold.values:.2f} ppb (95th percentile)")
        
        # Combine methods: use simple method if it finds hotspots, otherwise use statistical
        if n_simple > 0:
            logger.info("✅ Using simple threshold method")
            anomaly_mask = simple_mask
            method_used = "simple_threshold"
        elif n_statistical > 0:
            logger.info("✅ Using statistical threshold method")
            anomaly_mask = statistical_mask
            method_used = "statistical_threshold"
        elif n_percentile > 0:
            logger.info("✅ Using percentile threshold method")
            anomaly_mask = percentile_mask
            method_used = "percentile_threshold"
        else:
            # Very permissive fallback
            fallback_threshold = global_mean + 0.5 * global_std
            anomaly_mask = enhancement > fallback_threshold
            n_fallback = anomaly_mask.sum().values
            logger.info(f"🔄 Using fallback method: {n_fallback} pixels above {fallback_threshold.values:.2f} ppb")
            method_used = "fallback"
        
        # Calculate local statistics for additional context (but don't use for primary detection)
        window_size = self.detection_params.get('spatial_window', 3)
        
        # Local mean and std using rolling window
        local_mean = enhancement.rolling(
            lat=window_size, lon=window_size, center=True, min_periods=1
        ).mean()
        
        local_std = enhancement.rolling(
            lat=window_size, lon=window_size, center=True, min_periods=1
        ).std()
        
        # Calculate z-scores (for information, not primary detection)
        z_scores = (enhancement - local_mean) / (local_std + 1e-10)  # Add small epsilon to avoid division by zero
        
        # Final count
        final_count = anomaly_mask.sum().values
        logger.info(f"🎯 Final anomaly detection: {final_count} pixels detected using {method_used}")
        
        if final_count == 0:
            logger.warning("❌ No anomalies detected! Consider lowering thresholds.")
            logger.warning(f"   Try: min_enhancement < {enhancement.max().values:.1f} ppb")
            logger.warning(f"   Try: anomaly_threshold < {(enhancement.max() - global_mean).values / global_std.values:.1f}")
        
        # Store results
        ds_result = ds.copy()
        ds_result['z_scores'] = z_scores
        ds_result['local_mean'] = local_mean
        ds_result['local_std'] = local_std
        ds_result['anomaly_mask'] = anomaly_mask
        
        # Add detection metadata
        ds_result.attrs.update({
            'detection_method': method_used,
            'anomalies_detected': int(final_count),
            'detection_threshold_used': float(min_enhancement) if method_used == "simple_threshold" else float(statistical_threshold.values),
            'enhancement_max': float(enhancement.max().values),
            'enhancement_mean': float(global_mean.values),
            'enhancement_std': float(global_std.values)
        })
        
        return ds_result
    
    def cluster_anomalies(self, ds: xr.Dataset) -> xr.Dataset:
        """Cluster spatially connected anomalies."""
        logger.info("Clustering spatial anomalies")
        
        if 'anomaly_mask' not in ds.data_vars:
            raise ValueError("Anomaly mask not found in dataset")
        
        ds_result = ds.copy()
        cluster_labels = []
        total_clusters = 0
        
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
            min_cluster_size = 1  # More permissive - was 4
            valid_clusters = 0
            
            for cluster_id in range(1, num_features + 1):
                cluster_size = np.sum(labeled_array == cluster_id)
                if cluster_size < min_cluster_size:
                    labeled_array[labeled_array == cluster_id] = 0
                else:
                    valid_clusters += 1
            
            total_clusters += valid_clusters
            cluster_labels.append(labeled_array)
            
            if valid_clusters > 0:
                logger.info(f"Time {t}: {valid_clusters} clusters, largest: {np.max(np.bincount(labeled_array.flat)[1:])} pixels")
        
        # Convert to xarray
        cluster_array = np.stack(cluster_labels, axis=0)
        ds_result['cluster_labels'] = (['time', 'lat', 'lon'], cluster_array)
        
        logger.info(f"✅ Found {total_clusters} total clusters across all time steps")
        
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
        
        # More permissive persistence requirement
        min_persistence = 0.1  # Appear in at least 10% of time steps (was 20%)
        persistent_mask = persistence_fraction >= min_persistence
        
        n_persistent = persistent_mask.sum().values
        logger.info(f"Persistent locations (≥{min_persistence*100:.0f}% of time): {n_persistent}")
        
        # If no persistent hotspots, lower the requirement
        if n_persistent == 0:
            min_persistence = 0.05  # Try 5%
            persistent_mask = persistence_fraction >= min_persistence
            n_persistent = persistent_mask.sum().values
            logger.info(f"Lowered to ≥{min_persistence*100:.0f}% of time: {n_persistent}")
        
        # If still none, just take any hotspot that appears at least once
        if n_persistent == 0:
            persistent_mask = anomaly_count > 0
            n_persistent = persistent_mask.sum().values
            logger.info(f"Fallback - any hotspot: {n_persistent}")
        
        ds_result = ds.copy()
        ds_result['persistence_fraction'] = persistence_fraction
        ds_result['persistent_mask'] = persistent_mask
        
        # Create final hotspot mask
        final_hotspots = ds.cluster_labels.where(persistent_mask, 0)
        ds_result['hotspot_labels'] = final_hotspots
        
        # Count final hotspots
        final_hotspot_count = (final_hotspots > 0).sum().values
        logger.info(f"🎯 Final hotspots after persistence filtering: {final_hotspot_count} pixels")
        
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
                
                if features:  # Only add if features were calculated successfully
                    features['time'] = time_val
                    features['hotspot_id'] = int(hotspot_id)
                    features_list.append(features)
        
        logger.info(f"✅ Extracted features for {len(features_list)} hotspot instances")
        
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
        if len(lat_coords) == 0:
            return {}
            
        lats = ds.lat.values[lat_coords]
        lons = ds.lon.values[lon_coords]
        
        features = {
            'area_pixels': int(np.sum(mask)),
            'max_enhancement': float(np.max(valid_enh)),
            'mean_enhancement': float(np.mean(valid_enh)),
            'total_enhancement': float(np.sum(valid_enh)),
            'center_lat': float(np.mean(lats)),
            'center_lon': float(np.mean(lons)),
            'lat_extent': float(np.max(lats) - np.min(lats)) if len(lats) > 1 else 0.0,
            'lon_extent': float(np.max(lons) - np.min(lons)) if len(lons) > 1 else 0.0
        }
        
        return features