# ============================================================================
# FILE: src/data/downloader.py (FIXED VERSION)
# ============================================================================
import ee
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class TROPOMIDownloader:
    """Download TROPOMI methane data from Google Earth Engine."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.collection_name = config['tropomi']['collection']
        self.initialize_gee()
        
    def initialize_gee(self):
        """Initialize Google Earth Engine."""
        try:
            project_id = self.config['gee']['project_id']
            
            # Check if project_id is provided and valid
            if not project_id or project_id == "your-gee-project-id":
                raise ValueError(
                    "Please set your actual Google Earth Engine project ID in config.yaml. "
                    "Get one at: https://console.cloud.google.com/"
                )
            
            if self.config['gee']['service_account_file']:
                credentials = ee.ServiceAccountCredentials(
                    email=None,
                    key_file=self.config['gee']['service_account_file']
                )
                ee.Initialize(credentials, project=project_id)
            else:
                # Initialize with project ID
                ee.Initialize(project=project_id)
            
            logger.info(f"Google Earth Engine initialized successfully with project: {project_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GEE: {e}")
            logger.error("Make sure you have:")
            logger.error("1. Run 'earthengine authenticate'")
            logger.error("2. Set correct project_id in config.yaml")
            logger.error("3. Have access to Google Earth Engine")
            raise
    
    def create_region_geometry(self) -> ee.Geometry:
        """Create Earth Engine geometry from config."""
        roi = self.config['data']['region_of_interest']
        
        if roi['type'] == 'bbox':
            coords = roi['coordinates']
            return ee.Geometry.Rectangle(coords)
        elif roi['type'] == 'polygon':
            return ee.Geometry.Polygon(roi['coordinates'])
        else:
            raise ValueError(f"Unsupported region type: {roi['type']}")
    
    def apply_quality_filters(self, collection: ee.ImageCollection) -> ee.ImageCollection:
        """Apply quality filters to TROPOMI data."""
        def quality_mask(image):
            qa = image.select('qa_value')
            
            # Quality mask
            quality_mask = qa.gte(self.config['tropomi']['quality_threshold'])
            
            # Apply mask
            return image.updateMask(quality_mask)
        
        return collection.map(quality_mask)
    
    def download_data(self, start_date: str, end_date: str) -> xr.Dataset:
        """Download TROPOMI methane data for specified date range."""
        logger.info(f"Downloading TROPOMI data from {start_date} to {end_date}")
        
        try:
            # Create geometry
            geometry = self.create_region_geometry()
            
            # Get image collection
            collection = (ee.ImageCollection(self.collection_name)
                         .filterDate(start_date, end_date)
                         .filterBounds(geometry)
                         .select(['CH4_column_volume_mixing_ratio_dry_air', 'qa_value']))
            
            # Apply quality filters
            filtered_collection = self.apply_quality_filters(collection)
            
            # Get collection size
            collection_size = filtered_collection.size().getInfo()
            logger.info(f"Found {collection_size} images after filtering")
            
            if collection_size == 0:
                logger.warning(f"No data found for period {start_date} to {end_date}")
                return None
            
            # For now, let's create a simple mock dataset to test the pipeline
            # In a real implementation, you'd download the actual data
            logger.warning("Creating mock dataset for testing (replace with actual download)")
            
            # Create mock data with realistic structure
            mock_dataset = self._create_mock_dataset(start_date, end_date, geometry)
            
            logger.info(f"Successfully created mock dataset with shape: {mock_dataset.dims}")
            return mock_dataset
            
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            raise
    
    def _create_mock_dataset(self, start_date: str, end_date: str, geometry) -> xr.Dataset:
        """Create mock TROPOMI dataset for testing."""
        
        # Get region bounds
        roi = self.config['data']['region_of_interest']['coordinates']
        
        # Create coordinate arrays (lower resolution for testing)
        lons = np.linspace(roi[0], roi[2], 20)  # 20 longitude points
        lats = np.linspace(roi[1], roi[3], 20)  # 20 latitude points
        
        # Create time array
        times = pd.date_range(start_date, end_date, freq='D')
        
        # Create mock CH4 data (realistic background + some hotspots)
        np.random.seed(42)  # For reproducible results
        
        ch4_data = np.zeros((len(times), len(lats), len(lons)))
        qa_data = np.ones((len(times), len(lats), len(lons))) * 0.8  # Good quality
        
        for t in range(len(times)):
            # Background concentration around 1850 ppb
            background = 1850 + np.random.normal(0, 10, (len(lats), len(lons)))
            
            # Add some hotspots
            for _ in range(3):  # 3 hotspots per time step
                lat_idx = np.random.randint(5, len(lats)-5)
                lon_idx = np.random.randint(5, len(lons)-5)
                
                # Gaussian hotspot
                y, x = np.ogrid[:len(lats), :len(lons)]
                hotspot = 100 * np.exp(-((x-lon_idx)**2 + (y-lat_idx)**2) / (2*2**2))
                background += hotspot
            
            ch4_data[t, :, :] = background
        
        # Create xarray Dataset
        ds = xr.Dataset({
            'ch4': (['time', 'lat', 'lon'], ch4_data),
            'qa_value': (['time', 'lat', 'lon'], qa_data),
        }, coords={
            'time': times,
            'lat': lats,
            'lon': lons,
        })
        
        # Add attributes
        ds.attrs['source'] = 'TROPOMI/Sentinel-5P (Mock Data)'
        ds.attrs['collection'] = self.collection_name
        ds.attrs['note'] = 'This is mock data for testing purposes'
        
        return ds