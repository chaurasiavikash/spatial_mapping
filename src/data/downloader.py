

# ============================================================================
# FILE: src/data/downloader.py
# ============================================================================
import ee
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import xarray as xr
import pandas as pd
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
            if self.config['gee']['service_account_file']:
                credentials = ee.ServiceAccountCredentials(
                    email=None,
                    key_file=self.config['gee']['service_account_file']
                )
                ee.Initialize(credentials)
            else:
                ee.Initialize()
            logger.info("Google Earth Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GEE: {e}")
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
            cloud_fraction = image.select('cloud_fraction')
            
            # Quality mask
            quality_mask = qa.gte(self.config['tropomi']['quality_threshold'])
            
            # Cloud mask
            cloud_mask = cloud_fraction.lte(self.config['tropomi']['cloud_fraction_max'])
            
            # Combine masks
            mask = quality_mask.And(cloud_mask)
            
            return image.updateMask(mask)
        
        return collection.map(quality_mask)
    
    def download_data(self, start_date: str, end_date: str) -> xr.Dataset:
        """Download TROPOMI methane data for specified date range."""
        logger.info(f"Downloading TROPOMI data from {start_date} to {end_date}")
        
        # Create geometry
        geometry = self.create_region_geometry()
        
        # Get image collection
        collection = (ee.ImageCollection(self.collection_name)
                     .filterDate(start_date, end_date)
                     .filterBounds(geometry)
                     .select(['CH4_column_volume_mixing_ratio_dry_air', 
                             'qa_value', 'cloud_fraction']))
        
        # Apply quality filters
        filtered_collection = self.apply_quality_filters(collection)
        
        # Convert to list for processing
        image_list = filtered_collection.getInfo()
        
        if not image_list['features']:
            logger.warning(f"No data found for period {start_date} to {end_date}")
            return None
        
        logger.info(f"Found {len(image_list['features'])} images")
        
        # Download and process each image
        datasets = []
        for img_info in image_list['features']:
            try:
                # Get image data
                img = ee.Image(img_info['id'])
                
                # Get image metadata
                date_str = img_info['properties']['system:time_start']
                date = datetime.fromtimestamp(int(date_str) / 1000)
                
                # Sample the image over the region
                sample = img.sampleRectangle(
                    region=geometry,
                    defaultValue=0,
                    numPixels=1e8
                )
                
                # Get the data
                data = sample.getInfo()
                
                # Convert to xarray
                ds = self._ee_to_xarray(data, date)
                datasets.append(ds)
                
            except Exception as e:
                logger.warning(f"Failed to process image {img_info['id']}: {e}")
                continue
        
        if not datasets:
            logger.error("No valid datasets processed")
            return None
        
        # Combine all datasets
        combined_ds = xr.concat(datasets, dim='time')
        
        logger.info(f"Successfully downloaded data with shape: {combined_ds.dims}")
        return combined_ds
    
    def _ee_to_xarray(self, ee_data: Dict, timestamp: datetime) -> xr.Dataset:
        """Convert Earth Engine data to xarray Dataset."""
        
        # Extract arrays from Earth Engine format
        ch4_data = ee_data['properties']['CH4_column_volume_mixing_ratio_dry_air']
        qa_data = ee_data['properties']['qa_value']
        
        # Get dimensions
        height = len(ch4_data)
        width = len(ch4_data[0]) if height > 0 else 0
        
        if height == 0 or width == 0:
            raise ValueError("Empty data array")
        
        # Create coordinate arrays
        roi = self.config['data']['region_of_interest']['coordinates']
        lons = np.linspace(roi[0], roi[2], width)
        lats = np.linspace(roi[3], roi[1], height)  # Note: reversed for image coordinates
        
        # Create xarray Dataset
        ds = xr.Dataset({
            'ch4': (['lat', 'lon'], np.array(ch4_data)),
            'qa_value': (['lat', 'lon'], np.array(qa_data)),
        }, coords={
            'lat': lats,
            'lon': lons,
            'time': timestamp
        })
        
        # Add attributes
        ds.attrs['source'] = 'TROPOMI/Sentinel-5P'
        ds.attrs['collection'] = self.collection_name
        
        return ds.expand_dims('time')
