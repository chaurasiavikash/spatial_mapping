#!/usr/bin/env python3
"""
Diagnostic script to debug TROPOMI data availability and extraction issues.
"""

import ee
import yaml
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

def initialize_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize()
        print("✅ Google Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize GEE: {e}")
        return False

def check_collection_availability(start_date, end_date, bbox):
    """Check what data is available in the TROPOMI collection."""
    print(f"\n📊 Checking TROPOMI collection for {start_date} to {end_date}")
    
    # Create geometry
    geometry = ee.Geometry.Rectangle(bbox)
    
    # Get collection
    collection = (ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4')
                 .filterDate(start_date, end_date)
                 .filterBounds(geometry))
    
    # Get basic info
    collection_info = collection.getInfo()
    num_images = len(collection_info['features'])
    print(f"📷 Found {num_images} images in collection")
    
    if num_images == 0:
        print("❌ No images found for this region and time period")
        return False
    
    # Check first few images
    for i, img_info in enumerate(collection_info['features'][:3]):
        img_id = img_info['id']
        img_time = img_info['properties']['system:time_start']
        date = datetime.fromtimestamp(int(img_time) / 1000)
        
        print(f"\n🖼️  Image {i+1}: {date}")
        print(f"   ID: {img_id}")
        
        # Check bands available
        img = ee.Image(img_id)
        band_names = img.bandNames().getInfo()
        print(f"   Available bands: {band_names}")
        
        # Try to get some basic statistics
        try:
            ch4_band = img.select('CH4_column_volume_mixing_ratio_dry_air')
            
            stats = ch4_band.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.count(), sharedInputs=True
                ).combine(
                    ee.Reducer.minMax(), sharedInputs=True
                ),
                geometry=geometry,
                scale=10000,  # 10km resolution
                maxPixels=1e6
            ).getInfo()
            
            print(f"   Stats: {stats}")
            
        except Exception as e:
            print(f"   ⚠️ Failed to get stats: {e}")
    
    return True

def test_simple_extraction(start_date, end_date, bbox):
    """Test simple data extraction."""
    print(f"\n🔬 Testing simple data extraction")
    
    geometry = ee.Geometry.Rectangle(bbox)
    
    # Get first image
    collection = (ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4')
                 .filterDate(start_date, end_date)
                 .filterBounds(geometry)
                 .limit(1))
    
    first_image = ee.Image(collection.first())
    
    try:
        # Try different sampling methods
        print("\n🎯 Method 1: Single point sampling")
        center_lon = (bbox[0] + bbox[2]) / 2
        center_lat = (bbox[1] + bbox[3]) / 2
        center_point = ee.Geometry.Point([center_lon, center_lat])
        
        point_value = first_image.select('CH4_column_volume_mixing_ratio_dry_air').sample(
            region=center_point, 
            scale=5000
        ).first().get('CH4_column_volume_mixing_ratio_dry_air').getInfo()
        
        print(f"   Center point value: {point_value}")
        
    except Exception as e:
        print(f"   ❌ Point sampling failed: {e}")
    
    try:
        print("\n🎯 Method 2: Region reduction")
        stats = first_image.select('CH4_column_volume_mixing_ratio_dry_air').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=10000,
            maxPixels=1e6
        ).getInfo()
        
        print(f"   Regional mean: {stats}")
        
    except Exception as e:
        print(f"   ❌ Region reduction failed: {e}")

def test_different_regions():
    """Test different known active regions."""
    print(f"\n🌍 Testing different regions")
    
    regions = {
        "Permian Basin": [-103.5, 31.0, -101.0, 33.5],
        "Bakken Formation": [-103.5, 47.0, -102.0, 48.5],
        "Algeria (Known source)": [1.0, 28.0, 3.0, 30.0],
        "Netherlands": [4.0, 51.5, 6.0, 53.0]
    }
    
    for region_name, bbox in regions.items():
        print(f"\n📍 Testing {region_name}: {bbox}")
        try:
            geometry = ee.Geometry.Rectangle(bbox)
            
            # Get recent data
            collection = (ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4')
                         .filterDate('2023-06-01', '2023-06-03')
                         .filterBounds(geometry)
                         .limit(1))
            
            if collection.size().getInfo() > 0:
                first_image = ee.Image(collection.first())
                stats = first_image.select('CH4_column_volume_mixing_ratio_dry_air').reduceRegion(
                    reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
                    geometry=geometry,
                    scale=10000,
                    maxPixels=1e6
                ).getInfo()
                
                mean_val = stats.get('CH4_column_volume_mixing_ratio_dry_air_mean')
                count = stats.get('CH4_column_volume_mixing_ratio_dry_air_count', 0)
                
                print(f"   ✅ Mean CH4: {mean_val}, Valid pixels: {count}")
                
                if mean_val and count > 0:
                    print(f"   🎉 SUCCESS! Found valid data in {region_name}")
            else:
                print(f"   ❌ No images found for {region_name}")
                
        except Exception as e:
            print(f"   ❌ Error testing {region_name}: {e}")

def suggest_improvements():
    """Suggest improvements based on findings."""
    print(f"\n💡 Suggestions for improvement:")
    print("1. Try different time periods (TROPOMI has data gaps)")
    print("2. Use larger spatial regions")
    print("3. Check for known methane emission areas")
    print("4. Reduce quality filtering thresholds")
    print("5. Try different collection versions (NRTI vs OFFL)")
    print("6. Consider using the NO2 collection first to test GEE connectivity")

def main():
    """Run diagnostic tests."""
    print("🚀 TROPOMI Data Diagnostic Tool")
    print("=" * 50)
    
    # Initialize GEE
    if not initialize_gee():
        return
    
    # Load config if available
    config_path = "config/config.yaml"
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        start_date = config['data']['start_date']
        end_date = config['data']['end_date']
        bbox = config['data']['region_of_interest']['coordinates']
    else:
        # Default test parameters
        start_date = "2023-06-01"
        end_date = "2023-06-03"
        bbox = [-103.0, 31.5, -101.5, 33.0]  # Permian Basin
    
    print(f"📅 Test period: {start_date} to {end_date}")
    print(f"📍 Test region: {bbox}")
    
    # Run tests
    if check_collection_availability(start_date, end_date, bbox):
        test_simple_extraction(start_date, end_date, bbox)
    
    # Test different regions
    test_different_regions()
    
    # Suggestions
    suggest_improvements()
    
    print(f"\n✅ Diagnostic complete!")

if __name__ == "__main__":
    main()