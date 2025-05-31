#!/usr/bin/env python3
"""
Quick script to test detection on your existing data
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def analyze_existing_data():
    """Analyze the data that was just generated."""
    
    # Try to load the most recent data
    output_dir = Path("./data/outputs")
    
    # Check if we have saved netcdf data
    nc_files = list(output_dir.glob("**/*.nc"))
    if nc_files:
        print(f"Found NetCDF file: {nc_files[0]}")
        ds = xr.open_dataset(nc_files[0])
    else:
        print("No NetCDF found, will need to regenerate data")
        return
    
    print(f"Dataset structure: {ds}")
    print(f"CH4 data shape: {ds.ch4.shape}")
    print(f"CH4 statistics:")
    print(f"  Mean: {ds.ch4.mean().values:.2f} ppb")
    print(f"  Std:  {ds.ch4.std().values:.2f} ppb")
    print(f"  Min:  {ds.ch4.min().values:.2f} ppb")
    print(f"  Max:  {ds.ch4.max().values:.2f} ppb")
    
    # Calculate enhancements manually
    background = ds.ch4.median(dim='time')
    enhancement = ds.ch4 - background
    
    print(f"\nEnhancement statistics:")
    print(f"  Mean: {enhancement.mean().values:.2f} ppb")
    print(f"  Std:  {enhancement.std().values:.2f} ppb")
    print(f"  Min:  {enhancement.min().values:.2f} ppb")
    print(f"  Max:  {enhancement.max().values:.2f} ppb")
    
    # Find potential hotspots with very low threshold
    threshold = 2.0  # ppb
    hotspots = enhancement > threshold
    
    print(f"\nWith {threshold} ppb threshold:")
    print(f"  Hotspot pixels: {hotspots.sum().values}")
    print(f"  Total pixels: {hotspots.size}")
    print(f"  Hotspot fraction: {hotspots.sum().values / hotspots.size * 100:.1f}%")
    
    if hotspots.sum() > 0:
        print(f"  Max hotspot enhancement: {enhancement.where(hotspots).max().values:.2f} ppb")
    
    # Try even lower threshold
    threshold = 0.5  # ppb
    hotspots_low = enhancement > threshold
    
    print(f"\nWith {threshold} ppb threshold:")
    print(f"  Hotspot pixels: {hotspots_low.sum().values}")
    print(f"  Hotspot fraction: {hotspots_low.sum().values / hotspots_low.size * 100:.1f}%")
    
    # Create a quick plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    ds.ch4.isel(time=0).plot()
    plt.title('CH4 Concentrations (t=0)')
    
    plt.subplot(1, 3, 2)
    enhancement.isel(time=0).plot()
    plt.title('Enhancements (t=0)')
    
    plt.subplot(1, 3, 3)
    enhancement.max(dim='time').plot()
    plt.title('Max Enhancement')
    
    plt.tight_layout()
    plt.savefig('quick_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return enhancement

def run_simple_detection(enhancement):
    """Run a simple detection algorithm."""
    print(f"\n=== SIMPLE DETECTION TEST ===")
    
    # Very simple detection
    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for threshold in thresholds:
        hotspots = enhancement > threshold
        n_hotspots = hotspots.sum().values
        print(f"Threshold {threshold:3.1f} ppb: {n_hotspots:4d} hotspot pixels")
        
        if n_hotspots > 0:
            print(f"  → This threshold would detect hotspots!")
    
    # Recommend optimal threshold
    std_enh = enhancement.std().values
    mean_enh = enhancement.mean().values
    
    suggested_threshold = mean_enh + 1.0 * std_enh
    print(f"\nSuggested threshold: {suggested_threshold:.2f} ppb")
    print(f"(mean + 1*std = {mean_enh:.2f} + {std_enh:.2f})")

def main():
    print("🔍 Quick Detection Analysis")
    print("=" * 40)
    
    try:
        enhancement = analyze_existing_data()
        if enhancement is not None:
            run_simple_detection(enhancement)
        
        print(f"\n💡 Recommendations:")
        print(f"1. Use anomaly_threshold: 0.5 or lower")
        print(f"2. Use min_enhancement: 0.5 or lower") 
        print(f"3. Check the quick_analysis.png plot")
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"\nTo use this script, first run the main pipeline to generate data:")
        print(f"python src/main.py --start-date 2023-06-01 --end-date 2023-06-03")

if __name__ == "__main__":
    main()