# ============================================================================
# FILE: src/main.py
# ============================================================================
import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
import xarray as xr
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config.logging_config import setup_logging
from data.downloader import TROPOMIDownloader
from data.preprocessor import TROPOMIPreprocessor
from detection.anomaly_detector import MethaneAnomalyDetector
from detection.quantifier import EmissionQuantifier
from visualization.map_plotter import MethaneMapPlotter
from visualization.dashboard import MethaneDashboard

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise

def create_output_directories(base_path: str):
    """Create necessary output directories."""
    output_dirs = [
        'raw_data',
        'processed_data', 
        'detection_results',
        'visualizations',
        'reports'
    ]
    
    for dir_name in output_dirs:
        dir_path = Path(base_path, dir_name)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def save_results(ds: xr.Dataset, emission_data: pd.DataFrame, 
                output_path: str, config: dict):
    """Save results in multiple formats."""
    
    output_formats = config['output']['export_formats']
    
    try:
        # Save processed dataset
        if 'netcdf' in output_formats:
            nc_path = os.path.join(output_path, 'processed_data', 'tropomi_processed.nc')
            ds.to_netcdf(nc_path)
            logger.info(f"Saved processed dataset as NetCDF: {nc_path}")
        
        # Save emission results
        if 'csv' in output_formats:
            csv_path = os.path.join(output_path, 'detection_results', 'emissions.csv')
            emission_data.to_csv(csv_path, index=False)
            logger.info(f"Saved emission results as CSV: {csv_path}")
        
        if 'geojson' in output_formats and not emission_data.empty:
            # Convert to GeoDataFrame for GeoJSON export
            try:
                import geopandas as gpd
                from shapely.geometry import Point
                
                geometry = [Point(lon, lat) for lon, lat in 
                           zip(emission_data['center_lon'], emission_data['center_lat'])]
                
                gdf = gpd.GeoDataFrame(emission_data, geometry=geometry)
                geojson_path = os.path.join(output_path, 'detection_results', 'hotspots.geojson')
                gdf.to_file(geojson_path, driver='GeoJSON')
                logger.info(f"Saved hotspots as GeoJSON: {geojson_path}")
                
            except ImportError:
                logger.warning("GeoPandas not available, skipping GeoJSON export")
            except Exception as e:
                logger.warning(f"Failed to save GeoJSON: {e}")
                
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def validate_config(config: dict) -> bool:
    """Validate configuration parameters."""
    required_keys = [
        'gee.project_id',
        'data.start_date',
        'data.end_date',
        'data.region_of_interest',
        'output.base_path'
    ]
    
    for key_path in required_keys:
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
        except KeyError:
            logger.error(f"Missing required configuration key: {key_path}")
            return False
    
    # Validate date format
    try:
        datetime.strptime(config['data']['start_date'], '%Y-%m-%d')
        datetime.strptime(config['data']['end_date'], '%Y-%m-%d')
    except ValueError:
        logger.error("Invalid date format. Use YYYY-MM-DD")
        return False
    
    # Validate region of interest
    roi = config['data']['region_of_interest']
    if roi['type'] not in ['bbox', 'polygon']:
        logger.error("Region type must be 'bbox' or 'polygon'")
        return False
    
    if roi['type'] == 'bbox' and len(roi['coordinates']) != 4:
        logger.error("Bounding box must have 4 coordinates [west, south, east, north]")
        return False
    
    logger.info("Configuration validation passed")
    return True

def run_pipeline(config_path: str, start_date: str = None, end_date: str = None,
                verbose: bool = False) -> tuple:
    """Run the complete TROPOMI methane hotspot detection pipeline."""
    
    # Load configuration
    config = load_config(config_path)
    
    # Override dates if provided
    if start_date:
        config['data']['start_date'] = start_date
    if end_date:
        config['data']['end_date'] = end_date
    
    # Validate configuration
    if not validate_config(config):
        raise ValueError("Configuration validation failed")
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    log_file = os.path.join(config['output']['base_path'], 'pipeline.log')
    setup_logging(log_level=log_level, log_file=log_file)
    
    logger.info("=" * 60)
    logger.info("TROPOMI Methane Hotspot Detection Pipeline Started")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Date range: {config['data']['start_date']} to {config['data']['end_date']}")
    logger.info(f"Region: {config['data']['region_of_interest']}")
    
    # Create output directories
    create_output_directories(config['output']['base_path'])
    
    try:
        # Step 1: Download data
        logger.info("🛰️  Step 1: Downloading TROPOMI data")
        downloader = TROPOMIDownloader(config)
        raw_dataset = downloader.download_data(
            config['data']['start_date'],
            config['data']['end_date']
        )
        
        if raw_dataset is None:
            logger.error("Failed to download data. No data available for the specified parameters.")
            return None, None
        
        logger.info(f"Downloaded dataset with {len(raw_dataset.time)} time steps")
        
        # Step 2: Preprocess data
        logger.info("🔧 Step 2: Preprocessing data")
        preprocessor = TROPOMIPreprocessor(config)
        processed_dataset = preprocessor.preprocess_dataset(raw_dataset)
        
        # Log statistics
        stats = preprocessor.calculate_statistics(processed_dataset)
        logger.info("Dataset Statistics:")
        for var_name, var_stats in stats.items():
            logger.info(f"  {var_name}: mean={var_stats['mean']:.2f}, "
                       f"std={var_stats['std']:.2f}, count={var_stats['count']}")
        
        # Step 3: Detect hotspots
        logger.info("🔍 Step 3: Detecting methane hotspots")
        detector = MethaneAnomalyDetector(config)
        detected_dataset = detector.detect_hotspots(processed_dataset)
        
        # Extract hotspot features
        hotspot_features = detector.extract_hotspot_features(detected_dataset)
        logger.info(f"Detected {len(hotspot_features)} hotspot instances")
        
        if len(hotspot_features) == 0:
            logger.warning("No hotspots detected. Consider adjusting detection parameters.")
        
        # Step 4: Quantify emissions
        logger.info("📊 Step 4: Quantifying emissions")
        quantifier = EmissionQuantifier(config)
        
        if len(hotspot_features) > 0:
            emission_results = quantifier.quantify_emissions(detected_dataset, hotspot_features)
            emission_results = quantifier.estimate_uncertainty(emission_results)
            
            total_emissions = emission_results['emission_rate_kg_hr'].sum()
            max_emission = emission_results['emission_rate_kg_hr'].max()
            logger.info(f"Total estimated emissions: {total_emissions:.2f} kg/hr")
            logger.info(f"Maximum single source: {max_emission:.2f} kg/hr")
        else:
            emission_results = pd.DataFrame()
        
        # Step 5: Create visualizations
        if config['output']['create_plots']:
            logger.info("📈 Step 5: Creating visualizations")
            plotter = MethaneMapPlotter(config)
            
            viz_path = os.path.join(config['output']['base_path'], 'visualizations')
            
            try:
                # Enhancement map for first time step
                fig_enh = plotter.plot_enhancement_map(
                    detected_dataset, time_idx=0,
                    save_path=os.path.join(viz_path, 'enhancement_map.png')
                )
                plt.close(fig_enh)
                
                # Hotspots map if emissions detected
                if not emission_results.empty:
                    fig_hotspots = plotter.plot_hotspots_map(
                        detected_dataset, emission_results, time_idx=0,
                        save_path=os.path.join(viz_path, 'hotspots_map.png')
                    )
                    plt.close(fig_hotspots)
                    
                    # Time series
                    fig_ts = plotter.plot_time_series(
                        emission_results,
                        save_path=os.path.join(viz_path, 'emission_timeseries.png')
                    )
                    plt.close(fig_ts)
                    
                    # Interactive map
                    interactive_map = plotter.create_interactive_map(
                        detected_dataset, emission_results,
                        save_path=os.path.join(viz_path, 'interactive_map.html')
                    )
                    
                logger.info("Visualizations created successfully")
                
            except Exception as e:
                logger.warning(f"Some visualizations failed: {e}")
        
        # Step 6: Save results
        logger.info("💾 Step 6: Saving results")
        save_results(detected_dataset, emission_results, 
                    config['output']['base_path'], config)
        
        # Summary
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📁 Output directory: {config['output']['base_path']}")
        logger.info(f"🔍 Hotspots detected: {len(emission_results)}")
        if not emission_results.empty:
            logger.info(f"📊 Total emissions: {emission_results['emission_rate_kg_hr'].sum():.2f} kg/hr")
        logger.info("=" * 60)
        
        return detected_dataset, emission_results
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.exception("Full traceback:")
        raise

def run_dashboard_mode(config_path: str):
    """Run the pipeline and launch interactive dashboard."""
    try:
        # Run pipeline first
        dataset, emissions = run_pipeline(config_path)
        
        if dataset is not None:
            # Launch dashboard
            config = load_config(config_path)
            dashboard = MethaneDashboard(config)
            
            logger.info("🚀 Launching interactive dashboard...")
            dashboard.run_dashboard(dataset, emissions)
        else:
            logger.error("Cannot launch dashboard - no data available")
            
    except Exception as e:
        logger.error(f"Dashboard mode failed: {e}")
        raise

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='TROPOMI Methane Hotspot Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python main.py
  
  # Run with custom config and dates
  python main.py --config config/config.yaml --start-date 2023-06-01 --end-date 2023-06-07
  
  # Run and launch dashboard
  python main.py --dashboard
  
  # Verbose output
  python main.py --verbose
        """
    )
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file (default: config/config.yaml)')
    parser.add_argument('--start-date', type=str, 
                       help='Start date (YYYY-MM-DD), overrides config')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD), overrides config')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch interactive dashboard after pipeline')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--test', action='store_true',
                       help='Run with test data (small region, short time)')
    
    args = parser.parse_args()
    
    # Test mode - override with small parameters
    if args.test:
        print("🧪 Running in TEST mode with small dataset...")
        args.start_date = "2023-06-01"
        args.end_date = "2023-06-03"  # Just 3 days
    
    try:
        if args.dashboard:
            run_dashboard_mode(args.config)
        else:
            # Run pipeline
            dataset, emissions = run_pipeline(
                args.config, 
                args.start_date, 
                args.end_date,
                args.verbose
            )
            
            # Print summary
            if dataset is not None and emissions is not None:
                print(f"\n🎉 SUCCESS! Detected {len(emissions)} hotspots")
                if not emissions.empty:
                    print(f"📊 Total emissions: {emissions['emission_rate_kg_hr'].sum():.2f} kg/hr")
                print(f"📁 Results saved to: {Path(args.config).parent.parent / 'data' / 'outputs'}")
            else:
                print("\n⚠️  No hotspots detected. Try adjusting detection parameters.")
                
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()



