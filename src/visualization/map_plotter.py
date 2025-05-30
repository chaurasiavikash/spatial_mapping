# ============================================================================
# FILE: src/visualization/map_plotter.py (COMPLETE IMPLEMENTATION)
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import logging
from typing import Dict, List, Optional, Tuple
import xarray as xr
from pathlib import Path

logger = logging.getLogger(__name__)

class MethaneMapPlotter:
    """Create maps and visualizations for methane hotspots."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.viz_config = config['visualization']
        
    def plot_enhancement_map(self, ds: xr.Dataset, time_idx: int = 0, 
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot methane enhancement map for a specific time."""
        logger.info(f"Creating enhancement map for time index {time_idx}")
        
        fig, ax = plt.subplots(figsize=self.viz_config['figsize'], dpi=self.viz_config['dpi'])
        
        # Get data for specific time
        enhancement = ds.enhancement.isel(time=time_idx)
        
        # Create the plot
        im = ax.pcolormesh(
            ds.lon, ds.lat, enhancement,
            cmap=self.viz_config['colormap'],
            shading='auto'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label('CH₄ Enhancement (ppb)', fontsize=12)
        
        # Set title
        time_str = pd.to_datetime(ds.time.isel(time=time_idx).values).strftime('%Y-%m-%d')
        plt.title(f'TROPOMI Methane Enhancement - {time_str}', fontsize=14, fontweight='bold')
        
        # Set labels
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
            logger.info(f"Enhancement map saved to {save_path}")
        
        return fig
    
    def plot_hotspots_map(self, ds: xr.Dataset, emission_data: pd.DataFrame,
                         time_idx: int = 0, save_path: Optional[str] = None) -> plt.Figure:
        """Plot detected hotspots overlaid on enhancement map."""
        logger.info(f"Creating hotspots map for time index {time_idx}")
        
        fig, ax = plt.subplots(figsize=self.viz_config['figsize'], dpi=self.viz_config['dpi'])
        
        # Plot background enhancement
        enhancement = ds.enhancement.isel(time=time_idx)
        im = ax.pcolormesh(
            ds.lon, ds.lat, enhancement,
            cmap='Blues', alpha=0.7,
            shading='auto'
        )
        
        # Plot hotspots
        if 'hotspot_labels' in ds.data_vars:
            hotspots = ds.hotspot_labels.isel(time=time_idx)
            hotspot_mask = hotspots > 0
            
            # Contour the hotspots
            if hotspot_mask.any():
                ax.contour(
                    ds.lon, ds.lat, hotspots,
                    levels=np.arange(1, hotspots.max().values + 1),
                    colors='red', linewidths=2
                )
        
        # Plot emission estimates as circles
        time_val = ds.time.isel(time=time_idx).values
        time_emissions = emission_data[emission_data['time'] == time_val]
        
        if not time_emissions.empty:
            # Scale circle sizes by emission rate
            max_emission = time_emissions['emission_rate_kg_hr'].max()
            sizes = (time_emissions['emission_rate_kg_hr'] / max_emission) * 200 + 50
            
            scatter = ax.scatter(
                time_emissions['center_lon'], time_emissions['center_lat'],
                s=sizes, c=time_emissions['emission_rate_kg_hr'],
                cmap='Reds', alpha=0.8, edgecolors='black', linewidth=1
            )
            
            # Add colorbar for emissions
            cbar2 = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
            cbar2.set_label('Emission Rate (kg/hr)', fontsize=10)
        
        # Set title
        time_str = pd.to_datetime(time_val).strftime('%Y-%m-%d')
        plt.title(f'Detected Methane Hotspots - {time_str}', fontsize=14, fontweight='bold')
        
        # Set labels
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
            logger.info(f"Hotspots map saved to {save_path}")
        
        return fig
    
    def create_interactive_map(self, ds: xr.Dataset, emission_data: pd.DataFrame,
                             save_path: Optional[str] = None):
        """Create interactive Folium map with all hotspots."""
        logger.info("Creating interactive map")
        
        try:
            import folium
            from folium import plugins
        except ImportError:
            logger.warning("Folium not available, skipping interactive map")
            return None
        
        # Calculate map center
        roi = self.config['data']['region_of_interest']['coordinates']
        center_lat = (roi[1] + roi[3]) / 2
        center_lon = (roi[0] + roi[2]) / 2
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Add hotspot markers
        for _, hotspot in emission_data.iterrows():
            popup_html = f"""
            <div style="width: 200px;">
                <h4>Methane Hotspot</h4>
                <p><b>Date:</b> {pd.to_datetime(hotspot['time']).strftime('%Y-%m-%d')}</p>
                <p><b>Emission Rate:</b> {hotspot['emission_rate_kg_hr']:.2f} kg/hr</p>
                <p><b>Enhancement:</b> {hotspot['mean_enhancement']:.1f} ppb</p>
                <p><b>Area:</b> {hotspot['area_km2']:.2f} km²</p>
            </div>
            """
            
            # Color based on emission rate
            max_emission = emission_data['emission_rate_kg_hr'].max()
            emission_ratio = hotspot['emission_rate_kg_hr'] / max_emission
            
            if emission_ratio > 0.7:
                color = 'darkred'
            elif emission_ratio > 0.4:
                color = 'orange'
            else:
                color = 'green'
            
            folium.CircleMarker(
                location=[hotspot['center_lat'], hotspot['center_lon']],
                radius=8 + emission_ratio * 12,
                popup=folium.Popup(popup_html, max_width=300),
                color='black',
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
        
        if save_path:
            m.save(save_path)
            logger.info(f"Interactive map saved to {save_path}")
        
        return m
    
    def plot_time_series(self, emission_data: pd.DataFrame, 
                        save_path: Optional[str] = None) -> plt.Figure:
        """Plot time series of total emissions."""
        logger.info("Creating emission time series plot")
        
        # Group by time and sum emissions
        time_series = emission_data.groupby('time')['emission_rate_kg_hr'].sum()
        
        fig, ax = plt.subplots(figsize=self.viz_config['figsize'], dpi=self.viz_config['dpi'])
        
        # Plot time series
        ax.plot(time_series.index, time_series.values, 'b-', linewidth=2, marker='o')
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Total Emission Rate (kg/hr)', fontsize=12)
        ax.set_title('Methane Emissions Time Series', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
            logger.info(f"Time series plot saved to {save_path}")
        
        return fig
