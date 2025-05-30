
# ============================================================================
# FILE: src/detection/quantifier.py
# ============================================================================
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import xarray as xr

logger = logging.getLogger(__name__)

class EmissionQuantifier:
    """Quantify methane emissions from detected hotspots."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Physical constants
        self.MOLECULAR_WEIGHT_CH4 = 16.04  # g/mol
        self.AVOGADRO = 6.022e23
        self.STANDARD_PRESSURE = 1013.25  # hPa
        self.STANDARD_TEMPERATURE = 273.15  # K
        
    def quantify_emissions(self, ds: xr.Dataset, hotspot_features: pd.DataFrame) -> pd.DataFrame:
        """Quantify emissions for all detected hotspots."""
        logger.info("Quantifying methane emissions")
        
        emission_estimates = []
        
        for _, hotspot in hotspot_features.iterrows():
            try:
                # Calculate emission rate
                emission_rate = self.calculate_emission_rate(hotspot, ds)
                
                # Add to results
                result = hotspot.to_dict()
                result.update(emission_rate)
                emission_estimates.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to quantify emissions for hotspot {hotspot['hotspot_id']}: {e}")
                continue
        
        return pd.DataFrame(emission_estimates)
    
    def calculate_emission_rate(self, hotspot: pd.Series, ds: xr.Dataset) -> Dict:
        """Calculate emission rate for a single hotspot using mass balance approach."""
        
        # Simple emission estimation using enhancement and wind
        # This is a simplified approach - in practice, you'd use more sophisticated models
        
        enhancement = hotspot['mean_enhancement']  # ppb
        area = hotspot['area_pixels']
        
        # Convert area from pixels to km²
        # Assume TROPOMI pixel size ~7km x 3.5km
        pixel_area_km2 = 7.0 * 3.5  # km²
        total_area_km2 = area * pixel_area_km2
        
        # Estimate atmospheric column enhancement
        # Simplified approach - assumes well-mixed boundary layer
        boundary_layer_height = 1000.0  # meters (typical value)
        
        # Convert enhancement from ppb to kg/m²
        # This is a very simplified conversion
        enhancement_kg_m2 = self._ppb_to_kg_per_m2(enhancement, boundary_layer_height)
        
        # Estimate emission rate (very simplified)
        # In practice, you'd use atmospheric transport models
        wind_speed = 5.0  # m/s (assumed)
        emission_rate_kg_s = (enhancement_kg_m2 * total_area_km2 * 1e6 * wind_speed) / boundary_layer_height
        
        # Convert to common units
        emission_rate_kg_hr = emission_rate_kg_s * 3600
        emission_rate_tonnes_yr = emission_rate_kg_hr * 24 * 365 / 1000
        
        return {
            'emission_rate_kg_s': emission_rate_kg_s,
            'emission_rate_kg_hr': emission_rate_kg_hr,
            'emission_rate_tonnes_yr': emission_rate_tonnes_yr,
            'area_km2': total_area_km2,
            'enhancement_kg_m2': enhancement_kg_m2,
            'boundary_layer_height_m': boundary_layer_height,
            'assumed_wind_speed_ms': wind_speed
        }
    
    def _ppb_to_kg_per_m2(self, concentration_ppb: float, column_height_m: float) -> float:
        """Convert methane concentration from ppb to kg/m² column density."""
        
        # Convert ppb to molecules/m³ at standard conditions
        # ppb = parts per billion by volume
        molecules_per_m3 = (concentration_ppb * 1e-9 * self.AVOGADRO * 
                           self.STANDARD_PRESSURE * 100) / (8.314 * self.STANDARD_TEMPERATURE)
        
        # Convert to kg/m³
        kg_per_m3 = molecules_per_m3 * self.MOLECULAR_WEIGHT_CH4 / (self.AVOGADRO * 1000)
        
        # Multiply by column height to get kg/m²
        kg_per_m2 = kg_per_m3 * column_height_m
        
        return kg_per_m2