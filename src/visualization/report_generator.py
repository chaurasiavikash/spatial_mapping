
# ============================================================================
# FILE: src/visualization/report_generator.py (COMPLETE IMPLEMENTATION)
# ============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generate comprehensive reports for methane detection results."""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def generate_summary_report(self, emission_data: pd.DataFrame, 
                               output_path: Optional[str] = None) -> str:
        """Generate a summary report of detection results."""
        
        if emission_data.empty:
            report = """
# Methane Hotspot Detection Report

## Summary
No methane hotspots were detected in the analyzed dataset.

## Recommendations
- Check detection parameters (lower thresholds)
- Verify data quality and coverage
- Consider different time periods or regions
"""
        else:
            total_emissions = emission_data['emission_rate_kg_hr'].sum()
            max_emission = emission_data['emission_rate_kg_hr'].max()
            avg_emission = emission_data['emission_rate_kg_hr'].mean()
            
            report = f"""
# Methane Hotspot Detection Report

## Summary
- **Total Hotspots Detected**: {len(emission_data)}
- **Total Emission Rate**: {total_emissions:.2f} kg/hr
- **Maximum Single Source**: {max_emission:.2f} kg/hr
- **Average Emission Rate**: {avg_emission:.2f} kg/hr

## Spatial Distribution
- **Latitude Range**: {emission_data['center_lat'].min():.3f}° to {emission_data['center_lat'].max():.3f}°
- **Longitude Range**: {emission_data['center_lon'].min():.3f}° to {emission_data['center_lon'].max():.3f}°

## Temporal Coverage
- **Date Range**: {emission_data['time'].min()} to {emission_data['time'].max()}
- **Number of Time Steps**: {emission_data['time'].nunique()}

## Top 5 Emission Sources
"""
            
            # Add top 5 sources
            top_sources = emission_data.nlargest(5, 'emission_rate_kg_hr')
            for i, (_, source) in enumerate(top_sources.iterrows(), 1):
                report += f"\n{i}. **{source['emission_rate_kg_hr']:.2f} kg/hr** at ({source['center_lat']:.3f}°, {source['center_lon']:.3f}°)"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report

# ============================================================================
# CHECK YOUR FILES AND MAKE SURE THESE ARE COMPLETE
# ============================================================================

# Also make sure you have the missing numpy import in downloader.py
# Add this line at the top of src/data/downloader.py:
# import numpy as np

# And in src/data/preprocessor.py, make sure you have:
# import numpy as np