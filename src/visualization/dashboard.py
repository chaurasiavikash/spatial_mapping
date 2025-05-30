
# ============================================================================
# FILE: src/visualization/dashboard.py (COMPLETE IMPLEMENTATION)
# ============================================================================
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class MethaneDashboard:
    """Streamlit dashboard for methane hotspot analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def run_dashboard(self, ds, emission_data):
        """Run the complete dashboard."""
        try:
            import streamlit as st
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import folium
            from streamlit_folium import st_folium
        except ImportError:
            logger.warning("Streamlit or dashboard dependencies not available")
            print("Dashboard dependencies not installed. Install with:")
            print("pip install streamlit plotly streamlit-folium")
            return
        
        st.set_page_config(
            page_title="TROPOMI Methane Hotspot Dashboard",
            page_icon="🛰️",
            layout="wide"
        )
        
        st.title("🛰️ TROPOMI Methane Hotspot Detection Dashboard")
        st.markdown("---")
        
        # Basic metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Hotspots", len(emission_data))
        
        with col2:
            if not emission_data.empty:
                total_emissions = emission_data['emission_rate_kg_hr'].sum()
                st.metric("Total Emissions", f"{total_emissions:.1f} kg/hr")
            else:
                st.metric("Total Emissions", "0 kg/hr")
        
        with col3:
            if not emission_data.empty:
                max_emission = emission_data['emission_rate_kg_hr'].max()
                st.metric("Max Single Source", f"{max_emission:.1f} kg/hr")
            else:
                st.metric("Max Single Source", "0 kg/hr")
        
        # Data table
        if not emission_data.empty:
            st.subheader("Detected Hotspots")
            st.dataframe(emission_data)
        else:
            st.warning("No hotspots detected in this dataset.")
