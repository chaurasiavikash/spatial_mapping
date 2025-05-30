# ============================================================================
# FILE: src/visualization/dashboard.py (STANDALONE STREAMLIT APP)
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def load_data():
    """Load the latest pipeline results."""
    try:
        # Path to the results
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'outputs'
        
        # Load emission results
        emissions_file = output_dir / 'detection_results' / 'emissions.csv'
        
        if emissions_file.exists():
            emissions = pd.read_csv(emissions_file)
            st.success(f"✅ Loaded {len(emissions)} hotspots from latest run")
            return emissions
        else:
            st.error("❌ No results found. Run the pipeline first:")
            st.code("python src/main.py --test")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def main():
    """Main Streamlit dashboard."""
    
    # Page configuration
    st.set_page_config(
        page_title="TROPOMI Methane Hotspot Dashboard",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and header
    st.title("🛰️ TROPOMI Methane Hotspot Detection Dashboard")
    st.markdown("Interactive analysis of satellite-detected methane emissions")
    st.markdown("---")
    
    # Load data
    emission_data = load_data()
    
    if emission_data.empty:
        st.warning("No data available. Please run the pipeline first.")
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters & Controls")
    
    # Date filtering
    if 'time' in emission_data.columns:
        emission_data['time'] = pd.to_datetime(emission_data['time'])
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(emission_data['time'].min().date(), emission_data['time'].max().date()),
            min_value=emission_data['time'].min().date(),
            max_value=emission_data['time'].max().date()
        )
        
        # Filter by date
        if len(date_range) == 2:
            start_date, end_date = date_range
            emission_data = emission_data[
                (emission_data['time'].dt.date >= start_date) & 
                (emission_data['time'].dt.date <= end_date)
            ]
    
    # Emission rate filtering
    if 'emission_rate_kg_hr' in emission_data.columns:
        min_emission = float(emission_data['emission_rate_kg_hr'].min())
        max_emission = float(emission_data['emission_rate_kg_hr'].max())
        
        emission_range = st.sidebar.slider(
            "Emission Rate Range (kg/hr)",
            min_value=min_emission,
            max_value=max_emission,
            value=(min_emission, max_emission),
            format="%.1f"
        )
        
        # Filter by emission rate
        emission_data = emission_data[
            (emission_data['emission_rate_kg_hr'] >= emission_range[0]) &
            (emission_data['emission_rate_kg_hr'] <= emission_range[1])
        ]
    
    # Dataset info sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Dataset Info")
    st.sidebar.metric("Filtered Hotspots", len(emission_data))
    if not emission_data.empty:
        st.sidebar.metric("Date Range", f"{len(emission_data['time'].dt.date.unique())} days")
        st.sidebar.metric("Avg Emission", f"{emission_data['emission_rate_kg_hr'].mean():.1f} kg/hr")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Interactive Map", "📈 Analytics", "📋 Data"])
    
    with tab1:
        create_overview_tab(emission_data)
    
    with tab2:
        create_map_tab(emission_data)
    
    with tab3:
        create_analytics_tab(emission_data)
    
    with tab4:
        create_data_tab(emission_data)

def create_overview_tab(emission_data):
    """Create the overview tab with key metrics."""
    st.header("📊 Detection Overview")
    
    if emission_data.empty:
        st.warning("No data available after filtering.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Hotspots",
            len(emission_data),
            delta=f"+{len(emission_data)} detected"
        )
    
    with col2:
        total_emissions = emission_data['emission_rate_kg_hr'].sum()
        st.metric(
            "Total Emissions",
            f"{total_emissions:.0f} kg/hr",
            delta=f"{total_emissions/1000:.1f} tonnes/hr"
        )
    
    with col3:
        max_emission = emission_data['emission_rate_kg_hr'].max()
        st.metric(
            "Largest Source",
            f"{max_emission:.0f} kg/hr",
            delta="Peak detection"
        )
    
    with col4:
        avg_emission = emission_data['emission_rate_kg_hr'].mean()
        st.metric(
            "Average Rate",
            f"{avg_emission:.0f} kg/hr",
            delta="Per hotspot"
        )
    
    # Distribution plots
    st.subheader("📈 Emission Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig_hist = px.histogram(
            emission_data,
            x='emission_rate_kg_hr',
            nbins=min(20, len(emission_data)),
            title="Distribution of Emission Rates",
            labels={'emission_rate_kg_hr': 'Emission Rate (kg/hr)', 'count': 'Number of Hotspots'},
            color_discrete_sequence=['#FF6B6B']
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot
        fig_box = px.box(
            emission_data,
            y='emission_rate_kg_hr',
            title="Emission Rate Distribution",
            labels={'emission_rate_kg_hr': 'Emission Rate (kg/hr)'},
            color_discrete_sequence=['#4ECDC4']
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Top emitters table
    st.subheader("🔥 Top 10 Emission Sources")
    top_emitters = emission_data.nlargest(10, 'emission_rate_kg_hr')[
        ['time', 'emission_rate_kg_hr', 'center_lat', 'center_lon', 'mean_enhancement']
    ].round(2)
    st.dataframe(top_emitters, use_container_width=True)

# ============================================================================
# QUICK FIX: Replace the create_map_tab function in dashboard.py
# ============================================================================

def create_map_tab(emission_data):
    """Create the interactive map tab."""
    st.header("🗺️ Interactive Hotspot Map")
    
    if emission_data.empty:
        st.warning("No data available after filtering.")
        return
    
    # Map center calculation
    center_lat = emission_data['center_lat'].mean()
    center_lon = emission_data['center_lon'].mean()
    
    # Create folium map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add alternative tile layers with proper attribution
    folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)',
        name='Terrain',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='CartoDB positron',
        name='Light Theme',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='CartoDB dark_matter',
        name='Dark Theme', 
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add hotspot markers
    for _, hotspot in emission_data.iterrows():
        # Create popup content
        popup_html = f"""
        <div style="width: 250px; font-family: Arial;">
            <h4 style="color: #d32f2f; margin-bottom: 10px;">🔥 Methane Hotspot</h4>
            <p><b>📅 Date:</b> {pd.to_datetime(hotspot['time']).strftime('%Y-%m-%d')}</p>
            <p><b>💨 Emission Rate:</b> {hotspot['emission_rate_kg_hr']:.1f} kg/hr</p>
            <p><b>📈 Enhancement:</b> {hotspot['mean_enhancement']:.1f} ppb</p>
            <p><b>📍 Location:</b> {hotspot['center_lat']:.3f}°, {hotspot['center_lon']:.3f}°</p>
            {f"<p><b>📊 Area:</b> {hotspot['area_km2']:.1f} km²</p>" if 'area_km2' in hotspot else ""}
            {f"<p><b>🎯 Uncertainty:</b> ±{hotspot['emission_uncertainty_percent']:.0f}%</p>" if 'emission_uncertainty_percent' in hotspot else ""}
        </div>
        """
        
        # Color and size based on emission rate
        max_emission = emission_data['emission_rate_kg_hr'].max()
        emission_ratio = hotspot['emission_rate_kg_hr'] / max_emission
        
        if emission_ratio > 0.7:
            color = 'red'
        elif emission_ratio > 0.4:
            color = 'orange'
        else:
            color = 'yellow'
        
        # Add marker
        folium.CircleMarker(
            location=[hotspot['center_lat'], hotspot['center_lon']],
            radius=8 + emission_ratio * 15,
            popup=folium.Popup(popup_html, max_width=300),
            color='black',
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Display map
    map_data = st_folium(m, width=700, height=500)
    
    # Show selection info
    if map_data['last_object_clicked_popup']:
        st.info("💡 Click on markers to see detailed information about each hotspot")
    
    # Map statistics
    st.subheader("🗺️ Map Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Latitude Range", f"{emission_data['center_lat'].min():.3f}° to {emission_data['center_lat'].max():.3f}°")
    with col2:
        st.metric("Longitude Range", f"{emission_data['center_lon'].min():.3f}° to {emission_data['center_lon'].max():.3f}°")
    with col3:
        lat_span = emission_data['center_lat'].max() - emission_data['center_lat'].min()
        lon_span = emission_data['center_lon'].max() - emission_data['center_lon'].min()
        area_approx = lat_span * lon_span * 111 * 111  # Rough km² calculation
        st.metric("Coverage Area", f"~{area_approx:.0f} km²")

def create_analytics_tab(emission_data):
    """Create the analytics tab with charts."""
    st.header("📈 Advanced Analytics")
    
    if emission_data.empty:
        st.warning("No data available after filtering.")
        return
    
    # Time series analysis
    if 'time' in emission_data.columns:
        st.subheader("⏰ Time Series Analysis")
        
        # Daily emissions
        daily_emissions = emission_data.groupby(emission_data['time'].dt.date).agg({
            'emission_rate_kg_hr': ['sum', 'mean', 'count']
        }).round(2)
        daily_emissions.columns = ['Total (kg/hr)', 'Average (kg/hr)', 'Count']
        daily_emissions = daily_emissions.reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_daily = px.line(
                daily_emissions,
                x='time',
                y='Total (kg/hr)',
                title="Daily Total Emissions",
                markers=True,
                color_discrete_sequence=['#FF6B6B']
            )
            st.plotly_chart(fig_daily, use_container_width=True)
        
        with col2:
            fig_count = px.bar(
                daily_emissions,
                x='time',
                y='Count',
                title="Daily Hotspot Count",
                color_discrete_sequence=['#4ECDC4']
            )
            st.plotly_chart(fig_count, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Variable Relationships")
    
    # Select numeric columns for correlation
    numeric_cols = emission_data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("X-axis variable:", numeric_cols, index=0)
        with col2:
            y_var = st.selectbox("Y-axis variable:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        
        if x_var != y_var:
            fig_scatter = px.scatter(
                emission_data,
                x=x_var,
                y=y_var,
                size='emission_rate_kg_hr',
                color='emission_rate_kg_hr',
                title=f"{y_var} vs {x_var}",
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

def create_data_tab(emission_data):
    """Create the data table tab."""
    st.header("📋 Detailed Data")
    
    if emission_data.empty:
        st.warning("No data available after filtering.")
        return
    
    # Data summary
    st.subheader("📊 Summary Statistics")
    if 'emission_rate_kg_hr' in emission_data.columns:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Emission", f"{emission_data['emission_rate_kg_hr'].mean():.1f} kg/hr")
        with col2:
            st.metric("Median Emission", f"{emission_data['emission_rate_kg_hr'].median():.1f} kg/hr")
        with col3:
            st.metric("Std Deviation", f"{emission_data['emission_rate_kg_hr'].std():.1f} kg/hr")
    
    # Full data table
    st.subheader("🗃️ Complete Dataset")
    
    # Format columns for display
    display_data = emission_data.copy()
    
    # Format time column
    if 'time' in display_data.columns:
        display_data['time'] = pd.to_datetime(display_data['time']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Round numeric columns
    numeric_columns = display_data.select_dtypes(include=[np.number]).columns
    display_data[numeric_columns] = display_data[numeric_columns].round(3)
    
    # Display with search and sort
    st.dataframe(
        display_data,
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = display_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"methane_hotspots_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
    
    # Data info
    st.subheader("ℹ️ Dataset Information")
    buffer = []
    buffer.append(f"**Rows:** {len(display_data)}")
    buffer.append(f"**Columns:** {len(display_data.columns)}")
    buffer.append(f"**Memory Usage:** {display_data.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    st.markdown(" | ".join(buffer))

# Class wrapper for backward compatibility
class MethaneDashboard:
    """Streamlit dashboard for methane hotspot analysis."""
    
    def __init__(self, config: dict):
        self.config = config
        
    def run_dashboard(self, ds, emission_data):
        """Run the dashboard (calls main function)."""
        main()

# Run the app
if __name__ == "__main__":
    main()