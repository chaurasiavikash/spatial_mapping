# TROPOMI Methane Hotspot Detection Pipeline

A comprehensive Python pipeline for detecting and quantifying methane emission hotspots using TROPOMI satellite data from Google Earth Engine. This project combines atmospheric data processing, statistical anomaly detection, and emission quantification to support environmental monitoring and climate research.

## Overview

The TROPOspheric Monitoring Instrument (TROPOMI) aboard the Sentinel-5P satellite provides daily global measurements of atmospheric methane concentrations. This pipeline automates the process of identifying statistically significant methane enhancements that may indicate emission sources such as oil and gas facilities, landfills, or other industrial activities.

### Key Features

- **Automated data acquisition** from Google Earth Engine TROPOMI collections
- **Statistical anomaly detection** using local enhancement thresholds and spatial clustering
- **Emission quantification** with uncertainty estimation using simplified atmospheric transport models  
- **Interactive visualizations** including maps, time series, and web-based dashboards
- **Multi-format outputs** supporting NetCDF, CSV, and GeoJSON for downstream analysis
- **Production-ready deployment** with Docker containerization and comprehensive testing

## Scientific Background

Methane (CH₄) is the second most important anthropogenic greenhouse gas, with atmospheric concentrations that have more than doubled since pre-industrial times. Satellite-based monitoring using instruments like TROPOMI enables global-scale detection of methane emission sources, supporting both scientific research and regulatory compliance efforts.

The pipeline implements a multi-step detection algorithm:
1. **Background estimation** using temporal and spatial statistical methods
2. **Enhancement calculation** relative to local atmospheric baselines  
3. **Statistical significance testing** using configurable z-score thresholds
4. **Spatial clustering** to identify connected emission regions
5. **Temporal persistence filtering** to reduce false positive detections

## Installation

### Prerequisites

- Python 3.9 or higher
- Google Earth Engine account with project access
- Conda (recommended for environment management)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/chaurasiavikash/spatial_mapping.git
cd spatial_mapping

# Create conda environment
conda create -n tropomi-pipeline python=3.9 -y
conda activate tropomi-pipeline

# Install dependencies
pip install -r requirements.txt

# Authenticate with Google Earth Engine
earthengine authenticate
```

### Configuration

1. **Set up Google Earth Engine project**:
   - Create a project at [Google Cloud Console](https://console.cloud.google.com/)
   - Enable the Earth Engine API for your project
   - Note your project ID

2. **Configure the pipeline**:
   ```bash
   # Edit config/config.yaml with your settings
   cp config/config.yaml config/local_config.yaml
   # Update project_id and region_of_interest in local_config.yaml
   ```

## Usage

### Quick Start

Test the pipeline with a small dataset:

```bash
python src/main.py --test --verbose
```

### Full Analysis

Run analysis for a specific region and time period:

```bash
python src/main.py --config config/config.yaml \
                   --start-date 2023-06-01 \
                   --end-date 2023-06-30
```

### Interactive Dashboard

Launch the web-based dashboard for exploring results:

```bash
python src/main.py --dashboard
```

### Command Line Options

```bash
python src/main.py --help

Options:
  --config PATH          Configuration file path (default: config/config.yaml)
  --start-date DATE      Start date in YYYY-MM-DD format
  --end-date DATE        End date in YYYY-MM-DD format  
  --dashboard           Launch interactive dashboard
  --verbose             Enable verbose logging
  --test                Run with test data (small region, short time)
```

## Configuration

The pipeline behavior is controlled through YAML configuration files. Key parameters include:

### Detection Parameters
```yaml
detection:
  background_percentile: 50    # Percentile for background calculation
  anomaly_threshold: 2.0       # Z-score threshold for anomaly detection
  min_enhancement: 20.0        # Minimum enhancement (ppb) above background
  spatial_window: 5            # Spatial window size for local statistics
  temporal_window: 7           # Temporal persistence requirement (days)
```

### Region of Interest
```yaml
data:
  region_of_interest:
    type: "bbox"
    coordinates: [-95.0, 29.0, -94.0, 30.0]  # [west, south, east, north]
```

## Methodology

### Data Processing

The pipeline processes TROPOMI Level 3 methane data through several stages:

1. **Quality filtering**: Removes low-quality retrievals based on QA flags and cloud coverage
2. **Outlier removal**: Statistical filtering using interquartile range or z-score methods
3. **Background calculation**: Temporal and spatial baseline estimation
4. **Enhancement computation**: Difference between observed and background concentrations

### Anomaly Detection

Statistical anomalies are identified using a two-stage approach:

1. **Local enhancement analysis**: Computing z-scores within sliding spatial windows
2. **Significance testing**: Applying both statistical (z > threshold) and absolute (enhancement > minimum) criteria

### Spatial Clustering

Connected anomalous pixels are grouped using image processing techniques:
- Connected component labeling with 8-pixel connectivity
- Minimum cluster size filtering to reduce noise
- Feature extraction for each identified cluster

### Emission Quantification

Emission rates are estimated using a simplified mass balance approach:

```
E = (ΔC × A × U) / H
```

Where:
- E = emission rate (kg/hr)
- ΔC = methane enhancement (kg/m²)
- A = source area (m²)  
- U = wind speed (m/s)
- H = boundary layer height (m)

Uncertainty estimates incorporate measurement errors, meteorological variability, and model assumptions.

## Output Products

### Data Files
- **NetCDF**: Processed satellite data with all analysis variables
- **CSV**: Detected hotspot features and emission estimates
- **GeoJSON**: Geographic hotspot locations for GIS applications

### Visualizations  
- **Enhancement maps**: Spatial distribution of methane enhancements
- **Detection overlays**: Identified hotspots with emission estimates
- **Time series plots**: Temporal evolution of emissions
- **Interactive maps**: Web-based exploration interface

## Validation and Testing

The pipeline includes comprehensive testing capabilities:

### Unit Tests
```bash
# Run test suite
python -m pytest tests/

# Test specific components
python -m pytest tests/test_detector.py -v
```

### Validation Regions
- **Clean background areas** (e.g., Netherlands) for algorithm validation
- **Known emission sources** (e.g., Permian Basin, oil fields) for sensitivity testing
- **Urban areas** (e.g., Los Angeles) for complex source scenarios

## Performance and Limitations

### Computational Performance
- Processing time: ~2-5 minutes for weekly regional analysis
- Memory requirements: ~2-4 GB for typical datasets
- Google Earth Engine quotas: 15,000 requests/day (non-commercial use)

### Detection Limitations
- **Spatial resolution**: Limited by TROPOMI pixel size (~7×3.5 km)
- **Sensitivity threshold**: Minimum detectable sources ~1-10 tonnes CH₄/hr
- **Atmospheric conditions**: Reduced sensitivity under high cloud cover
- **Transport modeling**: Simplified assumptions may affect emission estimates

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository and create a feature branch
2. Add tests for new functionality
3. Ensure code follows PEP 8 style guidelines  
4. Update documentation for any API changes
5. Submit a pull request with a clear description

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code formatting
black src/ tests/
flake8 src/ tests/

# Run full test suite
pytest tests/ --cov=src
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{chaurasia2024tropomi,
  author = {Chaurasia, Vikash},
  title = {TROPOMI Methane Hotspot Detection Pipeline},
  year = {2024},
  url = {https://github.com/chaurasiavikash/spatial_mapping},
  note = {Software for satellite-based methane emission detection}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **ESA/Copernicus** for TROPOMI satellite data
- **Google Earth Engine** platform for data access and processing
- **SRON Netherlands Institute for Space Research** for TROPOMI algorithm development
- **TU Delft** Department of Biomechanical Engineering for institutional support

## Contact

**Vikash Chaurasia**  
Postdoctoral Researcher  
TU Delft, Netherlands  
Email: chaurasiavik@gmail.com  
GitHub: [@chaurasiavikash](https://github.com/chaurasiavikash)

## Related Publications

- Lorente, A., et al. (2021). Methane retrieved from TROPOMI: improvement of the data product and validation of the first 2 years of measurements. *Atmospheric Measurement Techniques*, 14(1), 665-684.
- Jacob, D. J., et al. (2022). Quantifying methane emissions from the global scale down to point sources using satellite observations of atmospheric methane. *Atmospheric Chemistry and Physics*, 22(14), 9617-9645.

---

For technical support, feature requests, or collaboration opportunities, please open an issue on GitHub or contact the maintainer directly.