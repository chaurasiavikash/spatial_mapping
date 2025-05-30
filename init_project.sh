#!/bin/bash

# You're already inside tropomi-methane-pipeline/
# So we don't need to prefix with $PROJECT_NAME/

# Define directory structure
DIRS=(
  "config"
  "src"
  "src/data"
  "src/detection"
  "src/visualization"
  "src/utils"
  "tests/test_data"
  "notebooks"
  "data/raw"
  "data/processed"
  "data/outputs"
  "docs"
  "deployment"
)

# Define files to create
FILES=(
  "README.md"
  "requirements.txt"
  "setup.py"

  "config/__init__.py"
  "config/config.yaml"
  "config/logging_config.py"

  "src/__init__.py"
  "src/data/__init__.py"
  "src/data/downloader.py"
  "src/data/preprocessor.py"
  "src/data/validator.py"

  "src/detection/__init__.py"
  "src/detection/anomaly_detector.py"
  "src/detection/hotspot_identifier.py"
  "src/detection/quantifier.py"

  "src/visualization/__init__.py"
  "src/visualization/map_plotter.py"
  "src/visualization/dashboard.py"
  "src/visualization/report_generator.py"

  "src/utils/__init__.py"
  "src/utils/geo_utils.py"
  "src/utils/atmospheric_utils.py"
  "src/utils/validation_utils.py"

  "src/main.py"

  "tests/__init__.py"
  "tests/test_downloader.py"
  "tests/test_preprocessor.py"
  "tests/test_detector.py"

  "notebooks/01_data_exploration.ipynb"
  "notebooks/02_detection_algorithm_development.ipynb"
  "notebooks/03_validation_analysis.ipynb"

  "docs/technical_report.md"
  "docs/user_guide.md"

  "deployment/Dockerfile"
  "deployment/docker-compose.yml"
  "deployment/app.py"
)

# Create directories
for dir in "${DIRS[@]}"; do
  mkdir -p "$dir"
done

# Create empty files
for file in "${FILES[@]}"; do
  touch "$file"
done

echo "✅ Project structure created successfully in $(pwd)"
