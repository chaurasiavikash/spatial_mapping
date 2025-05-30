
# ============================================================================
# FILE: src/detection/__init__.py
# ============================================================================
"""Detection modules for methane hotspots."""

from .anomaly_detector import MethaneAnomalyDetector
from .quantifier import EmissionQuantifier

__all__ = ['MethaneAnomalyDetector', 'EmissionQuantifier']
