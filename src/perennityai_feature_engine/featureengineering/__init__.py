# featureengineering/__init__.py

"""
Module Name: FeatureEngineering
Author: Oladayo Luke, Ph.D
Date: 2024-10-25
Revision: 1.0.0

Description:
This module contains the definition of FeatureEngineering.

"""

# Package-level variables
__version__ = "1.0.0"

# Import all functions
from .facial_features_processor import FacialExpressionProcessor
from .feature_extractor import LandmarkFeatureExtractor
from .hand_features_processor import HandFeaturesProcessor


# public classes that are available at the sub-package level
__all__ = ['FacialExpressionProcessor', 
           'LandmarkFeatureExtractor', 
           'HandFeaturesProcessor', 
           ]