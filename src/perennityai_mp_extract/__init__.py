# perennityai_mp_extract/__init__.py

"""
Module Name: perennityai-mp-extract
Author: PerennityAI
Date: 2024-11-05
Revision: 1.0.0

Description:
This module contains the definition of perennityai-mp-extract.

"""

# Package-level variables
__version__ = "1.0.0"

# Import all modules
from perennityai_mp_extract.data_extraction import DataExtractor

# public classes that are available at the sub-package level
__all__ = [
           'DataExtractor', 
           ]
