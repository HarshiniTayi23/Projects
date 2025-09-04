"""
Swiggy App Review Trend Analysis System

A comprehensive automated pipeline for analyzing Google Play Store reviews
and generating trend analysis reports.
"""

from .database_setup import DatabaseManager
from .data_ingestion import DataIngestion
from .agentic_topic_detection import AgenticTopicDetector
from .topic_consolidation import AdvancedTopicConsolidator
from .trend_generator import AdvancedTrendGenerator

__version__ = "1.0.0"
__author__ = " "

__all__ = [
    "DatabaseManager",
    "DataIngestion", 
    "AgenticTopicDetector",
    "AdvancedTopicConsolidator",
    "AdvancedTrendGenerator"
]