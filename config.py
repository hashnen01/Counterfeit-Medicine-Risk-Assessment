"""
Configuration settings and path definitions for Counterfeit Medicine Risk Assessment.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "medicine_dataset.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "counterfeit_model.pkl")

DB_PATH = os.path.join(BASE_DIR, "database.db")

EDA_DIR = os.path.join(BASE_DIR, "reports", "eda")
