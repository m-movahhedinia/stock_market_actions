# -*- coding: utf-8 -*-
"""
Created on January 09, 2024

@author: mansour
"""

from os import environ

# ------------------ Mandatory Imports ------------------
import gym_anytrading

# ------------------ Default Values ------------------
_DEFAULT_STOCK_SYMBOL = "GOOGL"
_DEFAULT_TIMEZONE = "UTC"
_DEFAULT_TRAINING_WINDOW_SIZE = 5
_DEFAULT_MODEL_ENV_ID = "stocks-v0"
_DEFAULT_VERBOSE = 1
_DEFAULT_MODEL_STRUCTURE_POLICY = "MlpLstmPolicy"
_DEFAULT_OUTPUT_LOCATION = "models"
_DEFAULT_TOTAL_STEPS = 100000
_DEFAULT_TRAINING_BOUNDARY_LIMIT = 100
_DEFAULT_CALLBACK_DELAY_STEPS = 10
_DEFAULT_CALLBACK_PATIENCE = 5
_DEFAULT_EVALUATION_FREQUENCY = 1000

# ---------------- Configured Values -----------------
STOCK_SYMBOL = environ.get("stock_symbol", _DEFAULT_STOCK_SYMBOL)
TIMEZONE = environ.get("timezone", _DEFAULT_TIMEZONE)
TRAINING_WINDOW_SIZE = int(environ.get("training_window_size", _DEFAULT_TRAINING_WINDOW_SIZE))
MODEL_ENV_ID = environ.get("model_env_id", _DEFAULT_MODEL_ENV_ID)
VERBOSE = int(environ.get("verbose", _DEFAULT_VERBOSE))
MODEL_STRUCTURE_POLICY = environ.get("model_structure_policy", _DEFAULT_MODEL_STRUCTURE_POLICY)
OUTPUT_LOCATION = environ.get("output_location", _DEFAULT_OUTPUT_LOCATION)
TOTAL_STEPS = int(environ.get("total_step", _DEFAULT_TOTAL_STEPS))
TRAINING_BOUNDARY_LIMIT = int(environ.get("training_boundary_limit", _DEFAULT_TRAINING_BOUNDARY_LIMIT))
CALLBACK_DELAY_STEPS = int(environ.get("callback_delay_steps", _DEFAULT_CALLBACK_DELAY_STEPS))
CALLBACK_PATIENCE = int(environ.get("callback_patience", _DEFAULT_CALLBACK_PATIENCE))
EVALUATION_FREQUENCY = int(environ.get("evaluation_frequency", _DEFAULT_EVALUATION_FREQUENCY))
