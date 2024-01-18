# -*- coding: utf-8 -*-
"""

@author: mansour
"""

from logging import INFO, basicConfig
from os import environ
from pathlib import Path

from gymnasium.envs.registration import register

# ------------------ Mandatory Imports ------------------
import gym_anytrading

# ------------------ Trading Environment ----------------
register(id="custom-stocks-v0", entry_point="modeler.environments:CustomStocksEnv")

# ------------------ Default Values ---------------------
_DEFAULT_STOCK_SYMBOL = "GOOGL"
_DEFAULT_TIMEZONE = "UTC"
_DEFAULT_TRAINING_WINDOW_SIZE = 5
_DEFAULT_MODEL_ENV_ID = "custom-stocks-v0"
_DEFAULT_VERBOSE = 1
_DEFAULT_MODEL_STRUCTURE_POLICY = "MlpLstmPolicy"
_DEFAULT_MODEL_LOCATION = "models"
_DEFAULT_TOTAL_STEPS = 100000
_DEFAULT_TRAINING_BOUNDARY_LIMIT = 100
_DEFAULT_CALLBACK_DELAY_STEPS = 10
_DEFAULT_CALLBACK_PATIENCE = 5
_DEFAULT_EVALUATION_FREQUENCY = 1000
_DEFAULT_LOG_LOCATION = "logs"

# ------------------ Configured Values ------------------
STOCK_SYMBOL = environ.get("stock_symbol", _DEFAULT_STOCK_SYMBOL)
TIMEZONE = environ.get("timezone", _DEFAULT_TIMEZONE)
TRAINING_WINDOW_SIZE = int(environ.get("training_window_size", _DEFAULT_TRAINING_WINDOW_SIZE))
MODEL_ENV_ID = environ.get("model_env_id", _DEFAULT_MODEL_ENV_ID)
VERBOSE = int(environ.get("verbose", _DEFAULT_VERBOSE))
MODEL_STRUCTURE_POLICY = environ.get("model_structure_policy", _DEFAULT_MODEL_STRUCTURE_POLICY)
MODEL_LOCATION = environ.get("output_location", _DEFAULT_MODEL_LOCATION)
TOTAL_STEPS = int(environ.get("total_step", _DEFAULT_TOTAL_STEPS))
TRAINING_BOUNDARY_LIMIT = int(environ.get("training_boundary_limit", _DEFAULT_TRAINING_BOUNDARY_LIMIT))
CALLBACK_DELAY_STEPS = int(environ.get("callback_delay_steps", _DEFAULT_CALLBACK_DELAY_STEPS))
CALLBACK_PATIENCE = int(environ.get("callback_patience", _DEFAULT_CALLBACK_PATIENCE))
EVALUATION_FREQUENCY = int(environ.get("evaluation_frequency", _DEFAULT_EVALUATION_FREQUENCY))
DATA_START_DATE = environ.get("data_start_date")
DATA_END_DATE = environ.get("data_end_date")
LOG_LOCATION = environ.get("log_location", _DEFAULT_LOG_LOCATION)

# ------------------ Logging Configs ------------------
_log_location = Path(LOG_LOCATION)
_log_location.mkdir(parents=True, exist_ok=True)
basicConfig(filename=_log_location.joinpath("logs"), encoding='utf-8', level=INFO)
