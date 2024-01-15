# -*- coding: utf-8 -*-
"""
Created on January 08, 2024

@author: mansour
"""

from json import dump
from logging import getLogger
from pathlib import Path

import gymnasium as gym
from pandas import DataFrame
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

from configs.constants import (CALLBACK_DELAY_STEPS, CALLBACK_PATIENCE, EVALUATION_FREQUENCY, MODEL_ENV_ID,
                               MODEL_STRUCTURE_POLICY, OUTPUT_LOCATION, STOCK_SYMBOL, TOTAL_STEPS,
                               TRAINING_BOUNDARY_LIMIT, TRAINING_WINDOW_SIZE, VERBOSE)
from data_handler.stock_data import get_stock_data

log = getLogger()


class Trainer:
    def __init__(self, data: DataFrame, frame_bound_max: int = TRAINING_BOUNDARY_LIMIT,
                 window_size: int = TRAINING_WINDOW_SIZE, env_id: str = MODEL_ENV_ID,
                 verbose: bool = VERBOSE, policy: str = MODEL_STRUCTURE_POLICY,
                 output_location: str or Path = OUTPUT_LOCATION, total_steps: int = TOTAL_STEPS):
        self.frame_bound = (window_size, frame_bound_max)
        self.window_size = window_size
        self.verbose = verbose
        self.env_id = env_id
        self.policy = policy
        self.output_location = Path(output_location)
        self.output_location.mkdir(parents=True, exist_ok=True)
        self.total_steps = total_steps
        self.env = gym.make(self.env_id, df=data, frame_bound=self.frame_bound, window_size=window_size)
        self.model = RecurrentPPO(policy, self.env, n_steps=frame_bound_max, verbose=self.verbose)

    def train(self, clean: bool = True):
        best_model_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=CALLBACK_PATIENCE,
                                                               min_evals=CALLBACK_DELAY_STEPS, verbose=self.verbose)
        eval_env = gym.make(self.env_id, window_size=self.window_size)
        eval_env = Monitor(eval_env, self.output_location.as_posix())

        eval_callback = EvalCallback(eval_env, eval_freq=EVALUATION_FREQUENCY, callback_after_eval=best_model_callback,
                                     verbose=self.verbose)
        self.model.learn(total_timesteps=self.total_steps, callback=eval_callback)

        if clean:
            self.output_location.joinpath("monitor.csv").unlink(missing_ok=True)

        return self

    def save_mmodel(self):
        configs = {"env_id": self.env_id, "frame_bound": self.frame_bound, "window_size": self.window_size}

        with open(self.output_location.joinpath(f"{STOCK_SYMBOL}_configs.json"), "w") as file:
            dump(configs, file, indent=4)

        self.model.save(self.output_location.joinpath(f"{STOCK_SYMBOL}.zip").as_posix())

        log.info(f"Stored the model and its configs in: {self.output_location.as_posix()}")


if __name__ == "__main__":
    fetched_data = get_stock_data(symbol="GOOG", start="2022-01-01", end="2023-01-01")
    trainer = Trainer(fetched_data)
    trainer.train().save_mmodel()
