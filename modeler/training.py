# -*- coding: utf-8 -*-
"""

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
                               MODEL_LOCATION, MODEL_STRUCTURE_POLICY, STOCK_SYMBOL, TOTAL_STEPS,
                               TRAINING_BOUNDARY_LIMIT, TRAINING_WINDOW_SIZE, VERBOSE)
from data_handler.stock_data import get_stock_data

log = getLogger()


class Trainer:
    def __init__(self, data: DataFrame, frame_bound_max: int = None,
                 window_size: int = None, env_id: str = None,
                 verbose: bool = None, policy: str = None,
                 model_location: str or Path = None, total_steps: int = None):
        self.window_size = window_size or TRAINING_WINDOW_SIZE
        self.frame_bound_max = frame_bound_max or TRAINING_BOUNDARY_LIMIT
        self.frame_bound = (self.window_size, self.frame_bound_max)
        self.verbose = verbose or VERBOSE
        self.env_id = env_id or MODEL_ENV_ID
        self.policy = policy or MODEL_STRUCTURE_POLICY
        self.output_location = Path(model_location or MODEL_LOCATION)
        self.output_location.mkdir(parents=True, exist_ok=True)
        self.total_steps = total_steps or TOTAL_STEPS
        self.data = data
        self.env = gym.make(self.env_id, df=data, frame_bound=self.frame_bound, window_size=self.window_size)
        self.model = RecurrentPPO(self.policy, self.env, n_steps=self.frame_bound_max, verbose=self.verbose)

    def train(self, clean: bool = True):
        best_model_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=CALLBACK_PATIENCE,
                                                               min_evals=CALLBACK_DELAY_STEPS, verbose=self.verbose)
        eval_env = gym.make(self.env_id, df=self.data, frame_bound=self.frame_bound, window_size=self.window_size)
        eval_env = Monitor(eval_env, self.output_location.as_posix())

        eval_callback = EvalCallback(eval_env, eval_freq=EVALUATION_FREQUENCY, callback_after_eval=best_model_callback,
                                     verbose=self.verbose)
        self.model.learn(total_timesteps=self.total_steps, callback=eval_callback)

        if clean:
            self.output_location.joinpath("monitor.csv").unlink(missing_ok=True)

        return self

    def save_mmodel(self):
        configs = {"env_id": self.env_id, "frame_bound": self.frame_bound, "window_size": self.window_size}

        output_location = self.output_location.joinpath(STOCK_SYMBOL)
        output_location.mkdir(parents=True, exist_ok=True)

        with open(output_location.joinpath("configs.json"), "w") as file:
            dump(configs, file, indent=4)

        self.model.save(output_location.joinpath("model.zip").as_posix())

        log.info(f"Stored the model and its configs in: {output_location.resolve().as_posix()}")


if __name__ == "__main__":
    fetched_data = get_stock_data(symbol="GOOG", start="2022-01-01", end="2023-01-01")
    trainer = Trainer(fetched_data)
    trainer.train().save_mmodel()
