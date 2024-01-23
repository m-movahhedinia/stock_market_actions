# -*- coding: utf-8 -*-
"""

@author: mansour
"""

from datetime import datetime
from json import load as json_load
from logging import getLogger
from pathlib import Path

import gymnasium as gym
from matplotlib import pyplot
from numpy import concatenate, full
from sb3_contrib import RecurrentPPO

from configs.constants import MODEL_LOCATION
from data_handler.stock_data import get_stock_data

log = getLogger()


class Inferencer:
    def __init__(self, stock_symbol: str, configs: dict = None, model_root_location: str | Path = None,
                 output_location: str = "temp_inference"):

        self.model_root_location = Path(model_root_location or MODEL_LOCATION, stock_symbol)

        if configs is None:
            configs_location = self.model_root_location.joinpath("configs.json")
            if configs_location.is_file():
                configs = json_load(configs_location.open())
            else:
                raise ValueError(f"Cannot find the config file at: {configs_location.resolve().as_posix()}")

        self.window_size = configs["window_size"]
        self.env_id = configs["env_id"]
        self.frame_bound = configs["frame_bound"]
        self.env = None
        self.observation = None
        self.model = RecurrentPPO.load(self.model_root_location.joinpath("model.zip").as_posix(), deterministic=True)
        self.output_location = Path(output_location)
        self.output_location.mkdir(parents=True, exist_ok=True)

    def update_environment(self, data):
        self.env = gym.make(id=self.env_id, frame_bound=self.frame_bound, window_size=self.window_size, df=data)
        return self

    def infer(self):
        done = False
        self.observation, info = self.env.reset()
        while not done:
            if self.observation.shape != self.env.observation_space.shape:
                padding = self.env.observation_space.shape[0] - self.observation.shape[0]
                self.observation = concatenate([self.observation,
                                                full((padding, self.observation.shape[1]), 0)], axis=0)
            action, states = self.model.predict(self.observation)
            self.observation, rewards, done, truncated, info = self.env.step(action)
        else:
            log.info(info)
        return self

    def get_action(self):
        self.observation, info = self.env.reset()
        action, _ = self.model.predict(self.observation)
        return action

    def plot_inference(self, location: str or Path = None):
        location = Path(location) if location else self.output_location
        location.parent.mkdir(exist_ok=True, parents=True)

        pyplot.figure(figsize=(30, 12))
        pyplot.cla()
        self.env.unwrapped.render_all()
        pyplot.savefig(self.output_location.joinpath(f"{datetime.today().isoformat()}.png"))


if __name__ == "__main__":
    fetched_data = get_stock_data(symbol="GOOG", start="2023-01-01", end="2024-01-01")
    inferencer = Inferencer(model_root_location="models", stock_symbol="GOOGL")
    inferencer.update_environment(fetched_data).infer().plot_inference("inferred")
    fetched_data = get_stock_data(symbol="GOOG", start="2023-01-01")
    print(inferencer.update_environment(fetched_data).get_action())
