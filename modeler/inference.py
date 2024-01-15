# -*- coding: utf-8 -*-
"""
Created on January 10, 2024

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

from data_handler.stock_data import get_stock_data

log = getLogger()


class Inferencer:
    def __init__(self, configs: str or dict, model: str):

        if isinstance(configs, str):
            configs = Path(configs)
            if configs.is_file():
                configs = json_load(configs.open())
            else:
                raise ValueError("Cannot find the config file.")

        self.window_size = configs["window_size"]
        self.env_id = configs["env_id"]
        self.frame_bound = configs["frame_bound"]
        self.env = None
        self.observation = None
        self.model = RecurrentPPO.load(model, deterministic=True)

    def update_environment(self, data):
        self.env = gym.make(id=self.env_id, frame_bound=self.frame_bound, window_size=self.window_size, df=data)
        return self

    def infer(self):
        done = False
        info = None
        self.observation, info = self.env.reset()
        while not done:
            # self.observation = self.observation[newaxis, ...]
            if self.observation.shape != self.env.observation_space.shape:
                padding = self.env.observation_space.shape[0] - self.observation.shape[0]
                self.observation = concatenate([self.observation,
                                                full((padding, self.observation.shape[1]), 0)], axis=0)
            action, states = self.model.predict(self.observation)
            self.observation, rewards, done, truncated, info = self.env.step(action)
        else:
            log.info(info)
        return self

    def plot_inference(self, location: str or Path):
        location = Path(location)
        location.parent.mkdir(exist_ok=True, parents=True)

        pyplot.figure(figsize=(30, 12))
        pyplot.cla()
        self.env.render_all()
        pyplot.savefig(f"{datetime.today().isoformat()}.png")


if __name__ == "__main__":
    fetched_data = get_stock_data(symbol="GOOG", start="2023-01-01", end="2024-01-01")
    print(len(fetched_data))
    inferencer = Inferencer("models/GOOGL_configs.json", "models/GOOGL.zip")
    inferencer.update_environment(fetched_data).infer().plot_inference("inferred")
