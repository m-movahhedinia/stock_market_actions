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
from numpy import array, newaxis, squeeze, zeros, concatenate
# from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

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
        self.model = PPO.load(model, deterministic=True)

    def update_environment(self, data):
        self.env = gym.make(id=self.env_id, frame_bound=(20, 120) ,window_size=self.window_size, df=data)
        return self

    def infer(self):
        done = False
        info = None
        self.observation, info = self.env.reset()
        # self.observation = array(self.observation, dtype=object)
        while not done:
            self.observation = self.observation[newaxis, ...]
            if (obs_shape := squeeze(self.observation).shape) != self.model.observation_space.shape:
                padding = zeros((self.observation.shape[0],
                                 self.model.observation_space.shape[0] - obs_shape[0],
                                 self.observation.shape[-1]))
                self.observation = concatenate((self.observation, padding), axis=1)
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
        pyplot.savefig(location.joinpath(datetime.today().isoformat()))


if __name__ == "__main__":
    fetched_data = get_stock_data(symbol="GOOG")
    print(len(fetched_data))
    inferencer = Inferencer("models/configs.json", "models/model.zip")
    inferencer.update_environment(fetched_data).infer().plot_inference("inferred")
