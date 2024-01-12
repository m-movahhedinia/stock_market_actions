# -*- coding: utf-8 -*-
"""
Created on January 08, 2024

@author: mansour
"""

from data_handler.stock_data import get_stock_data
from modeler.training import Trainer


if __name__ == "__main__":
    data = get_stock_data()
    trainer = Trainer(data)
    trainer.train().save_mmodel()
