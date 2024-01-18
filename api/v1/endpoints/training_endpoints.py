# -*- coding: utf-8 -*-
"""

@author: mansour
"""

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from data_handler.stock_data import get_stock_data
from modeler.training import Trainer

training_router = APIRouter()


class TrainingItem(BaseModel):
    start_date: str
    end_date: str = None
    time_zone: str = None
    frame_bound_max: int = None
    window_size: int = None
    env_id: str = None
    policy: str = None
    output_root_location: str = None
    total_steps: int = None


# TODO Make training async and return a valid status
@training_router.post("/{stock_symbol}")
async def fetch_data(stock_symbol: str, payload: TrainingItem):
    data = get_stock_data(start=payload.start_date, end=payload.end_date, time_zone=payload.time_zone,
                          symbol=stock_symbol)
    model_location = Path(payload.output_root_location, stock_symbol) if payload.output_root_location else None
    trainer = Trainer(data=data, frame_bound_max=payload.frame_bound_max, window_size=payload.window_size,
                      env_id=payload.env_id, verbose=True, policy=payload.policy,
                      model_location=model_location, total_steps=payload.total_steps)
    trainer.train().save_mmodel()
