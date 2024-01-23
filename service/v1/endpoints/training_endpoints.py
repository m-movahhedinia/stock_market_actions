# -*- coding: utf-8 -*-
"""

@author: mansour
"""

from json import dumps
from pathlib import Path

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse
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


async def training_job(start_date: str, end_date: str, time_zone: str, stock_symbol: str, output_root_location: str,
                       frame_bound_max: int, window_size: int, env_id: str, policy: str, total_steps: int):
    yield dumps({"status": "Training started"}) + "\n"
    data = get_stock_data(start=start_date, end=end_date, time_zone=time_zone, symbol=stock_symbol)
    model_location = Path(output_root_location, stock_symbol) if output_root_location else None
    trainer = Trainer(data=data, frame_bound_max=frame_bound_max, window_size=window_size, env_id=env_id, verbose=True,
                      policy=policy, model_location=model_location, total_steps=total_steps)
    trainer.train().save_mmodel()
    yield dumps({"status": "Training finished"}) + "\n"


@training_router.post("/{stock_symbol}")
async def fetch_data(stock_symbol: str, payload: TrainingItem):
    return StreamingResponse(training_job(payload.start_date, payload.end_date, payload.time_zone, stock_symbol,
                                          payload.output_root_location, payload.frame_bound_max, payload.window_size,
                                          payload.env_id, payload.policy, payload.total_steps),
                             status_code=status.HTTP_200_OK, media_type="application/json")
