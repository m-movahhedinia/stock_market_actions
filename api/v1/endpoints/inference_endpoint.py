# -*- coding: utf-8 -*-
"""

@author: mansour
"""

from fastapi import APIRouter
from pydantic import BaseModel

from data_handler.stock_data import get_stock_data
from modeler.inference import Inferencer

inference_router = APIRouter()


class InferenceItem(BaseModel):
    start_date: str
    end_date: str = None
    time_zone: str = None
    configs: str | dict = None
    model: str = None
    root_directory: str = None


models = {}


# TODO Make inference return final action
@inference_router.post("/{stock_symbol}")
async def fetch_data(stock_symbol: str, payload: InferenceItem):
    data = get_stock_data(start=payload.start_date, end=payload.end_date, time_zone=payload.time_zone,
                          symbol=stock_symbol)
    model_directory = payload.root_directory
    inference = models.setdefault(stock_symbol, Inferencer(stock_symbol=stock_symbol,
                                                           model_root_location=model_directory))
    print(models)
    inference.update_environment(data).infer().plot_inference()
