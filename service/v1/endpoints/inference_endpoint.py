# -*- coding: utf-8 -*-
"""

@author: mansour
"""

from fastapi import APIRouter
from pydantic import BaseModel

from data_handler.stock_data import get_stock_data
from modeler.inference import Inferencer

inference_router = APIRouter()


class InferencePlotItem(BaseModel):
    start_date: str
    end_date: str = None
    time_zone: str = None
    configs: str | dict = None
    model: str = None
    root_directory: str = None


class InferenceActionItem(BaseModel):
    start_date: str
    time_zone: str = None
    configs: str | dict = None
    model: str = None


models = {}


@inference_router.post("/plot/{stock_symbol}")
async def plot_inferences(stock_symbol: str, payload: InferencePlotItem):
    data = get_stock_data(start=payload.start_date, end=payload.end_date, time_zone=payload.time_zone,
                          symbol=stock_symbol)
    model_directory = payload.root_directory
    inference = models.setdefault(stock_symbol, Inferencer(stock_symbol=stock_symbol,
                                                           model_root_location=model_directory))
    print(models)
    inference.update_environment(data).infer().plot_inference()
    return {"status": "inference plotted."}


@inference_router.post("/action/{stock_symbol}")
async def get_action(stock_symbol: str, payload: InferenceActionItem):
    data = get_stock_data(start=payload.start_date, time_zone=payload.time_zone, symbol=stock_symbol)
    inference = models.setdefault(stock_symbol, Inferencer(stock_symbol=stock_symbol))
    return {"action": int(inference.update_environment(data).get_action())}
