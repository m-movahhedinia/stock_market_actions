# -*- coding: utf-8 -*-
"""

@author: mansour
"""

from fastapi import APIRouter
from pydantic import BaseModel

from data_handler.stock_data import get_stock_data

data_router = APIRouter()


class DataItem(BaseModel):
    start_date: str
    end_date: str = None
    time_zone: str = None


@data_router.post("/{stock_symbol}")
async def fetch_data(stock_symbol: str, payload: DataItem):
    return get_stock_data(start=payload.start_date, end=payload.end_date, time_zone=payload.time_zone,
                          symbol=stock_symbol).to_json(orient='index')
