# -*- coding: utf-8 -*-
"""
Created on January 05, 2024

@author: mansour
"""

from datetime import datetime
from logging import getLogger
from configs.constants import STOCK_SYMBOL, DATA_PERIOD, TIMEZONE
from pandas import read_csv

from yfinance import Ticker

log = getLogger()


def get_stock_data(start: str = None, end: str = None, symbol: str = STOCK_SYMBOL, period: str = DATA_PERIOD,
                   time_zone: str = TIMEZONE, local_data: str = None):
    if end is start is None:
        if period not in ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]:
            raise ValueError("The period can only be one of 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, or max.")
    elif start and end is None:
        end = datetime.today().strftime('%Y-%m-%d')
        period = None
    elif end and start is None:
        raise ValueError("If you pass a data end time, you have to pass a data start time as well.")
    else:
        period = None

    if local_data:
        data = read_csv(local_data, parse_dates=["Date"], index_col="Date")
    else:
        ticker = Ticker(symbol)
        data = ticker.history(**({"period": period} if period else {"start": start, "end": end}))

    if time_zone and len(data):
        data.index = data.index.tz_convert(time_zone)

    return data


if __name__ == "__main__":
    fetched_data = get_stock_data(symbol="GOOG")
    print(len(fetched_data))
    print(fetched_data.head())
    print(fetched_data.dtypes)
    print(fetched_data.index.dtype)
# Open            float64
# High            float64
# Low             float64
# Close           float64
# Volume            int64
# Dividends       float64
# Stock Splits    float64
# dtype: object
# datetime64[ns, UTC]
