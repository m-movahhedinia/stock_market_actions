# -*- coding: utf-8 -*-
"""

@author: mansour
"""

from datetime import datetime
from logging import getLogger

from pandas import to_datetime
from yfinance import download

from configs.constants import DATA_END_DATE, DATA_START_DATE, STOCK_SYMBOL, TIMEZONE

log = getLogger()


def get_stock_data(start: str = DATA_START_DATE, end: str = DATA_END_DATE, symbol: str = STOCK_SYMBOL,
                   time_zone: str = TIMEZONE):
    if start is None:
        raise ValueError("There must be a start value.")
    elif start and end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    data = download(symbol, start=start, end=end)

    data["record_date"] = to_datetime(data.index)

    if time_zone and len(data):
        data.index = data.index.tz_localize("UTC").tz_convert(time_zone)

    data = data.sort_values("record_date")

    return data


if __name__ == "__main__":
    fetched_data = get_stock_data("2023-01-01", symbol="GOOG")
    print(len(fetched_data))
    print(fetched_data.head())
    print(fetched_data.dtypes)
    print(fetched_data.index.dtype)
