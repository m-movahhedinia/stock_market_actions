# -*- coding: utf-8 -*-
"""

@author: mansour
"""

from uvicorn import run

from service.api import app

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)
