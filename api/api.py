# -*- coding: utf-8 -*-
"""

@author: mansour
"""

from fastapi import FastAPI

from api.v1.endpoints.data_endponts import data_router
from api.v1.endpoints.inference_endpoint import inference_router
from api.v1.endpoints.training_endpoints import training_router

app = FastAPI()

app.include_router(inference_router, prefix="/infer", tags=["Inference Endpoint"])
app.include_router(training_router, prefix="/train", tags=["Training Endpoint"])
app.include_router(data_router, prefix="/data", tags=["Data Endpoint"])
