# -*- coding: utf-8 -*-
"""

@author: mansour
"""

from data_handler.stock_data import get_stock_data
from modeler.training import Trainer
from modeler.inference import Inferencer
from sys import modules
from service.api import app
from httpx import AsyncClient
import pytest


def test_mandatory_imports():
    assert "gym_anytrading" in modules


def test_get_data():
    data = get_stock_data("2023-01-01", symbol="GOOG")
    assert len(data) > 0


def test_training():
    data = get_stock_data(symbol="GOOG", start="2022-01-01", end="2023-01-01").head(20)
    trainer = Trainer(data)
    trainer.train().save_mmodel()
    assert trainer.output_location.is_dir()
    assert trainer.output_location.joinpath("GOOGL", "configs.json").is_file()
    assert trainer.output_location.joinpath("GOOGL", "model.zip").is_file()


def test_inference():
    data = get_stock_data(symbol="GOOG", start="2023-01-01", end="2024-01-01").head(20)
    inferencer = Inferencer(model_root_location="models", stock_symbol="GOOGL")
    inferencer.update_environment(data).infer().plot_inference("inferred")
    assert inferencer.output_location.is_dir()


@pytest.mark.anyio
async def test_data_api():
    async with AsyncClient(app=app, base_url="http://0.0.0.0:8000") as client:
        response = await client.post("/data/GOOGL", json={"start_date": "2023-01-01", "end_date": "2024-01-01"})
        assert response.status_code == 200
        assert len(response.json()) > 0


@pytest.mark.anyio
async def test_train_api():
    payload = {"start_date": "2023-06-01", "end_date": "2023-07-01"}
    async with AsyncClient(app=app, base_url="http://0.0.0.0:8000") as client:
        first_response = await client.post("/train/GOOGL/", json=payload)
        assert first_response.status_code == 307
        second_response = await client.post(first_response.headers["Location"], json=payload)
        assert second_response.status_code == 200


@pytest.mark.anyio
async def test_infer_plot_api():
    payload = {"start_date": "2023-06-01", "end_date": "2023-07-01"}
    async with AsyncClient(app=app, base_url="http://0.0.0.0:8000") as client:
        first_response = await client.post("/infer/plot/GOOGL/", json=payload)
        assert first_response.status_code == 307
        second_response = await client.post(first_response.headers["Location"], json=payload)
        assert second_response.status_code == 200


@pytest.mark.anyio
async def test_infer_action_api():
    payload = {"start_date": "2023-06-01", "end_date": "2023-07-01"}
    async with AsyncClient(app=app, base_url="http://0.0.0.0:8000", follow_redirects=False) as client:
        first_response = await client.post("/infer/action/GOOGL/", json=payload)
        assert first_response.status_code == 307
        second_response = await client.post(first_response.headers["Location"], json=payload)
        assert second_response.status_code == 200
