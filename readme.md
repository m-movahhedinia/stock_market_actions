# Stock Market Actions
This is a simple example for training and implementation of a reinforcement learning stock 
market action decision maker model using gym, stable-baselines3, and yfinance.

# Warning
This is by no means a production ready code. Do not use it as is for real world 
transactions.

# Usage
First install the required packages.
```commandline
pip install -r requirements.txt
```

Then run just run the server with the command below from the project root.
```commandline
python serve.py
```

After you have the server up and running, you can use these three APIs.
1. Data fetching: `0.0.0.0:8000/data/{stock_symbol}`
2. Training: `0.0.0.0:8000/train/{stock_symbol}`
3. Getting inference: Training: `0.0.0.0:8000/infer/{stock_symbol}`

Use the examples below and refer to the code itself for more details.
Fetching data:
```commandline
curl --location '0.0.0.0:8000/data/GOOGL' \
--header 'Content-Type: application/json' \
--data '{
    "start_date": "2022-01-01",
    "end_date": "2023-01-01"
}'
```
Training:
```commandline
curl --location '0.0.0.0:8000/train/GOOGL' \
--header 'Content-Type: application/json' \
--data '{
    "start_date": "2022-01-01",
    "end_date": "2023-01-01"
}'
```
Getting inference:
```commandline
curl --location '0.0.0.0:8000/infer/GOOGL' \
--header 'Content-Type: application/json' \
--data '{
    "start_date": "2022-01-01",
    "end_date": "2023-01-01"
}'
```
Pay attention that you do not need to fetch the data for training and getting inference. 
Just passing the start and end dates will suffice. 
