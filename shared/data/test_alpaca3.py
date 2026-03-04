import os
from dotenv import load_dotenv
load_dotenv("/app/.env")
from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

client = StockHistoricalDataClient(
    api_key=os.environ.get("APCA_API_KEY_ID"),
    secret_key=os.environ.get("APCA_API_SECRET_KEY")
)

symbol = "AAPL"
req = StockBarsRequest(
    symbol_or_symbols=symbol,
    timeframe=TimeFrame(5, TimeFrameUnit.Minute),
    limit=5,
    feed="iex"
)
try:
    bars = client.get_stock_bars(req)
    print("Is empty?", bars.df.empty)
    print("Latest timestamps in Alpaca:")
    print(bars.df.tail())
except Exception as e:
    print(e)
