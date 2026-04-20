import os
import sys
from datetime import datetime, timedelta
sys.path.append("/Users/jflorezgaleano/Documents/JulianFlorez/Hapi")
from trading_bot.shared.data.market_data import get_alpaca_client
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pytz

client = get_alpaca_client()
tf = TimeFrame(5, TimeFrameUnit.Minute)
start_dt = datetime(2017, 1, 1, tzinfo=pytz.UTC)
end_dt = datetime(2021, 1, 1, tzinfo=pytz.UTC)

req = StockBarsRequest(
    symbol_or_symbols="AAPL",
    timeframe=tf,
    start=start_dt,
    end=end_dt,
    feed="iex"
)
try:
    bars = client.get_stock_bars(req)
    print("Got data")
    if not bars.df.empty:
        print("First index:", bars.df.index[0])
        print("Length:", len(bars.df))
except Exception as e:
    print("Error:", e)
