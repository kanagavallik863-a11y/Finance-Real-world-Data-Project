"""
generate_data.py
-----------------
Generates a synthetic daily stock price dataset (3 years) for a
fictional company "NOVA Technologies (NOVA)" using a geometric
random walk with drift + seasonal/volatility effects, plus trading
volume. Used as the raw dataset for the Real-World Data Project
(Finance domain).
"""

import numpy as np
import pandas as pd

np.random.seed(7)

dates = pd.bdate_range("2023-01-02", "2025-12-31")  # business days only
n = len(dates)

# --- Simulate price via geometric brownian motion with drift + regime shifts ---
mu = 0.0004          # daily drift
sigma = 0.018         # daily volatility
prices = [100.0]

for i in range(1, n):
    # occasional volatility regime shifts (market shocks)
    shock = 0
    if np.random.rand() < 0.003:
        shock = np.random.normal(0, 0.08)
    daily_return = np.random.normal(mu, sigma) + shock
    prices.append(prices[-1] * (1 + daily_return))

close = np.array(prices)
open_ = close * (1 + np.random.normal(0, 0.004, n))
high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.006, n)))
low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.006, n)))
volume = np.random.lognormal(mean=13, sigma=0.4, size=n).astype(int)

df = pd.DataFrame({
    "Date": dates,
    "Open": np.round(open_, 2),
    "High": np.round(high, 2),
    "Low": np.round(low, 2),
    "Close": np.round(close, 2),
    "Volume": volume,
})

df.to_csv("/home/claude/finance_project/data/nova_stock_prices.csv", index=False)
print("Dataset generated:", df.shape)
print(df.head())
print(df.tail())
