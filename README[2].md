# Real-World Data Project — Finance Domain

End-to-end applied data science project on 3 years of daily stock price data
for a fictional company, **NOVA Technologies (NOVA)**: feature engineering,
exploratory analysis, and two predictive modeling tasks (next-day price and
next-day direction).

## Project Structure

```
finance_project/
├── data/
│   ├── generate_data.py                   # Creates the synthetic price series
│   └── nova_stock_prices.csv              # OHLCV data, 783 trading days (2023-2025)
├── notebooks/
│   └── analyze_and_predict.py             # Full pipeline: features → EDA → models
├── outputs/
│   ├── engineered_features.csv            # Data with returns/MAs/volatility added
│   ├── regression_model_comparison.csv    # MAE/RMSE/R2 for price prediction
│   └── analysis_report.txt                # Full findings & conclusions
├── visuals/
│   ├── price_trend_moving_averages.png
│   ├── daily_returns_distribution.png
│   ├── rolling_volatility.png
│   ├── price_vs_volume.png
│   ├── actual_vs_predicted_price.png
│   ├── confusion_matrix_direction_logistic_regression.png
│   └── confusion_matrix_direction_random_forest_classifier.png
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

python data/generate_data.py              # regenerate dataset (optional, included)
python notebooks/analyze_and_predict.py   # run full analysis + modeling
```

## Dataset

Daily Open/High/Low/Close/Volume (OHLCV) data simulated with a geometric
random walk (drift + volatility + occasional shock events), producing a
realistic multi-year price series with trending and mean-reverting periods.

## Methodology

1. **Feature engineering** — daily returns, 7-day and 30-day moving
   averages, 30-day rolling volatility, next-day close (regression target),
   next-day direction up/down (classification target).
2. **Exploratory analysis** — price trend with moving averages, return
   distribution, rolling volatility over time, price vs. trading volume.
3. **Predictive modeling** (chronological train/test split — no shuffling,
   to respect time order):
   - **Next-day closing price** — Linear Regression vs. Random Forest
     Regressor, evaluated with MAE / RMSE / R².
   - **Next-day direction (up/down)** — Logistic Regression vs. Random
     Forest Classifier, evaluated with accuracy against a majority-class
     baseline, plus confusion matrices.

## Key Findings

- Over the 3-year window, NOVA's total return was **-4.2%**, with daily
  volatility of **~1.8%** and single-day moves as large as **+6.2% / -5.1%**.
- **Next-day closing price is highly predictable** from recent
  price/volume history — Linear Regression reached **R² ≈ 0.93** — because
  tomorrow's close tends to sit close to today's close plus a small drift.
- **Next-day direction is a much harder problem.** Both classifiers
  performed **at or below** the majority-class baseline (~53%), finding no
  real predictive edge on whether the price would go up or down.
- **Conclusion:** this gap illustrates a classic finance concept — price
  *levels* are easy to track day-to-day, but short-term price *direction*
  behaves close to a random walk, consistent with the efficient-market
  hypothesis.

## Expected Outcome (Task Goal)

Demonstrates applying data science skills to a real-world-style,
domain-specific problem end to end: choosing a dataset, engineering
features, performing analysis and prediction, and presenting findings with
visualizations and conclusions.
