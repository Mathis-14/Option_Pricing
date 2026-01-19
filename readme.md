# Option Pricing : Quant finance project — options pricing, volatility surfaces, delta hedging

Personal project to learn options theory and implement what I study.
The objective is to master the code behind the theory.
I want also to test the theory and see what it is like in practice with real data.
My objective is to develop this repo and then use it to price complex strategies myself.
Next step is implementing a sample portfolio game to simulate positions on the market and 
learn how to hedge properly.

## What's in here

**`data/`** — Data import scripts
- `import_options.py` — SPX options from Yahoo Finance
- `import_stocks.py` — Stock prices (NVDA, GOOGL, AAPL)
- `option_filtering.py` — Maturity filters

**`vol/`** — Volatility surface stuff
- `iv_surface_spx.py` — SPX implied vol surface
- `iv_2D.py` — Smile and term structure plots
- `ImpliedVolatilitySurface.py` — 3D surface calculator

**`option_basics/`** — Black-Scholes basics
- `Call_Strike_Spot.py` / `Put_Strike_Spot.py` — Value decomposition plots

**`options_strat/`** — Option strategies
- `option_strategies.py` — Payoff diagrams for spreads, straddles, butterflies, etc.

**`hedging&greeks/`** — Hedging and greeks. Effect and sign of greeks in several situations.
- `hedge_portfolio.py` — Backtesting hedging strategies on real data

Check the notebooks for guided code and to see specifically what I'm learning and implementing.


## Tech Stack

Python, pandas, numpy, scipy, matplotlib, plotly, yfinance

## Notes

I take sometimes inspiration from my previous work in the repo "Project_Python_Data_Science" on predictions
of gold log returns and use as a safe-haven.
