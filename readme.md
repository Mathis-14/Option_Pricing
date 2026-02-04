# Option Pricing : Quant finance project — options pricing, volatility surfaces, delta hedging

Personal project to learn options theory and implement what I study.
The objective is to master the code behind the theory.
I want also to test the theory and see what it is like in practice with real data.
My objective is to develop this repo and then use it to price complex strategies myself.
Next step is implementing a sample portfolio game to simulate positions on the market and 
learn how to hedge properly.

## What's in here

**`NVIDIA_vol_s/`** — NVIDIA volatility surface construction
- `import_data.py` — Refinitiv data import with RIC parsing for NVDA options
  - Fetches real-time option chains via Refinitiv Data API
  - Smart RIC parsing to extract Strike, Expiry, and Option Type
  - Exports clean CSV with Implied Volatility, Bid/Ask, and metadata

**`option_basics/`** — Black-Scholes fundamentals
- `Call_Strike_Spot.py` / `Put_Strike_Spot.py` — Value decomposition plots
- `option_basics.ipynb` — Core option pricing theory
- `greeks.ipynb` — Greeks computation and analysis
- `moneyness.ipynb` — Moneyness analysis

**`option_exotics/`** — Exotic options
- `barrier_options.ipynb` — Barrier options (up-and-out, down-and-in, etc.)
- `binary_options.ipynb` — Binary/digital options

**`option_strat/`** — Option strategies
- `option_strategies.py` — Payoff diagrams for spreads, straddles, butterflies, etc.
- `spread.ipynb` — Spread strategies analysis

**`sample_portefolio/`** — Portfolio hedging
- `hedge_portfolio.py` — Backtesting hedging strategies on real data

**`legacy_vol_scripts/`** — Legacy volatility surface scripts
- Historical SPX volatility surface implementations

Check the notebooks for guided code and to see specifically what I'm learning and implementing.


## Tech Stack

Python, pandas, numpy, scipy, matplotlib, plotly, yfinance, refinitiv-data

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. For Refinitiv data access, create a `.env` file:
   ```bash
   REFINITIV_API_KEY=your_api_key_here
   ```


