# Option Pricing - Learning & Implementation Repository

## Purpose

This repository serves as a practical learning environment where I test and implement concepts from options theory and quantitative finance. The goal is to build hands-on experience with:

- **Option pricing models** (Black-Scholes, implied volatility)
- **Volatility surface analysis** (3D visualization, interpolation)
- **Option strategies** (payoff/profit diagrams, Greeks)
- **Real-world data integration** (crypto and equity options markets)
- **Data processing pipelines** (filtering, quality checks, forward estimation)

Each component is implemented as I learn the underlying theory, making this an ongoing educational project.

---

## Repository Structure (for the moment)

### Data Import Modules

**`import_crypto_options.py`**
- Fetches BTC/ETH options data from Deribit API
- Handles authentication, rate limiting, and data caching
- Exports to CSV with automatic path resolution

**`import_other_options.py`**
- Fetches S&P 500 options from Yahoo Finance via `yfinance`
- Implements quality filters (bid-ask spread, open interest, volume)
- Uses file modification time for reliable CSV selection
- Supports date range filtering and forward price estimation via put-call parity

### Volatility Surface Analysis

**`iv_surface_spx.py`**
- `SPXIVSurface` class for processing and visualizing implied volatility surfaces
- RBF interpolation for smooth 3D surfaces
- Configurable quality filters and time-to-expiry ranges
- Plotly-based interactive 3D visualization

**`ImpliedVolatilitySurface.py`**
- Alternative implementation for BTC and ETH

**Notebooks:**
- `vol_sp500.ipynb` - Main analysis notebook for S&P 500 volatility surfaces
- `vol_surface_matplotlib.ipynb` - Matplotlib-based visualization
- `vol_surface_plotly.ipynb` - Plotly-based interactive visualization

### Option Strategies

**`option_strategies.py`**
- `OptionStrategies` class implementing common strategies:
  - Basic positions (long/short calls, puts)
  - Spreads (call/put spreads, butterflies, condors)
  - Combinations (straddles, strangles, risk reversals)
  - Complex strategies (ratios, ladders, synthetic forwards)
- Generates payoff and profit diagrams
- Saves plots to `plot/strategies/`

### Black-Scholes Implementation

**`Call_Strike_Spot.py`** & **`Put_Strike_Spot.py`**
- Academic-style decomposition of option value
- Visualizes intrinsic value vs. time value
- Shows convexity properties of option pricing

### Data Processing & Utilities

**`option_filtering.py`**
- Quality filters for options data
- Out-of-the-money (OTM) selection logic

**`issues_tests/`**
- Diagnostic scripts for debugging data import and visualization issues
- Test scripts for specific time ranges (e.g., T=1 to T=2 years)

---

## Key Features

### Data Quality Management
- Automatic filtering of low-quality quotes (wide spreads, low liquidity)
- Forward price estimation using put-call parity
- Log-moneyness calculation for volatility surface construction
- Handling of missing or invalid implied volatility data

### Visualization
- 3D volatility surfaces with interactive controls
- Strategy payoff/profit diagrams
- Option value decomposition (intrinsic vs. time value)
- Configurable time-to-expiry ranges for focused analysis

### Robustness
- Path resolution that works across different execution contexts (scripts, notebooks)
- CSV caching with modification-time-based selection
- Error handling for API rate limits and missing data
- Compatibility with Python 3.9 (JAX 0.4.23, SciPy 1.11.4)

---

## Dependencies

See `requirements.txt` for full list. Key packages:
- `jax` / `jaxlib` - Automatic differentiation for IV calculation
- `pandas` / `numpy` - Data manipulation
- `matplotlib` / `plotly` - Visualization
- `scipy` - Interpolation (RBF)
- `yfinance` - Yahoo Finance data

---

## Current Status

**Ongoing work:**
- Refining volatility surface interpolation methods
- Expanding strategy library
- Improving data quality filters
- Adding Greeks calculation
- Exploring alternative pricing models

**Known limitations:**
- Some implementations are experimental and may have edge cases
- Data import relies on third-party APIs (Deribit, Yahoo Finance)
- Focus is on European-style options for the moment

---

## Usage Example
thon
# Import S&P 500 options data
from import_other_options import import_sp500_options_data

df = import_sp500_options_data(
    start_date="2025-12-16",
    end_date="2027-06-30",
    ticker="^SPX"
)

# Create volatility surface
from iv_surface_spx import SPXIVSurface, SurfaceConfig

cfg = SurfaceConfig(
    min_T=0.1,  # Minimum 1 month
    max_T=2.0,  # Maximum 2 years
    r=0.05
)

surface = SPXIVSurface(df, cfg)
surface.plot()  # Interactive 3D plot---

## Notes

This is a learning repository. Code may be refactored as understanding improves. Documentation and error handling are added incrementally. The focus is on correctness and educational value rather than production-ready robustness.