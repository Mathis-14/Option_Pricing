"""
hedge_portfolio.py — Delta Hedging Portfolio Simulation

This module implements a comprehensive delta hedging system for options portfolios.
It provides tools to:
- Calculate Black-Scholes Greeks (Delta, Gamma, Theta, Vega, Rho)
- Manage portfolio positions (options + underlying stocks)
- Simulate delta hedging strategies with historical stock data
- Analyze hedging effectiveness and P&L

================================================================================
USAGE GUIDE
================================================================================

BASIC USAGE:
------------

1. Import the module and create a Greeks calculator:

    from hedge_portfolio import GreeksCalculator, Portfolio, DeltaHedgingSimulator
    
    # Calculate option price and Greeks
    greeks = GreeksCalculator(
        S=100,      # Current stock price
        K=100,      # Strike price
        T=0.25,     # Time to expiry (years)
        r=0.05,     # Risk-free rate
        sigma=0.20, # Volatility (20%)
        option_type="call"
    )
    print(f"Delta: {greeks.delta:.4f}")
    print(f"Gamma: {greeks.gamma:.4f}")

2. Create a portfolio with options and hedge:

    portfolio = Portfolio()
    portfolio.add_call_option(ticker="NVDA", K=180, T=0.25, quantity=-10)  # Short 10 calls
    portfolio.update_prices({"NVDA": 185.0})
    
    # Get hedge ratio (shares to buy to neutralize delta)
    hedge = portfolio.get_delta_hedge()
    print(f"Buy {hedge['NVDA']:.0f} shares of NVDA to delta hedge")

3. Run a delta hedging simulation:

    simulator = DeltaHedgingSimulator()
    results = simulator.run_simulation(
        ticker="NVDA",
        option_type="call",
        strike=180,
        days=30,
        volatility=0.50,
        rebalance_frequency="daily"
    )
    simulator.plot_results(results)

================================================================================
GREEKS FORMULAS (Black-Scholes)
================================================================================

Delta (Δ):
    Call: N(d1)
    Put:  N(d1) - 1
    
    Interpretation: Change in option price for $1 change in underlying
    
Gamma (Γ):
    n(d1) / (S * σ * √T)
    
    Interpretation: Rate of change of delta
    
Theta (Θ):
    Call: -S * n(d1) * σ / (2√T) - r * K * e^(-rT) * N(d2)
    Put:  -S * n(d1) * σ / (2√T) + r * K * e^(-rT) * N(-d2)
    
    Interpretation: Time decay (typically negative)
    
Vega (ν):
    S * √T * n(d1)
    
    Interpretation: Sensitivity to 1% change in volatility
    
Rho (ρ):
    Call: K * T * e^(-rT) * N(d2)
    Put:  -K * T * e^(-rT) * N(-d2)
    
    Interpretation: Sensitivity to 1% change in interest rate

Where:
    d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T
    N(x) = cumulative normal distribution
    n(x) = normal probability density function
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Literal, Tuple
from dataclasses import dataclass, field

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.facecolor'] = 'white'


# =============================================================================
# GREEKS CALCULATOR: Black-Scholes option pricing and Greeks
# =============================================================================

@dataclass
class GreeksCalculator:
    """
    Calculate Black-Scholes option price and Greeks.
    
    This class computes the theoretical price and all Greeks for a European
    option using the Black-Scholes-Merton model.
    
    Parameters
    ----------
    S : float
        Current price of the underlying asset
    K : float
        Strike price of the option
    T : float
        Time to expiration in years (e.g., 0.25 for 3 months)
    r : float
        Risk-free interest rate (annualized, e.g., 0.05 for 5%)
    sigma : float
        Volatility of the underlying (annualized, e.g., 0.20 for 20%)
    option_type : str
        "call" or "put"
    q : float
        Dividend yield (annualized, default 0)
        
    Attributes
    ----------
    price : float
        Theoretical option price
    delta : float
        First derivative w.r.t. underlying price
    gamma : float
        Second derivative w.r.t. underlying price
    theta : float
        Derivative w.r.t. time (daily theta, divide by 365)
    vega : float
        Derivative w.r.t. volatility (per 1% change)
    rho : float
        Derivative w.r.t. interest rate (per 1% change)
        
    Examples
    --------
    >>> calc = GreeksCalculator(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
    >>> print(f"Call price: ${calc.price:.2f}")
    >>> print(f"Delta: {calc.delta:.4f}")
    """
    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: str = "call"
    q: float = 0.0
    
    # Computed attributes (set in __post_init__)
    d1: float = field(init=False)
    d2: float = field(init=False)
    price: float = field(init=False)
    delta: float = field(init=False)
    gamma: float = field(init=False)
    theta: float = field(init=False)
    vega: float = field(init=False)
    rho: float = field(init=False)
    
    def __post_init__(self):
        """Calculate all Greeks upon initialization."""
        # Validate inputs
        if self.T <= 0:
            self.T = 1e-6  # Avoid division by zero for expired options
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive")
        if self.S <= 0 or self.K <= 0:
            raise ValueError("Stock price and strike must be positive")
        
        # Normalize option type
        self.option_type = self.option_type.lower()
        if self.option_type not in ["call", "put", "c", "p"]:
            raise ValueError(f"option_type must be 'call' or 'put', got '{self.option_type}'")
        if self.option_type == "c":
            self.option_type = "call"
        elif self.option_type == "p":
            self.option_type = "put"
        
        # Calculate d1 and d2
        sqrt_T = np.sqrt(self.T)
        self.d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * sqrt_T)
        self.d2 = self.d1 - self.sigma * sqrt_T
        
        # Calculate Greeks
        self._calculate_price()
        self._calculate_delta()
        self._calculate_gamma()
        self._calculate_theta()
        self._calculate_vega()
        self._calculate_rho()
    
    def _calculate_price(self):
        """Calculate Black-Scholes option price."""
        S_adj = self.S * np.exp(-self.q * self.T)  # Adjusted for dividends
        K_adj = self.K * np.exp(-self.r * self.T)  # Discounted strike
        
        if self.option_type == "call":
            self.price = S_adj * norm.cdf(self.d1) - K_adj * norm.cdf(self.d2)
        else:  # put
            self.price = K_adj * norm.cdf(-self.d2) - S_adj * norm.cdf(-self.d1)
    
    def _calculate_delta(self):
        """
        Calculate Delta: ∂C/∂S
        
        Delta measures how much the option price changes for a $1 change
        in the underlying stock price.
        
        Call delta: Between 0 and 1
        Put delta: Between -1 and 0
        """
        if self.option_type == "call":
            self.delta = np.exp(-self.q * self.T) * norm.cdf(self.d1)
        else:  # put
            self.delta = np.exp(-self.q * self.T) * (norm.cdf(self.d1) - 1)
    
    def _calculate_gamma(self):
        """
        Calculate Gamma: ∂²C/∂S²
        
        Gamma measures the rate of change of delta. It's highest for
        at-the-money options and decreases as options move ITM or OTM.
        
        Same for both calls and puts.
        """
        self.gamma = (np.exp(-self.q * self.T) * norm.pdf(self.d1)) / (self.S * self.sigma * np.sqrt(self.T))
    
    def _calculate_theta(self):
        """
        Calculate Theta: ∂C/∂t (time decay)
        
        Theta represents the daily decay of option value.
        Usually negative (options lose value over time).
        
        Returns theta per day (divided by 365).
        """
        sqrt_T = np.sqrt(self.T)
        S_adj = self.S * np.exp(-self.q * self.T)
        K_adj = self.K * np.exp(-self.r * self.T)
        
        # First term (same for calls and puts)
        term1 = -(S_adj * norm.pdf(self.d1) * self.sigma) / (2 * sqrt_T)
        
        if self.option_type == "call":
            term2 = -self.r * K_adj * norm.cdf(self.d2)
            term3 = self.q * S_adj * norm.cdf(self.d1)
        else:  # put
            term2 = self.r * K_adj * norm.cdf(-self.d2)
            term3 = -self.q * S_adj * norm.cdf(-self.d1)
        
        # Return daily theta (annual theta / 365)
        self.theta = (term1 + term2 + term3) / 365
    
    def _calculate_vega(self):
        """
        Calculate Vega: ∂C/∂σ
        
        Vega measures sensitivity to volatility.
        Returned per 1% (0.01) change in volatility.
        
        Same for both calls and puts.
        """
        S_adj = self.S * np.exp(-self.q * self.T)
        # Vega per 1% volatility change
        self.vega = S_adj * np.sqrt(self.T) * norm.pdf(self.d1) / 100
    
    def _calculate_rho(self):
        """
        Calculate Rho: ∂C/∂r
        
        Rho measures sensitivity to interest rate changes.
        Returned per 1% (0.01) change in interest rate.
        """
        K_adj = self.K * np.exp(-self.r * self.T)
        
        if self.option_type == "call":
            self.rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2) / 100
        else:  # put
            self.rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2) / 100
    
    def summary(self) -> str:
        """Return a formatted summary of the option and its Greeks."""
        lines = [
            f"\n{'='*50}",
            f"  {self.option_type.upper()} OPTION SUMMARY",
            f"{'='*50}",
            f"  Underlying (S):    ${self.S:,.2f}",
            f"  Strike (K):        ${self.K:,.2f}",
            f"  Time to Expiry:    {self.T:.4f} years ({self.T*365:.1f} days)",
            f"  Volatility (σ):    {self.sigma*100:.1f}%",
            f"  Risk-free rate:    {self.r*100:.2f}%",
            f"{'─'*50}",
            f"  PRICE:             ${self.price:,.4f}",
            f"{'─'*50}",
            f"  Delta (Δ):         {self.delta:+.4f}",
            f"  Gamma (Γ):         {self.gamma:.6f}",
            f"  Theta (Θ):         ${self.theta:,.4f} /day",
            f"  Vega (ν):          ${self.vega:,.4f} /1% vol",
            f"  Rho (ρ):           ${self.rho:,.4f} /1% rate",
            f"{'='*50}\n",
        ]
        return "\n".join(lines)
    
    @staticmethod
    def test_greeks():
        """Run tests to verify Greeks calculations are correct."""
        print("="*60)
        print("  GREEKS CALCULATOR UNIT TESTS")
        print("="*60)
        
        # Test case: ATM call option
        calc = GreeksCalculator(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
        
        # Known approximate values for ATM option
        tests_passed = 0
        tests_total = 0
        
        # Test 1: Delta should be around 0.5 for ATM call
        tests_total += 1
        if 0.45 < calc.delta < 0.60:
            print(f"✅ Test 1 PASSED: ATM call delta ~ 0.5 (got {calc.delta:.4f})")
            tests_passed += 1
        else:
            print(f"❌ Test 1 FAILED: ATM call delta should be ~0.5 (got {calc.delta:.4f})")
        
        # Test 2: Gamma should be positive
        tests_total += 1
        if calc.gamma > 0:
            print(f"✅ Test 2 PASSED: Gamma is positive ({calc.gamma:.6f})")
            tests_passed += 1
        else:
            print(f"❌ Test 2 FAILED: Gamma should be positive (got {calc.gamma:.6f})")
        
        # Test 3: Theta should be negative for long options
        tests_total += 1
        if calc.theta < 0:
            print(f"✅ Test 3 PASSED: Theta is negative ({calc.theta:.4f})")
            tests_passed += 1
        else:
            print(f"❌ Test 3 FAILED: Theta should be negative (got {calc.theta:.4f})")
        
        # Test 4: Vega should be positive
        tests_total += 1
        if calc.vega > 0:
            print(f"✅ Test 4 PASSED: Vega is positive ({calc.vega:.4f})")
            tests_passed += 1
        else:
            print(f"❌ Test 4 FAILED: Vega should be positive (got {calc.vega:.4f})")
        
        # Test 5: Put-Call Parity
        tests_total += 1
        call = GreeksCalculator(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call")
        put = GreeksCalculator(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put")
        parity = call.price - put.price - 100 + 100 * np.exp(-0.05 * 0.25)
        if abs(parity) < 0.01:
            print(f"✅ Test 5 PASSED: Put-Call parity holds (error = {parity:.6f})")
            tests_passed += 1
        else:
            print(f"❌ Test 5 FAILED: Put-Call parity violated (error = {parity:.6f})")
        
        # Test 6: Put delta should be negative
        tests_total += 1
        if put.delta < 0:
            print(f"✅ Test 6 PASSED: Put delta is negative ({put.delta:.4f})")
            tests_passed += 1
        else:
            print(f"❌ Test 6 FAILED: Put delta should be negative (got {put.delta:.4f})")
        
        print(f"\n{'─'*60}")
        print(f"  Results: {tests_passed}/{tests_total} tests passed")
        print(f"{'='*60}\n")
        
        return tests_passed == tests_total


# =============================================================================
# OPTION POSITION: Represents an option contract in a portfolio
# =============================================================================

@dataclass
class OptionPosition:
    """
    Represents an option position in a portfolio.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol of the underlying (e.g., "NVDA")
    option_type : str
        "call" or "put"
    strike : float
        Strike price
    expiry : datetime or float
        Expiration date or time to expiry in years
    quantity : int
        Number of contracts (positive = long, negative = short)
    volatility : float
        Implied volatility for pricing
    r : float
        Risk-free rate
    """
    ticker: str
    option_type: str
    strike: float
    expiry: float  # Time to expiry in years
    quantity: int
    volatility: float = 0.30
    r: float = 0.05
    
    # Current price of underlying (set externally)
    underlying_price: float = field(default=0.0, repr=False)
    
    def get_greeks(self) -> Optional[GreeksCalculator]:
        """Calculate Greeks for this position."""
        if self.underlying_price <= 0 or self.expiry <= 0:
            return None
        
        return GreeksCalculator(
            S=self.underlying_price,
            K=self.strike,
            T=self.expiry,
            r=self.r,
            sigma=self.volatility,
            option_type=self.option_type
        )
    
    def get_position_delta(self) -> float:
        """Get total delta for this position (delta * quantity * 100)."""
        greeks = self.get_greeks()
        if greeks is None:
            return 0.0
        # Each option contract is for 100 shares
        return greeks.delta * self.quantity * 100
    
    def get_position_value(self) -> float:
        """Get total value of this position."""
        greeks = self.get_greeks()
        if greeks is None:
            return 0.0
        return greeks.price * self.quantity * 100


# =============================================================================
# PORTFOLIO: Collection of positions with aggregated analytics
# =============================================================================

class Portfolio:
    """
    A portfolio of options and stock positions with integrated Greeks.
    
    This class manages multiple option positions across different underlyings
    and provides methods for calculating portfolio-level Greeks and hedge ratios.
    
    Examples
    --------
    >>> portfolio = Portfolio()
    >>> portfolio.add_call_option("NVDA", K=180, T=0.25, quantity=-10, sigma=0.50)
    >>> portfolio.update_prices({"NVDA": 185.0})
    >>> print(f"Portfolio delta: {portfolio.total_delta():.2f}")
    >>> hedge = portfolio.get_delta_hedge()
    >>> print(f"Buy {hedge['NVDA']:.0f} shares of NVDA to hedge")
    """
    
    def __init__(self, r: float = 0.05):
        """
        Initialize an empty portfolio.
        
        Parameters
        ----------
        r : float
            Risk-free rate to use for all positions
        """
        self.r = r
        self.option_positions: List[OptionPosition] = []
        self.stock_positions: Dict[str, float] = {}  # ticker -> shares
        self.current_prices: Dict[str, float] = {}   # ticker -> price
    
    def add_call_option(
        self,
        ticker: str,
        K: float,
        T: float,
        quantity: int,
        sigma: float = 0.30
    ):
        """
        Add a call option position to the portfolio.
        
        Parameters
        ----------
        ticker : str
            Underlying ticker symbol
        K : float
            Strike price
        T : float
            Time to expiry in years
        quantity : int
            Number of contracts (positive = long, negative = short)
        sigma : float
            Implied volatility
        """
        position = OptionPosition(
            ticker=ticker,
            option_type="call",
            strike=K,
            expiry=T,
            quantity=quantity,
            volatility=sigma,
            r=self.r
        )
        self.option_positions.append(position)
        print(f"📈 Added {'long' if quantity > 0 else 'short'} {abs(quantity)} "
              f"{ticker} calls @ K=${K:.2f}, T={T:.3f}y, σ={sigma*100:.1f}%")
    
    def add_put_option(
        self,
        ticker: str,
        K: float,
        T: float,
        quantity: int,
        sigma: float = 0.30
    ):
        """Add a put option position to the portfolio."""
        position = OptionPosition(
            ticker=ticker,
            option_type="put",
            strike=K,
            expiry=T,
            quantity=quantity,
            volatility=sigma,
            r=self.r
        )
        self.option_positions.append(position)
        print(f"📉 Added {'long' if quantity > 0 else 'short'} {abs(quantity)} "
              f"{ticker} puts @ K=${K:.2f}, T={T:.3f}y, σ={sigma*100:.1f}%")
    
    def add_stock(self, ticker: str, shares: float):
        """
        Add or update a stock position.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        shares : float
            Number of shares (can be fractional)
        """
        current = self.stock_positions.get(ticker, 0)
        self.stock_positions[ticker] = current + shares
        print(f"🔶 {'Bought' if shares > 0 else 'Sold'} {abs(shares):.2f} shares of {ticker}")
    
    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices for all underlying assets.
        
        Parameters
        ----------
        prices : dict
            Mapping of ticker -> current price
        """
        self.current_prices.update(prices)
        
        # Update option positions with new underlying prices
        for pos in self.option_positions:
            if pos.ticker in prices:
                pos.underlying_price = prices[pos.ticker]
    
    def total_delta(self, ticker: Optional[str] = None) -> float:
        """
        Calculate total portfolio delta.
        
        Parameters
        ----------
        ticker : str, optional
            If provided, calculate delta only for this underlying
            
        Returns
        -------
        float
            Total delta (in shares equivalent)
        """
        total = 0.0
        
        # Option deltas
        for pos in self.option_positions:
            if ticker is None or pos.ticker == ticker:
                total += pos.get_position_delta()
        
        # Stock positions (delta = 1 per share)
        for tick, shares in self.stock_positions.items():
            if ticker is None or tick == ticker:
                total += shares
        
        return total
    
    def get_delta_hedge(self) -> Dict[str, float]:
        """
        Calculate shares needed to delta hedge the portfolio.
        
        Returns
        -------
        dict
            Mapping of ticker -> shares to buy (positive) or sell (negative)
            to make the portfolio delta-neutral
        """
        hedge = {}
        
        # Get all unique tickers
        tickers = set()
        for pos in self.option_positions:
            tickers.add(pos.ticker)
        for tick in self.stock_positions:
            tickers.add(tick)
        
        # Calculate hedge for each ticker
        for ticker in tickers:
            current_delta = self.total_delta(ticker)
            # To hedge, we need to offset the delta
            hedge[ticker] = -current_delta
        
        return hedge
    
    def apply_delta_hedge(self):
        """Apply delta hedge by adjusting stock positions."""
        hedge = self.get_delta_hedge()
        
        print("\n🔄 Applying Delta Hedge:")
        for ticker, shares in hedge.items():
            if abs(shares) > 0.01:  # Only hedge if meaningful
                self.add_stock(ticker, shares)
        
        # Verify hedge
        print("\n✅ Post-hedge portfolio deltas:")
        for ticker in set(pos.ticker for pos in self.option_positions):
            delta = self.total_delta(ticker)
            print(f"   {ticker}: Δ = {delta:+.2f}")
    
    def summary(self) -> str:
        """Generate a comprehensive portfolio summary."""
        lines = [
            "\n" + "="*60,
            "  PORTFOLIO SUMMARY",
            "="*60,
        ]
        
        # Option positions
        if self.option_positions:
            lines.append("\n  OPTION POSITIONS:")
            lines.append("  " + "-"*56)
            for pos in self.option_positions:
                greeks = pos.get_greeks()
                if greeks:
                    lines.append(
                        f"  {pos.ticker} {pos.option_type.upper()} K=${pos.strike:.0f} "
                        f"| Qty: {pos.quantity:+d} | Δ: {pos.get_position_delta():+.2f} "
                        f"| Value: ${pos.get_position_value():,.2f}"
                    )
        
        # Stock positions
        if self.stock_positions:
            lines.append("\n  STOCK POSITIONS:")
            lines.append("  " + "-"*56)
            for ticker, shares in self.stock_positions.items():
                price = self.current_prices.get(ticker, 0)
                value = shares * price
                lines.append(
                    f"  {ticker} | Shares: {shares:+.2f} | "
                    f"Price: ${price:.2f} | Value: ${value:,.2f}"
                )
        
        # Portfolio totals
        lines.append("\n  PORTFOLIO GREEKS:")
        lines.append("  " + "-"*56)
        
        # Calculate totals by ticker
        tickers = set(pos.ticker for pos in self.option_positions)
        for ticker in tickers:
            delta = self.total_delta(ticker)
            lines.append(f"  {ticker} Total Δ: {delta:+.2f}")
        
        lines.append("="*60 + "\n")
        
        return "\n".join(lines)


# =============================================================================
# DELTA HEDGING SIMULATOR: Backtest hedging strategies
# =============================================================================

class DeltaHedgingSimulator:
    """
    Simulate and analyze delta hedging strategies using historical data.
    
    This class loads stock price data and simulates the performance of
    delta-hedged option positions over time.
    
    Examples
    --------
    >>> sim = DeltaHedgingSimulator()
    >>> results = sim.run_simulation(
    ...     ticker="NVDA",
    ...     option_type="call", 
    ...     strike=180,
    ...     days=30
    ... )
    >>> sim.plot_results(results)
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the simulator.
        
        Parameters
        ----------
        data_dir : str
            Directory containing stock data CSV files
        """
        self.data_dir = self._resolve_data_dir(data_dir)
        self.stock_data: Dict[str, pd.DataFrame] = {}
    
    def _resolve_data_dir(self, data_dir: str) -> str:
        """Resolve data directory path."""
        if os.path.isabs(data_dir):
            return data_dir
        
        # Try relative to script location
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(script_dir, data_dir)
            if os.path.exists(candidate):
                return candidate
        except NameError:
            pass
        
        # Try relative to current directory
        if os.path.exists(data_dir):
            return data_dir
        
        # Search for Option_Pricing directory
        current = os.getcwd()
        for _ in range(10):
            if "Option_Pricing" in current:
                candidate = os.path.join(current, data_dir)
                if os.path.exists(candidate):
                    return candidate
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        
        return data_dir
    
    def load_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load historical stock data for a ticker.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame with stock price history
        """
        if ticker in self.stock_data:
            return self.stock_data[ticker]
        
        # Search for matching CSV file
        if not os.path.exists(self.data_dir):
            print(f"⚠️  Data directory not found: {self.data_dir}")
            return None
        
        pattern = f"stock_data_{ticker.upper()}_"
        matching_files = [
            f for f in os.listdir(self.data_dir)
            if f.startswith(pattern) and f.endswith(".csv")
        ]
        
        if not matching_files:
            print(f"⚠️  No data file found for {ticker}")
            return None
        
        # Load most recent file
        latest_file = sorted(matching_files)[-1]
        file_path = os.path.join(self.data_dir, latest_file)
        
        print(f"📂 Loading {ticker} data from: {latest_file}")
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        self.stock_data[ticker] = df
        return df
    
    def run_simulation(
        self,
        ticker: str,
        option_type: str = "call",
        strike: Optional[float] = None,
        days: int = 30,
        volatility: Optional[float] = None,
        r: float = 0.05,
        rebalance_frequency: Literal["daily", "weekly", "threshold"] = "daily",
        delta_threshold: float = 0.05,
        initial_option_qty: int = -10,
    ) -> Dict:
        """
        Run a delta hedging simulation.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        option_type : str
            "call" or "put"
        strike : float, optional
            Strike price (default: ATM based on starting price)
        days : int
            Number of days to simulate
        volatility : float, optional
            Implied volatility (default: calculated from historical data)
        r : float
            Risk-free rate
        rebalance_frequency : str
            "daily", "weekly", or "threshold"
        delta_threshold : float
            For threshold rebalancing: rebalance when delta drifts by this much
        initial_option_qty : int
            Initial option position (negative = short)
            
        Returns
        -------
        dict
            Simulation results with P&L, Greeks over time, etc.
        """
        # Load data
        df = self.load_stock_data(ticker)
        if df is None or len(df) < days:
            print(f"❌ Insufficient data for {ticker}")
            return {}
        
        # Use last 'days' of data for simulation
        df = df.tail(days + 1).reset_index(drop=True)
        
        # Set strike (ATM if not specified)
        start_price = df['close'].iloc[0]
        if strike is None:
            strike = round(start_price / 5) * 5  # Round to nearest 5
        
        # Calculate historical volatility if not provided
        if volatility is None:
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            print(f"📊 Calculated historical volatility: {volatility*100:.1f}%")
        
        print(f"\n{'='*60}")
        print(f"  DELTA HEDGING SIMULATION: {ticker}")
        print(f"{'='*60}")
        print(f"  Option: {option_type.upper()} @ K=${strike:.0f}")
        print(f"  Quantity: {initial_option_qty:+d} contracts")
        print(f"  Volatility: {volatility*100:.1f}%")
        print(f"  Simulation period: {days} days")
        print(f"  Rebalance: {rebalance_frequency}")
        print(f"{'─'*60}\n")
        
        # Initialize tracking variables
        results = {
            'dates': [],
            'stock_prices': [],
            'option_values': [],
            'hedge_shares': [],
            'hedge_values': [],
            'total_pnl': [],
            'delta': [],
            'gamma': [],
            'theta': [],
            'rebalance_days': [],
            'cumulative_shares_traded': 0,
        }
        
        current_hedge_shares = 0
        initial_cash = 0  # Cash from option premium if short
        last_rebalance_delta = None
        
        # Simulate each day
        for i, row in df.iterrows():
            date = row['date']
            price = row['close']
            
            # Time to expiry (starts at T and decreases)
            initial_T = days / 365
            T = max(initial_T - (i / 365), 1e-6)
            
            # Calculate Greeks
            greeks = GreeksCalculator(
                S=price,
                K=strike,
                T=T,
                r=r,
                sigma=volatility,
                option_type=option_type
            )
            
            # Option position value
            option_value = greeks.price * initial_option_qty * 100
            
            # Position delta (in shares)
            position_delta = greeks.delta * initial_option_qty * 100
            
            # Determine if we should rebalance
            should_rebalance = False
            if i == 0:
                should_rebalance = True  # Initial hedge
            elif rebalance_frequency == "daily":
                should_rebalance = True
            elif rebalance_frequency == "weekly":
                should_rebalance = (i % 5 == 0)
            elif rebalance_frequency == "threshold":
                if last_rebalance_delta is not None:
                    delta_drift = abs(position_delta - last_rebalance_delta) / 100
                    should_rebalance = (delta_drift > delta_threshold)
            
            # Execute rebalancing
            if should_rebalance:
                target_hedge = -position_delta  # Offset the delta
                shares_to_trade = target_hedge - current_hedge_shares
                
                if abs(shares_to_trade) > 0.5:
                    results['cumulative_shares_traded'] += abs(shares_to_trade)
                    current_hedge_shares = target_hedge
                    last_rebalance_delta = position_delta
                    results['rebalance_days'].append(i)
            
            # Calculate hedge position value
            hedge_value = current_hedge_shares * price
            
            # Total portfolio value
            total_value = option_value + hedge_value + initial_cash
            
            # Track results
            results['dates'].append(date)
            results['stock_prices'].append(price)
            results['option_values'].append(option_value)
            results['hedge_shares'].append(current_hedge_shares)
            results['hedge_values'].append(hedge_value)
            results['total_pnl'].append(total_value)
            results['delta'].append(position_delta)
            results['gamma'].append(greeks.gamma * initial_option_qty * 100)
            results['theta'].append(greeks.theta * initial_option_qty * 100)
        
        # Calculate performance metrics
        results['final_pnl'] = results['total_pnl'][-1] - results['total_pnl'][0]
        results['pnl_std'] = np.std(np.diff(results['total_pnl']))
        results['stock_return'] = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        results['num_rebalances'] = len(results['rebalance_days'])
        
        # Summary
        print(f"📊 SIMULATION RESULTS:")
        print(f"   Stock return: {results['stock_return']:+.2f}%")
        print(f"   Final P&L: ${results['final_pnl']:+,.2f}")
        print(f"   P&L volatility: ${results['pnl_std']:.2f}")
        print(f"   Rebalances: {results['num_rebalances']}")
        print(f"   Total shares traded: {results['cumulative_shares_traded']:.0f}")
        
        return results
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """
        Create visualization of hedging simulation results.
        
        Parameters
        ----------
        results : dict
            Output from run_simulation()
        save_path : str, optional
            Path to save the figure
        """
        if not results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Stock price and hedge shares
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        ax1.plot(results['dates'], results['stock_prices'], 'b-', label='Stock Price', linewidth=2)
        ax1_twin.fill_between(results['dates'], results['hedge_shares'], alpha=0.3, color='orange', label='Hedge Shares')
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price ($)', color='blue')
        ax1_twin.set_ylabel('Hedge Shares', color='orange')
        ax1.set_title('Stock Price & Hedge Position')
        ax1.legend(loc='upper left')
        
        # Plot 2: P&L over time
        ax2 = axes[0, 1]
        ax2.plot(results['dates'], results['total_pnl'], 'g-', linewidth=2, label='Hedged P&L')
        ax2.plot(results['dates'], results['option_values'], 'r--', linewidth=1.5, alpha=0.7, label='Unhedged Option P&L')
        
        # Mark rebalance points
        for day in results['rebalance_days'][1:]:  # Skip initial
            if day < len(results['dates']):
                ax2.axvline(results['dates'][day], color='gray', linestyle=':', alpha=0.3)
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('P&L ($)')
        ax2.set_title('Portfolio P&L (Hedged vs Unhedged)')
        ax2.legend()
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 3: Delta over time
        ax3 = axes[1, 0]
        ax3.plot(results['dates'], results['delta'], 'purple', linewidth=2)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Position Delta')
        ax3.set_title('Position Delta Over Time')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 4: Gamma and Theta
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()
        
        ax4.plot(results['dates'], results['gamma'], 'red', linewidth=2, label='Gamma')
        ax4_twin.plot(results['dates'], results['theta'], 'blue', linewidth=2, label='Theta')
        
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Gamma', color='red')
        ax4_twin.set_ylabel('Theta ($/day)', color='blue')
        ax4.set_title('Gamma & Theta Over Time')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            # Make path relative to script directory if not absolute
            if not os.path.isabs(save_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                save_path = os.path.join(script_dir, save_path)
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved plot to: {save_path}")
        
        plt.show()
    
    def compare_strategies(
        self,
        ticker: str,
        strategies: List[str] = ["daily", "weekly", "threshold"],
        **kwargs
    ) -> pd.DataFrame:
        """
        Compare different rebalancing strategies.
        
        Parameters
        ----------
        ticker : str
            Stock ticker
        strategies : list
            List of rebalancing strategies to compare
            
        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        print("\n" + "="*60)
        print("  STRATEGY COMPARISON")
        print("="*60 + "\n")
        
        comparison_data = []
        
        for strategy in strategies:
            print(f"\n{'─'*40}")
            print(f"Running {strategy} strategy...")
            print('─'*40)
            
            results = self.run_simulation(
                ticker=ticker,
                rebalance_frequency=strategy,
                **kwargs
            )
            
            if results:
                comparison_data.append({
                    'Strategy': strategy.capitalize(),
                    'Final P&L': f"${results['final_pnl']:+,.2f}",
                    'P&L Std Dev': f"${results['pnl_std']:.2f}",
                    'Num Rebalances': results['num_rebalances'],
                    'Shares Traded': f"{results['cumulative_shares_traded']:.0f}",
                })
        
        df = pd.DataFrame(comparison_data)
        print("\n" + "="*60)
        print("  COMPARISON SUMMARY")
        print("="*60)
        print(df.to_string(index=False))
        
        return df


# =============================================================================
# MAIN: Demo and testing
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "🚀"*30)
    print("  DELTA HEDGING PORTFOLIO DEMO")
    print("🚀"*30 + "\n")
    
    # ==========================================================================
    # Part 1: Test Greeks Calculator
    # ==========================================================================
    
    print("\n" + "═"*60)
    print("  PART 1: GREEKS CALCULATOR TEST")
    print("═"*60 + "\n")
    
    # Run unit tests
    GreeksCalculator.test_greeks()
    
    # Show example for NVDA
    print("\nExample: NVDA Call Option")
    print("─"*40)
    nvda_call = GreeksCalculator(
        S=185,      # Current NVDA price
        K=180,      # Strike
        T=30/365,   # 30 days to expiry
        r=0.05,     
        sigma=0.50, # 50% volatility
        option_type="call"
    )
    print(nvda_call.summary())
    
    # ==========================================================================
    # Part 2: Portfolio Demo
    # ==========================================================================
    
    print("\n" + "═"*60)
    print("  PART 2: PORTFOLIO MANAGEMENT")
    print("═"*60 + "\n")
    
    # Create a portfolio
    portfolio = Portfolio(r=0.05)
    
    # Add short calls on tech stocks (simulating a covered call strategy)
    portfolio.add_call_option("NVDA", K=190, T=30/365, quantity=-5, sigma=0.50)
    portfolio.add_call_option("GOOGL", K=330, T=30/365, quantity=-3, sigma=0.30)
    portfolio.add_call_option("AAPL", K=260, T=30/365, quantity=-5, sigma=0.28)
    
    # Update with current prices
    portfolio.update_prices({
        "NVDA": 185.0,
        "GOOGL": 325.0,
        "AAPL": 259.0,
    })
    
    # Show portfolio before hedging
    print(portfolio.summary())
    
    # Calculate and display hedge requirements
    print("\n📊 DELTA HEDGE REQUIREMENTS:")
    print("─"*40)
    hedge = portfolio.get_delta_hedge()
    for ticker, shares in hedge.items():
        print(f"  {ticker}: Buy {shares:+.0f} shares")
    
    # Apply the hedge
    portfolio.apply_delta_hedge()
    
    # Show final portfolio
    print(portfolio.summary())
    
    # ==========================================================================
    # Part 3: Delta Hedging Simulation
    # ==========================================================================
    
    print("\n" + "═"*60)
    print("  PART 3: DELTA HEDGING SIMULATION")
    print("═"*60 + "\n")
    
    simulator = DeltaHedgingSimulator()
    
    # Run simulation for NVDA
    results = simulator.run_simulation(
        ticker="NVDA",
        option_type="call",
        strike=180,
        days=60,
        volatility=0.50,
        rebalance_frequency="daily",
        initial_option_qty=-10,  # Short 10 calls
    )
    
    if results:
        # Plot results
        simulator.plot_results(results, save_path="plots/delta_hedge_nvda.png")
        
        # Compare different strategies
        print("\n")
        comparison = simulator.compare_strategies(
            ticker="NVDA",
            days=60,
            volatility=0.50,
            strike=180,
            initial_option_qty=-10,
        )
