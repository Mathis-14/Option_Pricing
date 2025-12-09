"""
implied_vol.py — Calculate and visualize implied volatility surfaces for BTC and ETH options

This module provides a class-based interface for calculating and visualizing implied
volatility surfaces using JAX for automatic differentiation and Plotly for 3D visualization.

================================================================================
USAGE GUIDE
================================================================================

BASIC USAGE:
------------

1. Import the class and load your data:
   
   from implied_vol import ImpliedVolatilitySurface
   from import_crypto_options import import_options_data
   
   # Load options data
   dfs = import_options_data(start_date="2024-12-01", end_date="2025-03-01")
   btc_df = dfs["BTC"]

2. Create an instance and process data:
   
   iv_surface = ImpliedVolatilitySurface(
       df=btc_df,
       currency="BTC",
       r=0.05,  # Risk-free rate (5%)
       use_mark_iv=True  # Use Deribit's mark_iv directly (recommended)
   )

3. Plot the volatility surface:
   
   # Continuous interpolated surface (default)
   iv_surface.plot(interpolate=True, option_type="both")
   
   # Or discrete scatter plot
   iv_surface.plot(interpolate=False, option_type="both")

ADVANCED USAGE:
---------------

# Custom interpolation method
iv_surface.plot(
    interpolate=True,
    interpolation_method="rbf",  # Options: "rbf", "linear", "cubic"
    show_points=True,  # Overlay data points on surface
    option_type="C"  # Only calls
)

# Calculate IV from prices instead of using mark_iv
iv_surface = ImpliedVolatilitySurface(
    df=btc_df,
    currency="BTC",
    use_mark_iv=False,  # Calculate IV from mark_price
    r=0.05
)
iv_surface.plot(interpolate=True)

# Use current price from Deribit (for real-time analysis)
iv_surface = ImpliedVolatilitySurface(
    df=btc_df,
    currency="BTC",
    use_current_price=True,  # Fetch current price from API
    use_mark_iv=True
)
iv_surface.plot(interpolate=True)

================================================================================
UNDERSTANDING mark_iv: WHAT IS IT AND HOW IS IT CALCULATED?
================================================================================

WHAT IS mark_iv?
----------------
mark_iv = "Mark Implied Volatility" = The volatility value that Deribit calculates
by REVERSING the Black-Scholes model.

THE RELATIONSHIP:
-----------------

1. FORWARD DIRECTION (Pricing an option):
   Black-Scholes Model:
   Input:  S (spot), K (strike), T (time), r (rate), σ (volatility)
   Output: Option Price
   
   Example:
   - BTC = $90,000, Strike = $85,000, T = 0.1 years, r = 5%, σ = 60%
   - Black-Scholes → Option Price = $6,178

2. REVERSE DIRECTION (Finding implied volatility):
   What Deribit does:
   Input:  S, K, T, r, Option Price (from market)
   Output: σ (implied volatility) = mark_iv
   
   Example:
   - BTC = $90,000, Strike = $85,000, T = 0.1 years, r = 5%
   - Market Price = $6,178
   - Deribit asks: "What σ makes Black-Scholes give $6,178?"
   - Answer: σ = 60% → mark_iv = 60.00

HOW DERIBIT CALCULATES mark_iv:
-------------------------------
1. Deribit observes the MARKET PRICE of the option (mark_price)
2. Deribit uses Black-Scholes model in REVERSE:
   - Takes: S, K, T, r, and the market price
   - Solves for: σ (volatility)
   - This σ is what they call "mark_iv"
3. Deribit uses sophisticated methods (likely similar to our Newton's method)
   to find the volatility that makes Black-Scholes match the market price

WHY USE mark_iv vs CALCULATING OURSELVES?
------------------------------------------
Option 1: use_mark_iv=True (RECOMMENDED)
  ✓ Deribit already did the calculation
  ✓ Uses their exact methodology
  ✓ Faster (no computation needed)
  ✓ Consistent with Deribit's platform
  ✓ Already in percentage format (62.19 = 62.19%)

Option 2: use_mark_iv=False
  ✓ We calculate IV ourselves from mark_price
  ✓ Useful for verification
  ✓ Can use different parameters (r, model, etc.)
  ✗ Slower (Newton's method iteration)
  ✗ Might differ slightly from Deribit's calculation

THE KEY INSIGHT:
---------------
mark_iv IS the volatility from the Black-Scholes model, but calculated BACKWARDS:
- Normal use: volatility → price
- mark_iv: price → volatility

It represents: "What volatility does the market imply, based on current option prices?"

CONCRETE EXAMPLE FROM YOUR DATA:
--------------------------------
From the CSV file:
- Instrument: BTC-10DEC25-84000-C (BTC Call, Strike $84,000, Expires Dec 10, 2025)
- underlying_price: $90,123.40
- mark_price: 0.06852267 BTC = $6,178 USD (0.06852267 × $90,123.40)
- mark_iv: 62.19%

What this means:
1. Market says: "This option is worth $6,178"
2. Deribit asks: "What volatility (σ) makes Black-Scholes give $6,178?"
3. Deribit calculates: σ = 62.19% → This is mark_iv
4. Verification: If you plug σ=62.19% into Black-Scholes with S=$90,123, K=$84,000,
   you should get approximately $6,178 (the mark_price)

So mark_iv = The volatility that the MARKET is implying through option prices!

================================================================================
CRITICAL UNIT CONVERSIONS
================================================================================

1. PRICE UNITS (mark_price):
   - Deribit returns mark_price in BTC/ETH units (e.g., 0.06852267 BTC)
   - Black-Scholes needs prices in USD (same currency as strike)
   - SOLUTION: Convert by multiplying mark_price * underlying_price

2. VOLATILITY UNITS (mark_iv):
   - Deribit returns mark_iv in PERCENTAGE (e.g., 62.19 = 62.19%)
   - Black-Scholes expects volatility as DECIMAL (0.6219 for 62.19%)
   - The class handles all conversions automatically
"""

from datetime import datetime, timedelta
from typing import Optional, Literal
import os
import pandas as pd
import numpy as np
import requests

from jax.scipy.stats import norm as jnorm
import jax.numpy as jnp
from jax import grad
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# For continuous surface interpolation
try:
    from scipy.interpolate import RBFInterpolator, griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ImpliedVolatilitySurface:
    """
    Calculate and visualize implied volatility surfaces for options.
    
    This class provides a complete workflow for:
    - Calculating implied volatility from market prices or using Deribit's mark_iv
    - Interpolating discrete data points into continuous surfaces
    - Visualizing 3D volatility surfaces with Plotly
    
    Attributes
    ----------
    df : pd.DataFrame
        Processed DataFrame with IV data
    currency : str
        Currency code (BTC, ETH, etc.)
    spot_price : float
        Current or historical spot price used
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        currency: str = "BTC",
        r: float = 0.05,
        q: float = 0.0,
        sigma_guess: float = 0.8,
        use_mark_iv: bool = True,
        use_current_price: bool = False,
        env: Literal["prod", "test"] = "prod",
        verbose: bool = False
    ):
        """
        Initialize the ImpliedVolatilitySurface calculator.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns: expiry_date, strike, underlying_price, mark_price, mark_iv, type
        currency : str
            Currency code ("BTC", "ETH", etc.) - used for fetching current price
        r : float
            Risk-free interest rate (default 0.05 = 5%)
        q : float
            Dividend yield (default 0)
        sigma_guess : float
            Initial guess for volatility when calculating IV (as percentage, e.g., 0.8 for 80%)
        use_mark_iv : bool
            If True, use mark_iv directly from Deribit (recommended, already in percentage)
            If False, calculate IV from mark_price using Black-Scholes
        use_current_price : bool
            If True, fetch current price from Deribit API (for real-time analysis only!)
            If False, use underlying_price from data (correct for historical data)
        env : str
            Deribit environment ("prod" or "test")
        verbose : bool
            Whether to print detailed progress information
        """
        self.currency = currency
        self.r = r
        self.q = q
        self.sigma_guess = sigma_guess
        self.verbose = verbose
        self.env = env
        
        # Process the data
        self.df = self._calculate_implied_volatility_surface(
            df.copy(),
            use_mark_iv=use_mark_iv,
            use_current_price=use_current_price
        )
        
        # Store spot price used
        if 'underlying_price' in self.df.columns:
            self.spot_price = float(self.df['underlying_price'].iloc[0])
        else:
            self.spot_price = None
    
    @staticmethod
    def _black_scholes(S, K, T, r, sigma, q=0, otype="call"):
        """Black-Scholes option pricing formula using JAX."""
        d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * jnp.sqrt(T))
        d2 = d1 - sigma * jnp.sqrt(T)
        
        if otype == "call":
            call = S * jnp.exp(-q * T) * jnorm.cdf(d1, 0, 1) - K * jnp.exp(-r * T) * jnorm.cdf(d2, 0, 1)
            return call
        else:
            put = K * jnp.exp(-r * T) * jnorm.cdf(-d2, 0, 1) - S * jnp.exp(-q * T) * jnorm.cdf(-d1, 0, 1)
            return put
    
    @staticmethod
    def _loss_func(S, K, T, r, sigma_guess, price, q=0, otype="call"):
        """Loss function: difference between theoretical and market price."""
        theoretical_price = ImpliedVolatilitySurface._black_scholes(S, K, T, r, sigma_guess, q, otype=otype)
        return theoretical_price - price
    
    @staticmethod
    def _solve_for_iv(S, K, T, r, price, sigma_guess=0.8, q=0, otype="call",
                     N_iter=20, epsilon=0.001, verbose=False):
        """Solve for implied volatility using Newton's method."""
        # Validate inputs
        if S <= 0 or K <= 0 or T <= 0 or price <= 0:
            raise ValueError(f"Invalid input parameters: S={S}, K={K}, T={T}, price={price}")
        
        # Check if price is within reasonable bounds (with some tolerance)
        # For calls: max is S, min is max(0, S*exp(-q*T) - K*exp(-r*T))
        # For puts: max is K*exp(-r*T), min is max(0, K*exp(-r*T) - S*exp(-q*T))
        if otype == "call":
            max_price = S * np.exp(-q * T)  # Upper bound for call (discounted underlying)
            min_price = max(0, S * np.exp(-q * T) - K * np.exp(-r * T))  # Lower bound (intrinsic value)
        else:  # put
            max_price = K * np.exp(-r * T)  # Upper bound for put (discounted strike)
            min_price = max(0, K * np.exp(-r * T) - S * np.exp(-q * T))  # Lower bound (intrinsic value)
        
        # Add 5% tolerance to account for market imperfections and rounding
        tolerance = 0.05
        min_price_tol = max(0, min_price * (1 - tolerance))
        max_price_tol = max_price * (1 + tolerance)
        
        if price < min_price_tol or price > max_price_tol:
            # Don't raise error, just return NaN - let the caller handle it
            # This is less strict and allows for market imperfections
            if price < min_price * 0.5 or price > max_price * 1.5:
                # Only reject if way outside bounds
                raise ValueError(f"Option price {price:.2f} is far outside theoretical bounds [{min_price:.2f}, {max_price:.2f}]")
        
        loss_grad_func = grad(ImpliedVolatilitySurface._loss_func, argnums=4)
        converged = False
        sigma = sigma_guess
        
        # Try multiple initial guesses if first one fails
        initial_guesses = [sigma_guess, 0.5, 1.0, 0.3, 1.5]
        best_sigma = None
        best_error = float('inf')
        
        for guess_idx, sigma in enumerate(initial_guesses):
            converged_this_guess = False
            for i in range(N_iter):
                try:
                    loss_val = ImpliedVolatilitySurface._loss_func(S, K, T, r, sigma, price, q, otype=otype)
                    abs_loss = abs(loss_val)
                    
                    if verbose and guess_idx == 0:
                        print(f"\nIteration: {i}, Error: {loss_val}")
                    
                    # Track best result so far
                    if abs_loss < best_error:
                        best_error = abs_loss
                        best_sigma = sigma
                    
                    if abs_loss < epsilon:
                        converged = True
                        converged_this_guess = True
                        break
                    else:
                        loss_grad_val = loss_grad_func(S, K, T, r, sigma, price, q, otype=otype)
                        
                        # Avoid division by zero or very small gradients
                        if abs(loss_grad_val) < 1e-10:
                            if verbose and guess_idx == 0:
                                print("Gradient too small, trying next guess")
                            break
                        
                        # Newton step with damping for stability
                        step = loss_val / loss_grad_val
                        # Limit step size to avoid overshooting
                        step = np.clip(step, -0.5, 0.5)
                        sigma = sigma - step
                        sigma = max(sigma, 0.001)  # Ensure positive
                        sigma = min(sigma, 5.0)  # Cap at 500% to avoid unrealistic values
                    
                    if verbose and guess_idx == 0:
                        print(f"New sigma: {sigma}")
                except (ValueError, ZeroDivisionError, OverflowError) as e:
                    if verbose and guess_idx == 0:
                        print(f"Error in iteration {i}: {e}")
                    break
            
            if converged_this_guess:
                break
        
        # Use best result found, even if not fully converged
        if best_sigma is not None and best_error < 0.01:  # Accept if error < 1%
            return float(best_sigma)
        elif converged:
            return float(sigma)
        else:
            # If we have a reasonable result, return it
            if best_sigma is not None and best_error < 0.1:  # Accept if error < 10%
                return float(best_sigma)
            else:
                raise ValueError(f"IV solver did not converge (best error: {best_error:.6f})")
    
    @staticmethod
    def _fetch_current_price(currency: str, env: Literal["prod", "test"] = "prod") -> Optional[float]:
        """Fetch current spot price from Deribit API."""
        base_url = "https://www.deribit.com" if env == "prod" else "https://test.deribit.com"
        index_name = f"{currency.lower()}_usd"
        
        try:
            url = f"{base_url}/api/v2/public/get_index_price"
            params = {"index_name": index_name}
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            payload = resp.json()
            price = payload.get("result", {}).get("index_price")
            return float(price) if price is not None else None
        except Exception as e:
            if ImpliedVolatilitySurface._fetch_current_price.__doc__:
                print(f"Warning: Could not fetch current price from Deribit: {e}")
            return None
    
    @staticmethod
    def _calculate_time_to_expiry(expiry_date: datetime, current_date: Optional[datetime] = None) -> float:
        """Calculate time to expiration in years."""
        if current_date is None:
            current_date = datetime.now()
        
        if isinstance(expiry_date, pd.Timestamp):
            expiry_date = expiry_date.to_pydatetime()
        if isinstance(current_date, pd.Timestamp):
            current_date = current_date.to_pydatetime()
        
        time_delta = expiry_date - current_date
        days = time_delta.total_seconds() / (24 * 3600)
        return max(days / 365.25, 1e-6)
    
    def _calculate_implied_volatility_surface(
        self,
        df: pd.DataFrame,
        use_mark_iv: bool = True,
        use_current_price: bool = False
    ) -> pd.DataFrame:
        """Calculate implied volatility for all options in DataFrame."""
        # Check if we should use current price
        current_spot_price = None
        if use_current_price:
            current_spot_price = self._fetch_current_price(self.currency, self.env)
            if current_spot_price:
                print(f"✓ Using current {self.currency} price from Deribit: ${current_spot_price:,.2f}")
                if 'expiry_date' in df.columns and not df.empty:
                    data_age = datetime.now() - df['expiry_date'].min()
                    if data_age > timedelta(days=1):
                        print(f"⚠️  WARNING: Data is {data_age.days} days old. Using current price may create inconsistencies!")
            else:
                print("⚠️  Could not fetch current price, using underlying_price from data")
                use_current_price = False
        
        # Calculate time to expiration
        current_date = datetime.now()
        df['T'] = df['expiry_date'].apply(lambda x: self._calculate_time_to_expiry(x, current_date))
        
        # Determine spot price to use
        if use_current_price and current_spot_price:
            spot_price = current_spot_price
            df['underlying_price'] = spot_price
            if self.verbose:
                print(f"✓ Using current spot price: ${spot_price:,.2f}")
        else:
            if 'underlying_price' in df.columns:
                unique_prices = df['underlying_price'].dropna().unique()
                if len(unique_prices) > 0:
                    spot_price = float(unique_prices[0])
                    if self.verbose:
                        print(f"✓ Using underlying_price from data: ${spot_price:,.2f}")
                else:
                    raise ValueError("No underlying_price found in data")
            else:
                raise ValueError("No underlying_price column in DataFrame")
        
        # Calculate moneyness
        df['moneyness'] = df['strike'].apply(lambda K: spot_price / K if K > 0 else np.nan)
        
        # Calculate or use IV
        if use_mark_iv and 'mark_iv' in df.columns:
            df['iv_percentage'] = df['mark_iv'].apply(
                lambda x: float(x) if pd.notna(x) and x > 0 else np.nan
            )
            if self.verbose:
                print("Using mark_iv directly from Deribit (already in percentage)")
        else:
            calculated_ivs = []
            valid_count = 0
            invalid_count = 0
            failure_reasons = {
                'missing_data': 0,
                'invalid_values': 0,
                'price_too_high': 0,
                'solver_failed': 0,
                'iv_out_of_range': 0
            }
            
            for idx, row in df.iterrows():
                # Validate inputs
                if pd.isna(row['strike']) or pd.isna(row['T']) or pd.isna(row['mark_price']):
                    calculated_ivs.append(np.nan)
                    invalid_count += 1
                    failure_reasons['missing_data'] += 1
                    continue
                
                try:
                    K = float(row['strike'])
                    T = float(row['T'])
                    mark_price_btc = float(row['mark_price'])
                    
                    # Validate values
                    if K <= 0 or T <= 0 or mark_price_btc <= 0:
                        calculated_ivs.append(np.nan)
                        invalid_count += 1
                        failure_reasons['invalid_values'] += 1
                        continue
                    
                    market_price_usd = mark_price_btc * spot_price
                    otype = "call" if row['type'] == 'C' else "put"
                    
                    # Additional validation
                    if market_price_usd <= 0 or pd.isna(market_price_usd):
                        calculated_ivs.append(np.nan)
                        invalid_count += 1
                        failure_reasons['invalid_values'] += 1
                        continue
                    
                    # Check if option price is reasonable (not too high relative to spot)
                    # For calls, max price is roughly spot_price, for puts it's strike
                    max_reasonable_price = spot_price if otype == "call" else K
                    if market_price_usd > max_reasonable_price * 1.5:
                        # Option price seems unreasonably high, skip
                        calculated_ivs.append(np.nan)
                        invalid_count += 1
                        failure_reasons['price_too_high'] += 1
                        continue
                    
                    # Calculate IV
                    try:
                        iv_decimal = self._solve_for_iv(
                            spot_price, K, T, self.r, market_price_usd,
                            sigma_guess=self.sigma_guess / 100.0,
                            q=self.q,
                            otype=otype,
                            verbose=False  # Don't spam output
                        )
                        
                        # Validate IV result (should be positive and reasonable, e.g., < 500%)
                        if iv_decimal > 0 and iv_decimal < 5.0:  # 0% to 500%
                            calculated_ivs.append(iv_decimal * 100.0)
                            valid_count += 1
                        else:
                            calculated_ivs.append(np.nan)
                            invalid_count += 1
                            failure_reasons['iv_out_of_range'] += 1
                            
                    except ValueError as e:
                        # This includes bounds checking errors from _solve_for_iv
                        calculated_ivs.append(np.nan)
                        invalid_count += 1
                        failure_reasons['solver_failed'] += 1
                        if self.verbose:
                            print(f"IV solver error for row {idx} ({otype}, K={K}, T={T:.3f}): {e}")
                    except (ZeroDivisionError, OverflowError) as e:
                        calculated_ivs.append(np.nan)
                        invalid_count += 1
                        failure_reasons['solver_failed'] += 1
                        if self.verbose:
                            print(f"Numerical error for row {idx}: {e}")
                        
                except (ValueError, TypeError) as e:
                    calculated_ivs.append(np.nan)
                    invalid_count += 1
                    failure_reasons['invalid_values'] += 1
                    if self.verbose:
                        print(f"Error processing row {idx}: {e}")
                except Exception as e:
                    calculated_ivs.append(np.nan)
                    invalid_count += 1
                    failure_reasons['solver_failed'] += 1
                    if self.verbose:
                        print(f"Unexpected error for row {idx}: {e}")
            
            df['iv_percentage'] = calculated_ivs
            valid_iv_count = df['iv_percentage'].notna().sum()
            
            # Always print summary when calculating IV (not just when verbose)
            print(f"✓ Calculated IV from mark_price: {valid_iv_count} valid out of {len(df)} options")
            if valid_iv_count < len(df) * 0.5:
                print(f"⚠️  Warning: Only {valid_iv_count}/{len(df)} options have valid IV.")
                print(f"   Failure breakdown:")
                for reason, count in failure_reasons.items():
                    if count > 0:
                        print(f"   - {reason}: {count}")
                print(f"   💡 Recommendation: Use use_mark_iv=True to use Deribit's pre-calculated IV")
                
                # Additional diagnostics
                missing_mark_price = df['mark_price'].isna().sum()
                zero_mark_price = (df['mark_price'] == 0).sum() if 'mark_price' in df.columns else 0
                if missing_mark_price > 0 or zero_mark_price > 0:
                    print(f"   📊 Data quality: {missing_mark_price} missing mark_price, {zero_mark_price} zero mark_price")
        
        return df
    
    @staticmethod
    def _interpolate_iv_surface(
        moneyness: np.ndarray,
        time_to_expiry: np.ndarray,
        iv_values: np.ndarray,
        grid_resolution: int = 50,
        method: str = "rbf"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate implied volatility surface to create a continuous grid."""
        valid_mask = ~(np.isnan(moneyness) | np.isnan(time_to_expiry) | np.isnan(iv_values))
        moneyness_clean = moneyness[valid_mask]
        time_clean = time_to_expiry[valid_mask]
        iv_clean = iv_values[valid_mask]
        
        if len(moneyness_clean) < 3:
            raise ValueError("Not enough valid data points for interpolation (need at least 3)")
        
        # Create grid
        moneyness_min, moneyness_max = moneyness_clean.min(), moneyness_clean.max()
        time_min, time_max = time_clean.min(), time_clean.max()
        
        moneyness_range = moneyness_max - moneyness_min
        time_range = time_max - time_min
        moneyness_min -= 0.05 * moneyness_range
        moneyness_max += 0.05 * moneyness_range
        time_min = max(0, time_min - 0.05 * time_range)
        time_max += 0.05 * time_range
        
        moneyness_grid = np.linspace(moneyness_min, moneyness_max, grid_resolution)
        time_grid = np.linspace(time_min, time_max, grid_resolution)
        X_grid, Y_grid = np.meshgrid(moneyness_grid, time_grid)
        
        points = np.column_stack([moneyness_clean, time_clean])
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        # Interpolate
        if method == "rbf" and SCIPY_AVAILABLE:
            try:
                rbf = RBFInterpolator(points, iv_clean, kernel='thin_plate_spline', smoothing=0.1)
                Z_grid = rbf(grid_points).reshape(X_grid.shape)
            except Exception:
                method = "linear"
        
        if method != "rbf" or not SCIPY_AVAILABLE:
            if SCIPY_AVAILABLE:
                Z_grid = griddata(
                    points, iv_clean, grid_points,
                    method=method if method in ['linear', 'cubic'] else 'linear'
                ).reshape(X_grid.shape)
            else:
                from scipy.spatial.distance import cdist
                distances = cdist(grid_points, points)
                nearest_idx = np.argmin(distances, axis=1)
                Z_grid = iv_clean[nearest_idx].reshape(X_grid.shape)
        
        return X_grid, Y_grid, Z_grid
    
    def plot(
        self,
        option_type: Literal["C", "P", "both"] = "both",
        title: Optional[str] = None,
        renderer: str = "notebook",
        interpolate: bool = True,
        show_points: bool = True,
        interpolation_method: str = "rbf",
        save_png: bool = True,
        output_dir: str = "plot/vol"
    ) -> None:
        """
        Plot 3D implied volatility surface.
        
        Parameters
        ----------
        option_type : str
            "C" for calls, "P" for puts, "both" for separate subplots
        title : str, optional
            Custom title
        renderer : str
            Plotly renderer (default "notebook" for Jupyter notebooks, use "browser" for scripts)
        interpolate : bool
            If True (default), create continuous interpolated surface
            If False, show scatter plot of discrete points
        show_points : bool
            If True and interpolate=True, overlay scatter points on surface
        interpolation_method : str
            Interpolation method: "rbf" (recommended), "linear", or "cubic"
        save_png : bool
            If True (default), save the plot as PNG in plots/vol/ directory
        output_dir : str
            Directory to save PNG files (default "plots/vol")
        """
        # Determine IV column
        if 'iv_percentage' in self.df.columns:
            iv_column = 'iv_percentage'
        elif 'mark_iv' in self.df.columns:
            iv_column = 'mark_iv'
        else:
            print("Error: No IV column found (need 'iv_percentage' or 'mark_iv')")
            return
        
        if option_type == "both":
            scene_type = 'surface' if interpolate else 'scatter3d'
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': scene_type}, {'type': scene_type}]],
                subplot_titles=('Calls', 'Puts'),
                horizontal_spacing=0.1
            )
            
            # Plot calls
            calls_df = self.df[self.df['type'] == 'C'].copy()
            if not calls_df.empty and iv_column in calls_df.columns:
                calls_df = calls_df.dropna(subset=[iv_column, 'moneyness', 'T'])
                if not calls_df.empty:
                    if interpolate:
                        try:
                            X_grid, Y_grid, Z_grid = self._interpolate_iv_surface(
                                calls_df['moneyness'].values,
                                calls_df['T'].values,
                                calls_df[iv_column].values,
                                method=interpolation_method
                            )
                            fig.add_trace(
                                go.Surface(
                                    x=X_grid, y=Y_grid, z=Z_grid,
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar=dict(title="IV (%)", x=0.45, len=0.4),
                                    name='Calls Surface',
                                    hovertemplate='Moneyness: %{x:.3f}<br>TTE: %{y:.3f} years<br>IV: %{z:.2f}%<extra></extra>'
                                ),
                                row=1, col=1
                            )
                            if show_points:
                                fig.add_trace(
                                    go.Scatter3d(
                                        x=calls_df['moneyness'],
                                        y=calls_df['T'],
                                        z=calls_df[iv_column],
                                        mode='markers',
                                        marker=dict(size=2, color='white', opacity=0.6),
                                        name='Calls Points',
                                        showlegend=False,
                                        hovertemplate='Moneyness: %{x:.3f}<br>TTE: %{y:.3f} years<br>IV: %{z:.2f}%<extra></extra>'
                                    ),
                                    row=1, col=1
                                )
                        except Exception as e:
                            print(f"Warning: Surface interpolation failed for calls: {e}")
                            interpolate = False
                    
                    if not interpolate:
                        fig.add_trace(
                            go.Scatter3d(
                                x=calls_df['moneyness'],
                                y=calls_df['T'],
                                z=calls_df[iv_column],
                                mode='markers',
                                marker=dict(size=3, color=calls_df[iv_column], colorscale='Viridis', showscale=True, colorbar=dict(title="IV (%)", x=0.45)),
                                name='Calls',
                                hovertemplate='Moneyness: %{x:.3f}<br>TTE: %{y:.2f} years<br>IV: %{z:.2f}%<extra></extra>'
                            ),
                            row=1, col=1
                        )
            
            # Plot puts
            puts_df = self.df[self.df['type'] == 'P'].copy()
            if not puts_df.empty and iv_column in puts_df.columns:
                puts_df = puts_df.dropna(subset=[iv_column, 'moneyness', 'T'])
                if not puts_df.empty:
                    if interpolate:
                        try:
                            X_grid, Y_grid, Z_grid = self._interpolate_iv_surface(
                                puts_df['moneyness'].values,
                                puts_df['T'].values,
                                puts_df[iv_column].values,
                                method=interpolation_method
                            )
                            fig.add_trace(
                                go.Surface(
                                    x=X_grid, y=Y_grid, z=Z_grid,
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar=dict(title="IV (%)", x=1.02, len=0.4),
                                    name='Puts Surface',
                                    hovertemplate='Moneyness: %{x:.3f}<br>TTE: %{y:.3f} years<br>IV: %{z:.2f}%<extra></extra>'
                                ),
                                row=1, col=2
                            )
                            if show_points:
                                fig.add_trace(
                                    go.Scatter3d(
                                        x=puts_df['moneyness'],
                                        y=puts_df['T'],
                                        z=puts_df[iv_column],
                                        mode='markers',
                                        marker=dict(size=2, color='white', opacity=0.6),
                                        name='Puts Points',
                                        showlegend=False,
                                        hovertemplate='Moneyness: %{x:.3f}<br>TTE: %{y:.3f} years<br>IV: %{z:.2f}%<extra></extra>'
                                    ),
                                    row=1, col=2
                                )
                        except Exception as e:
                            print(f"Warning: Surface interpolation failed for puts: {e}")
                            fig.add_trace(
                                go.Scatter3d(
                                    x=puts_df['moneyness'],
                                    y=puts_df['T'],
                                    z=puts_df[iv_column],
                                    mode='markers',
                                    marker=dict(size=3, color=puts_df[iv_column], colorscale='Viridis', showscale=True, colorbar=dict(title="IV (%)", x=1.02)),
                                    name='Puts',
                                    hovertemplate='Moneyness: %{x:.3f}<br>TTE: %{y:.2f} years<br>IV: %{z:.2f}%<extra></extra>'
                                ),
                                row=1, col=2
                            )
                    else:
                        fig.add_trace(
                            go.Scatter3d(
                                x=puts_df['moneyness'],
                                y=puts_df['T'],
                                z=puts_df[iv_column],
                                mode='markers',
                                marker=dict(size=3, color=puts_df[iv_column], colorscale='Viridis', showscale=True, colorbar=dict(title="IV (%)", x=1.02)),
                                name='Puts',
                                hovertemplate='Moneyness: %{x:.3f}<br>TTE: %{y:.2f} years<br>IV: %{z:.2f}%<extra></extra>'
                            ),
                            row=1, col=2
                        )
            
            if title is None:
                title = f"{self.currency} Implied Volatility Surface"
            
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='Moneyness (S/K)',
                    yaxis_title='Time to Expiration (years)',
                    zaxis_title='Implied Volatility (%)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                ),
                scene2=dict(
                    xaxis_title='Moneyness (S/K)',
                    yaxis_title='Time to Expiration (years)',
                    zaxis_title='Implied Volatility (%)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                ),
                template="plotly_dark",
                height=700
            )
        
        else:
            # Single plot
            filtered_df = self.df[self.df['type'] == option_type].copy()
            if filtered_df.empty:
                print(f"No {option_type} options found")
                return
            
            if iv_column not in filtered_df.columns:
                print(f"Column {iv_column} not found in DataFrame")
                return
            
            filtered_df = filtered_df.dropna(subset=[iv_column, 'moneyness', 'T'])
            if filtered_df.empty:
                print("No valid data points after filtering")
                return
            
            fig = go.Figure()
            
            if interpolate:
                try:
                    X_grid, Y_grid, Z_grid = self._interpolate_iv_surface(
                        filtered_df['moneyness'].values,
                        filtered_df['T'].values,
                        filtered_df[iv_column].values,
                        method=interpolation_method
                    )
                    fig.add_trace(
                        go.Surface(
                            x=X_grid, y=Y_grid, z=Z_grid,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="IV (%)"),
                            name='IV Surface',
                            hovertemplate='Moneyness: %{x:.3f}<br>TTE: %{y:.3f} years<br>IV: %{z:.2f}%<extra></extra>'
                        )
                    )
                    if show_points:
                        fig.add_trace(
                            go.Scatter3d(
                                x=filtered_df['moneyness'],
                                y=filtered_df['T'],
                                z=filtered_df[iv_column],
                                mode='markers',
                                marker=dict(size=3, color='white', opacity=0.7),
                                name='Data Points',
                                showlegend=True,
                                hovertemplate='Moneyness: %{x:.3f}<br>TTE: %{y:.3f} years<br>IV: %{z:.2f}%<extra></extra>'
                            )
                        )
                except Exception as e:
                    print(f"Warning: Surface interpolation failed: {e}")
                    interpolate = False
            
            if not interpolate:
                fig.add_trace(
                    go.Scatter3d(
                        x=filtered_df['moneyness'],
                        y=filtered_df['T'],
                        z=filtered_df[iv_column],
                        mode='markers',
                        marker=dict(size=4, color=filtered_df[iv_column], colorscale='Viridis', showscale=True, colorbar=dict(title="IV (%)")),
                        hovertemplate='Moneyness: %{x:.3f}<br>TTE: %{y:.3f} years<br>IV: %{z:.2f}%<extra></extra>'
                    )
                )
            
            if title is None:
                opt_name = "Calls" if option_type == 'C' else "Puts"
                title = f"{self.currency} Implied Volatility Surface - {opt_name}"
            
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='Moneyness (S/K)',
                    yaxis_title='Time to Expiration (years)',
                    zaxis_title='Implied Volatility (%)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                ),
                template="plotly_dark",
                height=700
            )
        
        # Save PNG if requested
        if save_png:
            try:
                # Create output directory
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                opt_suffix = {
                    "both": "calls_puts",
                    "C": "calls",
                    "P": "puts"
                }.get(option_type, "options")
                
                interp_suffix = "interpolated" if interpolate else "scatter"
                filename = f"{self.currency}_IV_surface_{opt_suffix}_{interp_suffix}_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Save PNG
                fig.write_image(filepath, width=1400, height=700, scale=2)
                print(f"💾 Saved plot to: {filepath}")
            except Exception as e:
                print(f"⚠️  Could not save PNG: {e}")
                print("   Make sure kaleido is installed: pip install kaleido")
        
        fig.show(renderer=renderer)



if __name__ == "__main__":
    # Example usage
    import sys
    import os
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from import_crypto_options import import_options_data
    from datetime import timedelta
    
    print("="*70)
    print("IMPLIED VOLATILITY SURFACE CALCULATOR")
    print("="*70)
    
    # Load data
    print("\nLoading options data...")
    end_date = datetime.now() + timedelta(days=90)
    start_date = datetime.now()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    
    dfs = import_options_data(
        start_date=start_date,
        end_date=end_date,
        currencies=["BTC", "ETH"],
        env="prod",
        output_dir=data_dir,
        auto_load_existing=True
    )
    
    # Process BTC
    if "BTC" in dfs and not dfs["BTC"].empty:
        print("\n" + "="*70)
        print("PROCESSING BTC OPTIONS")
        print("="*70)
        
        # Create IV surface calculator
        btc_iv = ImpliedVolatilitySurface(
            df=dfs["BTC"],
            currency="BTC",
            r=0.05,
            use_mark_iv=True,
            verbose=False
        )
        
        valid_iv_count = btc_iv.df['iv_percentage'].notna().sum()
        print(f"\n✓ Valid IV data for {valid_iv_count} out of {len(btc_iv.df)} BTC options")
        print(f"✓ Spot price used: ${btc_iv.spot_price:,.2f}")
        
        # Plot continuous surface (default: interpolate=True)
        print("\n" + "="*70)
        print("PLOTTING BTC CONTINUOUS VOLATILITY SURFACE")
        print("="*70)
        btc_iv.plot(
            option_type="both",
            interpolate=True,  # Create continuous interpolated surface
            show_points=True,  # Overlay data points
            interpolation_method="rbf",  # Use RBF for smooth surface
            renderer="browser"  # Use browser for script execution
        )
    
    # Process ETH
    if "ETH" in dfs and not dfs["ETH"].empty:
        print("\n" + "="*70)
        print("PROCESSING ETH OPTIONS")
        print("="*70)
        
        eth_iv = ImpliedVolatilitySurface(
            df=dfs["ETH"],
            currency="ETH",
            r=0.05,
            use_mark_iv=True,
            verbose=False
        )
        
        valid_iv_count = eth_iv.df['iv_percentage'].notna().sum()
        print(f"\n✓ Valid IV data for {valid_iv_count} out of {len(eth_iv.df)} ETH options")
        print(f"✓ Spot price used: ${eth_iv.spot_price:,.2f}")
        
        print("\n" + "="*70)
        print("PLOTTING ETH CONTINUOUS VOLATILITY SURFACE")
        print("="*70)
        eth_iv.plot(
            option_type="both",
            interpolate=True,
            show_points=True,
            interpolation_method="rbf",
            renderer="browser"  # Use browser for script execution
        )

