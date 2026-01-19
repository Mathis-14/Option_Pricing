"""
import_stocks.py — Import historical stock price data from Yahoo Finance

This module fetches historical stock prices for major US stocks (NVIDIA, GOOGLE, APPLE)
using Yahoo Finance API via yfinance. It follows the same robust architecture as
the options import modules for consistency across the project.

Features:
- Automatic caching: If a CSV already exists for the ticker, it loads from disk.
- Flexible date range filtering for historical data.
- Consistent output directory resolution (works from any script location).
- Detailed logging with emojis for clear feedback.

Usage:
    from data.import_stocks import import_stock_data, import_major_stocks

    # Import a single stock
    df_nvda = import_stock_data(
        ticker="NVDA",
        start_date="2023-01-01",
        end_date="2024-12-31",
    )

    # Import all major US stocks (NVDA, GOOGL, AAPL)
    stocks_dict = import_major_stocks(
        start_date="2023-01-01",
        end_date="2024-12-31",
    )
    # stocks_dict = {"NVDA": df_nvda, "GOOGL": df_googl, "AAPL": df_aapl}
"""

from __future__ import annotations

import os
from datetime import datetime, date
from typing import Union, Optional, Dict, List

import pandas as pd
import yfinance as yf


# =============================================================================
# CONFIGURATION: Major US stocks to track
# =============================================================================

# Default list of major US stocks to import
MAJOR_STOCKS = {
    "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc. (Google)",
    "AAPL": "Apple Inc.",
}


# =============================================================================
# HELPER FUNCTIONS: Directory resolution, date handling, file management
# =============================================================================

def _resolve_output_dir(output_dir: str) -> str:
    """
    Resolve the output directory path robustly.
    
    This function ensures that the output directory is always relative to the
    Option_Pricing project root, regardless of where the script is called from.
    This allows the module to work correctly from:
    - Direct script execution
    - Jupyter notebooks in any subdirectory
    - Other scripts importing this module
    
    Parameters
    ----------
    output_dir : str
        Relative or absolute path to the output directory.
        
    Returns
    -------
    str
        Absolute path to the output directory.
    """
    # If already an absolute path, use it directly
    if os.path.isabs(output_dir):
        final_dir = output_dir
    else:
        # Step 1: Try to get the script's directory from __file__
        script_dir = None
        try:
            file_path = os.path.abspath(__file__)
            script_dir = os.path.dirname(file_path)
            # This file is in data/, so go up one level to get project root
            script_dir = os.path.dirname(script_dir)
            # Verify we're in Option_Pricing by checking for a known file
            if not os.path.exists(os.path.join(script_dir, "import_options.py")):
                script_dir = None
        except NameError:
            # __file__ not available (e.g., in Jupyter notebook)
            pass

        # Step 2: If __file__ didn't work, search from current working directory
        if script_dir is None:
            script_dir = os.getcwd()
            # Check if we're already in the project root
            if not os.path.exists(os.path.join(script_dir, "import_options.py")):
                # Try to find the project root by traversing up
                current = script_dir
                for _ in range(10):  # Maximum 10 levels up
                    test_path = os.path.join(current, "import_options.py")
                    if os.path.exists(test_path):
                        script_dir = current
                        break
                    parent = os.path.dirname(current)
                    if parent == current:  # Reached filesystem root
                        break
                    current = parent

        # Step 3: Ensure we're at the Option_Pricing directory level
        if script_dir and "Option_Pricing" in script_dir:
            parts = script_dir.split(os.sep)
            if "Option_Pricing" in parts:
                option_pricing_idx = None
                for i, part in enumerate(parts):
                    if part == "Option_Pricing":
                        option_pricing_idx = i
                        break
                if option_pricing_idx is not None:
                    # Reconstruct path up to and including Option_Pricing
                    script_dir = os.sep.join(parts[:option_pricing_idx + 1])

        # Build the final output directory path
        if script_dir:
            final_dir = os.path.join(script_dir, output_dir)
        else:
            # Fallback: use current directory
            final_dir = os.path.join(os.getcwd(), output_dir)
            print(f"⚠️  Warning: Could not find project root, using current directory: {os.getcwd()}")

    # Normalize the path (resolve any .. or . components)
    final_dir = os.path.normpath(final_dir)

    # Verify the final path contains "Option_Pricing"
    if "Option_Pricing" not in final_dir:
        print(f"⚠️  Warning: Data directory path doesn't contain 'Option_Pricing': {final_dir}")
        print(f"   Expected path should contain: .../Option_Pricing/data/")
        
        # Attempt to fix by searching from current directory
        cwd = os.getcwd()
        if "Option_Pricing" in cwd:
            parts = cwd.split(os.sep)
            if "Option_Pricing" in parts:
                option_pricing_idx = None
                for i, part in enumerate(parts):
                    if part == "Option_Pricing":
                        option_pricing_idx = i
                        break
                if option_pricing_idx is not None:
                    project_root = os.sep.join(parts[:option_pricing_idx + 1])
                    final_dir = os.path.join(project_root, output_dir)
                    final_dir = os.path.normpath(final_dir)
                    print(f"   ✅ Fixed path to: {final_dir}")

    print(f"📁 Stock data will be saved to / loaded from: {final_dir}")
    return final_dir


def _ensure_date(d: Union[str, datetime, date]) -> date:
    """
    Convert string/datetime/date to a date object.
    
    Parameters
    ----------
    d : str, datetime, or date
        Date to convert. Strings must be in "YYYY-MM-DD" format.
        
    Returns
    -------
    date
        Python date object.
        
    Raises
    ------
    TypeError
        If the input type is not supported.
    """
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        # Expected format: YYYY-MM-DD
        return datetime.strptime(d, "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date type: {type(d)}")


def _find_existing_csv(
    output_dir: str,
    ticker: str,
    filename: Optional[str] = None,
) -> Optional[str]:
    """
    Search for an existing CSV file for the given ticker.
    
    This function implements smart caching by detecting previously downloaded
    data to avoid redundant API calls.
    
    Parameters
    ----------
    output_dir : str
        Directory to search for CSV files.
    ticker : str
        Stock ticker symbol (e.g., "NVDA").
    filename : str, optional
        Specific filename to look for (without .csv extension).
        If None, searches for auto-generated filenames.
        
    Returns
    -------
    str or None
        Full path to the existing CSV file, or None if not found.
    """
    if not os.path.exists(output_dir):
        return None

    # Case 1: User provided a specific filename
    if filename is not None:
        path = os.path.join(output_dir, f"{filename}.csv")
        if os.path.exists(path):
            return path
        return None

    # Case 2: Search for auto-generated filenames
    # Pattern: stock_data_{TICKER}_YYYYMMDD_HHMMSS.csv
    ticker_clean = ticker.upper()
    prefix = f"stock_data_{ticker_clean}_"

    candidates = [
        f for f in os.listdir(output_dir)
        if f.startswith(prefix) and f.endswith(".csv")
    ]
    if not candidates:
        return None

    # Return the most recent file based on modification time (more reliable than filename)
    candidates_with_mtime = [
        (f, os.path.getmtime(os.path.join(output_dir, f)))
        for f in candidates
    ]
    latest = max(candidates_with_mtime, key=lambda x: x[1])[0]
    return os.path.join(output_dir, latest)


# =============================================================================
# MAIN FUNCTIONS: Stock data import
# =============================================================================

def import_stock_data(
    ticker: str,
    start_date: Union[str, datetime, date],
    end_date: Union[str, datetime, date],
    output_dir: str = "data",
    filename: Optional[str] = None,
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Import historical stock price data from Yahoo Finance.
    
    This function downloads historical OHLCV (Open, High, Low, Close, Volume) data
    for a given stock ticker. It implements intelligent caching: if data has been
    previously downloaded, it loads from the local CSV instead of making API calls.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "NVDA", "GOOGL", "AAPL").
    start_date : str, datetime, or date
        Start date for historical data (inclusive). Format: "YYYY-MM-DD".
    end_date : str, datetime, or date
        End date for historical data (inclusive). Format: "YYYY-MM-DD".
    output_dir : str, default="data"
        Directory where CSV files are saved. Relative paths are resolved
        relative to the Option_Pricing project root.
    filename : str, optional
        Custom filename for the CSV (without extension).
        If None, an auto-generated name is used: stock_data_{TICKER}_YYYYMMDD_HHMMSS.csv
    interval : str, default="1d"
        Data frequency. Common values:
        - "1m", "2m", "5m", "15m", "30m", "60m" (minute-level, limited history)
        - "1d" (daily, most common)
        - "1wk" (weekly)
        - "1mo" (monthly)
    force_refresh : bool, default=False
        If True, always download fresh data even if a cached CSV exists.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - date: Trading date (datetime)
        - date_str: Date as string "YYYY-MM-DD"
        - ticker: Stock symbol
        - open: Opening price
        - high: Day's high price
        - low: Day's low price
        - close: Closing price (adjusted)
        - volume: Trading volume
        - adj_close: Adjusted closing price (same as close for yfinance)
        
    Examples
    --------
    >>> df = import_stock_data("NVDA", "2023-01-01", "2023-12-31")
    >>> print(df.head())
    
    Notes
    -----
    - The function uses yfinance which gets data from Yahoo Finance.
    - For minute-level data, Yahoo Finance limits history to ~30 days.
    - All prices are adjusted for splits and dividends.
    """
    # Normalize dates
    start_date = _ensure_date(start_date)
    end_date = _ensure_date(end_date)
    
    # Validate date range
    if start_date > end_date:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")
    
    # Normalize ticker to uppercase
    ticker = ticker.upper()
    
    # Resolve output directory
    output_dir = _resolve_output_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for existing cached data (unless force_refresh is True)
    if not force_refresh:
        existing_csv = _find_existing_csv(output_dir, ticker, filename)
        if existing_csv is not None:
            print(f"📂 Loading cached stock data for {ticker} from:\n   {existing_csv}")
            df = pd.read_csv(existing_csv)
            
            # Ensure date column is datetime
            if "date" in df.columns:
                # Handle timezone-aware date strings (e.g., "2024-01-09 00:00:00-05:00")
                try:
                    df["date"] = pd.to_datetime(df["date"], utc=True)
                except Exception:
                    # Fallback for simple date strings
                    df["date"] = pd.to_datetime(df["date"])
            
            # Filter by requested date range
            total_rows = len(df)
            # Convert to date for comparison (remove timezone info)
            df_date = df["date"].dt.tz_localize(None).dt.date if df["date"].dt.tz is not None else df["date"].dt.date
            df = df[(df_date >= start_date) & (df_date <= end_date)].copy()
            
            print(f"✅ Loaded {len(df)} rows (filtered from {total_rows} total)")
            if not df.empty:
                print(f"   Date range in data: {df['date'].min()} to {df['date'].max()}")
            return df.reset_index(drop=True)
    
    # Download fresh data from Yahoo Finance
    print(f"🔎 Fetching stock data from Yahoo Finance...")
    print(f"   Ticker: {ticker}")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Interval: {interval}")
    
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Get company info for logging
        try:
            info = stock.info
            company_name = info.get("longName", info.get("shortName", ticker))
            print(f"   Company: {company_name}")
        except Exception:
            company_name = ticker
        
        # Download historical data
        # Note: yfinance end_date is exclusive, so we add 1 day
        end_date_for_yf = end_date
        df_raw = stock.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date_for_yf + pd.Timedelta(days=1)).strftime("%Y-%m-%d") 
                if hasattr(end_date_for_yf, 'strftime') 
                else (datetime.strptime(str(end_date_for_yf), "%Y-%m-%d") + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval=interval,
        )
        
        if df_raw.empty:
            print(f"⚠️  No data found for {ticker} in the specified date range.")
            return pd.DataFrame()
        
    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        raise
    
    # Process and format the DataFrame
    df = df_raw.reset_index()
    
    # Rename columns to lowercase and more descriptive names
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Dividends": "dividends",
        "Stock Splits": "stock_splits",
    })
    
    # Add derived columns
    df["ticker"] = ticker
    df["company_name"] = company_name if 'company_name' in dir() else ticker
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    
    # For compatibility, add adj_close (yfinance already adjusts Close)
    df["adj_close"] = df["close"]
    
    # Reorder columns for clarity
    cols_order = [
        "date",
        "date_str",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
        "company_name",
    ]
    # Keep only columns that exist
    cols_order = [c for c in cols_order if c in df.columns]
    df = df[cols_order].copy()
    
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    
    # Save to CSV
    if filename is None:
        # Auto-generate filename: stock_data_{TICKER}_YYYYMMDD_HHMMSS.csv
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_to_save = f"stock_data_{ticker}_{timestamp}"
    else:
        filename_to_save = filename
    
    csv_path = os.path.join(output_dir, f"{filename_to_save}.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {len(df)} rows to {csv_path}")
    
    # Print summary statistics
    print(f"\n📊 Summary for {ticker}:")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Trading days: {len(df)}")
    print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"   Latest close: ${df['close'].iloc[-1]:.2f}")
    
    return df


def import_major_stocks(
    start_date: Union[str, datetime, date],
    end_date: Union[str, datetime, date],
    tickers: Optional[List[str]] = None,
    output_dir: str = "data",
    interval: str = "1d",
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Import historical data for multiple major US stocks.
    
    This is a convenience function that imports data for several stocks at once.
    By default, it imports NVIDIA (NVDA), Google (GOOGL), and Apple (AAPL).
    
    Parameters
    ----------
    start_date : str, datetime, or date
        Start date for historical data. Format: "YYYY-MM-DD".
    end_date : str, datetime, or date
        End date for historical data. Format: "YYYY-MM-DD".
    tickers : list of str, optional
        List of ticker symbols to import. If None, uses the default:
        ["NVDA", "GOOGL", "AAPL"]
    output_dir : str, default="data"
        Directory where CSV files are saved.
    interval : str, default="1d"
        Data frequency ("1d", "1wk", "1mo", etc.)
    force_refresh : bool, default=False
        If True, always download fresh data even if cached.
        
    Returns
    -------
    dict
        Dictionary mapping ticker symbols to their DataFrames.
        Example: {"NVDA": df_nvda, "GOOGL": df_googl, "AAPL": df_aapl}
        
    Examples
    --------
    >>> stocks = import_major_stocks("2023-01-01", "2023-12-31")
    >>> for ticker, df in stocks.items():
    ...     print(f"{ticker}: {len(df)} trading days")
    """
    # Use default tickers if none provided
    if tickers is None:
        tickers = list(MAJOR_STOCKS.keys())
    
    print("=" * 60)
    print("🏢 IMPORTING MAJOR US STOCKS")
    print("=" * 60)
    print(f"📅 Date range: {start_date} to {end_date}")
    print(f"📈 Stocks to import: {', '.join(tickers)}")
    print()
    
    results = {}
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(tickers)}] Processing {ticker}...")
        print("─" * 60)
        
        try:
            df = import_stock_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir,
                interval=interval,
                force_refresh=force_refresh,
            )
            results[ticker] = df
            print(f"✅ Successfully imported {ticker}")
        except Exception as e:
            print(f"❌ Failed to import {ticker}: {e}")
            results[ticker] = pd.DataFrame()  # Empty DataFrame for failed imports
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 IMPORT SUMMARY")
    print("=" * 60)
    for ticker in tickers:
        df = results.get(ticker, pd.DataFrame())
        if not df.empty:
            print(f"  ✅ {ticker}: {len(df)} rows "
                  f"({df['date'].min().date()} to {df['date'].max().date()})")
        else:
            print(f"  ❌ {ticker}: No data")
    print()
    
    return results


def get_combined_stocks_df(
    start_date: Union[str, datetime, date],
    end_date: Union[str, datetime, date],
    tickers: Optional[List[str]] = None,
    output_dir: str = "data",
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Import and combine multiple stocks into a single DataFrame.
    
    This function imports data for multiple stocks and concatenates them
    into a single DataFrame, useful for comparative analysis.
    
    Parameters
    ----------
    start_date : str, datetime, or date
        Start date for historical data.
    end_date : str, datetime, or date
        End date for historical data.
    tickers : list of str, optional
        List of ticker symbols to import.
    output_dir : str, default="data"
        Directory where CSV files are saved.
    interval : str, default="1d"
        Data frequency.
    force_refresh : bool, default=False
        If True, always download fresh data.
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all stocks, identifiable by the 'ticker' column.
    """
    stocks_dict = import_major_stocks(
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        output_dir=output_dir,
        interval=interval,
        force_refresh=force_refresh,
    )
    
    # Filter out empty DataFrames and concatenate
    non_empty_dfs = [df for df in stocks_dict.values() if not df.empty]
    
    if not non_empty_dfs:
        print("⚠️  No data available for any of the requested stocks.")
        return pd.DataFrame()
    
    combined_df = pd.concat(non_empty_dfs, ignore_index=True)
    combined_df = combined_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    print(f"\n📊 Combined DataFrame: {len(combined_df)} total rows "
          f"for {len(non_empty_dfs)} stocks")
    
    return combined_df


# =============================================================================
# MAIN EXECUTION: Demo and testing
# =============================================================================

if __name__ == "__main__":
    from datetime import timedelta
    
    # Demo: Import the last 2 years of data for major US stocks
    print("\n" + "=" * 60)
    print("🚀 STOCK DATA IMPORT DEMO")
    print("=" * 60 + "\n")
    
    # Calculate date range: last 2 years
    end = datetime.now().date()
    start = end - timedelta(days=730)  # ~2 years
    
    # Import all major stocks
    stocks = import_major_stocks(
        start_date=start,
        end_date=end,
    )
    
    # Display individual stock summaries
    print("\n" + "=" * 60)
    print("📊 DETAILED STOCK STATISTICS")
    print("=" * 60)
    
    for ticker, df in stocks.items():
        if df.empty:
            print(f"\n{ticker}: No data available")
            continue
            
        print(f"\n📈 {ticker} ({MAJOR_STOCKS.get(ticker, ticker)}):")
        print(f"   Trading days: {len(df)}")
        print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        print(f"   Start price: ${df['close'].iloc[0]:.2f}")
        print(f"   End price: ${df['close'].iloc[-1]:.2f}")
        
        # Calculate returns
        total_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
        print(f"   Total return: {total_return:+.2f}%")
        
        # Calculate volatility (annualized)
        daily_returns = df['close'].pct_change().dropna()
        volatility = daily_returns.std() * (252 ** 0.5) * 100
        print(f"   Annualized volatility: {volatility:.2f}%")
