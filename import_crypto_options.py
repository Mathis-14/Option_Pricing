"""
import_derebit.py — Import BTC and ETH options data from Deribit API

This module fetches options data for BTC and ETH from Deribit, filters by
maturity range, and exports to CSV in the "data" folder.

Usage:
    from import_derebit import import_options_data
    
    # Import options with expiry between 2024-11-01 and 2024-12-31
    # Returns a dictionary with separate DataFrames for BTC and ETH
    dfs = import_options_data(
        start_date="2024-11-01",
        end_date="2024-12-31",
        env="prod"
    )
    btc_df = dfs["BTC"]
    eth_df = dfs["ETH"]
    
    # Or use datetime objects
    from datetime import datetime
    dfs = import_options_data(
        start_date=datetime(2024, 11, 1),
        end_date=datetime(2024, 12, 31)
    )
"""

from __future__ import annotations

import os
from datetime import datetime, date
from typing import Optional, Literal, Union

import requests
import pandas as pd


def parse_expiry_date(expiry_str: str) -> Optional[datetime]:
    """
    Parse expiry string like "29NOV24" into a datetime object.
    
    Parameters
    ----------
    expiry_str : str
        Expiry string in format like "29NOV24" (DDMMMYY)
    
    Returns
    -------
    datetime or None
        Parsed datetime object, or None if parsing fails
    """
    try:
        # Format: DDMMMYY (e.g., "29NOV24")
        day = int(expiry_str[:2])
        month_str = expiry_str[2:5].upper()
        year = int(expiry_str[5:7])
        
        # Month mapping
        month_map = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
            "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
            "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
        }
        
        month = month_map.get(month_str)
        if month is None:
            return None
        
        # Convert 2-digit year to 4-digit (assuming 20XX)
        full_year = 2000 + year
        
        return datetime(full_year, month, day)
    except (ValueError, KeyError, IndexError):
        return None


def fetch_options_data(
    currency: str,
    env: Literal["prod", "test"] = "prod"
) -> list[dict]:
    """
    Fetch raw options data from Deribit API.
    
    Parameters
    ----------
    currency : str
        Currency code: "BTC" or "ETH"
    env : Literal["prod", "test"]
        Environment: "prod" for live, "test" for testnet
    
    Returns
    -------
    list[dict]
        List of raw option book entries from Deribit API
    """
    base_url = (
        "https://www.deribit.com" if env == "prod"
        else "https://test.deribit.com"
    )
    
    url = f"{base_url}/api/v2/public/get_book_summary_by_currency"
    params = {"currency": currency, "kind": "option"}
    
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    payload = resp.json()
    
    return payload.get("result", [])


def parse_option_name(name: str) -> Optional[dict]:
    """
    Parse option instrument name like: BTC-29NOV24-90000-C
    
    Returns
    -------
    dict with keys: currency, expiry, strike, type
    or None if parsing fails
    """
    parts = name.split("-")
    if len(parts) != 4:
        return None
    
    currency, expiry_str, strike_str, opt_type = parts
    
    try:
        strike = float(strike_str)
    except ValueError:
        return None
    
    if opt_type not in ("C", "P"):
        return None
    
    expiry_date = parse_expiry_date(expiry_str)
    
    return {
        "currency": currency,
        "expiry_str": expiry_str,
        "expiry_date": expiry_date,
        "strike": strike,
        "type": opt_type,
    }


def check_existing_csvs(
    output_dir: str,
    currencies: list[str]
) -> dict[str, Optional[str]]:
    """
    Check if CSV files already exist for the given currencies.
    
    Returns
    -------
    dict[str, Optional[str]]
        Dictionary mapping currency to the most recent CSV file path, or None if not found
    """
    existing_files = {}
    
    if not os.path.exists(output_dir):
        return {currency: None for currency in currencies}
    
    for currency in currencies:
        csv_files = [
            f for f in os.listdir(output_dir)
            if f.endswith(f'_{currency}.csv') and f.startswith('options_data_')
        ]
        
        if csv_files:
            # Sort by filename (which includes timestamp) and get the most recent
            latest_file = sorted(csv_files)[-1]
            existing_files[currency] = os.path.join(output_dir, latest_file)
        else:
            existing_files[currency] = None
    
    return existing_files


def import_options_data(
    start_date: Union[str, datetime, date],
    end_date: Union[str, datetime, date],
    currencies: Optional[list[str]] = None,
    env: Literal["prod", "test"] = "prod",
    output_dir: str = "data",
    filename: Optional[str] = None,
    force_new_import: bool = False,
    auto_load_existing: bool = False
) -> dict[str, pd.DataFrame]:
    """
    Import BTC and ETH options data from Deribit for a given maturity range.
    Returns separate DataFrames for each currency.
    
    Parameters
    ----------
    start_date : str, datetime, or date
        Start date for maturity filter (inclusive). Can be string like "2024-11-01"
        or datetime/date object.
    end_date : str, datetime, or date
        End date for maturity filter (inclusive). Can be string like "2024-12-31"
        or datetime/date object.
    currencies : list[str], optional
        List of currencies to fetch. Defaults to ["BTC", "ETH"].
    env : Literal["prod", "test"]
        Environment: "prod" for live, "test" for testnet. Defaults to "prod".
    output_dir : str
        Directory to save CSV files. Defaults to "data".
    filename : str, optional
        CSV filename prefix. If None, auto-generated with timestamp.
        Files will be named: {filename}_BTC.csv and {filename}_ETH.csv
    force_new_import : bool
        If True, skip checking for existing files and always do a new import.
        If False (default), check for existing CSV files and ask user confirmation.
    auto_load_existing : bool
        If True, automatically load existing CSV files without asking for confirmation.
        Useful in notebooks where input() may not work well. Defaults to False.
    
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with currency keys ("BTC", "ETH") and corresponding DataFrames.
        Each DataFrame contains columns:
        - expiry_str: Expiry string (e.g., "29NOV24")
        - expiry_date: Parsed expiry date
        - strike: Strike price
        - type: Option type (C or P)
        - mark_iv: Implied volatility
        - mark_price: Mark price
        - underlying_price: Underlying asset price
        - open_interest: Open interest
        - bid_price: Bid price
        - ask_price: Ask price
        - best_bid_price: Best bid price
        - best_ask_price: Best ask price
        - volume: 24h volume
        - instrument_name: Full instrument name
    
    Examples
    --------
    >>> # Using string dates
    >>> dfs = import_options_data("2024-11-01", "2024-12-31")
    >>> btc_df = dfs["BTC"]
    >>> eth_df = dfs["ETH"]
    >>> 
    >>> # Using datetime objects
    >>> from datetime import datetime
    >>> dfs = import_options_data(
    ...     datetime(2024, 11, 1),
    ...     datetime(2024, 12, 31)
    ... )
    """
    # Parse dates
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    elif isinstance(start_date, datetime):
        start_date = start_date.date()
    
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    elif isinstance(end_date, datetime):
        end_date = end_date.date()
    
    if currencies is None:
        currencies = ["BTC", "ETH"]
    
    # Check for existing CSV files
    if not force_new_import:
        existing_files = check_existing_csvs(output_dir, currencies)
        existing_currencies = [c for c, f in existing_files.items() if f is not None]
        
        if existing_currencies:
            print(f"\n⚠️  Existing CSV files found in '{output_dir}':")
            for currency in existing_currencies:
                print(f"   - {os.path.basename(existing_files[currency])}")
            
            # Auto-load or ask user
            if auto_load_existing:
                print("\n📂 Auto-loading existing CSV files (auto_load_existing=True)...")
                choice = "1"
            else:
                # Ask user if they want to load existing data or do a new import
                print("\nOptions:")
                print("  1. Load existing CSV files (recommended if data is recent)")
                print("  2. Do a new import from Deribit API")
                
                try:
                    choice = input("\nEnter your choice (1 or 2, default=1): ").strip()
                except (KeyboardInterrupt, EOFError):
                    # In notebooks or non-interactive environments, default to loading existing
                    print("\n⚠️  Input not available, loading existing files...")
                    choice = "1"
            
            if choice == "" or choice == "1":
                    # Load existing CSV files
                    print("\n📂 Loading existing CSV files...")
                    dataframes = {}
                    for currency in currencies:
                        if existing_files.get(currency):
                            df = pd.read_csv(existing_files[currency])
                            # Convert expiry_date to datetime if it's a string
                            if 'expiry_date' in df.columns:
                                df['expiry_date'] = pd.to_datetime(df['expiry_date'])
                            dataframes[currency] = df
                            print(f"✅ Loaded {len(df)} {currency} options from {os.path.basename(existing_files[currency])}")
                        else:
                            print(f"⚠️  No existing CSV found for {currency}, fetching from API...")
                            # Fetch for this currency only
                            raw_data = fetch_options_data(currency, env)
                            currency_rows = []
                            
                            for row in raw_data:
                                instrument_name = row.get("instrument_name")
                                if not instrument_name:
                                    continue
                                
                                parsed = parse_option_name(instrument_name)
                                if not parsed:
                                    continue
                                
                                expiry_date = parsed["expiry_date"]
                                if expiry_date is None:
                                    continue
                                
                                expiry_date_only = expiry_date.date() if isinstance(expiry_date, datetime) else expiry_date
                                
                                if not (start_date <= expiry_date_only <= end_date):
                                    continue
                                
                                option_row = {
                                    "expiry_str": parsed["expiry_str"],
                                    "expiry_date": parsed["expiry_date"],
                                    "strike": parsed["strike"],
                                    "type": parsed["type"],
                                    "mark_iv": row.get("mark_iv"),
                                    "mark_price": row.get("mark_price"),
                                    "underlying_price": row.get("underlying_price"),
                                    "open_interest": row.get("open_interest"),
                                    "bid_price": row.get("bid_price"),
                                    "ask_price": row.get("ask_price"),
                                    "best_bid_price": row.get("best_bid_price"),
                                    "best_ask_price": row.get("best_ask_price"),
                                    "volume": row.get("volume"),
                                    "instrument_name": instrument_name,
                                }
                                currency_rows.append(option_row)
                            
                            if currency_rows:
                                df = pd.DataFrame(currency_rows)
                                df = df.sort_values(["expiry_date", "strike", "type"]).reset_index(drop=True)
                                dataframes[currency] = df
                                
                                # Export this new currency
                                os.makedirs(output_dir, exist_ok=True)
                                if filename is None:
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename_prefix = f"options_data_{timestamp}"
                                else:
                                    filename_prefix = filename
                                csv_path = os.path.join(output_dir, f"{filename_prefix}_{currency}.csv")
                                df.to_csv(csv_path, index=False)
                                print(f"✅ Exported {len(df)} {currency} options to {csv_path}")
                            else:
                                dataframes[currency] = pd.DataFrame()
                                print(f"  No {currency} options found in range")
                    
                    return dataframes
            else:
                print("\n🔄 Proceeding with new import from Deribit API...")
    
    # Fetch data for each currency separately
    dataframes = {}
    
    for currency in currencies:
        print(f"Fetching {currency} options data...")
        raw_data = fetch_options_data(currency, env)
        
        currency_rows = []
        
        for row in raw_data:
            instrument_name = row.get("instrument_name")
            if not instrument_name:
                continue
            
            # Parse option name
            parsed = parse_option_name(instrument_name)
            if not parsed:
                continue
            
            # Filter by maturity range
            expiry_date = parsed["expiry_date"]
            if expiry_date is None:
                continue
            
            expiry_date_only = expiry_date.date() if isinstance(expiry_date, datetime) else expiry_date
            
            if not (start_date <= expiry_date_only <= end_date):
                continue
            
            # Extract all available fields (remove currency from parsed as it's redundant)
            option_row = {
                "expiry_str": parsed["expiry_str"],
                "expiry_date": parsed["expiry_date"],
                "strike": parsed["strike"],
                "type": parsed["type"],
                "mark_iv": row.get("mark_iv"),
                "mark_price": row.get("mark_price"),
                "underlying_price": row.get("underlying_price"),
                "open_interest": row.get("open_interest"),
                "bid_price": row.get("bid_price"),
                "ask_price": row.get("ask_price"),
                "best_bid_price": row.get("best_bid_price"),
                "best_ask_price": row.get("best_ask_price"),
                "volume": row.get("volume"),
                "instrument_name": instrument_name,
            }
            
            currency_rows.append(option_row)
        
        # Create DataFrame for this currency
        if currency_rows:
            df = pd.DataFrame(currency_rows)
            # Sort by expiry, strike, type
            df = df.sort_values(["expiry_date", "strike", "type"]).reset_index(drop=True)
            dataframes[currency] = df
            print(f"  Found {len(df)} {currency} options in range")
        else:
            dataframes[currency] = pd.DataFrame()
            print(f"  No {currency} options found in range")
    
    # Export to CSV (separate files for each currency)
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"options_data_{timestamp}"
    else:
        filename_prefix = filename
    
    # Export each DataFrame to separate CSV
    for currency, df in dataframes.items():
        if not df.empty:
            csv_path = os.path.join(output_dir, f"{filename_prefix}_{currency}.csv")
            df.to_csv(csv_path, index=False)
            print(f"✅ Exported {len(df)} {currency} options to {csv_path}")
        else:
            print(f"⚠️  No {currency} options to export")
    
    return dataframes


if __name__ == "__main__":
    # Example usage
    from datetime import datetime, timedelta
    
    # Import options expiring in the next 3 months
    end_date = datetime.now() + timedelta(days=90)
    start_date = datetime.now()
    
    dfs = import_options_data(
        start_date=start_date,
        end_date=end_date,
        currencies=["BTC", "ETH"],
        env="prod"
    )
    
    print(f"\nSummary:")
    for currency, df in dfs.items():
        if not df.empty:
            print(f"\n{currency}:")
            print(f"  Total options: {len(df)}")
            print(f"  Expiry range: {df['expiry_date'].min()} to {df['expiry_date'].max()}")
            print(f"  Option types: {df['type'].value_counts().to_dict()}")
        else:
            print(f"\n{currency}: No options found")

