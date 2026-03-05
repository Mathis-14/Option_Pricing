"""
import_basket_data.py — Import market data for a two-stock basket call

Fetches from Refinitiv:
    1. Current spot prices for both stocks
    2. Historical closing prices (to compute realized correlation & volatilities)
    3. ATM implied volatilities from each stock's option chain

Data is saved as a JSON file in cross_gamma/refinitiv_data/ for downstream use.

Usage:
    python import_basket_data.py
    python import_basket_data.py --lookback 252   # 1 year of history
    python import_basket_data.py --tickers NVDA.O,GOOG.O
"""

import refinitiv.data as rd
import pandas as pd
import numpy as np
import os
import json
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default tickers (Refinitiv RICs)
DEFAULT_TICKERS = ["NVDA.O", "GOOG.O"]

# Default lookback for historical data (trading days)
DEFAULT_LOOKBACK_DAYS = 180


# =============================================================================
# REFINITIV SESSION MANAGEMENT
# =============================================================================

def init_session() -> str:
    """
    Load API key from .env and open a Refinitiv session.
    
    Returns
    -------
    str
        The API key used for the session.
        
    Raises
    ------
    ValueError
        If REFINITIV_API_KEY is not found in .env.
    """
    # Load .env from project root (two levels up from this script)
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(env_path)
    
    api_key = os.getenv('REFINITIV_API_KEY')
    if not api_key:
        raise ValueError(
            "REFINITIV_API_KEY not found in .env file. "
            "Please create a .env file in the project root with: "
            "REFINITIV_API_KEY=your_key_here"
        )
    
    rd.open_session(app_key=api_key)
    print("✅ Refinitiv session opened successfully.")
    return api_key


def close_session():
    """Close the Refinitiv session."""
    rd.close_session()
    print("🔒 Refinitiv session closed.")


# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def fetch_spot_prices(tickers: list) -> dict:
    """
    Fetch current spot prices for the given tickers.
    
    Parameters
    ----------
    tickers : list of str
        Refinitiv RICs (e.g., ['NVDA.O', 'GOOG.O']).
        
    Returns
    -------
    dict
        Mapping of ticker -> spot price.
    """
    print(f"\n📊 Fetching spot prices for {tickers}...")
    
    df = rd.get_data(
        universe=tickers,
        fields=['CF_LAST', 'DSPLY_NAME']
    )
    
    spots = {}
    for _, row in df.iterrows():
        ric = row['Instrument']
        price = row['CF_LAST']
        name = row.get('DSPLY_NAME', ric)
        spots[ric] = float(price)
        print(f"   {name} ({ric}): ${price:.2f}")
    
    return spots


def fetch_historical_prices(tickers: list, lookback_days: int) -> pd.DataFrame:
    """
    Fetch historical closing prices to compute correlation and volatility.
    
    Parameters
    ----------
    tickers : list of str
        Refinitiv RICs.
    lookback_days : int
        Number of calendar days to look back.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with Date index and columns for each ticker's close price.
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    
    print(f"\n📈 Fetching historical prices ({start_date} → {end_date})...")
    
    # rd.get_history returns DataFrame with tickers as columns
    hist = rd.get_history(
        universe=tickers,
        fields=['TR.PriceClose'],
        start=start_date,
        end=end_date
    )
    
    # Ensure we have a clean DataFrame with ticker columns
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    
    print(f"   Retrieved {len(hist)} data points per ticker.")
    return hist


def compute_correlation_and_vols(hist_prices: pd.DataFrame, tickers: list) -> dict:
    """
    Compute realized correlation and annualized volatilities from historical prices.
    
    Uses log returns for accurate estimation of GBM parameters.
    
    Parameters
    ----------
    hist_prices : pd.DataFrame
        Historical closing prices.
    tickers : list of str
        Ticker columns to use.
        
    Returns
    -------
    dict
        Contains: 'correlation', 'vol_1', 'vol_2', 'log_returns' DataFrame.
    """
    print("\n🔬 Computing correlation and volatilities...")
    
    # Compute log returns: ln(S_t / S_{t-1})
    log_returns = np.log(hist_prices / hist_prices.shift(1)).dropna()
    
    # Annualized volatilities (√252 scaling for daily → annual)
    vols = log_returns.std() * np.sqrt(252)
    
    # Correlation matrix
    corr_matrix = log_returns.corr()
    rho = corr_matrix.iloc[0, 1]  # Off-diagonal element
    
    print(f"   {tickers[0]} annualized vol: {vols.iloc[0]*100:.1f}%")
    print(f"   {tickers[1]} annualized vol: {vols.iloc[1]*100:.1f}%")
    print(f"   Realized correlation (ρ):    {rho:.4f}")
    
    return {
        'correlation': float(rho),
        'vol_1': float(vols.iloc[0]),
        'vol_2': float(vols.iloc[1]),
        'log_returns': log_returns
    }


def fetch_atm_implied_vol(ticker: str, spot: float) -> float:
    """
    Fetch ATM implied volatility for a given stock from its option chain.
    
    Searches for call options near the current spot price with nearest expiry
    and extracts the implied volatility.
    
    Parameters
    ----------
    ticker : str
        Refinitiv RIC for the stock (e.g., 'NVDA.O').
    spot : float
        Current spot price of the stock.
        
    Returns
    -------
    float
        ATM implied volatility (as decimal, e.g., 0.45 for 45%).
    """
    # Extract root ticker for option search (NVDA.O -> NVDA)
    root = ticker.split('.')[0]
    
    print(f"\n🎯 Fetching ATM implied vol for {ticker} (spot=${spot:.2f})...")
    
    try:
        # Search for options on this underlying
        search_response = rd.discovery.search(
            view=rd.discovery.Views.SEARCH_ALL,
            filter=f"RIC eq '{root}*.U'",
            select="RIC, DSPLY_NAME",
            top=500
        )
        
        if len(search_response) == 0:
            print(f"   ⚠️  No options found for {root}. Using historical vol.")
            return None
        
        option_rics = search_response['RIC'].tolist()
        print(f"   Found {len(option_rics)} options for {root}.")
        
        # Get IV and strike for these options
        opt_data = rd.get_data(
            universe=option_rics[:200],  # Limit to avoid timeout
            fields=[
                'PUTCALLIND',
                'TR.ImpliedVolatility',
                'TR.StrikePrice',
                'TR.ExpiryDate'
            ]
        )
        
        # Filter for calls only and valid data
        calls = opt_data[
            (opt_data['PUTCALLIND'].isin(['CALL', 'Call', 'C'])) &
            (opt_data['Implied Volatility'].notna()) &
            (opt_data['Strike Price'].notna())
        ].copy()
        
        if calls.empty:
            print(f"   ⚠️  No valid call options with IV found. Using historical vol.")
            return None
        
        # Find the nearest-to-ATM option (minimize |Strike - Spot|)
        calls['ATM_distance'] = abs(calls['Strike Price'].astype(float) - spot)
        atm_option = calls.loc[calls['ATM_distance'].idxmin()]
        
        iv = float(atm_option['Implied Volatility']) / 100.0  # Convert from % to decimal
        strike = float(atm_option['Strike Price'])
        
        print(f"   ATM option: K=${strike:.2f}, IV={iv*100:.1f}%")
        return iv
        
    except Exception as e:
        print(f"   ⚠️  Error fetching options for {ticker}: {e}")
        return None


# =============================================================================
# DATA EXPORT
# =============================================================================

def save_basket_data(
    spots: dict,
    stats: dict,
    atm_ivs: dict,
    tickers: list,
    lookback_days: int
) -> str:
    """
    Save all imported data to a JSON file in refinitiv_data/.
    
    Parameters
    ----------
    spots : dict
        Current spot prices.
    stats : dict
        Correlation and vol statistics.
    atm_ivs : dict
        ATM implied volatilities.
    tickers : list
        Ticker RICs.
    lookback_days : int
        Lookback period used.
        
    Returns
    -------
    str
        Path to the saved JSON file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), 'refinitiv_data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Use realized vol as fallback if ATM IV not available
    data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'tickers': tickers,
            'lookback_days': lookback_days,
        },
        'spot_prices': spots,
        'volatilities': {
            'realized_vol_1': stats['vol_1'],
            'realized_vol_2': stats['vol_2'],
            'atm_iv_1': atm_ivs.get(tickers[0], stats['vol_1']),
            'atm_iv_2': atm_ivs.get(tickers[1], stats['vol_2']),
        },
        'correlation': stats['correlation'],
        'basket': {
            'weights': [0.5, 0.5],
            'basket_value': 0.5 * spots[tickers[0]] + 0.5 * spots[tickers[1]],
            'strike_atm': 0.5 * spots[tickers[0]] + 0.5 * spots[tickers[1]],
        }
    }
    
    file_path = os.path.join(output_dir, f'basket_data_{timestamp}.json')
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n💾 Data saved to: {file_path}")
    return file_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point: import basket data from Refinitiv."""
    
    parser = argparse.ArgumentParser(
        description="Import market data for a two-stock basket call option."
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"Historical lookback in calendar days (default: {DEFAULT_LOOKBACK_DAYS})"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help=f"Comma-separated Refinitiv RICs (default: {','.join(DEFAULT_TICKERS)})"
    )
    args = parser.parse_args()
    
    tickers = [t.strip() for t in args.tickers.split(",")]
    if len(tickers) != 2:
        raise ValueError(f"Exactly 2 tickers required, got {len(tickers)}: {tickers}")
    
    print("=" * 60)
    print("  BASKET CALL DATA IMPORT")
    print(f"  Tickers: {tickers[0]} + {tickers[1]}")
    print(f"  Lookback: {args.lookback} days")
    print("=" * 60)
    
    try:
        # 1. Open Refinitiv session
        init_session()
        
        # 2. Fetch current spot prices
        spots = fetch_spot_prices(tickers)
        
        # 3. Fetch historical prices & compute stats
        hist_prices = fetch_historical_prices(tickers, args.lookback)
        stats = compute_correlation_and_vols(hist_prices, tickers)
        
        # 4. Fetch ATM implied volatilities
        atm_ivs = {}
        for ticker in tickers:
            iv = fetch_atm_implied_vol(ticker, spots[ticker])
            if iv is not None:
                atm_ivs[ticker] = iv
            else:
                # Fallback to realized vol
                idx = tickers.index(ticker)
                fallback = stats[f'vol_{idx + 1}']
                atm_ivs[ticker] = fallback
                print(f"   → Using realized vol ({fallback*100:.1f}%) as fallback for {ticker}")
        
        # 5. Save everything
        file_path = save_basket_data(spots, stats, atm_ivs, tickers, args.lookback)
        
        # 6. Print summary
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        basket_val = 0.5 * spots[tickers[0]] + 0.5 * spots[tickers[1]]
        print(f"  {tickers[0]} spot:  ${spots[tickers[0]]:.2f}")
        print(f"  {tickers[1]} spot:  ${spots[tickers[1]]:.2f}")
        print(f"  Basket (50/50):    ${basket_val:.2f}")
        print(f"  σ₁ (ATM IV):       {atm_ivs[tickers[0]]*100:.1f}%")
        print(f"  σ₂ (ATM IV):       {atm_ivs[tickers[1]]*100:.1f}%")
        print(f"  ρ (realized):      {stats['correlation']:.4f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    
    finally:
        close_session()


if __name__ == "__main__":
    main()
