"""
Get NVIDIA stock price at the time of options data import
"""
import refinitiv.data as rd
import os
from dotenv import load_dotenv
from datetime import datetime

# Load API key
load_dotenv()
api_key = os.getenv('REFINITIV_API_KEY')
rd.open_session(app_key=api_key)

try:
    print("Timestamp from CSV: 2026-02-04 21:04:56 (after market close)")
    print("Getting closing price for Feb 4, 2026...\n")
    
    # Get closing price from Feb 4, 2026
    # Options were captured at 9:04 PM, after market close
    # So we use the close price from that day
    
    close_data = rd.get_data(
        universe='NVDA.O',
        fields=['TR.PriceClose', 'TR.PriceClose.date'],
        parameters={'SDate': '2026-02-04', 'EDate': '2026-02-04'}
    )
    
    print("Feb 4, 2026 Close Price:")
    print(close_data)
    
    if not close_data.empty and 'Price Close' in close_data.columns:
        S = close_data['Price Close'].iloc[0]
        print(f"\n NVDA Stock Price (Feb 4, 2026 close): ${S:.2f}")
        print(f"   Use this value: S = {S:.2f}")
    else:
        # Fallback: use current price
        current = rd.get_data(universe='NVDA.O', fields=['CF_LAST'])
        S = current['CF_LAST'].iloc[0]
        print(f"\n  Could not get Feb 4 close, using current price: ${S:.2f}")
        print(f"   Use this value: S = {S:.2f}")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    rd.close_session()
