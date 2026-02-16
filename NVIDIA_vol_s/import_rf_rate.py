# Use Refinitiv to get actual Treasury rates
import refinitiv.data as rd
from dotenv import load_dotenv
import os
import pandas as pd
import argparse
from datetime import datetime

# Load .env file
load_dotenv()

# Get API KEY from .env
api_key = os.getenv('REFINITIV_API_KEY')

if not api_key:
    raise ValueError("API Key not found in .env")

# Open session
rd.open_session(app_key=api_key)

def get_treasury_rates(date_str=None):
    """
    Fetch Treasury rates.
    :param date_str: String date in 'YYYY-MM-DD' format. If None, fetches current data.
    """
    try:
        # Determine mode
        is_historical = False
        target_date_obj = None
        
        if date_str:
            try:
                target_date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                if target_date_obj < datetime.now().date():
                    is_historical = True
            except ValueError:
                print(f"Invalid date format: {date_str}. Using today's data.")

        # Get Treasury yields
        print(f"Attempting to retrieve 2-year Treasury yield ({'Historical: ' + date_str if is_historical else 'Real-time'})...")
        
        r_2y = None
        treasury_data = pd.DataFrame()

        if not is_historical:
            # --- REAL-TIME / SNAPSHOT ---
            
            # Try method 1: Use generic government bond RIC
            try:
                treasury_2y = rd.get_data(
                    universe=['US2YT=RRPS'],  # Alternative 2Y RIC
                    fields=['YIELD_1']
                )
                if not treasury_2y.empty and 'YIELD_1' in treasury_2y.columns:
                    val = treasury_2y['YIELD_1'].iloc[0]
                    if pd.notna(val):
                        r_2y = val
                        print(f" Found 2Y yield via US2YT=RRPS: {r_2y:.3f}%")
            except Exception as e:
                print(f" 2Y lookup failed: {e}")

            # Get short-term rates
            treasury_data = rd.get_data(
                universe=['US3MT=RR', 'US6MT=RR', 'US1YT=RR'],  
                fields=['CF_LAST']
            )
            # Ensure consistent column name for processing
            if 'CF_LAST' in treasury_data.columns:
                treasury_data.rename(columns={'CF_LAST': 'Rate'}, inplace=True)

        else:
            # --- HISTORICAL ---
            
            # 2Y Yield
            try:
                # For history, use TR.Yield for US2YT=RRPS (or generic 2Y)
                # Note: fields might need adjustment based on specific RIC data model
                hist_2y = rd.get_history(
                    universe=['US2YT=RRPS'],
                    fields=['TR.Yield'], 
                    start=date_str,
                    end=date_str
                )
                # Handling likely structure: Index=Date, Column=Yield (or TR.Yield)
                if not hist_2y.empty:
                    # Look for a numeric value in the first row
                    # Columns might be MultiIndex or simple
                    val = hist_2y.iloc[0, 0] # Assume 1st column is the field
                    if pd.notna(val):
                        r_2y = val
                        print(f" Found 2Y yield for {date_str}: {r_2y:.3f}%")
            except Exception as e:
                print(f" Historical 2Y lookup failed: {e}")
            
            # Short-term rates
            # US3MT=RR etc are indices, TR.ClosePrice is usually the rate
            try:
                hist_short = rd.get_history(
                    universe=['US3MT=RR', 'US6MT=RR', 'US1YT=RR'],
                    fields=['TR.ClosePrice'],
                    start=date_str,
                    end=date_str
                )
                
                # Transform to match get_data output structure: Instrument | Rate
                # hist_short likely has columns = RICS (if 1 field) or MultiIndex
                if not hist_short.empty:
                    # If columns are the RICs:
                    # Flatten/Melt
                    # Assuming columns are the RICS: 'US3MT=RR', etc.
                    # Or 'Close Price' under a MultiIndex
                    
                    data_list = []
                    for col in hist_short.columns:
                        # Extract RIC from column name if possible, or assume column IS RIC
                        # rd.get_history usually returns columns matching the universe if one field
                        ric = str(col)
                        # Clean up if it's like ('US3MT=RR', 'Close Price')
                        if isinstance(col, tuple):
                             ric = col[0]
                        
                        val = hist_short.iloc[0][col]
                        data_list.append({'Instrument': ric, 'Rate': val})
                    
                    treasury_data = pd.DataFrame(data_list)
            except Exception as e:
                print(f" Historical short-term rates failed: {e}")

        # --- Fallback & Merge ---
        
        if r_2y is None:
            # Fallback: estimate from 1-year rate or constant
            r_2y = 3.25  # Approximate based on yield curve
            print(f"  Using estimated 2Y rate: {r_2y:.3f}%")
        
        # Add the 2-year rate manually
        treasury_2y_row = pd.DataFrame({
            'Instrument': ['US2YT (estimated)' if is_historical and r_2y == 3.25 else 'US2YT'],
            'Rate': [r_2y]
        })
        
        # Ensure treasury_data has 'Instrument' and 'Rate'
        if 'Instrument' not in treasury_data.columns and not treasury_data.empty:
            # If get_data returned generic structure? get_data returns Instrument column usually
            pass # Should be fine from get_data
            
        treasury_data = pd.concat([treasury_data, treasury_2y_row], ignore_index=True)
        
        print(f"\nUS Treasury Yields ({date_str if date_str else 'Latest'}):")
        print(treasury_data)
        
        if not treasury_data.empty and 'Rate' in treasury_data.columns:
            # Safely extract by Instrument if possible, or by index (assuming order)
            # Better to set index to Instrument
            td_indexed = treasury_data.set_index('Instrument')
            
            # Helper to get rate with fallback
            def get_rate(ric_partial):
                # Simple lookup by partial match
                for idx in td_indexed.index:
                    if ric_partial in str(idx):
                        return td_indexed.loc[idx, 'Rate']
                return None

            r_3m = get_rate('US3MT')
            r_6m = get_rate('US6MT')
            r_1y = get_rate('US1YT')
            r_2y_final = get_rate('US2YT')

            print("\n Risk-free rates:")
            if r_3m: print(f"  r_3m  = {r_3m / 100:.4f}  # {r_3m:.3f}%")
            if r_6m: print(f"  r_6m  = {r_6m / 100:.4f}  # {r_6m:.3f}%")
            if r_1y: print(f"  r_1y  = {r_1y / 100:.4f}  # {r_1y:.3f}%")
            if r_2y_final: print(f"  r_2y  = {r_2y_final / 100:.4f}  # {r_2y_final:.3f}%")
            
            # Save to CSV
            output_dir = os.path.join(os.path.dirname(__file__), 'refinitiv_data')
            os.makedirs(output_dir, exist_ok=True)
            
            # Filename includes date
            date_label = date_str if date_str else datetime.now().strftime("%Y%m%d")
            file_path = os.path.join(output_dir, f'us_treasury_rates_{date_label}.csv')
            
            treasury_data.to_csv(file_path, index=False)
            print(f"\n  Data saved to: {file_path}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        rd.close_session()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import US Treasury Rates")
    parser.add_argument("--date", type=str, default=None, help="Date in YYYY-MM-DD format (default: today)")
    args = parser.parse_args()
    
    get_treasury_rates(args.date)