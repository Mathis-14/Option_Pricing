import refinitiv.data as rd
import pandas as pd
import os
from dotenv import load_dotenv
import datetime

# Load .env file
load_dotenv()

# Get API KEY from .env
api_key = os.getenv('REFINITIV_API_KEY')

if not api_key:
    raise ValueError("API Key not found in .env")

def parse_option_ric(ric):
    """
    Parses a Refinitiv Option RIC to extract metadata.
    Format assumption: [Root][MonthCode][Day][Year][Strike].U
    Example: NVDAB202619000.U
    """
    try:
        details = ric.split('.')[0]
        if len(details) < 11: return None

        strike_str = details[-5:]
        year_str = details[-7:-5]
        day_str = details[-9:-7]
        month_code = details[-10]
        
        # Strike parsing: RIC strike encoding is ambiguous
        # This is a fallback - prefer API-returned Strike Price when available
        strike = float(strike_str) / 100.0 if strike_str.isdigit() else None
        year = int("20" + year_str) if year_str.isdigit() else None
        day = int(day_str) if day_str.isdigit() else None
        
        month_codes = "ABCDEFGHIJKL" # Calls
        month_codes_puts = "MNOPQRSTUVWX" # Puts
        
        month_num = 0
        type_str = ''
        if month_code in month_codes:
            month_num = month_codes.index(month_code) + 1
            type_str = 'CALL'
        elif month_code in month_codes_puts:
            month_num = month_codes_puts.index(month_code) + 1
            type_str = 'PUT'
            
        if year and month_num and day:
            expiry = datetime.date(year, month_num, day)
            return {
                'Strike': strike, 
                'Expiry': expiry.strftime('%Y-%m-%d'),
                'Type': type_str
            }
            
    except Exception:
        pass
    return None

try:
    # Open session
    rd.open_session(app_key=api_key)
    print("Session opened successfully.")

    # 1. Discover Option RICs using Search
    print("Searching for NVDA options...")
    search_response = rd.discovery.search(
        view = rd.discovery.Views.SEARCH_ALL,
        filter = "RIC eq 'NVDA*.U'",
        select = "RIC, DSPLY_NAME",
        top = 1000 # Increased to get meaningful surface data
    )
    
    if len(search_response) == 0:
        print("No options found.")
    else:
        option_rics = search_response['RIC'].tolist()
        print(f"Found {len(option_rics)} options.")

        # 2. Retrieve Data
        fields = [
            'BID', 'ASK', 'CF_LAST',
            'PUTCALLIND',
            'TR.ImpliedVolatility',
            'TR.StrikePrice',
            'TR.ExpiryDate'
        ]
        
        print(f"Retrieving data for {len(option_rics)} options...")
        df = rd.get_data(universe=option_rics, fields=fields)
        
        # 3. Fill missing data using RIC Parsing
        print("Parsing RICs to fill missing Strike/Expiry...")
        
        parsed_strikes = []
        parsed_expiries = []
        parsed_types = []
        
        for index, row in df.iterrows():
            ric = row['Instrument']
            parsed = parse_option_ric(ric)
            
            # Fill logic: Use parsed if API return is null/NaN
            
            # Strike
            val_strike = row.get('Strike Price') # API maps TR.StrikePrice -> Strike Price normally
            if (pd.isna(val_strike) or val_strike == '') and parsed:
                parsed_strikes.append(parsed['Strike'])
            else:
                parsed_strikes.append(val_strike)
                
            # Expiry
            val_expiry = row.get('Expiry Date')
            if (pd.isna(val_expiry) or val_expiry == '') and parsed:
                parsed_expiries.append(parsed['Expiry'])
            else:
                parsed_expiries.append(val_expiry)
                
            # Type (PUTCALLIND)
            val_type = row.get('PUTCALLIND')
            if (pd.isna(val_type) or val_type == '') and parsed:
                parsed_types.append(parsed['Type'])
            else:
                parsed_types.append(val_type)

        # Update DataFrame
        df['Strike Price'] = parsed_strikes
        df['Expiry Date'] = parsed_expiries
        df['PUTCALLIND'] = parsed_types
        
        print("\nData Retrieved Successfully:")
        print(df.head())
        
        # 4. Save to CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), 'refinitiv_data')
        os.makedirs(output_dir, exist_ok=True)
        
        file_path = os.path.join(output_dir, f'nvda_options_{timestamp}.csv')
        df.to_csv(file_path, index=False)
        print(f"\nData saved to: {file_path}")

except Exception as e:
    print(f"\nAn error occurred: {e}")

finally:
    rd.close_session()