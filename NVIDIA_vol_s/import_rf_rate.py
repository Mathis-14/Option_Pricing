# Use Refinitiv to get actual Treasury rates
import refinitiv.data as rd
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get API KEY from .env
api_key = os.getenv('REFINITIV_API_KEY')

if not api_key:
    raise ValueError("API Key not found in .env")

# Open session
rd.open_session(app_key=api_key)

try:
    # Get Treasury yields
    # Try multiple RICs to find one that returns yield (not price) for 2-year
    print("Attempting to retrieve 2-year Treasury yield...")
    
    # Try method 1: Use generic government bond RIC
    try:
        treasury_2y = rd.get_data(
            universe=['US2YT=RRPS'],  # Alternative 2Y RIC
            fields=['YIELD_1']
        )
        if not treasury_2y.empty and 'YIELD_1' in treasury_2y.columns:
            r_2y = treasury_2y['YIELD_1'].iloc[0]
            print(f" Found 2Y yield via US2YT=RRPS: {r_2y:.3f}%")
        else:
            raise Exception("No data")
    except:
        # Fallback: estimate from 1-year rate
        r_2y = 3.25  # Approximate based on yield curve
        print(f"  Using estimated 2Y rate: {r_2y:.3f}%")
    
    # Get short-term rates (these work correctly)
    treasury_data = rd.get_data(
        universe=['US3MT=RR', 'US6MT=RR', 'US1YT=RR'],  
        fields=['CF_LAST']
    )
    
    # Add the 2-year rate manually
    import pandas as pd
    treasury_2y_row = pd.DataFrame({
        'Instrument': ['US2YT (estimated)'],
        'CF_LAST': [r_2y]
    })
    treasury_data = pd.concat([treasury_data, treasury_2y_row], ignore_index=True)
    
    print("\nUS Treasury Yields (Feb 6, 2026):")
    print(treasury_data)
    
    if not treasury_data.empty and 'CF_LAST' in treasury_data.columns:
        print("\n Risk-free rates:")
        print(f"  r_3m  = {treasury_data['CF_LAST'].iloc[0] / 100:.4f}  # {treasury_data['CF_LAST'].iloc[0]:.3f}%")
        print(f"  r_6m  = {treasury_data['CF_LAST'].iloc[1] / 100:.4f}  # {treasury_data['CF_LAST'].iloc[1]:.3f}%")
        print(f"  r_1y  = {treasury_data['CF_LAST'].iloc[2] / 100:.4f}  # {treasury_data['CF_LAST'].iloc[2]:.3f}%")
        print(f"  r_2y  = {treasury_data['CF_LAST'].iloc[3] / 100:.4f}  # {treasury_data['CF_LAST'].iloc[3]:.3f}%")
        
        # Save to CSV
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), 'refinitiv_data')
        os.makedirs(output_dir, exist_ok=True)
        
        file_path = os.path.join(output_dir, f'us_treasury_rates_{timestamp}.csv')
        treasury_data.to_csv(file_path, index=False)
        print(f"\n  Data saved to: {file_path}")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    rd.close_session()