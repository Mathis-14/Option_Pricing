"""
Get NVIDIA stock price for a given options data CSV file.
Parses the timestamp from the filename and fetches the closing price for that date.
"""
import refinitiv.data as rd
import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import re
import sys

def list_csv_files(data_dir):
    """List all NVDA options CSV files in the directory"""
    csv_files = []
    for file in Path(data_dir).glob('nvda_options_*.csv'):
        csv_files.append(file.name)
    return sorted(csv_files, reverse=True)  # Most recent first

def parse_timestamp_from_filename(filename):
    """
    Parse timestamp from filename like: nvda_options_20260204_210456.csv
    Returns: (date_str, datetime_obj)
    """
    match = re.search(r'nvda_options_(\d{8})_(\d{6})\.csv', filename)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMMSS
        
        # Convert to datetime
        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        date_for_api = dt.strftime("%Y-%m-%d")
        
        return date_for_api, dt
    return None, None

def get_closing_price(date_str, api_key):
    """Fetch NVDA closing price for a given date"""
    rd.open_session(app_key=api_key)
    
    try:
        close_data = rd.get_data(
            universe='NVDA.O',
            fields=['TR.PriceClose', 'TR.PriceClose.date'],
            parameters={'SDate': date_str, 'EDate': date_str}
        )
        
        if not close_data.empty and 'Price Close' in close_data.columns:
            price = close_data['Price Close'].iloc[0]
            return price
        else:
            # Fallback to current price
            current = rd.get_data(universe='NVDA.O', fields=['CF_LAST'])
            return current['CF_LAST'].iloc[0]
    
    finally:
        rd.close_session()

def main():
    # Load API key
    load_dotenv()
    api_key = os.getenv('REFINITIV_API_KEY')
    
    if not api_key:
        print("Error: REFINITIV_API_KEY not found in .env")
        sys.exit(1)
    
    # Get data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'refinitiv_data'
    
    # List available CSV files
    csv_files = list_csv_files(data_dir)
    
    if not csv_files:
        print(f"No NVDA options CSV files found in {data_dir}")
        sys.exit(1)
    
    print("Available NVDA options data files:")
    print("=" * 60)
    for i, filename in enumerate(csv_files, 1):
        date_str, dt = parse_timestamp_from_filename(filename)
        if dt:
            print(f"{i}. {filename}")
            print(f"   Captured: {dt.strftime('%Y-%m-%d at %H:%M:%S')}")
        else:
            print(f"{i}. {filename} (invalid format)")
    print("=" * 60)
    
    # Get user selection
    if len(sys.argv) > 1:
        # Command line argument provided
        try:
            selection = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid selection '{sys.argv[1]}'")
            sys.exit(1)
    else:
        # Interactive selection
        try:
            selection = int(input(f"\nSelect file (1-{len(csv_files)}): "))
        except (ValueError, EOFError):
            print("\nNo selection made. Exiting.")
            sys.exit(0)
    
    if selection < 1 or selection > len(csv_files):
        print(f"Error: Selection must be between 1 and {len(csv_files)}")
        sys.exit(1)
    
    # Get selected file
    selected_file = csv_files[selection - 1]
    date_str, dt = parse_timestamp_from_filename(selected_file)
    
    if not date_str:
        print(f"Error: Could not parse timestamp from {selected_file}")
        sys.exit(1)
    
    print(f"\nSelected: {selected_file}")
    print(f"Date: {date_str}")
    print(f"Time: {dt.strftime('%H:%M:%S')} (after market close)")
    print(f"\nFetching NVDA closing price for {date_str}...")
    
    # Get closing price
    price = get_closing_price(date_str, api_key)
    
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"NVDA closing price on {date_str}: ${price:.2f}")
    print(f"\n Price to use:")
    print(f"  S = {price:.2f}")
    print(f"  capture_date = pd.to_datetime('{date_str}')")
    print("=" * 60)

if __name__ == "__main__":
    main()
