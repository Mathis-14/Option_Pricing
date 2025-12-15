import pandas as pd
from datetime import datetime

# Charger le CSV
csv_path = "data/sp500_options_SPX_20251215_231925.csv"
df = pd.read_csv(csv_path)

print(f"Total rows in CSV: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Convertir expiry_date
df["expiry_date"] = pd.to_datetime(df["expiry_date"])

# Vérifier les dates d'expiration
print(f"\nExpiry date range:")
print(f"  Min: {df['expiry_date'].min()}")
print(f"  Max: {df['expiry_date'].max()}")

# Date de référence
as_of = datetime.now()
print(f"\nReference date (as_of): {as_of}")

# Calculer T
df["T"] = (df["expiry_date"] - pd.Timestamp(as_of)).dt.total_seconds() / (365.25 * 24 * 3600)

print(f"\nT (time to expiration) range:")
print(f"  Min: {df['T'].min():.6f} years")
print(f"  Max: {df['T'].max():.6f} years")
print(f"  Mean: {df['T'].mean():.6f} years")

# Vérifier combien ont T > 1
df_long = df[df["T"] > 1.0].copy()
print(f"\nOptions with T > 1.0 year: {len(df_long)}")

if len(df_long) > 0:
    print(f"\nSample of options with T > 1:")
    print(df_long[['expiry_date', 'strike', 'type', 'T', 'bid_price', 'ask_price', 'open_interest', 'volume', 'mark_iv']].head(10))
else:
    print("\n⚠️  No options found with T > 1.0 year")
    print("\nChecking why:")
    print(f"  - Options with T > 0.9: {len(df[df['T'] > 0.9])}")
    print(f"  - Options with T > 0.95: {len(df[df['T'] > 0.95])}")
    print(f"  - Options with T > 0.99: {len(df[df['T'] > 0.99])}")
    
    # Vérifier les dates d'expiration les plus lointaines
    print(f"\nTop 10 furthest expiration dates:")
    furthest = df.nlargest(10, 'T')[['expiry_date', 'T']]
    print(furthest)