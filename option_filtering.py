import pandas as pd
from datetime import datetime

def load_options_with_min_maturity(csv_path: str, min_maturity_years: float = 1.0, print_df=False):
    """
    Charge les options depuis un CSV et filtre par maturité minimale.
    
    Parameters:
    -----------
    csv_path : str
        Chemin vers le fichier CSV
    min_maturity_years : float
        Maturité minimale en années (défaut: 1.0)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame filtré avec les options ayant maturité > min_maturity_years
    """
    df = pd.read_csv(csv_path)
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    
    # Calculer le temps jusqu'à l'expiration en années
    as_of = datetime.now()
    df["T"] = (df["expiry_date"] - pd.Timestamp(as_of)).dt.total_seconds() / (365.25 * 24 * 3600)
    
    # Filtrer
    df_filtered = df[df["T"] > min_maturity_years].copy()
    
    print(f"Total options: {len(df)}")
    print(f"Options with maturity > {min_maturity_years} year(s): {len(df_filtered)}")
    if len(df_filtered) > 0:
        print(f"Maturity range: {df_filtered['T'].min():.3f} to {df_filtered['T'].max():.3f} years")

    if print_df==True :
        print(df_filtered)
    
    return df_filtered

# Utilisation
df_long_term = load_options_with_min_maturity(
    "data/sp500_options_SPX_20251215_231925.csv",
    min_maturity_years=1.0, print_df=True
)