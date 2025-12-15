import pandas as pd
from datetime import datetime
from iv_surface_spx import SPXIVSurface, SurfaceConfig

# Simuler le chargement des données
csv_path = "data/sp500_options_SPX_20251215_231925.csv"
spx_df = pd.read_csv(csv_path)

# Filtrer par date comme dans votre code
start_date = pd.Timestamp("2025-12-16")
end_date = pd.Timestamp("2027-04-30")
spx_df["expiry_date"] = pd.to_datetime(spx_df["expiry_date"])
spx_df = spx_df[(spx_df["expiry_date"] >= start_date) & (spx_df["expiry_date"] <= end_date)]

print(f"📊 Après filtrage par date:")
print(f"   Total options: {len(spx_df)}")
print(f"   Date min: {spx_df['expiry_date'].min()}")
print(f"   Date max: {spx_df['expiry_date'].max()}")

# Calculer T avec la date actuelle
as_of = datetime.now()
spx_df["T"] = (spx_df["expiry_date"] - pd.Timestamp(as_of)).dt.total_seconds() / (365.25 * 24 * 3600)

print(f"\n📅 Date de référence (as_of): {as_of}")
print(f"\n⏱️  Maturité (T):")
print(f"   T min: {spx_df['T'].min():.4f} ans")
print(f"   T max: {spx_df['T'].max():.4f} ans")
print(f"   Options avec T > 1 an: {len(spx_df[spx_df['T'] > 1.0])}")
print(f"   Options avec T > 1.5 an: {len(spx_df[spx_df['T'] > 1.5])}")

if len(spx_df[spx_df['T'] > 1.0]) > 0:
    print(f"\n✅ Exemples d'options avec T > 1 an AVANT filtrage:")
    long_term = spx_df[spx_df['T'] > 1.0]
    print(long_term[['expiry_date', 'strike', 'type', 'T', 'bid_price', 'ask_price', 'open_interest', 'volume', 'mark_iv']].head(10).to_string())

# Créer la surface avec la config
cfg = SurfaceConfig(
    r=0.05,
    min_bid=0.01,
    max_rel_spread=0.25,
    min_oi=10,
    min_volume=1,
    grid_n=60,
    rbf_smoothing=0.5,
    min_T=1/365,
    max_T=2.0
)

print(f"\n🔧 Création de la surface avec filtres...")
spx_surface = SPXIVSurface(spx_df, cfg)

print(f"\n📊 Après tous les filtres de SPXIVSurface:")
print(f"   Total options: {len(spx_surface.df)}")
print(f"   T min: {spx_surface.df['T'].min():.4f} ans")
print(f"   T max: {spx_surface.df['T'].max():.4f} ans")
print(f"   Options avec T > 1 an: {len(spx_surface.df[spx_surface.df['T'] > 1.0])}")
print(f"   Options avec T > 1.5 an: {len(spx_surface.df[spx_surface.df['T'] > 1.5])}")

if len(spx_surface.df[spx_surface.df['T'] > 1.0]) > 0:
    print(f"\n✅ Exemples d'options avec T > 1 an APRÈS filtrage:")
    long_term_filtered = spx_surface.df[spx_surface.df['T'] > 1.0]
    print(long_term_filtered[['expiry_date', 'strike', 'type', 'T', 'x', 'iv_pct']].head(10).to_string())
else:
    print(f"\n❌ AUCUNE option avec T > 1 an après filtrage!")
    print(f"\n🔍 Analyse des raisons d'exclusion:")
    
    # Vérifier chaque filtre
    df_test = spx_df.copy()
    df_test["T"] = (df_test["expiry_date"] - pd.Timestamp(as_of)).dt.total_seconds() / (365.25 * 24 * 3600)
    df_test = df_test[df_test["T"] > 1.0]
    
    print(f"\n   1. Filtre bid/ask > 0:")
    df1 = df_test[(df_test["bid_price"] > 0) & (df_test["ask_price"] > 0)]
    print(f"      Conservées: {len(df1)} / {len(df_test)}")
    
    print(f"\n   2. Filtre bid >= 0.01:")
    df2 = df1[df1["bid_price"] >= 0.01]
    print(f"      Conservées: {len(df2)} / {len(df1)}")
    
    print(f"\n   3. Filtre spread relatif <= 25%:")
    df2["mid"] = (df2["bid_price"] + df2["ask_price"]) / 2.0
    df2["rel_spread"] = (df2["ask_price"] - df2["bid_price"]) / df2["mid"].replace(0, pd.NA)
    df3 = df2[df2["rel_spread"] <= 0.25]
    print(f"      Conservées: {len(df3)} / {len(df2)}")
    
    print(f"\n   4. Filtre liquidité (OI >= 10 OU volume >= 1):")
    df4 = df3[(df3["open_interest"] >= 10) | (df3["volume"] >= 1)]
    print(f"      Conservées: {len(df4)} / {len(df3)}")
    
    print(f"\n   5. Filtre IV valide:")
    df5 = df4[df4["mark_iv"].notna()]
    print(f"      Conservées: {len(df5)} / {len(df4)}")
    
    # Normaliser IV
    import numpy as np
    iv_series = pd.to_numeric(df5["mark_iv"], errors="coerce")
    med = np.nanmedian(iv_series.values) if np.isfinite(iv_series).any() else np.nan
    if np.isfinite(med) and med < 3.0:
        df5["iv_pct"] = iv_series * 100.0
    else:
        df5["iv_pct"] = iv_series
    
    df6 = df5[df5["iv_pct"].between(0.01, 300.0)]
    print(f"\n   6. Filtre IV entre 0.01% et 300%:")
    print(f"      Conservées: {len(df6)} / {len(df5)}")
    
    if len(df6) > 0:
        print(f"\n   ⚠️  Il reste {len(df6)} options après filtres de qualité.")
        print(f"   Le problème vient probablement du calcul du forward ou du filtre OTM-only.")

print(f"\n" + "=" * 70)

