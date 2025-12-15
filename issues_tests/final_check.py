import pandas as pd
from datetime import datetime
from iv_surface_spx import SPXIVSurface, SurfaceConfig

# Charger le CSV
csv_path = "data/sp500_options_SPX_20251215_231925.csv"
spx_df = pd.read_csv(csv_path)

print(f"📊 DataFrame initial: {len(spx_df)} options")
print(f"=" * 70)

# Configurer comme dans le notebook
cfg = SurfaceConfig(
    r=0.05,
    min_bid=0.01,
    max_rel_spread=0.25,
    min_oi=10,
    min_volume=1,
    grid_n=60,
    rbf_smoothing=0.5,
    min_T=1/365,  # Min 3-4 days
    max_T=2       # Max 2 ans
)

# Créer la surface
spx_surface = SPXIVSurface(spx_df, cfg)

print(f"\n✅ Options après tous les filtres: {len(spx_surface.df)}")
print(f"\n🔍 Analyse des options avec T > 1 an:")

# Filtrer les options avec T > 1
df_long = spx_surface.df[spx_surface.df["T"] > 1.0].copy()

print(f"   Options avec T > 1.0 an: {len(df_long)}")
print(f"   Pourcentage du total: {100*len(df_long)/len(spx_surface.df):.1f}%")

if len(df_long) > 0:
    print(f"\n   Plage de maturité:")
    print(f"     Min T: {df_long['T'].min():.4f} ans")
    print(f"     Max T: {df_long['T'].max():.4f} ans")
    print(f"     Mean T: {df_long['T'].mean():.4f} ans")
    
    print(f"\n   Répartition par expiration:")
    expiry_counts = df_long.groupby("expiry_date").size().sort_index()
    print(f"     Nombre d'expirations: {len(expiry_counts)}")
    print(f"     Options par expiration:")
    for exp_date, count in expiry_counts.items():
        print(f"       {exp_date.date()}: {count} options")
    
    print(f"\n   Exemples d'options avec T > 1 an:")
    print(df_long[['expiry_date', 'strike', 'type', 'T', 'x', 'iv_pct']].head(10).to_string())
else:
    print(f"\n   ⚠️  AUCUNE option avec T > 1 an dans le DataFrame final!")
    print(f"\n   Vérification de la plage T dans le DataFrame:")
    print(f"     Min T: {spx_surface.df['T'].min():.4f} ans")
    print(f"     Max T: {spx_surface.df['T'].max():.4f} ans")
    print(f"     Options avec T > 0.9: {len(spx_surface.df[spx_surface.df['T'] > 0.9])}")
    print(f"     Options avec T > 0.95: {len(spx_surface.df[spx_surface.df['T'] > 0.95])}")
    print(f"     Options avec T > 0.99: {len(spx_surface.df[spx_surface.df['T'] > 0.99])}")

print(f"\n" + "=" * 70)
print(f"\n💡 CONCLUSION:")
if len(df_long) > 0:
    print(f"   ✅ Les options avec T > 1 an SONT présentes dans la surface ({len(df_long)} options)")
    print(f"   Elles devraient apparaître dans le plot si max_T >= 2")
else:
    print(f"   ❌ Les options avec T > 1 an NE SONT PAS présentes dans la surface")
    print(f"   Raison possible: elles sont filtrées par les critères de qualité ou OTM-only")

