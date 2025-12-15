"""
Script de test pour visualiser la surface de volatilité entre T=1 an et T=2 ans.
Sauvegarde le graphique en PNG.
"""

import pandas as pd
from datetime import datetime
from iv_surface_spx import SPXIVSurface, SurfaceConfig
from pathlib import Path

# Configuration
CSV_PATH = "data/sp500_options_SPX_20251215_231925.csv"
OUTPUT_DIR = Path("plot/vol")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Créer le dossier s'il n'existe pas
OUTPUT_PNG = OUTPUT_DIR / "vol_surface_t1_t2.png"

print("=" * 70)
print("🔍 TEST: Surface de volatilité entre T=1 an et T=2 ans (PNG)")
print("=" * 70)

# Charger les données
print(f"\n📂 Chargement des données depuis: {CSV_PATH}")
spx_df = pd.read_csv(CSV_PATH)
print(f"   Total options dans CSV: {len(spx_df)}")

# Filtrer par date si nécessaire
start_date = pd.Timestamp("2025-12-16")
end_date = pd.Timestamp("2027-04-30")
spx_df["expiry_date"] = pd.to_datetime(spx_df["expiry_date"])
spx_df = spx_df[(spx_df["expiry_date"] >= start_date) & (spx_df["expiry_date"] <= end_date)]
print(f"   Options après filtrage par date: {len(spx_df)}")

# Configuration pour filtrer uniquement T entre 1 et 2 ans
cfg = SurfaceConfig(
    r=0.05,
    min_bid=0.01,
    max_rel_spread=0.25,
    min_oi=10,
    min_volume=1,
    grid_n=60,
    rbf_smoothing=0.5,
    min_T=1.0,   # Commencer à 1 an
    max_T=2.0   # Aller jusqu'à 2 ans
)

print(f"\n🔧 Configuration:")
print(f"   min_T: {cfg.min_T} ans")
print(f"   max_T: {cfg.max_T} ans")

# Créer la surface
print(f"\n🔨 Création de la surface...")
spx_surface = SPXIVSurface(spx_df, cfg)

print(f"\n📊 Résultats après filtrage:")
print(f"   Total options: {len(spx_surface.df)}")
if len(spx_surface.df) > 0:
    print(f"   T min: {spx_surface.df['T'].min():.4f} ans")
    print(f"   T max: {spx_surface.df['T'].max():.4f} ans")
    print(f"   Options avec T > 1 an: {len(spx_surface.df[spx_surface.df['T'] > 1.0])}")
    print(f"   Options avec T > 1.5 an: {len(spx_surface.df[spx_surface.df['T'] > 1.5])}")
else:
    print("   ⚠️  AUCUNE option dans cette plage de maturité!")
    print("   Vérifiez que vos données contiennent des options avec T entre 1 et 2 ans.")

# Créer le plot
print(f"\n🎨 Création du graphique...")
fig = spx_surface.plot(
    title="SPX Implied Volatility Surface (T = 1-2 years)",
    interpolate=True
)

# Sauvegarder en PNG
print(f"\n💾 Sauvegarde en PNG: {OUTPUT_PNG}")

try:
    # Sauvegarder le graphique en PNG
    fig.write_image(str(OUTPUT_PNG), width=1920, height=1080, scale=2)
    print(f"\n✅ Graphique sauvegardé avec succès!")
    print(f"   Fichier: {OUTPUT_PNG}")
    print(f"   Dossier: {OUTPUT_DIR.absolute()}")
except Exception as e:
    print(f"\n❌ Erreur lors de la sauvegarde: {e}")
    print(f"   💡 Assurez-vous que 'kaleido' est installé: pip install kaleido")
    print(f"   Ou utilisez 'orca': pip install plotly-orca")

