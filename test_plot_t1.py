import pandas as pd
from datetime import datetime
from iv_surface_spx import SPXIVSurface, SurfaceConfig

# Charger les données
csv_path = "data/sp500_options_SPX_20251215_231925.csv"
spx_df = pd.read_csv(csv_path)

# Filtrer par date
start_date = pd.Timestamp("2025-12-16")
end_date = pd.Timestamp("2027-04-30")
spx_df["expiry_date"] = pd.to_datetime(spx_df["expiry_date"])
spx_df = spx_df[(spx_df["expiry_date"] >= start_date) & (spx_df["expiry_date"] <= end_date)]

print(f"📊 Données après filtrage par date: {len(spx_df)} options")

# Config
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

# Créer la surface
spx_surface = SPXIVSurface(spx_df, cfg)

print(f"\n📊 Après filtrage SPXIVSurface:")
print(f"   Total: {len(spx_surface.df)} options")
print(f"   T min: {spx_surface.df['T'].min():.4f} ans")
print(f"   T max: {spx_surface.df['T'].max():.4f} ans")

# Vérifier les options avec T > 1 an
df_long = spx_surface.df[spx_surface.df["T"] > 1.0]
print(f"\n🔍 Options avec T > 1 an:")
print(f"   Nombre: {len(df_long)}")
if len(df_long) > 0:
    print(f"   T range: {df_long['T'].min():.4f} à {df_long['T'].max():.4f} ans")
    print(f"   Exemples:")
    print(df_long[['expiry_date', 'strike', 'type', 'T', 'x', 'iv_pct']].head(5).to_string())
    
    # Vérifier les valeurs de x (log-moneyness)
    print(f"\n   Plage de x (log-moneyness):")
    print(f"     Min: {df_long['x'].min():.4f}")
    print(f"     Max: {df_long['x'].max():.4f}")
    print(f"     Mean: {df_long['x'].mean():.4f}")
    
    # Vérifier les valeurs de IV
    print(f"\n   Plage de IV:")
    print(f"     Min: {df_long['iv_pct'].min():.2f}%")
    print(f"     Max: {df_long['iv_pct'].max():.2f}%")
    print(f"     Mean: {df_long['iv_pct'].mean():.2f}%")
else:
    print("   ❌ AUCUNE option avec T > 1 an!")

# Créer le plot et vérifier les traces
print(f"\n🎨 Création du plot...")
fig = spx_surface.plot(interpolate=True)

# Vérifier les traces ajoutées
print(f"\n📊 Traces dans le graphique:")
for i, trace in enumerate(fig.data):
    print(f"   Trace {i}: {trace.type}")
    if hasattr(trace, 'name'):
        print(f"      Nom: {trace.name}")
    if hasattr(trace, 'y'):
        if hasattr(trace.y, '__len__'):
            y_data = trace.y
            if hasattr(y_data, 'flatten'):
                y_flat = y_data.flatten()
            else:
                y_flat = y_data
            print(f"      Y range: {min(y_flat):.4f} à {max(y_flat):.4f} ans")
            print(f"      Points avec Y > 1: {sum(y_flat > 1.0) if hasattr(y_flat, '__len__') else 0}")

# Vérifier les limites de l'axe Y
print(f"\n📐 Configuration de l'axe Y:")
scene = fig.layout.scene
if hasattr(scene, 'yaxis') and hasattr(scene.yaxis, 'range'):
    print(f"   Range configuré: {scene.yaxis.range}")
else:
    print(f"   ⚠️  Pas de range configuré pour l'axe Y!")

print(f"\n💡 Si vous ne voyez pas les points rouges:")
print(f"   1. Vérifiez que la caméra montre bien la zone T > 1 an")
print(f"   2. Zoom/pan sur l'axe Y pour voir jusqu'à 2 ans")
print(f"   3. Les points rouges peuvent être masqués par la surface")

