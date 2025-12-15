import pandas as pd
import numpy as np
from datetime import datetime
from iv_surface_spx import SPXIVSurface, SurfaceConfig

# Charger le CSV
csv_path = "data/sp500_options_SPX_20251215_231925.csv"
spx_df = pd.read_csv(csv_path)

print("=" * 70)
print("🔍 DIAGNOSTIC DE L'INTERPOLATION")
print("=" * 70)

# Configurer comme dans le notebook
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

print(f"\n📊 Données après filtrage:")
print(f"   Total options: {len(spx_surface.df)}")
print(f"   Options avec T > 1 an: {len(spx_surface.df[spx_surface.df['T'] > 1.0])}")
print(f"   Plage T: {spx_surface.df['T'].min():.4f} à {spx_surface.df['T'].max():.4f} ans")

# Analyser la distribution de T
t_values = spx_surface.df["T"].values
print(f"\n📈 Distribution de T:")
print(f"   Percentile 1%: {np.nanpercentile(t_values, 1):.4f} ans")
print(f"   Percentile 50%: {np.nanpercentile(t_values, 50):.4f} ans")
print(f"   Percentile 99%: {np.nanpercentile(t_values, 99):.4f} ans")
print(f"   Max: {np.nanmax(t_values):.4f} ans")

# Vérifier l'interpolation
print(f"\n🔧 Paramètres d'interpolation:")
print(f"   min_T configuré: {cfg.min_T}")
print(f"   max_T configuré: {cfg.max_T}")
print(f"   grid_n: {cfg.grid_n}")

# Simuler l'interpolation
try:
    from scipy.interpolate import RBFInterpolator
    
    x = spx_surface.df["x"].values
    t = spx_surface.df["T"].values
    z = spx_surface.df["iv_pct"].values
    
    x_min, x_max = np.nanpercentile(x, [1, 99])
    
    if cfg.min_T is not None and cfg.max_T is not None:
        t_min, t_max = cfg.min_T, cfg.max_T
    else:
        t_min, t_max = np.nanpercentile(t, [1, 99])
    
    print(f"\n📐 Grille d'interpolation:")
    print(f"   x_min: {x_min:.4f}, x_max: {x_max:.4f}")
    print(f"   t_min: {t_min:.4f}, t_max: {t_max:.4f}")
    
    X = np.linspace(x_min, x_max, cfg.grid_n)
    TT = np.linspace(t_min, t_max, cfg.grid_n)
    XX, YY = np.meshgrid(X, TT)
    
    print(f"\n🔍 Points de données pour l'interpolation:")
    print(f"   Nombre de points (x, t): {len(x)}")
    print(f"   Points avec T > 1 an: {np.sum(t > 1.0)}")
    print(f"   Points avec T > 1.5 an: {np.sum(t > 1.5)}")
    
    # Vérifier combien de points de la grille sont dans la zone T > 1 an
    grid_points_t_above_1 = np.sum(YY > 1.0)
    print(f"\n📊 Points de la grille avec T > 1 an:")
    print(f"   {grid_points_t_above_1} / {YY.size} points de grille ({100*grid_points_t_above_1/YY.size:.1f}%)")
    
    # Créer l'interpolateur
    pts = np.column_stack([x, t])
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
    
    print(f"\n🔧 Création de l'interpolateur RBF...")
    rbf = RBFInterpolator(
        pts, z,
        kernel=cfg.rbf_kernel,
        smoothing=cfg.rbf_smoothing
    )
    
    print(f"   ✅ Interpolateur créé")
    
    # Interpoler
    print(f"\n📈 Interpolation sur la grille...")
    ZZ = rbf(grid_pts).reshape(XX.shape)
    
    print(f"   ✅ Interpolation terminée")
    print(f"   Shape de ZZ: {ZZ.shape}")
    print(f"   Valeurs min/max de ZZ: {np.nanmin(ZZ):.2f}% / {np.nanmax(ZZ):.2f}%")
    
    # Vérifier les valeurs interpolées dans la zone T > 1 an
    mask_t_above_1 = YY > 1.0
    zz_above_1 = ZZ[mask_t_above_1]
    
    print(f"\n🎯 Zone T > 1 an dans l'interpolation:")
    print(f"   Nombre de points interpolés avec T > 1 an: {len(zz_above_1)}")
    if len(zz_above_1) > 0:
        print(f"   IV min dans cette zone: {np.nanmin(zz_above_1):.2f}%")
        print(f"   IV max dans cette zone: {np.nanmax(zz_above_1):.2f}%")
        print(f"   IV moyenne dans cette zone: {np.nanmean(zz_above_1):.2f}%")
        print(f"   Nombre de NaN: {np.sum(np.isnan(zz_above_1))}")
        print(f"   Nombre de valeurs infinies: {np.sum(np.isinf(zz_above_1))}")
    else:
        print(f"   ⚠️  AUCUN point interpolé avec T > 1 an!")
    
    # Comparer avec les données réelles dans cette zone
    real_data_above_1 = spx_surface.df[spx_surface.df["T"] > 1.0]
    if len(real_data_above_1) > 0:
        print(f"\n📊 Données réelles avec T > 1 an:")
        print(f"   Nombre: {len(real_data_above_1)}")
        print(f"   IV min: {real_data_above_1['iv_pct'].min():.2f}%")
        print(f"   IV max: {real_data_above_1['iv_pct'].max():.2f}%")
        print(f"   IV moyenne: {real_data_above_1['iv_pct'].mean():.2f}%")
    
except Exception as e:
    print(f"\n❌ Erreur lors de l'interpolation: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("\n💡 CONCLUSION:")
print("   Si l'interpolation fonctionne mais que vous ne voyez rien,")
print("   le problème peut venir de:")
print("   1. L'échelle de l'axe Y dans Plotly")
print("   2. La densité de points (les options T > 1 an sont peu nombreuses)")
print("   3. Les valeurs interpolées sont NaN ou infinies dans cette zone")

