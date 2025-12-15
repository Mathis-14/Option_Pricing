import pandas as pd
import numpy as np
from datetime import datetime

# Charger le CSV
csv_path = "data/sp500_options_SPX_20251215_231925.csv"
df = pd.read_csv(csv_path)

# Convertir expiry_date
df["expiry_date"] = pd.to_datetime(df["expiry_date"])

# Date de référence
as_of = datetime.now()
df["T"] = (df["expiry_date"] - pd.Timestamp(as_of)).dt.total_seconds() / (365.25 * 24 * 3600)

# Filtrer T > 1 et appliquer les filtres de base
df_long = df[df["T"] > 1.0].copy()

# Appliquer les mêmes filtres que iv_surface_spx.py jusqu'au calcul du forward
df_long = df_long[(df_long["bid_price"] > 0) & (df_long["ask_price"] > 0)]
df_long = df_long[df_long["bid_price"] >= 0.01]
df_long["mid"] = (df_long["bid_price"] + df_long["ask_price"]) / 2.0
df_long["rel_spread"] = (df_long["ask_price"] - df_long["bid_price"]) / df_long["mid"].replace(0, pd.NA)
df_long = df_long[df_long["rel_spread"] <= 0.25]
df_long = df_long[(df_long["open_interest"] >= 10) | (df_long["volume"] >= 1)]
df_long = df_long[df_long["mark_iv"].notna()]

# Normaliser IV
iv_series = pd.to_numeric(df_long["mark_iv"], errors="coerce")
med = np.nanmedian(iv_series.values) if np.isfinite(iv_series).any() else np.nan
if np.isfinite(med) and med < 3.0:
    df_long["iv_pct"] = iv_series * 100.0
else:
    df_long["iv_pct"] = iv_series
df_long = df_long[df_long["iv_pct"].between(0.01, 300.0)]

print(f"📊 Options avec T > 1 an après filtres de base: {len(df_long)}")
print(f"=" * 70)

# Simuler le calcul du forward (comme dans iv_surface_spx.py)
def estimate_forward_group(g, r=0.05):
    T = float(g["T"].iloc[0])
    S = float(g["underlying_price"].iloc[0])
    
    pvt = g.pivot_table(index="strike", columns="type", values="mid", aggfunc="first")
    if "C" not in pvt.columns or "P" not in pvt.columns:
        return pd.Series(index=g.index, data=np.nan)
    
    pvt = pvt.dropna(subset=["C", "P"])
    if pvt.empty:
        return pd.Series(index=g.index, data=np.nan)
    
    # take near-ATM strikes for stability
    pvt = pvt.assign(abs_diff=np.abs(pvt.index.values - S)).sort_values("abs_diff").head(12)
    
    disc = np.exp(r * T)
    K = pvt.index.values.astype(float)
    C = pvt["C"].values.astype(float)
    P = pvt["P"].values.astype(float)
    
    F_est = K + disc * (C - P)
    F = np.nanmedian(F_est) if len(F_est) else np.nan
    return pd.Series(index=g.index, data=F)

# Calculer le forward
df_long["F"] = df_long.groupby("expiry_date", group_keys=False).apply(estimate_forward_group)
df_with_forward = df_long[df_long["F"].notna()].copy()

print(f"\n🔍 Après calcul du forward:")
print(f"   ✅ Options avec forward calculé: {len(df_with_forward)} / {len(df_long)} ({100*len(df_with_forward)/len(df_long):.1f}%)")
print(f"   ❌ Options sans forward: {len(df_long) - len(df_with_forward)}")

if len(df_long) > len(df_with_forward):
    no_forward = df_long[df_long["F"].isna()]
    print(f"\n   Expirations sans forward calculable:")
    for exp_date in no_forward["expiry_date"].unique():
        exp_df = no_forward[no_forward["expiry_date"] == exp_date]
        has_calls = (exp_df["type"] == "C").any()
        has_puts = (exp_df["type"] == "P").any()
        print(f"     - {exp_date.date()}: {len(exp_df)} options (Calls: {has_calls}, Puts: {has_puts})")

# Calculer log-moneyness
df_with_forward["x"] = np.log(df_with_forward["strike"] / df_with_forward["F"])

# Appliquer filtre OTM-only
is_put_otm = (df_with_forward["strike"] < df_with_forward["F"]) & (df_with_forward["type"] == "P")
is_call_otm = (df_with_forward["strike"] > df_with_forward["F"]) & (df_with_forward["type"] == "C")
is_atm = (np.abs(df_with_forward["strike"] - df_with_forward["F"]) / df_with_forward["F"] < 0.002)
df_otm = df_with_forward[is_put_otm | is_call_otm | is_atm].copy()

print(f"\n🔍 Après filtre OTM-only:")
print(f"   ✅ Options OTM/ATM conservées: {len(df_otm)} / {len(df_with_forward)} ({100*len(df_otm)/len(df_with_forward):.1f}%)")
print(f"   ❌ Options ITM éliminées: {len(df_with_forward) - len(df_otm)}")

if len(df_with_forward) > len(df_otm):
    itm = df_with_forward[~(is_put_otm | is_call_otm | is_atm)]
    print(f"\n   Exemples d'options ITM éliminées:")
    print(itm[['expiry_date', 'strike', 'type', 'F', 'x']].head(10).to_string())

print(f"\n" + "=" * 70)
print(f"\n📈 RÉSUMÉ FINAL:")
print(f"   Options initiales avec T > 1 an: {len(df[df['T'] > 1.0])}")
print(f"   Après filtres de qualité: {len(df_long)}")
print(f"   Après calcul du forward: {len(df_with_forward)}")
print(f"   Après filtre OTM-only: {len(df_otm)}")
print(f"   Taux de conservation final: {100*len(df_otm)/len(df[df['T'] > 1.0]):.1f}%")

if len(df_otm) > 0:
    print(f"\n✅ Exemples d'options finales conservées:")
    print(df_otm[['expiry_date', 'strike', 'type', 'T', 'F', 'x', 'iv_pct']].head(10).to_string())

