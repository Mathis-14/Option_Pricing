import pandas as pd
from datetime import datetime

# Charger le CSV
csv_path = "data/sp500_options_SPX_20251215_231925.csv"
df = pd.read_csv(csv_path)

# Convertir expiry_date
df["expiry_date"] = pd.to_datetime(df["expiry_date"])

# Date de référence
as_of = datetime.now()
df["T"] = (df["expiry_date"] - pd.Timestamp(as_of)).dt.total_seconds() / (365.25 * 24 * 3600)

# Filtrer T > 1
df_long = df[df["T"] > 1.0].copy()
print(f"📊 Options avec T > 1.0 an: {len(df_long)}")
print(f"=" * 70)

# Simuler les filtres de iv_surface_spx.py
print("\n🔍 Analyse des filtres appliqués par iv_surface_spx.py:\n")

# 1. Filtre bid/ask > 0
df_step1 = df_long[(df_long["bid_price"] > 0) & (df_long["ask_price"] > 0)].copy()
print(f"1️⃣  Après filtre bid > 0 ET ask > 0:")
print(f"   ✅ Conservées: {len(df_step1)} / {len(df_long)} ({100*len(df_step1)/len(df_long):.1f}%)")
print(f"   ❌ Éliminées: {len(df_long) - len(df_step1)}")
print(f"   Exemples éliminées: {len(df_long[(df_long['bid_price'] == 0) | (df_long['ask_price'] == 0)])} avec bid=0 ou ask=0")

# 2. Filtre bid >= min_bid (0.01)
df_step2 = df_step1[df_step1["bid_price"] >= 0.01].copy()
print(f"\n2️⃣  Après filtre bid >= 0.01:")
print(f"   ✅ Conservées: {len(df_step2)} / {len(df_step1)} ({100*len(df_step2)/len(df_step1):.1f}%)")
print(f"   ❌ Éliminées: {len(df_step1) - len(df_step2)}")

# 3. Filtre spread relatif <= 25%
df_step2["mid"] = (df_step2["bid_price"] + df_step2["ask_price"]) / 2.0
df_step2["rel_spread"] = (df_step2["ask_price"] - df_step2["bid_price"]) / df_step2["mid"].replace(0, pd.NA)
df_step3 = df_step2[df_step2["rel_spread"] <= 0.25].copy()
print(f"\n3️⃣  Après filtre spread relatif <= 25%:")
print(f"   ✅ Conservées: {len(df_step3)} / {len(df_step2)} ({100*len(df_step3)/len(df_step2):.1f}%)")
print(f"   ❌ Éliminées: {len(df_step2) - len(df_step3)}")
if len(df_step2) > len(df_step3):
    bad_spread = df_step2[df_step2["rel_spread"] > 0.25]
    print(f"   Exemple de spread élevé: {bad_spread[['strike', 'type', 'bid_price', 'ask_price', 'rel_spread']].head(3).to_string()}")

# 4. Filtre liquidité (OI >= 10 OU volume >= 1)
df_step4 = df_step3[(df_step3["open_interest"] >= 10) | (df_step3["volume"] >= 1)].copy()
print(f"\n4️⃣  Après filtre liquidité (OI >= 10 OU volume >= 1):")
print(f"   ✅ Conservées: {len(df_step4)} / {len(df_step3)} ({100*len(df_step4)/len(df_step3):.1f}%)")
print(f"   ❌ Éliminées: {len(df_step3) - len(df_step4)}")
if len(df_step3) > len(df_step4):
    no_liquidity = df_step3[(df_step3["open_interest"] < 10) & (df_step3["volume"] < 1)]
    print(f"   Exemples sans liquidité: {len(no_liquidity)} options")

# 5. Filtre IV valide
df_step5 = df_step4[df_step4["mark_iv"].notna()].copy()
print(f"\n5️⃣  Après filtre IV non-null:")
print(f"   ✅ Conservées: {len(df_step5)} / {len(df_step4)} ({100*len(df_step5)/len(df_step4):.1f}%)")
print(f"   ❌ Éliminées: {len(df_step4) - len(df_step5)}")

# Normaliser IV (comme dans iv_surface_spx.py)
import numpy as np
iv_series = pd.to_numeric(df_step5["mark_iv"], errors="coerce")
med = np.nanmedian(iv_series.values) if np.isfinite(iv_series).any() else np.nan
if np.isfinite(med) and med < 3.0:
    df_step5["iv_pct"] = iv_series * 100.0
else:
    df_step5["iv_pct"] = iv_series

# Filtre IV entre 0.01% et 300%
df_step6 = df_step5[df_step5["iv_pct"].between(0.01, 300.0)].copy()
print(f"\n6️⃣  Après filtre IV entre 0.01% et 300%:")
print(f"   ✅ Conservées: {len(df_step6)} / {len(df_step5)} ({100*len(df_step6)/len(df_step5):.1f}%)")
print(f"   ❌ Éliminées: {len(df_step5) - len(df_step6)}")
if len(df_step5) > len(df_step6):
    bad_iv = df_step5[~df_step5["iv_pct"].between(0.01, 300.0)]
    print(f"   Exemples IV hors plage: {bad_iv[['strike', 'type', 'mark_iv', 'iv_pct']].head(3).to_string()}")

print(f"\n" + "=" * 70)
print(f"\n📈 RÉSUMÉ:")
print(f"   Options initiales avec T > 1 an: {len(df_long)}")
print(f"   Options finales après tous les filtres: {len(df_step6)}")
print(f"   Taux de conservation: {100*len(df_step6)/len(df_long):.1f}%")

if len(df_step6) > 0:
    print(f"\n✅ Exemples d'options conservées:")
    print(df_step6[['expiry_date', 'strike', 'type', 'T', 'bid_price', 'ask_price', 'open_interest', 'volume', 'mark_iv']].head(10).to_string())
else:
    print(f"\n⚠️  AUCUNE option avec T > 1 an ne passe tous les filtres!")
    print(f"\n💡 Solution: Ajuster les paramètres de SurfaceConfig pour être moins restrictif:")
    print(f"   - Réduire min_bid (actuellement 0.01)")
    print(f"   - Augmenter max_rel_spread (actuellement 0.25)")
    print(f"   - Réduire min_oi (actuellement 10)")

