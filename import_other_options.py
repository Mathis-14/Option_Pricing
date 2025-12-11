"""
import_sp500_options_yahoo.py — Import S&P 500 options data from Yahoo Finance

Ce module récupère les options sur un sous-jacent Yahoo Finance (par défaut ^SPX),
filtre par maturité, et exporte un CSV dans le dossier "data" à la racine du projet.

Comportement :
- S'il existe déjà un CSV correspondant dans `data/`, il est chargé directement.
- Sinon, les données sont téléchargées depuis Yahoo Finance puis sauvegardées.

Utilisation :
    from import_other_options import import_sp500_options_data

    # Importer les options avec expiration entre 2024-11-01 et 2024-12-31
    df = import_sp500_options_data(
        start_date="2024-11-01",
        end_date="2024-12-31",
        ticker="^SPX",     # ou "SPY" si tu préfères l'ETF
    )

    # df est un DataFrame pandas avec toutes les options sélectionnées
"""

from __future__ import annotations

import os
from datetime import datetime, date
from typing import Union, Optional

import pandas as pd
import yfinance as yf


def _resolve_output_dir(output_dir: str) -> str:
    """
    Résout le chemin du dossier data de manière similaire à ton module Deribit.
    Si `output_dir` est relatif, on essaie de remonter jusqu'au dossier 'Option_Pricing'
    puis on ajoute `output_dir` derrière, sinon on utilise le répertoire du script.
    """
    if os.path.isabs(output_dir):
        final_dir = output_dir
    else:
        # 1) Essayer d'utiliser le répertoire du fichier courant
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # __file__ n'existe pas (par ex. dans un notebook)
            script_dir = os.getcwd()

        # 2) Si on est dans ou sous "Option_Pricing", on remonte jusqu'à lui
        if "Option_Pricing" in script_dir:
            parts = script_dir.split(os.sep)
            if "Option_Pricing" in parts:
                idx = parts.index("Option_Pricing")
                project_root = os.sep.join(parts[: idx + 1])
            else:
                project_root = script_dir
        else:
            # Sinon, on essaie de remonter quelques niveaux pour trouver "Option_Pricing"
            current = script_dir
            project_root = script_dir
            for _ in range(10):
                if "Option_Pricing" in os.path.basename(current):
                    project_root = current
                    break
                parent = os.path.dirname(current)
                if parent == current:
                    break
                current = parent

        final_dir = os.path.join(project_root, output_dir)

    final_dir = os.path.normpath(final_dir)

    if "Option_Pricing" not in final_dir:
        print(f"⚠️  Warning: Data directory path doesn't contain 'Option_Pricing': {final_dir}")
        print("    Expected path should look like: .../Option_Pricing/data/")

    print(f"📁 Data will be saved to / loaded from: {final_dir}")
    return final_dir


def _ensure_date(d: Union[str, datetime, date]) -> date:
    """Convertit str/datetime/date en objet date."""
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        # Format attendu : YYYY-MM-DD
        return datetime.strptime(d, "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date type: {type(d)}")


def _find_existing_csv(
    output_dir: str,
    ticker: str,
    filename: Optional[str] = None,
) -> Optional[str]:
    """
    Cherche un CSV existant pour ce ticker dans `output_dir`.

    - Si `filename` est fourni, on vérifie directement {filename}.csv.
    - Sinon, on cherche les fichiers auto-générés :
      sp500_options_{TICKER_CLEAN}_YYYYMMDD_HHMMSS.csv
    """
    if not os.path.exists(output_dir):
        return None

    # Cas 1 : l'utilisateur a donné un nom de fichier spécifique
    if filename is not None:
        path = os.path.join(output_dir, f"{filename}.csv")
        if os.path.exists(path):
            return path
        return None

    # Cas 2 : nom auto-généré
    ticker_clean = ticker.replace("^", "").upper()
    prefix = f"sp500_options_{ticker_clean}_"

    candidates = [
        f for f in os.listdir(output_dir)
        if f.startswith(prefix) and f.endswith(".csv")
    ]
    if not candidates:
        return None

    # On prend le plus récent (ordre lexicographique vu que timestamp dans le nom)
    latest = sorted(candidates)[-1]
    return os.path.join(output_dir, latest)


def import_sp500_options_data(
    start_date: Union[str, datetime, date],
    end_date: Union[str, datetime, date],
    ticker: str = "^SPX",
    output_dir: str = "data",
    filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Importe les options S&P 500 (ou autre sous-jacent Yahoo) via yfinance
    pour un intervalle de maturité donné, sauvegarde un CSV dans `data/`
    et renvoie un DataFrame unique.

    Comportement :
    - Si un CSV existe déjà dans `output_dir` pour ce ticker (et ce filename
      si fourni), il est chargé et renvoyé.
    - Sinon, les données sont téléchargées, sauvegardées, puis renvoyées.

    Paramètres
    ----------
    start_date : str, datetime ou date
        Date de début (incluse) du filtre de maturité, au format "YYYY-MM-DD"
        si string.
    end_date : str, datetime ou date
        Date de fin (incluse) du filtre de maturité.
    ticker : str
        Ticker Yahoo Finance de l'underlying. Exemples :
        - "^SPX" pour l'indice S&P 500
        - "^GSPC" (autre ticker S&P 500)
        - "SPY" pour l'ETF S&P 500
    output_dir : str
        Dossier de sauvegarde (par défaut "data", à la racine du projet).
    filename : str, optionnel
        Nom de fichier (sans extension). Si None, un nom auto-généré sera utilisé.

    Retour
    ------
    pd.DataFrame
        Un DataFrame avec les colonnes harmonisées avec ton module Deribit :
        - expiry_str      : date d'expiration au format YYYY-MM-DD
        - expiry_date     : datetime de l'expiration
        - strike          : prix d'exercice
        - type            : "C" (call) ou "P" (put)
        - mark_iv         : implied volatility (yfinance -> impliedVolatility)
        - mark_price      : dernier prix (lastPrice)
        - underlying_price: dernier cours du sous-jacent (même valeur pour toutes les lignes)
        - open_interest   : open interest
        - bid_price       : bid
        - ask_price       : ask
        - best_bid_price  : = bid_price
        - best_ask_price  : = ask_price
        - volume          : volume
        - instrument_name : ticker-expiry-strike-type (synthetique)
    """
    # Normaliser les dates
    start_date = _ensure_date(start_date)
    end_date = _ensure_date(end_date)

    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    # Résoudre le chemin du dossier data
    output_dir = _resolve_output_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 🔎 1) Vérifier s'il existe déjà un CSV pour ce ticker
    existing_csv = _find_existing_csv(output_dir, ticker, filename)
    if existing_csv is not None:
        print(f"📂 Existing CSV found, loading instead of fetching from Yahoo Finance:\n   {existing_csv}")
        df = pd.read_csv(existing_csv)

        # S'assurer que expiry_date est bien en datetime
        if "expiry_date" in df.columns:
            df["expiry_date"] = pd.to_datetime(df["expiry_date"])

        return df

    # 🔄 2) Sinon : téléchargement depuis Yahoo Finance
    print(f"🔎 Fetching options from Yahoo Finance for ticker {ticker}...")
    tk = yf.Ticker(ticker)

    # Liste des maturités disponibles sur Yahoo
    expirations = getattr(tk, "options", [])
    if not expirations:
        print("⚠️  No option expirations available for this ticker.")
        return pd.DataFrame()

    # Récupérer un prix du sous-jacent (close le plus récent)
    try:
        hist = tk.history(period="1d")
        if hist.empty:
            underlying_price = None
        else:
            underlying_price = float(hist["Close"].iloc[-1])
    except Exception as e:
        print(f"⚠️  Could not fetch underlying price: {e}")
        underlying_price = None

    all_rows = []

    for exp_str in expirations:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except ValueError:
            # Format inattendu, on ignore
            continue

        if not (start_date <= exp_date <= end_date):
            continue

        print(f"  ⏱  Fetching option chain for expiry {exp_str}...")
        try:
            chain = tk.option_chain(exp_str)
        except Exception as e:
            print(f"  ⚠️  Failed to fetch chain for {exp_str}: {e}")
            continue

        for opt_df, opt_type in [(chain.calls, "C"), (chain.puts, "P")]:
            if opt_df is None or opt_df.empty:
                continue

            tmp = opt_df.copy()
            tmp["type"] = opt_type
            tmp["expiry_str"] = exp_str
            tmp["expiry_date"] = datetime.strptime(exp_str, "%Y-%m-%d")
            tmp["underlying_ticker"] = ticker
            all_rows.append(tmp)

    if not all_rows:
        print("⚠️  No options found in the requested expiry range.")
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)

    # Renommer les colonnes pour coller à ton schéma Deribit
    rename_map = {
        "impliedVolatility": "mark_iv",
        "lastPrice": "mark_price",
        "bid": "bid_price",
        "ask": "ask_price",
        "openInterest": "open_interest",
    }
    df = df.rename(columns=rename_map)

    # Colonnes calculées / ajoutées
    df["underlying_price"] = underlying_price
    df["best_bid_price"] = df.get("bid_price")
    df["best_ask_price"] = df.get("ask_price")

    # Construire un instrument_name synthétique
    # ex: "^SPX-2024-12-20-5000.0-C"
    df["instrument_name"] = (
        df["underlying_ticker"].astype(str)
        + "-"
        + df["expiry_str"].astype(str)
        + "-"
        + df["strike"].astype(str)
        + "-"
        + df["type"].astype(str)
    )

    # Ordonner les colonnes de manière cohérente
    cols_order = [
        "expiry_str",
        "expiry_date",
        "strike",
        "type",
        "mark_iv",
        "mark_price",
        "underlying_price",
        "open_interest",
        "bid_price",
        "ask_price",
        "best_bid_price",
        "best_ask_price",
        "volume",
        "instrument_name",
        "contractSymbol",
        "underlying_ticker",
    ]
    # Garder uniquement les colonnes qui existent
    cols_order = [c for c in cols_order if c in df.columns]
    df = df[cols_order].sort_values(["expiry_date", "strike", "type"]).reset_index(drop=True)

    # Sauvegarde CSV
    if filename is None:
        # Petit nom de fichier propre : sp500_options_SPX_YYYYMMDD_HHMMSS.csv
        ticker_clean = ticker.replace("^", "").upper()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"sp500_options_{ticker_clean}_{timestamp}"
    else:
        filename_prefix = filename

    csv_path = os.path.join(output_dir, f"{filename_prefix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Exported {len(df)} options to {csv_path}")

    return df


if __name__ == "__main__":
    from datetime import timedelta

    # Exemple : toutes les options qui expirent dans les 6 prochains mois
    end = datetime.now().date() + timedelta(days=180)
    start = datetime.now().date()

    df_spx = import_sp500_options_data(
        start_date=start,
        end_date=end,
        ticker="^SPX",  # change en "SPY" si tu préfères l'ETF
        output_dir="data",
    )

    print("\nSummary:")
    if not df_spx.empty:
        print(f"  Total options: {len(df_spx)}")
        print(f"  Expiry range: {df_spx['expiry_date'].min()} to {df_spx['expiry_date'].max()}")
        print(f"  Option types: {df_spx['type'].value_counts().to_dict()}")
    else:
        print("  No options found.")
