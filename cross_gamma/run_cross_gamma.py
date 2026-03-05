"""
run_cross_gamma.py — Main orchestration script for cross gamma analysis

Loads market data (from Refinitiv import), computes the cross gamma surface,
and generates both 3D surface and heatmap visualizations.

Usage:
    python run_cross_gamma.py                    # uses latest imported data
    python run_cross_gamma.py --data basket_data_20260220_235000.json
    python run_cross_gamma.py --grid-size 15     # finer grid (slower)
    python run_cross_gamma.py --paths 500000     # more MC paths (more accurate)
"""

import json
import os
import argparse
import numpy as np
from pathlib import Path

from cross_gamma_model import BasketCallPricer
from plot_cross_gamma import plot_3d_surface, plot_heatmap


# =============================================================================
# CONFIGURATION
# =============================================================================

# Grid range: how far from current spots to explore (as % of spot)
GRID_RANGE_PCT = 0.30      # ±30% of current spot price

# Default grid size (number of points per axis)
DEFAULT_GRID_SIZE = 12      # 12×12 = 144 grid points → 576 MC evaluations

# Default time to maturity (3 months)
DEFAULT_T = 0.25

# Risk-free rate (approximate, or import from Refinitiv)
DEFAULT_R = 0.045


# =============================================================================
# DATA LOADING
# =============================================================================

def find_latest_data(data_dir: str) -> str:
    """
    Find the most recent basket data JSON file in the data directory.
    
    Parameters
    ----------
    data_dir : str
        Path to the refinitiv_data directory.
        
    Returns
    -------
    str
        Path to the latest JSON file.
        
    Raises
    ------
    FileNotFoundError
        If no basket data files are found.
    """
    json_files = sorted(Path(data_dir).glob('basket_data_*.json'), reverse=True)
    
    if not json_files:
        raise FileNotFoundError(
            f"No basket data files found in {data_dir}.\n"
            f"Run 'python import_basket_data.py' first to import market data."
        )
    
    return str(json_files[0])


def load_market_data(file_path: str) -> dict:
    """
    Load market data from a JSON file.
    
    Parameters
    ----------
    file_path : str
        Path to the basket data JSON file.
        
    Returns
    -------
    dict
        Market data dictionary.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"📂 Loaded data from: {os.path.basename(file_path)}")
    print(f"   Timestamp: {data['metadata']['timestamp']}")
    
    return data


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_analysis(
    data: dict,
    grid_size: int = DEFAULT_GRID_SIZE,
    n_paths: int = 200_000,
    T: float = DEFAULT_T,
    r: float = DEFAULT_R,
    show_plots: bool = True
):
    """
    Run the complete cross gamma analysis pipeline.
    
    Steps:
    1. Extract parameters from market data
    2. Create the BasketCallPricer
    3. Compute the cross gamma surface
    4. Generate both plots
    
    Parameters
    ----------
    data : dict
        Market data from load_market_data().
    grid_size : int
        Number of points per axis in the surface grid.
    n_paths : int
        Number of Monte Carlo paths.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    show_plots : bool
        Whether to display plots interactively.
    """
    # ─── Extract parameters ─────────────────────────────────────────────
    tickers = data['metadata']['tickers']
    S1 = data['spot_prices'][tickers[0]]
    S2 = data['spot_prices'][tickers[1]]
    sigma1 = data['volatilities']['atm_iv_1']
    sigma2 = data['volatilities']['atm_iv_2']
    rho = data['correlation']
    w1, w2 = data['basket']['weights']
    K = data['basket']['strike_atm']
    
    print("\n" + "=" * 60)
    print("  CROSS GAMMA ANALYSIS")
    print("=" * 60)
    print(f"  {tickers[0]}: S₁ = ${S1:.2f}, σ₁ = {sigma1*100:.1f}%")
    print(f"  {tickers[1]}: S₂ = ${S2:.2f}, σ₂ = {sigma2*100:.1f}%")
    print(f"  Correlation ρ = {rho:.4f}")
    print(f"  Basket (50/50) = ${w1*S1 + w2*S2:.2f}")
    print(f"  Strike K = ${K:.2f}")
    print(f"  T = {T} years, r = {r*100:.1f}%")
    print(f"  MC paths = {n_paths:,}, Grid = {grid_size}×{grid_size}")
    print("=" * 60)
    
    # ─── Create pricer ──────────────────────────────────────────────────
    pricer = BasketCallPricer(
        S1=S1, S2=S2, K=K, T=T, r=r,
        sigma1=sigma1, sigma2=sigma2, rho=rho,
        w1=w1, w2=w2,
        n_paths=n_paths
    )
    
    # ─── Print option summary ───────────────────────────────────────────
    print(pricer.summary())
    
    # ─── Build the evaluation grid ──────────────────────────────────────
    S1_lo = S1 * (1 - GRID_RANGE_PCT)
    S1_hi = S1 * (1 + GRID_RANGE_PCT)
    S2_lo = S2 * (1 - GRID_RANGE_PCT)
    S2_hi = S2 * (1 + GRID_RANGE_PCT)
    
    S1_range = np.linspace(S1_lo, S1_hi, grid_size)
    S2_range = np.linspace(S2_lo, S2_hi, grid_size)
    
    print(f"\n  Grid: S₁ ∈ [${S1_lo:.0f}, ${S1_hi:.0f}]")
    print(f"         S₂ ∈ [${S2_lo:.0f}, ${S2_hi:.0f}]")
    
    # ─── Compute cross gamma surface ────────────────────────────────────
    gamma_surface = pricer.cross_gamma_surface(S1_range, S2_range)
    
    # ─── Print surface statistics ───────────────────────────────────────
    print(f"\n  Surface Statistics:")
    print(f"    max(Γ_cross)  = {np.max(gamma_surface):.6f}")
    print(f"    min(Γ_cross)  = {np.min(gamma_surface):.6f}")
    print(f"    mean(Γ_cross) = {np.mean(gamma_surface):.6f}")
    
    # ─── Generate plots ─────────────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(__file__), 'plots')
    
    print("\n📊 Generating plots...\n")
    
    plot_3d_surface(
        S1_range, S2_range, gamma_surface,
        spot_S1=S1, spot_S2=S2, K=K,
        output_dir=output_dir, show=show_plots
    )
    
    plot_heatmap(
        S1_range, S2_range, gamma_surface,
        spot_S1=S1, spot_S2=S2, K=K,
        output_dir=output_dir, show=show_plots
    )
    
    # ─── Financial interpretation ───────────────────────────────────────
    i_s1 = np.argmin(np.abs(S1_range - S1))
    i_s2 = np.argmin(np.abs(S2_range - S2))
    atm_gamma = gamma_surface[i_s2, i_s1]
    
    print("\n" + "=" * 60)
    print("  FINANCIAL INTERPRETATION")
    print("=" * 60)
    print(f"""
  Cross gamma at current spots: Γ_cross = {atm_gamma:.6f}

  What this means:
  • If {tickers[1]} moves by $1, the delta of the basket call with 
    respect to {tickers[0]} changes by approximately {atm_gamma:.6f}.
  
  • Positive cross gamma (typical for correlated stocks near ATM)
    means: when one stock rises, the delta w.r.t. the other also 
    increases — reinforcing the directional exposure.
  
  • Cross gamma is highest near ATM because that's where the option
    transitions between worthless and in-the-money, and both stocks
    contribute to that transition.
  
  • For hedging: if cross gamma is large, you must re-hedge BOTH
    stocks simultaneously when EITHER moves. Ignoring cross gamma
    leads to under-hedging of interaction risk.
    """)
    print("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Parse arguments and run the cross gamma analysis."""
    
    parser = argparse.ArgumentParser(
        description="Compute and visualize cross gamma for a basket call option."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Specific data file name (e.g., basket_data_20260220_235000.json)"
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Grid points per axis (default: {DEFAULT_GRID_SIZE})"
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=200_000,
        help="Number of Monte Carlo paths (default: 200,000)"
    )
    parser.add_argument(
        "--T",
        type=float,
        default=DEFAULT_T,
        help=f"Time to maturity in years (default: {DEFAULT_T})"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plots without displaying them"
    )
    args = parser.parse_args()
    
    # Locate data file
    data_dir = os.path.join(os.path.dirname(__file__), 'refinitiv_data')
    
    if args.data:
        file_path = os.path.join(data_dir, args.data)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
    else:
        file_path = find_latest_data(data_dir)
    
    # Load and run
    data = load_market_data(file_path)
    run_analysis(
        data,
        grid_size=args.grid_size,
        n_paths=args.paths,
        T=args.T,
        show_plots=not args.no_show
    )


if __name__ == "__main__":
    main()
