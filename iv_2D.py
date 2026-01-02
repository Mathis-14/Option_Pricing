import argparse
import pandas as pd
import plotly.graph_objects as go
import glob
import os
import sys
from datetime import datetime

# Import existing classes
# Assuming this script is in the same directory as iv_surface_spx.py
from iv_surface_spx import SPXIVSurface, SurfaceConfig

def get_latest_data(data_dir="data"):
    """Finds the latest CSV file in the data directory."""
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        return None
    # Sort by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def plot_smile(surface: SPXIVSurface, 
               max_moneyness: float = 0.3,
               min_volume: int = 10,
               min_oi: int = 100,
               max_spread_pct: float = 0.20,
               iv_min: float = 5.0,
               iv_max: float = 100.0,
               outlier_std: float = 2.0,
               smooth: bool = True,
               diagnostics: bool = True,
               detailed_diagnostics: bool = False,
               atm_blend_threshold: float = 0.10,
               save_plot: bool = True,
               output_dir: str = "plot/vol/vol_2D",
               maturities: list = None,
               show_data_points: bool = False,
               show_true_data_points: bool = False):
    """Plots production-quality Volatility Smile curves with smooth ATM transition.
    
    Strategy to eliminate ATM discontinuity:
    1. Far OTM puts (ln(K/F) < -atm_blend_threshold): Use put IV only
    2. ATM region (|ln(K/F)| <= atm_blend_threshold): Average put & call IVs
    3. Far OTM calls (ln(K/F) > atm_blend_threshold): Use call IV only
    
    Args:
        surface: SPXIVSurface instance with processed options data
        max_moneyness: Maximum absolute log-moneyness (default 0.3)
        min_volume: Minimum volume required (default 10)
        min_oi: Minimum open interest required (default 100)
        max_spread_pct: Maximum bid-ask spread as % of mid (default 0.20 = 20%)
        iv_min: Minimum IV threshold in % (default 5.0)
        iv_max: Maximum IV threshold in % (default 100.0)
        outlier_std: Number of standard deviations for outlier removal (default 2.0)
        smooth: Whether to apply cubic spline smoothing (default True)
        diagnostics: Whether to print filtering statistics (default True)
        detailed_diagnostics: Print sample data and parity checks (default False)
        atm_blend_threshold: ATM region for blending put/call IVs (default 0.10)
        save_plot: Whether to save the plot (default True)
        output_dir: Directory to save plots (default "plot/vol/vol_2D")
        maturities: List of maturities to plot (e.g., ["2D", "30D"]). Default None = all
        show_data_points: Whether to show blended data point markers (default False)
        show_true_data_points: Whether to show raw option data points (default False)
    """
    import numpy as np
    from scipy.interpolate import CubicSpline
    import os
    from datetime import datetime
    
    df = surface.df.copy()
    
    if diagnostics:
        print("\n" + "="*80)
        print("VOLATILITY SMILE DIAGNOSTICS (ATM BLENDING STRATEGY)")
        print("="*80)
    
    # Initial count
    initial_count = len(df)
    
    # 1. MONEYNESS FILTER
    df = df[df['x'].abs() <= max_moneyness]
    if diagnostics:
        print(f"\n1. Moneyness filter (|ln(K/F)| ≤ {max_moneyness:.2f}):")
        print(f"   Kept {len(df):,} / {initial_count:,} options")
    
    # 2. VOLUME & OPEN INTEREST FILTER
    count_before = len(df)
    df = df[(df['volume'] >= min_volume) | (df['open_interest'] >= min_oi)]
    if diagnostics:
        print(f"\n2. Liquidity filter (volume ≥ {min_volume} OR OI ≥ {min_oi}):")
        print(f"   Kept {len(df):,} / {count_before:,} options")
    
    # 3. BID-ASK SPREAD FILTER
    count_before = len(df)
    df['spread_pct'] = (df['ask'] - df['bid']) / df['mid']
    df = df[df['spread_pct'] <= max_spread_pct]
    if diagnostics:
        print(f"\n3. Spread filter (spread ≤ {max_spread_pct*100:.0f}% of mid):")
        print(f"   Kept {len(df):,} / {count_before:,} options")
    
    # 4. IV RANGE FILTER
    count_before = len(df)
    df = df[(df['iv_pct'] >= iv_min) & (df['iv_pct'] <= iv_max)]
    if diagnostics:
        print(f"\n4. IV range filter ({iv_min:.0f}% ≤ IV ≤ {iv_max:.0f}%):")
        print(f"   Kept {len(df):,} / {count_before:,} options")
    
    # 5. VALIDATE LOG-MONEYNESS
    count_before = len(df)
    df = df[np.isfinite(df['x'])]
    if diagnostics and count_before > len(df):
        print(f"\n5. Removed {count_before - len(df)} options with invalid log-moneyness")
    
    if df.empty:
        print("\n❌ ERROR: No data remaining after filtering!")
        return
    
    # Get unique expiries
    expiries = sorted(df['expiry_date'].unique())
    
    # MATURITY SELECTION
    if maturities is not None and len(maturities) > 0:
        # Parse user-specified maturities (e.g., ["2D", "30D", "90D"])
        selected_expiries = []
        
        for mat_spec in maturities:
            # Parse maturity specification (e.g., "2D" -> 2 days)
            if isinstance(mat_spec, str) and mat_spec.upper().endswith('D'):
                try:
                    target_days = int(mat_spec[:-1])
                except:
                    if diagnostics:
                        print(f"⚠️  Invalid maturity format: {mat_spec}, skipping")
                    continue
                
                # Find closest maturity by DTE
                closest_expiry = None
                min_diff = float('inf')
                
                for expiry in expiries:
                    expiry_subset = df[df['expiry_date'] == expiry]
                    if not expiry_subset.empty:
                        T_val = expiry_subset['T'].iloc[0]
                        dte = int(T_val * 365)
                        diff = abs(dte - target_days)
                        
                        if diff < min_diff:
                            min_diff = diff
                            closest_expiry = expiry
                            closest_dte = dte
                
                if closest_expiry is not None:
                    selected_expiries.append(closest_expiry)
                    if diagnostics and closest_dte != target_days:
                        print(f"ℹ️  Requested {mat_spec} → Using {closest_dte}D (closest available)")
            else:
                if diagnostics:
                    print(f"⚠️  Maturity must be in format 'xD' (e.g., '2D', '30D'), got: {mat_spec}")
        
        if not selected_expiries:
            if diagnostics:
                print("⚠️  No valid maturities selected, using all available")
            selected_expiries = expiries
    else:
        # Select representative expiries (max 12)
        selected_expiries = expiries
        if len(expiries) > 12:
            step = max(1, len(expiries) // 10)
            selected_expiries = expiries[::step][:12]
    
    if diagnostics:
        print(f"\n6. Selected {len(selected_expiries)} maturities (from {len(expiries)} available)")
        print(f"\n7. ATM blending strategy: Average put/call IVs within |ln(K/F)| ≤ {atm_blend_threshold:.3f}")
        print("="*80)
    
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#ff33cc',
              '#33ff57', '#3357ff']
    
    for idx, expiry in enumerate(selected_expiries):
        subset = df[df['expiry_date'] == expiry].copy()
        
        if subset.empty:
            continue
        
        T_val = subset['T'].iloc[0]
        DTE = int(T_val * 365)
        F_val = subset['F'].iloc[0]
        S_val = subset['S'].iloc[0]
        
        # FORWARD PRICE DIAGNOSTICS
        if diagnostics:
            forward_premium_pct = ((F_val - S_val) / S_val) * 100
            r_used = surface.cfg.r
            
            print(f"\n{'─'*80}")
            print(f"Maturity: {pd.Timestamp(expiry).date()} ({DTE}D)")
            print(f"{'─'*80}")
            print(f"  Spot (S): {S_val:,.2f} | Forward (F): {F_val:,.2f} | Premium: {forward_premium_pct:+.3f}%")
        
        # SEPARATE PUTS AND CALLS
        puts = subset[subset['type'] == 'P'].copy()
        calls = subset[subset['type'] == 'C'].copy()
        
        if diagnostics:
            print(f"  Total: {len(subset)} ({len(puts)} puts, {len(calls)} calls)")
        
        # OUTLIER REMOVAL (do this before interpolation)
        count_before = len(subset)
        median_iv = subset['iv_pct'].median()
        std_iv = subset['iv_pct'].std()
        
        if std_iv > 0:
            subset = subset[np.abs(subset['iv_pct'] - median_iv) <= outlier_std * std_iv]
            puts = subset[subset['type'] == 'P'].copy()
            calls = subset[subset['type'] == 'C'].copy()
        
        if diagnostics and count_before > len(subset):
            print(f"  Outliers removed: {count_before - len(subset)}")
        
        if len(subset) < 5:
            if diagnostics:
                print(f"  ⚠️  Too few points (< 5), skipping")
            continue
        
        # INTERPOLATION-BASED BLENDING STRATEGY
        # Instead of blending at common strikes, interpolate both curves then blend
        
        # Aggregate puts and calls by moneyness
        if not puts.empty:
            puts_agg = puts.groupby('x')['iv_pct'].median().reset_index()
            puts_agg = puts_agg.sort_values('x')
        else:
            puts_agg = pd.DataFrame(columns=['x', 'iv_pct'])
        
        if not calls.empty:
            calls_agg = calls.groupby('x')['iv_pct'].median().reset_index()
            calls_agg = calls_agg.sort_values('x')
        else:
            calls_agg = pd.DataFrame(columns=['x', 'iv_pct'])
        
        if len(puts_agg) < 3 and len(calls_agg) < 3:
            if diagnostics:
                print(f"  ⚠️  Insufficient data for interpolation")
            continue
        
        # Create common moneyness grid
        x_min = min(puts_agg['x'].min() if not puts_agg.empty else 0,
                    calls_agg['x'].min() if not calls_agg.empty else 0)
        x_max = max(puts_agg['x'].max() if not puts_agg.empty else 0,
                    calls_agg['x'].max() if not calls_agg.empty else 0)
        
        x_grid = np.linspace(x_min, x_max, 100)
        
        # Interpolate put and call curves WITH EXTRAPOLATION
        # This ensures both curves have values everywhere, enabling smooth blending
        from scipy.interpolate import interp1d
        
        put_interp = None
        call_interp = None
        
        if not puts_agg.empty and len(puts_agg) >= 2:
            # Use linear extrapolation beyond data range
            put_interp = interp1d(puts_agg['x'].values, puts_agg['iv_pct'].values,
                                 kind='linear', fill_value='extrapolate', bounds_error=False)
        
        if not calls_agg.empty and len(calls_agg) >= 2:
            call_interp = interp1d(calls_agg['x'].values, calls_agg['iv_pct'].values,
                                  kind='linear', fill_value='extrapolate', bounds_error=False)
        
        if put_interp is None and call_interp is None:
            if diagnostics:
                print(f"  ⚠️  No valid interpolators")
            continue
        
        # Generate interpolated values
        iv_blended = np.full_like(x_grid, np.nan)
        source_type = np.array([''] * len(x_grid), dtype=object)
        
        for i, x_val in enumerate(x_grid):
            put_iv = put_interp(x_val) if put_interp is not None else None
            call_iv = call_interp(x_val) if call_interp is not None else None
            
            # SMOOTH BLENDING EVERYWHERE
            if put_iv is not None and call_iv is not None:
                # Both available: blend based on moneyness
                if x_val < -atm_blend_threshold:
                    # Far OTM put: 100% put
                    iv_blended[i] = put_iv
                    source_type[i] = 'P'
                elif x_val > atm_blend_threshold:
                    # Far OTM call: 100% call
                    iv_blended[i] = call_iv
                    source_type[i] = 'C'
                else:
                    # ATM region: smooth blend
                    # weight goes from 1.0 (left) to 0.0 (right)
                    weight = (atm_blend_threshold - x_val) / (2 * atm_blend_threshold)
                    weight = np.clip(weight, 0, 1)
                    iv_blended[i] = weight * put_iv + (1 - weight) * call_iv
                    source_type[i] = 'B'
            elif put_iv is not None:
                # Only put available
                iv_blended[i] = put_iv
                source_type[i] = 'P'
            elif call_iv is not None:
                # Only call available
                iv_blended[i] = call_iv
                source_type[i] = 'C'
        
        # Remove NaN and clip to reasonable range
        valid_mask = np.isfinite(iv_blended)
        x_grid = x_grid[valid_mask]
        iv_blended = iv_blended[valid_mask]
        source_type = source_type[valid_mask]
        
        # Clip extreme extrapolated values
        iv_blended = np.clip(iv_blended, iv_min, iv_max)
        
        if len(x_grid) < 3:
            if diagnostics:
                print(f"  ⚠️  Too few valid points after blending")
            continue
        
        # DATA VALIDATION
        if len(iv_blended) > 1:
            iv_diff = np.abs(np.diff(iv_blended))
            max_jump = np.max(iv_diff) if len(iv_diff) > 0 else 0
            if diagnostics:
                print(f"  Max IV jump after blending: {max_jump:.2f}%")
        
        atm_mask = np.abs(x_grid) < 0.02
        if diagnostics and np.any(atm_mask):
            print(f"  ATM IV (|ln(K/F)|<0.02): {iv_blended[atm_mask].mean():.2f}%")
        
        color = colors[idx % len(colors)]
        
        # PLOT RAW TRUE DATA POINTS (if requested)
        if show_true_data_points:
            # Plot the actual raw option data points
            subset_puts = subset[subset['type'] == 'P']
            subset_calls = subset[subset['type'] == 'C']
            
            if not subset_puts.empty:
                fig.add_trace(go.Scatter(
                    x=subset_puts['x'],
                    y=subset_puts['iv_pct'],
                    mode='markers',
                    marker=dict(size=4, color=color, symbol='x', opacity=0.5),
                    name=f"{DTE}D raw puts",
                    legendgroup=f"group{idx}",
                    showlegend=False,
                    hovertemplate=f'{DTE}D PUT<br>Strike: %{{customdata[0]:.0f}}<br>ln(K/F): %{{x:.3f}}<br>IV: %{{y:.2f}}%<extra></extra>',
                    customdata=subset_puts[['strike']].values
                ))
            
            if not subset_calls.empty:
                fig.add_trace(go.Scatter(
                    x=subset_calls['x'],
                    y=subset_calls['iv_pct'],
                    mode='markers',
                    marker=dict(size=4, color=color, symbol='cross', opacity=0.5),
                    name=f"{DTE}D raw calls",
                    legendgroup=f"group{idx}",
                    showlegend=False,
                    hovertemplate=f'{DTE}D CALL<br>Strike: %{{customdata[0]:.0f}}<br>ln(K/F): %{{x:.3f}}<br>IV: %{{y:.2f}}%<extra></extra>',
                    customdata=subset_calls[['strike']].values
                ))
        
        # PLOT DATA POINTS (if requested)
        if show_data_points:
            # Plot subset of points to avoid clutter
            marker_step = max(1, len(x_grid) // 20)
            fig.add_trace(go.Scatter(
                x=x_grid[::marker_step],
                y=iv_blended[::marker_step],
                mode='markers',
                marker=dict(size=5, color=color, symbol='circle', opacity=0.6,
                           line=dict(width=0.5, color='white')),
                name=f"{DTE}D data",
                legendgroup=f"group{idx}",
                showlegend=False,
                hovertemplate=f'{DTE}D<br>ln(K/F): %{{x:.3f}}<br>IV: %{{y:.2f}}%<extra></extra>'
            ))
        
        # PLOT SMOOTH LINE ONLY (no markers for cleaner look)
        # Cubic spline on the blended curve
        if smooth and len(x_grid) >= 4:
            try:
                cs = CubicSpline(x_grid, iv_blended)
                x_smooth = np.linspace(x_grid.min(), x_grid.max(), 200)
                y_smooth = cs(x_smooth)
                y_smooth = np.clip(y_smooth, iv_min, iv_max)
                
                fig.add_trace(go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines',
                    line=dict(color=color, width=3),
                    name=f"{DTE}D",
                    legendgroup=f"group{idx}",
                    showlegend=True,
                    hovertemplate=f'{DTE}D<br>ln(K/F): %{{x:.3f}}<br>IV: %{{y:.2f}}%<extra></extra>'
                ))
            except Exception as e:
                if diagnostics:
                    print(f"  ⚠️  Spline failed: {e}")
    
    if diagnostics:
        print("="*80 + "\n")
    
    # ATM REFERENCE LINE
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.4)", line_width=2,
                  annotation_text="ATM", annotation_position="top")
    
    fig.update_layout(
        title=dict(
            text=f"Implied Volatility Smile (Extrapolated Blending)<br>"
                 f"<sub>Smooth curves via extrapolation + cubic spline | |ln(K/F)| ≤ {max_moneyness:.2f}</sub>",
            font=dict(size=20)
        ),
        xaxis_title="Log-Moneyness ln(K/F)",
        yaxis_title="Implied Volatility (%)",
        template="plotly_dark",
        legend=dict(title="Maturity (DTE)", font=dict(size=12)),
        hovermode='closest',
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='rgba(255,255,255,0.2)',
                   gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=700,
        plot_bgcolor='rgba(15,15,25,1)',
        paper_bgcolor='rgba(10,10,15,1)'
    )
    
    # SAVE PLOT
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"smile_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        fig.write_image(filepath, width=1920, height=800, scale=2)
        if diagnostics:
            print(f"📊 Plot saved to: {filepath}\n")
    
    fig.show()


def plot_vol_term_structure(surface: SPXIVSurface):
    """Plots the Volatility Term Structure (IV vs Time to Maturity) for ATM options."""
    df = surface.df
    
    # Filter for At-The-Money (ATM) options
    # We define ATM as log-moneyness 'x' being close to 0.
    # Adjust tolerance as needed.
    tolerance = 0.02
    atm_df = df[df['x'].abs() < tolerance].copy()
    
    if atm_df.empty:
        print("No ATM data found (abs(x) < 0.02). widening search...")
        atm_df = df[df['x'].abs() < 0.05].copy()
    
    if atm_df.empty:
        print("Still no ATM data found.")
        return

    # Sort by T
    atm_df = atm_df.sort_values('T')

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=atm_df['T'],
        y=atm_df['iv_pct'],
        mode='markers',
        marker=dict(size=6, color='cyan'),
        name="ATM Quotes"
    ))
    
    # Optional: Add a smoothed line (e.g. grouped median per expiry)
    term_structure = atm_df.groupby('T')['iv_pct'].median().reset_index()
    fig.add_trace(go.Scatter(
        x=term_structure['T'],
        y=term_structure['iv_pct'],
        mode='lines',
        line=dict(color='orange', width=2),
        name="Median ATM Term Structure"
    ))

    fig.update_layout(
        title="Volatility Term Structure (ATM IV vs Time to Maturity)",
        xaxis_title="Time to Maturity (Years)",
        yaxis_title="Implied Volatility (%)",
        template="plotly_dark"
    )
    fig.show()

def main():
    parser = argparse.ArgumentParser(description="Plot Implied Volatility 2D Curves")
    parser.add_argument("type", nargs="?", choices=["smile", "vol_term_structure"], default="smile",
                        help="Type of plot to generate: 'smile' (default) or 'vol_term_structure'")
    parser.add_argument("--file", type=str, help="Path to CSV data file", default=None)
    
    args = parser.parse_args()

    # Inform user about default valid if args.type was inferred
    if len(sys.argv) == 1:
        print("No plot type specified. Defaulting to 'smile'.")
        print("Usage: python iv_2D.py [smile|vol_term_structure]")
    
    # Resolve file path
    csv_path = args.file
    if csv_path is None:
        csv_path = get_latest_data()
        if csv_path is None:
            print("Error: No CSV file found in data/ directory and none provided.")
            sys.exit(1)
            
    print(f"Loading data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
        
    # Prepare Surface
    # We can use default config or customize if needed
    cfg = SurfaceConfig(
        min_T=1/365,      # Filter very short dated
        max_T=3.0,        # Reasonable max T
        min_oi=10,
        min_volume=1
    )
    
    print("Processing data...")
    surface = SPXIVSurface(df, cfg)
    print(f"Loaded {len(surface.df)} valid options.")

    if args.type == "smile":
        plot_smile(surface)
    elif args.type == "vol_term_structure":
        plot_vol_term_structure(surface)

if __name__ == "__main__":
    main()
