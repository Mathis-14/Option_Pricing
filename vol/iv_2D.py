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

def get_latest_data(data_dir="../data"):
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
               output_dir: str = "plot_vol/vol_2D/smile",
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
        output_dir: Directory to save plots (default "plot/vol/vol_2D/smile")
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
        # Make path relative to script directory
        if not os.path.isabs(output_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"smile_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        fig.write_image(filepath, width=1920, height=800, scale=2)
        if diagnostics:
            print(f"📊 Plot saved to: {filepath}\n")
    
    fig.show()


def plot_vol_term_structure(surface: SPXIVSurface,
                            min_volume: int = 10,
                            min_oi: int = 100,
                            max_spread_pct: float = 0.20,
                            iv_min: float = 5.0,
                            iv_max: float = 100.0,
                            outlier_std: float = 2.0,
                            atm_tolerance: float = 0.05,
                            atm_blend_threshold: float = 0.02,
                            smooth: bool = True,
                            diagnostics: bool = True,
                            save_plot: bool = True,
                            output_dir: str = "plot_vol/vol_2D/term_structure",
                            show_raw_points: bool = True):
    """Plots production-quality Volatility Term Structure (ATM IV vs Time to Maturity).
    
    Strategy to compute robust ATM IV per expiry:
    1. Filter data by quality (volume, OI, spread)
    2. For each expiry, extract near-ATM puts and calls
    3. Interpolate to exact ATM (x=0) for both puts and calls
    4. Blend put/call ATM IVs using smooth weighting
    5. Remove outliers and apply cubic spline smoothing
    
    Args:
        surface: SPXIVSurface instance with processed options data
        min_volume: Minimum volume required (default 10)
        min_oi: Minimum open interest required (default 100)
        max_spread_pct: Maximum bid-ask spread as % of mid (default 0.20 = 20%)
        iv_min: Minimum IV threshold in % (default 5.0)
        iv_max: Maximum IV threshold in % (default 100.0)
        outlier_std: Number of standard deviations for outlier removal (default 2.0)
        atm_tolerance: Max |ln(K/F)| to consider for ATM interpolation (default 0.05)
        atm_blend_threshold: ATM region for blending put/call IVs (default 0.02)
        smooth: Whether to apply cubic spline smoothing (default True)
        diagnostics: Whether to print filtering statistics (default True)
        save_plot: Whether to save the plot (default True)
        output_dir: Directory to save plots (default "plot/vol/vol_2D/term_structure")
        show_raw_points: Whether to show individual ATM option quotes (default True)
    """
    import numpy as np
    from scipy.interpolate import CubicSpline, interp1d
    import os
    from datetime import datetime
    
    df = surface.df.copy()
    
    if diagnostics:
        print("\n" + "="*80)
        print("VOLATILITY TERM STRUCTURE DIAGNOSTICS (ROBUST ATM EXTRACTION)")
        print("="*80)
    
    # Initial count
    initial_count = len(df)
    
    # 1. VOLUME & OPEN INTEREST FILTER
    df = df[(df['volume'] >= min_volume) | (df['open_interest'] >= min_oi)]
    if diagnostics:
        print(f"\n1. Liquidity filter (volume ≥ {min_volume} OR OI ≥ {min_oi}):")
        print(f"   Kept {len(df):,} / {initial_count:,} options")
    
    # 2. BID-ASK SPREAD FILTER
    count_before = len(df)
    df['spread_pct'] = (df['ask'] - df['bid']) / df['mid']
    df = df[df['spread_pct'] <= max_spread_pct]
    if diagnostics:
        print(f"\n2. Spread filter (spread ≤ {max_spread_pct*100:.0f}% of mid):")
        print(f"   Kept {len(df):,} / {count_before:,} options")
    
    # 3. IV RANGE FILTER
    count_before = len(df)
    df = df[(df['iv_pct'] >= iv_min) & (df['iv_pct'] <= iv_max)]
    if diagnostics:
        print(f"\n3. IV range filter ({iv_min:.0f}% ≤ IV ≤ {iv_max:.0f}%):")
        print(f"   Kept {len(df):,} / {count_before:,} options")
    
    # 4. ATM REGION FILTER (keep only near-ATM for term structure)
    count_before = len(df)
    df = df[df['x'].abs() <= atm_tolerance]
    if diagnostics:
        print(f"\n4. ATM region filter (|ln(K/F)| ≤ {atm_tolerance}):")
        print(f"   Kept {len(df):,} / {count_before:,} options")
    
    # 5. VALIDATE DATA
    df = df[np.isfinite(df['x']) & np.isfinite(df['iv_pct'])]
    
    if df.empty:
        print("\n❌ ERROR: No ATM data remaining after filtering!")
        return
    
    # Get unique expiries
    expiries = sorted(df['expiry_date'].unique())
    
    if diagnostics:
        print(f"\n5. Found {len(expiries)} unique expiries")
        print("="*80)
    
    # EXTRACT ATM IV PER EXPIRY using interpolation + blending
    term_data = []
    raw_points = []  # For showing individual quotes
    
    for expiry in expiries:
        subset = df[df['expiry_date'] == expiry].copy()
        
        if subset.empty or len(subset) < 2:
            continue
        
        T_val = subset['T'].iloc[0]
        DTE = int(T_val * 365)
        
        # Separate puts and calls
        puts = subset[subset['type'] == 'P'].copy()
        calls = subset[subset['type'] == 'C'].copy()
        
        # Store raw points for visualization
        for _, row in subset.iterrows():
            raw_points.append({
                'T': T_val,
                'DTE': DTE,
                'iv_pct': row['iv_pct'],
                'type': row['type'],
                'x': row['x']
            })
        
        # Try to interpolate ATM IV (at x=0)
        put_atm_iv = None
        call_atm_iv = None
        
        # Interpolate put IV to x=0
        if len(puts) >= 2:
            puts_sorted = puts.sort_values('x')
            x_puts = puts_sorted['x'].values
            iv_puts = puts_sorted['iv_pct'].values
            
            # Check if x=0 is within interpolation range
            if x_puts.min() <= 0 <= x_puts.max():
                try:
                    f_put = interp1d(x_puts, iv_puts, kind='linear', fill_value='extrapolate')
                    put_atm_iv = float(f_put(0))
                except:
                    pass
            elif len(puts) >= 1:
                # Use closest put
                closest_idx = puts['x'].abs().idxmin()
                put_atm_iv = puts.loc[closest_idx, 'iv_pct']
        elif len(puts) == 1:
            put_atm_iv = puts['iv_pct'].iloc[0]
        
        # Interpolate call IV to x=0
        if len(calls) >= 2:
            calls_sorted = calls.sort_values('x')
            x_calls = calls_sorted['x'].values
            iv_calls = calls_sorted['iv_pct'].values
            
            if x_calls.min() <= 0 <= x_calls.max():
                try:
                    f_call = interp1d(x_calls, iv_calls, kind='linear', fill_value='extrapolate')
                    call_atm_iv = float(f_call(0))
                except:
                    pass
            elif len(calls) >= 1:
                closest_idx = calls['x'].abs().idxmin()
                call_atm_iv = calls.loc[closest_idx, 'iv_pct']
        elif len(calls) == 1:
            call_atm_iv = calls['iv_pct'].iloc[0]
        
        # Blend put/call ATM IVs
        if put_atm_iv is not None and call_atm_iv is not None:
            # Average of put and call ATM IVs (they should be equal by put-call parity)
            atm_iv = (put_atm_iv + call_atm_iv) / 2
            source = 'Blended'
        elif put_atm_iv is not None:
            atm_iv = put_atm_iv
            source = 'Put'
        elif call_atm_iv is not None:
            atm_iv = call_atm_iv
            source = 'Call'
        else:
            # Fallback: median of all near-ATM options
            atm_iv = subset['iv_pct'].median()
            source = 'Median'
        
        if np.isfinite(atm_iv) and iv_min <= atm_iv <= iv_max:
            term_data.append({
                'T': T_val,
                'DTE': DTE,
                'atm_iv': atm_iv,
                'source': source,
                'n_puts': len(puts),
                'n_calls': len(calls)
            })
            
            if diagnostics:
                print(f"  {DTE:3d}D: ATM IV = {atm_iv:.2f}% ({source}, {len(puts)}P/{len(calls)}C)")
    
    if not term_data:
        print("\n❌ ERROR: Could not extract ATM IV for any expiry!")
        return
    
    term_df = pd.DataFrame(term_data)
    raw_df = pd.DataFrame(raw_points)
    
    # 6. OUTLIER REMOVAL on term structure
    count_before = len(term_df)
    median_iv = term_df['atm_iv'].median()
    std_iv = term_df['atm_iv'].std()
    
    if std_iv > 0:
        term_df = term_df[np.abs(term_df['atm_iv'] - median_iv) <= outlier_std * std_iv]
    
    if diagnostics and count_before > len(term_df):
        print(f"\n6. Removed {count_before - len(term_df)} outlier expiries")
    
    if len(term_df) < 2:
        print("\n❌ ERROR: Too few data points for term structure!")
        return
    
    term_df = term_df.sort_values('T')
    
    if diagnostics:
        print(f"\n7. Final term structure: {len(term_df)} expiries")
        print(f"   T range: {term_df['T'].min():.3f}y to {term_df['T'].max():.3f}y")
        print(f"   IV range: {term_df['atm_iv'].min():.2f}% to {term_df['atm_iv'].max():.2f}%")
        print("="*80 + "\n")
    
    # BUILD PLOT
    fig = go.Figure()
    
    # Show raw ATM quotes (if requested)
    if show_raw_points and not raw_df.empty:
        # Puts
        raw_puts = raw_df[raw_df['type'] == 'P']
        if not raw_puts.empty:
            fig.add_trace(go.Scatter(
                x=raw_puts['T'],
                y=raw_puts['iv_pct'],
                mode='markers',
                marker=dict(size=4, color='rgba(255,100,100,0.4)', symbol='x'),
                name="ATM Puts",
                hovertemplate='Put<br>DTE: %{customdata[0]}D<br>IV: %{y:.2f}%<br>ln(K/F): %{customdata[1]:.3f}<extra></extra>',
                customdata=np.column_stack([raw_puts['DTE'], raw_puts['x']])
            ))
        
        # Calls  
        raw_calls = raw_df[raw_df['type'] == 'C']
        if not raw_calls.empty:
            fig.add_trace(go.Scatter(
                x=raw_calls['T'],
                y=raw_calls['iv_pct'],
                mode='markers',
                marker=dict(size=4, color='rgba(100,255,100,0.4)', symbol='cross'),
                name="ATM Calls",
                hovertemplate='Call<br>DTE: %{customdata[0]}D<br>IV: %{y:.2f}%<br>ln(K/F): %{customdata[1]:.3f}<extra></extra>',
                customdata=np.column_stack([raw_calls['DTE'], raw_calls['x']])
            ))
    
    # Extracted ATM IV points
    fig.add_trace(go.Scatter(
        x=term_df['T'],
        y=term_df['atm_iv'],
        mode='markers',
        marker=dict(size=10, color='cyan', symbol='circle',
                   line=dict(width=1, color='white')),
        name="ATM IV (Blended)",
        hovertemplate='%{customdata[0]}D<br>ATM IV: %{y:.2f}%<br>T: %{x:.3f}y<br>Source: %{customdata[1]}<extra></extra>',
        customdata=np.column_stack([term_df['DTE'], term_df['source']])
    ))
    
    # Smoothed term structure curve
    if smooth and len(term_df) >= 4:
        try:
            T_vals = term_df['T'].values
            iv_vals = term_df['atm_iv'].values
            
            cs = CubicSpline(T_vals, iv_vals)
            T_smooth = np.linspace(T_vals.min(), T_vals.max(), 200)
            iv_smooth = cs(T_smooth)
            iv_smooth = np.clip(iv_smooth, iv_min, iv_max)
            
            fig.add_trace(go.Scatter(
                x=T_smooth,
                y=iv_smooth,
                mode='lines',
                line=dict(color='orange', width=3),
                name="Smoothed Term Structure",
                hovertemplate='T: %{x:.3f}y<br>IV: %{y:.2f}%<extra></extra>'
            ))
        except Exception as e:
            if diagnostics:
                print(f"⚠️  Spline smoothing failed: {e}")
            # Fallback to connected line
            fig.add_trace(go.Scatter(
                x=term_df['T'],
                y=term_df['atm_iv'],
                mode='lines',
                line=dict(color='orange', width=2),
                name="Term Structure"
            ))
    else:
        # Just connect the points
        fig.add_trace(go.Scatter(
            x=term_df['T'],
            y=term_df['atm_iv'],
            mode='lines',
            line=dict(color='orange', width=2),
            name="Term Structure"
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text="Volatility Term Structure (ATM IV vs Time to Maturity)<br>"
                 "<sub>Robust ATM extraction via interpolation + put/call blending</sub>",
            font=dict(size=20)
        ),
        xaxis_title="Time to Maturity (Years)",
        yaxis_title="Implied Volatility (%)",
        template="plotly_dark",
        legend=dict(title="Data Source", font=dict(size=12)),
        hovermode='closest',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=600,
        plot_bgcolor='rgba(15,15,25,1)',
        paper_bgcolor='rgba(10,10,15,1)'
    )
    
    # SAVE PLOT
    if save_plot:
        # Make path relative to script directory
        if not os.path.isabs(output_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"term_structure_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        fig.write_image(filepath, width=1920, height=800, scale=2)
        if diagnostics:
            print(f"📊 Plot saved to: {filepath}\n")
    
    fig.show()
    
    return term_df  # Return the term structure data for further analysis

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
