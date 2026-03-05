"""
plot_cross_gamma.py — Visualization of basket call cross gamma

Generates two complementary plots:

1. **3D Surface Plot**: Shows the full shape of cross gamma Γ_cross(S₁, S₂).
   - Relevance: Reveals how the interaction risk between the two stocks peaks
     near ATM and flattens far ITM/OTM. The shape is influenced by correlation
     and relative volatilities.

2. **2D Heatmap**: Cross gamma as color intensity over the (S₁, S₂) plane.
   - Relevance: Easier to read for practical use — directly identifies the
     "hot zone" where cross gamma is highest, meaning your delta hedge on one
     stock is most sensitive to moves in the other. Traders use this to know
     when re-hedging is critical.

Usage:
    from plot_cross_gamma import plot_3d_surface, plot_heatmap
    
    plot_3d_surface(S1_range, S2_range, gamma_surface, spots, K, output_dir)
    plot_heatmap(S1_range, S2_range, gamma_surface, spots, K, output_dir)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
import os


# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

# Use a clean, professional style
plt.rcParams.update({
    'figure.facecolor': '#0f0f0f',
    'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#333355',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#aaaaaa',
    'ytick.color': '#aaaaaa',
    'grid.color': '#333355',
    'grid.alpha': 0.3,
    'figure.figsize': (12, 9),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


# =============================================================================
# 3D SURFACE PLOT
# =============================================================================

def plot_3d_surface(
    S1_range: np.ndarray,
    S2_range: np.ndarray,
    gamma_surface: np.ndarray,
    spot_S1: float,
    spot_S2: float,
    K: float,
    output_dir: str = None,
    show: bool = True
) -> str:
    """
    Plot cross gamma as a 3D surface over the (S₁, S₂) plane.
    
    This visualization reveals the FULL SHAPE of cross gamma:
    - The peak near ATM shows where interaction risk is highest
    - The falloff to zero far ITM/OTM shows where interaction becomes negligible
    - Asymmetries reveal the impact of different volatilities
    
    Parameters
    ----------
    S1_range, S2_range : np.ndarray
        Grid axis values.
    gamma_surface : np.ndarray
        2D array of cross gamma values (shape: len(S2_range) × len(S1_range)).
    spot_S1, spot_S2 : float
        Current spot prices (for marking the ATM point).
    K : float
        Basket strike price.
    output_dir : str, optional
        Directory to save the plot.
    show : bool
        Whether to display the plot.
        
    Returns
    -------
    str
        Path to saved image (if output_dir provided).
    """
    S1_grid, S2_grid = np.meshgrid(S1_range, S2_range)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set 3D-specific background colors
    ax.set_facecolor('#1a1a2e')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#333355')
    ax.yaxis.pane.set_edgecolor('#333355')
    ax.zaxis.pane.set_edgecolor('#333355')
    
    # Color normalization centered on zero
    vmax = np.max(np.abs(gamma_surface))
    if vmax == 0:
        vmax = 1e-6
    norm = TwoSlopeNorm(vmin=-vmax * 0.3, vcenter=0, vmax=vmax)
    
    # Surface plot with gradient colormap
    surf = ax.plot_surface(
        S1_grid, S2_grid, gamma_surface,
        cmap='RdYlGn',
        norm=norm,
        alpha=0.85,
        linewidth=0.2,
        edgecolor='#ffffff20',
        antialiased=True
    )
    
    # Mark the current ATM point
    # Find closest grid indices to current spots
    i_s1 = np.argmin(np.abs(S1_range - spot_S1))
    i_s2 = np.argmin(np.abs(S2_range - spot_S2))
    atm_gamma = gamma_surface[i_s2, i_s1]
    
    ax.scatter(
        [spot_S1], [spot_S2], [atm_gamma],
        color='#00ffff', s=100, zorder=10,
        edgecolors='white', linewidth=1.5,
        label=f'Current ATM: Γ_cross = {atm_gamma:.6f}'
    )
    
    # Labels and title
    ax.set_xlabel(f'S₁ (Stock 1)', fontsize=12, labelpad=10)
    ax.set_ylabel(f'S₂ (Stock 2)', fontsize=12, labelpad=10)
    ax.set_zlabel('Cross Gamma (∂²C/∂S₁∂S₂)', fontsize=11, labelpad=10)
    
    ax.set_title(
        f'Cross Gamma Surface — Basket Call\n'
        f'K = ${K:.0f}  |  S₁ = ${spot_S1:.0f}  |  S₂ = ${spot_S2:.0f}',
        fontsize=14, fontweight='bold', pad=20
    )
    
    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.08)
    cbar.set_label('Γ_cross', fontsize=11)
    
    ax.legend(loc='upper left', fontsize=10)
    ax.view_init(elev=30, azim=-45)
    
    plt.tight_layout()
    
    # Save
    save_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'cross_gamma_3d_surface.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"   📊 3D surface saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return save_path


# =============================================================================
# 2D HEATMAP
# =============================================================================

def plot_heatmap(
    S1_range: np.ndarray,
    S2_range: np.ndarray,
    gamma_surface: np.ndarray,
    spot_S1: float,
    spot_S2: float,
    K: float,
    output_dir: str = None,
    show: bool = True
) -> str:
    """
    Plot cross gamma as a 2D heatmap with contour lines.
    
    This visualization is PRACTICAL for trading:
    - The "hot zone" (bright colors) shows where cross gamma is highest
    - A trader sees at a glance: if S₁ is at X and S₂ at Y, how sensitive
      is my hedge to a move in the other stock?
    - Contour lines highlight key cross gamma levels
    
    Parameters
    ----------
    S1_range, S2_range : np.ndarray
        Grid axis values.
    gamma_surface : np.ndarray
        2D array of cross gamma values.
    spot_S1, spot_S2 : float
        Current spot prices.
    K : float
        Basket strike price.
    output_dir : str, optional
        Directory to save the plot.
    show : bool
        Whether to display the plot.
        
    Returns
    -------
    str
        Path to saved image (if output_dir provided).
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Color normalization centered on zero
    vmax = np.max(np.abs(gamma_surface))
    if vmax == 0:
        vmax = 1e-6
    norm = TwoSlopeNorm(vmin=-vmax * 0.3, vcenter=0, vmax=vmax)
    
    # Heatmap
    extent = [S1_range[0], S1_range[-1], S2_range[0], S2_range[-1]]
    im = ax.imshow(
        gamma_surface,
        extent=extent,
        origin='lower',
        aspect='auto',
        cmap='magma',
        norm=norm,
        interpolation='bicubic'
    )
    
    # Contour lines
    S1_grid, S2_grid = np.meshgrid(S1_range, S2_range)
    n_contours = 8
    contour_levels = np.linspace(
        np.min(gamma_surface),
        np.max(gamma_surface),
        n_contours
    )
    contours = ax.contour(
        S1_grid, S2_grid, gamma_surface,
        levels=contour_levels,
        colors='white',
        alpha=0.4,
        linewidths=0.8
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.5f', colors='white')
    
    # Mark current spot (ATM point)
    ax.plot(
        spot_S1, spot_S2,
        marker='*', markersize=18,
        color='#00ffff', markeredgecolor='white', markeredgewidth=1.5,
        label=f'Current: S₁=${spot_S1:.0f}, S₂=${spot_S2:.0f}',
        zorder=10
    )
    
    # Mark the ATM basket line: w1*S1 + w2*S2 = K
    # → S2 = (K - w1*S1) / w2  (for 50/50: S2 = 2K - S1)
    atm_line_S1 = np.linspace(S1_range[0], S1_range[-1], 100)
    atm_line_S2 = (K - 0.5 * atm_line_S1) / 0.5  # assuming w1=w2=0.5
    mask = (atm_line_S2 >= S2_range[0]) & (atm_line_S2 <= S2_range[-1])
    ax.plot(
        atm_line_S1[mask], atm_line_S2[mask],
        '--', color='#ff6b6b', linewidth=2, alpha=0.7,
        label=f'ATM line: 0.5·S₁ + 0.5·S₂ = K (${K:.0f})'
    )
    
    # Labels and title
    ax.set_xlabel('S₁ (Stock 1)', fontsize=13)
    ax.set_ylabel('S₂ (Stock 2)', fontsize=13)
    ax.set_title(
        f'Cross Gamma Heatmap — Basket Call\n'
        f'K = ${K:.0f}  |  Shows ∂²C/∂S₁∂S₂ intensity across spot prices',
        fontsize=14, fontweight='bold'
    )
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=25)
    cbar.set_label('Γ_cross (∂²C/∂S₁∂S₂)', fontsize=11)
    
    ax.legend(loc='upper right', fontsize=10, facecolor='#1a1a2e', edgecolor='#333355')
    
    # Annotation: explain what the hot zone means
    ax.text(
        0.02, 0.02,
        'Hot zone = delta hedge on one stock is\n'
        'most sensitive to moves in the other stock',
        transform=ax.transAxes,
        fontsize=9, color='#aaaaaa',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e', edgecolor='#333355', alpha=0.9)
    )
    
    plt.tight_layout()
    
    # Save
    save_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'cross_gamma_heatmap.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"   📊 Heatmap saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return save_path
