# iv_surface_spx.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from scipy.interpolate import RBFInterpolator
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


@dataclass
class SurfaceConfig:
    r: float = 0.05
    as_of: Optional[datetime] = None

    # quote quality
    min_bid: float = 0.01
    max_rel_spread: float = 0.25
    min_oi: int = 10
    min_volume: int = 1

    # interpolation
    grid_n: int = 60
    rbf_smoothing: float = 0.5
    rbf_kernel: str = "thin_plate_spline"

     # Time filtering
    min_T: Optional[float] = None  # Temps minimum en années (None = pas de filtre)
    max_T: Optional[float] = None  # Temps maximum en années (None = pas de filtre)

class SPXIVSurface:
    def __init__(self, df: pd.DataFrame, cfg: SurfaceConfig = SurfaceConfig()):
        self.cfg = cfg
        self.raw = df.copy()
        self.df = self._prepare(self.raw)

    @staticmethod
    def _normalize_iv_to_percent(iv: pd.Series) -> pd.Series:
        iv = pd.to_numeric(iv, errors="coerce")
        med = np.nanmedian(iv.values) if np.isfinite(iv).any() else np.nan
        # Yahoo donne souvent iv en décimal (0.20). Si median < 3 => décimal.
        return iv * 100.0 if np.isfinite(med) and med < 3.0 else iv

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")

        

        as_of = self.cfg.as_of or datetime.now()
        df["T"] = (df["expiry_date"] - pd.Timestamp(as_of)).dt.total_seconds() / (365.25 * 24 * 3600)
        df["T"] = df["T"].clip(lower=1e-6)

        if self.cfg.min_T is not None:
            df = df[df["T"] >= self.cfg.min_T]
        if self.cfg.max_T is not None:
            df = df[df["T"] <= self.cfg.max_T]

        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["S"] = pd.to_numeric(df.get("underlying_price"), errors="coerce")

        df["bid"] = pd.to_numeric(df.get("bid_price"), errors="coerce")
        df["ask"] = pd.to_numeric(df.get("ask_price"), errors="coerce")
        df["mid"] = (df["bid"] + df["ask"]) / 2.0

        df["open_interest"] = pd.to_numeric(df.get("open_interest"), errors="coerce").fillna(0).astype(int)
        df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0).astype(int)

        # base clean
        df = df.dropna(subset=["expiry_date", "T", "strike", "type", "S", "bid", "ask", "mid"])
        df = df[(df["strike"] > 0) & (df["S"] > 0)]

        # remove stub quotes (ton gros problème)
        df = df[(df["bid"] > 0) & (df["ask"] > 0)]
        df = df[df["bid"] >= self.cfg.min_bid]
        df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)
        df = df[df["rel_spread"] <= self.cfg.max_rel_spread]

        # liquidity (keep if OI or traded)
        df = df[(df["open_interest"] >= self.cfg.min_oi) | (df["volume"] >= self.cfg.min_volume)]

        # IV from Yahoo
        df["iv_pct"] = self._normalize_iv_to_percent(df["mark_iv"])
        df = df.dropna(subset=["iv_pct"])
        df = df[df["iv_pct"].between(0.01, 300.0)]

        # forward per expiry via parity: F ≈ K + exp(rT)(C-P)
        df["F"] = df.groupby("expiry_date", group_keys=False).apply(self._estimate_forward_group)
        df = df.dropna(subset=["F"])

        # log-moneyness
        df["x"] = np.log(df["strike"] / df["F"])

        # OTM-only selection (standard)
        is_put_otm = (df["strike"] < df["F"]) & (df["type"] == "P")
        is_call_otm = (df["strike"] > df["F"]) & (df["type"] == "C")
        is_atm = (np.abs(df["strike"] - df["F"]) / df["F"] < 0.002)
        df = df[is_put_otm | is_call_otm | is_atm].copy()

        return df.reset_index(drop=True)

    def _estimate_forward_group(self, g: pd.DataFrame) -> pd.Series:
        r = self.cfg.r
        T = float(g["T"].iloc[0])
        S = float(g["S"].iloc[0])

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

    def _interpolate(self):
        if not SCIPY_OK:
            return None

        x = self.df["x"].values
        t = self.df["T"].values
        z = self.df["iv_pct"].values

        x_min, x_max = np.nanpercentile(x, [1, 99])

        if self.cfg.min_T is not None and self.cfg.max_T is not None:
            t_min, t_max = self.cfg.min_T, self.cfg.max_T
        else:
            t_min, t_max = np.nanpercentile(t, [1, 99])

        X = np.linspace(x_min, x_max, self.cfg.grid_n)
        TT = np.linspace(t_min, t_max, self.cfg.grid_n)
        XX, YY = np.meshgrid(X, TT)

        pts = np.column_stack([x, t])
        grid_pts = np.column_stack([XX.ravel(), YY.ravel()])

        rbf = RBFInterpolator(
            pts, z,
            kernel=self.cfg.rbf_kernel,
            smoothing=self.cfg.rbf_smoothing
        )
        ZZ = rbf(grid_pts).reshape(XX.shape)
        return XX, YY, ZZ

    def plot(self, title: str = "SPX Implied Volatility Surface (OTM)", interpolate: bool = True):
        fig = go.Figure()

        if interpolate:
            out = self._interpolate()
            if out is not None:
                XX, YY, ZZ = out
                fig.add_trace(go.Surface(
                    x=XX, y=YY, z=ZZ,
                    colorscale="Viridis",
                    colorbar=dict(title="IV (%)"),
                    hovertemplate="ln(K/F): %{x:.3f}<br>T: %{y:.3f}y<br>IV: %{z:.2f}%<extra></extra>"
                ))

        fig.add_trace(go.Scatter3d(
            x=self.df["x"], y=self.df["T"], z=self.df["iv_pct"],
            mode="markers",
            marker=dict(size=2, color="white", opacity=0.7),
            name="OTM quotes",
            hovertemplate="ln(K/F): %{x:.3f}<br>T: %{y:.3f}y<br>IV: %{z:.2f}%<extra></extra>"
        ))

        fig.update_layout(
            template="plotly_dark",
            title=title,
            scene=dict(
                xaxis_title="log-moneyness ln(K/F)",
                yaxis_title="Time to expiry (years)",
                zaxis_title="Implied vol (%)",
                camera=dict(eye=dict(x=1.6, y=1.4, z=1.2))
            ),
            height=720
        )
        return fig
