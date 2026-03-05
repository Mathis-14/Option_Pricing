"""
cross_gamma_model.py — Monte Carlo pricing and cross gamma computation

This module prices a two-asset basket call option and computes its cross gamma
(∂²C/∂S₁∂S₂) using Monte Carlo simulation with finite-difference bumps.

Theory
------
A basket call on two stocks with weights w₁, w₂ has payoff at maturity:
    max(w₁·S₁(T) + w₂·S₂(T) − K, 0)

The stock prices follow correlated Geometric Brownian Motions (GBM):
    dSᵢ = r·Sᵢ·dt + σᵢ·Sᵢ·dWᵢ

where dW₁·dW₂ = ρ·dt (Wiener processes with correlation ρ).

Since w₁·S₁ + w₂·S₂ is NOT lognormally distributed, there is no closed-form
Black-Scholes formula. Monte Carlo is the standard approach.

Cross gamma is computed via central finite differences:
    Γ_cross = [C(S₁+h₁,S₂+h₂) − C(S₁+h₁,S₂) − C(S₁,S₂+h₂) + C(S₁,S₂)] / (h₁·h₂)

Usage
-----
    from cross_gamma_model import BasketCallPricer
    
    pricer = BasketCallPricer(S1=130, S2=170, K=150, T=0.25, r=0.05,
                              sigma1=0.50, sigma2=0.30, rho=0.6)
    print(f"Price: ${pricer.price():.4f}")
    print(f"Cross Gamma: {pricer.cross_gamma():.6f}")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


# =============================================================================
# BASKET CALL PRICER
# =============================================================================

@dataclass
class BasketCallPricer:
    """
    Monte Carlo pricer for a two-asset basket call option.
    
    Parameters
    ----------
    S1, S2 : float
        Current spot prices of the two underlyings.
    K : float
        Strike price of the basket call.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate (annualized).
    sigma1, sigma2 : float
        Volatilities of the two underlyings (annualized).
    rho : float
        Correlation between the two stocks' returns.
    w1, w2 : float
        Basket weights (default: 0.5 each).
    n_paths : int
        Number of Monte Carlo simulation paths.
    seed : int
        Random seed for reproducibility.
    """
    S1: float
    S2: float
    K: float
    T: float
    r: float
    sigma1: float
    sigma2: float
    rho: float
    w1: float = 0.5
    w2: float = 0.5
    n_paths: int = 200_000
    seed: int = 42
    
    def _simulate_terminal_prices(
        self, S1: float = None, S2: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate terminal stock prices using correlated GBM.
        
        Uses Cholesky decomposition to generate correlated Brownian motions:
            W₁ = Z₁
            W₂ = ρ·Z₁ + √(1−ρ²)·Z₂
        
        Terminal price under GBM (exact solution, no discretization error):
            S(T) = S(0) · exp[(r − σ²/2)·T + σ·√T·W]
        
        Uses antithetic variates for variance reduction:
            For each Z, also use −Z, effectively doubling the paths.
        
        Parameters
        ----------
        S1, S2 : float, optional
            Override spot prices (used for finite-difference bumps).
            
        Returns
        -------
        tuple of np.ndarray
            (S1_terminal, S2_terminal) arrays of shape (2 * n_paths,).
        """
        if S1 is None:
            S1 = self.S1
        if S2 is None:
            S2 = self.S2
        
        rng = np.random.default_rng(self.seed)
        
        # Generate independent standard normals
        Z1 = rng.standard_normal(self.n_paths)
        Z2 = rng.standard_normal(self.n_paths)
        
        # Cholesky: correlate Z2 with Z1
        W1 = Z1
        W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
        
        # Antithetic variates: use both Z and -Z
        W1_anti = np.concatenate([W1, -W1])
        W2_anti = np.concatenate([W2, -W2])
        
        # GBM terminal prices (exact, single-step)
        drift1 = (self.r - 0.5 * self.sigma1**2) * self.T
        drift2 = (self.r - 0.5 * self.sigma2**2) * self.T
        
        S1_T = S1 * np.exp(drift1 + self.sigma1 * np.sqrt(self.T) * W1_anti)
        S2_T = S2 * np.exp(drift2 + self.sigma2 * np.sqrt(self.T) * W2_anti)
        
        return S1_T, S2_T
    
    def price(self, S1: float = None, S2: float = None) -> float:
        """
        Compute the basket call price via Monte Carlo.
        
        Price = e^{-rT} · E[max(w₁·S₁(T) + w₂·S₂(T) − K, 0)]
        
        Parameters
        ----------
        S1, S2 : float, optional
            Override spot prices (used internally for bumped prices).
            
        Returns
        -------
        float
            Monte Carlo estimate of the basket call price.
        """
        S1_T, S2_T = self._simulate_terminal_prices(S1, S2)
        
        # Basket value at maturity
        basket_T = self.w1 * S1_T + self.w2 * S2_T
        
        # Call payoff
        payoffs = np.maximum(basket_T - self.K, 0.0)
        
        # Discounted expected payoff
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return price
    
    def cross_gamma(self, dS_frac: float = 0.01) -> float:
        """
        Compute cross gamma ∂²C/∂S₁∂S₂ using central finite differences.
        
        Formula:
            Γ_cross = [C(S₁+h₁,S₂+h₂) − C(S₁+h₁,S₂) − C(S₁,S₂+h₂) + C(S₁,S₂)]
                      / (h₁ · h₂)
        
        Parameters
        ----------
        dS_frac : float
            Bump size as fraction of spot price (default: 1%).
            
        Returns
        -------
        float
            Cross gamma estimate.
        """
        h1 = dS_frac * self.S1
        h2 = dS_frac * self.S2
        
        # Four price evaluations with different bump combinations
        C_pp = self.price(self.S1 + h1, self.S2 + h2)  # both bumped up
        C_p0 = self.price(self.S1 + h1, self.S2)         # only S1 bumped
        C_0p = self.price(self.S1, self.S2 + h2)         # only S2 bumped
        C_00 = self.price(self.S1, self.S2)               # no bump (base)
        
        gamma_cross = (C_pp - C_p0 - C_0p + C_00) / (h1 * h2)
        return gamma_cross
    
    def cross_gamma_surface(
        self,
        S1_range: np.ndarray,
        S2_range: np.ndarray,
        dS_frac: float = 0.01,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Compute cross gamma over a 2D grid of (S₁, S₂) values.
        
        Parameters
        ----------
        S1_range : np.ndarray
            Array of S₁ values to evaluate.
        S2_range : np.ndarray
            Array of S₂ values to evaluate.
        dS_frac : float
            Bump size as fraction of each point's spot price.
        verbose : bool
            If True, print progress updates.
            
        Returns
        -------
        np.ndarray
            2D array of shape (len(S2_range), len(S1_range)) with cross gamma values.
            Row index corresponds to S₂, column index to S₁.
        """
        n1 = len(S1_range)
        n2 = len(S2_range)
        surface = np.zeros((n2, n1))
        
        total = n1 * n2
        count = 0
        
        if verbose:
            print(f"\n🔄 Computing cross gamma surface ({n1}×{n2} = {total} points)...")
        
        for j, s2 in enumerate(S2_range):
            for i, s1 in enumerate(S1_range):
                # Temporarily set S1, S2 for this grid point
                h1 = dS_frac * s1
                h2 = dS_frac * s2
                
                C_pp = self.price(s1 + h1, s2 + h2)
                C_p0 = self.price(s1 + h1, s2)
                C_0p = self.price(s1, s2 + h2)
                C_00 = self.price(s1, s2)
                
                surface[j, i] = (C_pp - C_p0 - C_0p + C_00) / (h1 * h2)
                
                count += 1
                if verbose and count % max(1, total // 10) == 0:
                    pct = count / total * 100
                    print(f"   Progress: {pct:.0f}% ({count}/{total})")
        
        if verbose:
            print("   ✅ Surface computation complete.")
        
        return surface
    
    def delta_1(self, dS_frac: float = 0.01) -> float:
        """Compute delta w.r.t. S₁ using central finite differences."""
        h = dS_frac * self.S1
        return (self.price(self.S1 + h, self.S2) - self.price(self.S1 - h, self.S2)) / (2 * h)
    
    def delta_2(self, dS_frac: float = 0.01) -> float:
        """Compute delta w.r.t. S₂ using central finite differences."""
        h = dS_frac * self.S2
        return (self.price(self.S1, self.S2 + h) - self.price(self.S1, self.S2 - h)) / (2 * h)
    
    def gamma_1(self, dS_frac: float = 0.01) -> float:
        """Compute gamma w.r.t. S₁ using central finite differences."""
        h = dS_frac * self.S1
        C_up = self.price(self.S1 + h, self.S2)
        C_dn = self.price(self.S1 - h, self.S2)
        C_00 = self.price(self.S1, self.S2)
        return (C_up - 2 * C_00 + C_dn) / (h**2)
    
    def gamma_2(self, dS_frac: float = 0.01) -> float:
        """Compute gamma w.r.t. S₂ using central finite differences."""
        h = dS_frac * self.S2
        C_up = self.price(self.S1, self.S2 + h)
        C_dn = self.price(self.S1, self.S2 - h)
        C_00 = self.price(self.S1, self.S2)
        return (C_up - 2 * C_00 + C_dn) / (h**2)
    
    def summary(self) -> str:
        """
        Print a full summary of the basket option pricing and Greeks.
        
        Returns
        -------
        str
            Formatted summary string.
        """
        price = self.price()
        d1 = self.delta_1()
        d2 = self.delta_2()
        g1 = self.gamma_1()
        g2 = self.gamma_2()
        gc = self.cross_gamma()
        
        basket_val = self.w1 * self.S1 + self.w2 * self.S2
        moneyness = basket_val / self.K
        
        lines = [
            "",
            "=" * 60,
            "  BASKET CALL OPTION — PRICING & GREEKS",
            "=" * 60,
            f"  S₁ = ${self.S1:.2f}  (weight: {self.w1:.0%})",
            f"  S₂ = ${self.S2:.2f}  (weight: {self.w2:.0%})",
            f"  Basket value: ${basket_val:.2f}",
            f"  Strike (K):   ${self.K:.2f}",
            f"  Moneyness:    {moneyness:.4f}",
            "─" * 60,
            f"  T = {self.T:.4f} years ({self.T*365:.0f} days)",
            f"  r = {self.r*100:.2f}%",
            f"  σ₁ = {self.sigma1*100:.1f}%    σ₂ = {self.sigma2*100:.1f}%",
            f"  ρ  = {self.rho:.4f}",
            f"  Paths = {self.n_paths:,} (×2 with antithetic)",
            "─" * 60,
            f"  PRICE:        ${price:.4f}",
            "─" * 60,
            f"  Δ₁ (∂C/∂S₁):      {d1:+.6f}",
            f"  Δ₂ (∂C/∂S₂):      {d2:+.6f}",
            f"  Γ₁ (∂²C/∂S₁²):    {g1:.6f}",
            f"  Γ₂ (∂²C/∂S₂²):    {g2:.6f}",
            f"  Γ_cross (∂²C/∂S₁∂S₂): {gc:.6f}",
            "=" * 60,
            "",
        ]
        return "\n".join(lines)


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    # Example: two stocks at $130 and $170, basket strike at ATM
    pricer = BasketCallPricer(
        S1=130.0, S2=170.0,
        K=150.0,  # ATM: 0.5*130 + 0.5*170 = 150
        T=0.25,
        r=0.05,
        sigma1=0.50,
        sigma2=0.30,
        rho=0.6
    )
    
    print(pricer.summary())
