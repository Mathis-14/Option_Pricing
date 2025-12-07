"""
Option Strategies Class
=======================

This module implements a comprehensive class for analyzing various option trading strategies.
Each strategy is implemented as a method that calculates both payoff and profit profiles.

The class provides two types of plots:
1. Payoff diagrams: Show the intrinsic value at expiration (ignoring option premiums)
2. Profit diagrams: Show the actual profit/loss including the cost of options

All strategies are based on European options and assume expiration at maturity.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import math
from typing import Tuple, Optional, List


class OptionStrategies:
    """
    A class to calculate and visualize various option trading strategies.
    
    This class implements common option strategies including:
    - Basic positions (long/short calls and puts)
    - Spreads (call spreads, put spreads)
    - Combinations (straddles, strangles)
    - Complex strategies (ratios, ladders, flies, condors, etc.)
    
    Each strategy method calculates the payoff and profit profiles and can generate
    visualization plots.
    """
    
    def __init__(self, S_range: Tuple[float, float] = (0, 200), n_points: int = 1000):
        """
        Initialize the OptionStrategies class.
        
        Parameters:
        -----------
        S_range : tuple of float
            The range of underlying asset prices to evaluate (min, max)
        n_points : int
            Number of points in the price grid for calculations
        """
        self.S_min, self.S_max = S_range
        self.n_points = n_points
        self.S = np.linspace(self.S_min, self.S_max, n_points)
        self.plot_dir = "plot/strategies"
        os.makedirs(self.plot_dir, exist_ok=True)
    
    def _call_payoff(self, S: np.ndarray, K: float) -> np.ndarray:
        """
        Calculate the payoff of a long call option at expiration.
        
        Payoff = max(0, S - K)
        
        Parameters:
        -----------
        S : np.ndarray
            Array of underlying asset prices
        K : float
            Strike price
            
        Returns:
        --------
        np.ndarray
            Payoff values
        """
        return np.maximum(0, S - K)
    
    def _put_payoff(self, S: np.ndarray, K: float) -> np.ndarray:
        """
        Calculate the payoff of a long put option at expiration.
        
        Payoff = max(0, K - S)
        
        Parameters:
        -----------
        S : np.ndarray
            Array of underlying asset prices
        K : float
            Strike price
            
        Returns:
        --------
        np.ndarray
            Payoff values
        """
        return np.maximum(0, K - S)
    
    def _plot_strategy(self, S: np.ndarray, payoff: np.ndarray, profit: np.ndarray,
                      strategy_name: str, strikes: List[float], premiums: Optional[float] = None):
        """
        Create and save plots for a strategy showing both payoff and profit.
        
        Parameters:
        -----------
        S : np.ndarray
            Underlying asset prices
        payoff : np.ndarray
            Payoff values (intrinsic value at expiration)
        profit : np.ndarray
            Profit values (payoff - net premium)
        strategy_name : str
            Name of the strategy for file naming
        strikes : list of float
            List of strike prices used (for vertical lines on plot)
        premiums : float, optional
            Net premium paid/received (for annotation)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Payoff plot
        ax1.plot(S, payoff, linewidth=2, color='steelblue', label='Payoff')
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
        for K in strikes:
            ax1.axvline(K, color='orange', linestyle='--', linewidth=1, alpha=0.7)
            ax1.text(K, ax1.get_ylim()[0] * 0.05, f'K={K}', ha='center', va='bottom', 
                    rotation=90, fontsize=8)
        ax1.set_xlabel('Underlying Asset Price ($S_T$)', fontsize=12)
        ax1.set_ylabel('Payoff', fontsize=12)
        ax1.set_title(f'{strategy_name} - Payoff Diagram', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Profit plot
        ax2.plot(S, profit, linewidth=2, color='green', label='Profit')
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
        for K in strikes:
            ax2.axvline(K, color='orange', linestyle='--', linewidth=1, alpha=0.7)
            ax2.text(K, ax2.get_ylim()[0] * 0.05, f'K={K}', ha='center', va='bottom', 
                    rotation=90, fontsize=8)
        if premiums is not None:
            ax2.text(0.02, 0.98, f'Net Premium: {premiums:.2f}', 
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_xlabel('Underlying Asset Price ($S_T$)', fontsize=12)
        ax2.set_ylabel('Profit/Loss', fontsize=12)
        ax2.set_title(f'{strategy_name} - Profit Diagram', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{self.plot_dir}/{strategy_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {filename}")
    
    # ========================================================================
    # BASIC OPTIONS
    # ========================================================================
    
    def long_call(self, K: float, premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Long Call Option Strategy
        
        Buy a call option. This gives the right (but not obligation) to buy
        the underlying asset at strike price K.
        
        Payoff at expiration: max(0, S_T - K)
        Profit: Payoff - premium
        
        Parameters:
        -----------
        K : float
            Strike price
        premium : float
            Premium paid for the call option
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        payoff = self._call_payoff(self.S, K)
        profit = payoff - premium
        self._plot_strategy(self.S, payoff, profit, "Long CALL", [K], premium)
        return payoff, profit
    
    def long_put(self, K: float, premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Long Put Option Strategy
        
        Buy a put option. This gives the right (but not obligation) to sell
        the underlying asset at strike price K.
        
        Payoff at expiration: max(0, K - S_T)
        Profit: Payoff - premium
        
        Parameters:
        -----------
        K : float
            Strike price
        premium : float
            Premium paid for the put option
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        payoff = self._put_payoff(self.S, K)
        profit = payoff - premium
        self._plot_strategy(self.S, payoff, profit, "Long PUT", [K], premium)
        return payoff, profit
    
    def short_call(self, K: float, premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Short Call Option Strategy
        
        Sell a call option. This obligates you to sell the underlying asset
        at strike price K if the option is exercised.
        
        Payoff at expiration: -max(0, S_T - K)
        Profit: Payoff + premium
        
        Parameters:
        -----------
        K : float
            Strike price
        premium : float
            Premium received for selling the call option
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        payoff = -self._call_payoff(self.S, K)
        profit = payoff + premium
        self._plot_strategy(self.S, payoff, profit, "Short CALL", [K], -premium)
        return payoff, profit
    
    def short_put(self, K: float, premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Short Put Option Strategy
        
        Sell a put option. This obligates you to buy the underlying asset
        at strike price K if the option is exercised.
        
        Payoff at expiration: -max(0, K - S_T)
        Profit: Payoff + premium
        
        Parameters:
        -----------
        K : float
            Strike price
        premium : float
            Premium received for selling the put option
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        payoff = -self._put_payoff(self.S, K)
        profit = payoff + premium
        self._plot_strategy(self.S, payoff, profit, "Short PUT", [K], -premium)
        return payoff, profit
    
    # ========================================================================
    # SPREADS AND COMBINATIONS
    # ========================================================================
    
    def straddle(self, K: float, call_premium: float, put_premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Straddle Strategy
        
        Buy both a call and a put with the same strike price K and expiration.
        Profitable when the underlying moves significantly in either direction.
        
        Payoff: max(0, S_T - K) + max(0, K - S_T) = |S_T - K|
        Profit: Payoff - (call_premium + put_premium)
        
        Parameters:
        -----------
        K : float
            Strike price (same for both call and put)
        call_premium : float
            Premium paid for the call option
        put_premium : float
            Premium paid for the put option
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        call_payoff = self._call_payoff(self.S, K)
        put_payoff = self._put_payoff(self.S, K)
        payoff = call_payoff + put_payoff
        net_premium = call_premium + put_premium
        profit = payoff - net_premium
        self._plot_strategy(self.S, payoff, profit, "Straddle (CALL + PUT)", [K], net_premium)
        return payoff, profit
    
    def strangle(self, K1: float, K2: float, call_premium: float, put_premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Strangle Strategy
        
        Buy a call with strike K1 (higher) and a put with strike K2 (lower).
        Cheaper than a straddle but requires larger price movements to be profitable.
        
        Payoff: max(0, S_T - K1) + max(0, K2 - S_T)
        Profit: Payoff - (call_premium + put_premium)
        
        Note: K1 > K2 (call strike is higher than put strike)
        
        Parameters:
        -----------
        K1 : float
            Strike price of the call option (higher strike)
        K2 : float
            Strike price of the put option (lower strike)
        call_premium : float
            Premium paid for the call option
        put_premium : float
            Premium paid for the put option
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        call_payoff = self._call_payoff(self.S, K1)
        put_payoff = self._put_payoff(self.S, K2)
        payoff = call_payoff + put_payoff
        net_premium = call_premium + put_premium
        profit = payoff - net_premium
        self._plot_strategy(self.S, payoff, profit, f"Strangle (CALL K1={K1} + PUT K2={K2})", [K1, K2], net_premium)
        return payoff, profit
    
    def call_spread(self, K1: float, K2: float, call1_premium: float, call2_premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Call Spread (Bull Call Spread) Strategy
        
        Buy a call at lower strike K1 and sell a call at higher strike K2.
        Bullish strategy with limited profit and limited loss.
        
        Payoff: max(0, S_T - K1) - max(0, S_T - K2)
        Profit: Payoff - (call1_premium - call2_premium)
        
        Note: K1 < K2 (buy lower strike, sell higher strike)
        
        Parameters:
        -----------
        K1 : float
            Strike price of the long call (lower strike)
        K2 : float
            Strike price of the short call (higher strike)
        call1_premium : float
            Premium paid for the long call at K1
        call2_premium : float
            Premium received for the short call at K2
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        long_call_payoff = self._call_payoff(self.S, K1)
        short_call_payoff = -self._call_payoff(self.S, K2)
        payoff = long_call_payoff + short_call_payoff
        net_premium = call1_premium - call2_premium  # Net cost
        profit = payoff - net_premium
        self._plot_strategy(self.S, payoff, profit, f"Call Spread (CALL K1={K1} - CALL K2={K2})", [K1, K2], net_premium)
        return payoff, profit
    
    def put_spread(self, K1: float, K2: float, put1_premium: float, put2_premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Put Spread (Bear Put Spread) Strategy
        
        Buy a put at higher strike K1 and sell a put at lower strike K2.
        Bearish strategy with limited profit and limited loss.
        
        Payoff: max(0, K1 - S_T) - max(0, K2 - S_T)
        Profit: Payoff - (put1_premium - put2_premium)
        
        Note: K1 > K2 (buy higher strike, sell lower strike)
        
        Parameters:
        -----------
        K1 : float
            Strike price of the long put (higher strike)
        K2 : float
            Strike price of the short put (lower strike)
        put1_premium : float
            Premium paid for the long put at K1
        put2_premium : float
            Premium received for the short put at K2
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        long_put_payoff = self._put_payoff(self.S, K1)
        short_put_payoff = -self._put_payoff(self.S, K2)
        payoff = long_put_payoff + short_put_payoff
        net_premium = put1_premium - put2_premium  # Net cost
        profit = payoff - net_premium
        self._plot_strategy(self.S, payoff, profit, f"Put Spread (PUT K1={K1} - PUT K2={K2})", [K1, K2], net_premium)
        return payoff, profit
    
    # ========================================================================
    # RATIO AND LADDER STRATEGIES
    # ========================================================================
    
    def call_ratio(self, K1: float, K2: float, call1_premium: float, call2_premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Call Ratio Strategy
        
        Buy one call at strike K1 and sell two calls at strike K2.
        Limited profit potential, unlimited risk on the upside.
        
        Payoff: max(0, S_T - K1) - 2 * max(0, S_T - K2)
        Profit: Payoff - (call1_premium - 2 * call2_premium)
        
        Note: K1 < K2 (buy lower strike, sell higher strike)
        
        Parameters:
        -----------
        K1 : float
            Strike price of the long call
        K2 : float
            Strike price of the short calls (2x)
        call1_premium : float
            Premium paid for the long call at K1
        call2_premium : float
            Premium received per short call at K2 (total received = 2 * call2_premium)
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        long_call_payoff = self._call_payoff(self.S, K1)
        short_calls_payoff = -2 * self._call_payoff(self.S, K2)
        payoff = long_call_payoff + short_calls_payoff
        net_premium = call1_premium - 2 * call2_premium
        profit = payoff - net_premium
        self._plot_strategy(self.S, payoff, profit, f"Call Ratio (CALL K1={K1} - 2×CALL K2={K2})", [K1, K2], net_premium)
        return payoff, profit
    
    def put_ratio(self, K1: float, K2: float, put1_premium: float, put2_premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Put Ratio Strategy
        
        Buy one put at strike K1 and sell two puts at strike K2.
        Limited profit potential, unlimited risk on the downside.
        
        Payoff: max(0, K1 - S_T) - 2 * max(0, K2 - S_T)
        Profit: Payoff - (put1_premium - 2 * put2_premium)
        
        Note: K1 > K2 (buy higher strike, sell lower strike)
        
        Parameters:
        -----------
        K1 : float
            Strike price of the long put
        K2 : float
            Strike price of the short puts (2x)
        put1_premium : float
            Premium paid for the long put at K1
        put2_premium : float
            Premium received per short put at K2 (total received = 2 * put2_premium)
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        long_put_payoff = self._put_payoff(self.S, K1)
        short_puts_payoff = -2 * self._put_payoff(self.S, K2)
        payoff = long_put_payoff + short_puts_payoff
        net_premium = put1_premium - 2 * put2_premium
        profit = payoff - net_premium
        self._plot_strategy(self.S, payoff, profit, f"Put Ratio (PUT K1={K1} - 2×PUT K2={K2})", [K1, K2], net_premium)
        return payoff, profit
    
    def call_ladder(self, K1: float, K2: float, K3: float, 
                   call1_premium: float, call2_premium: float, call3_premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Call Ladder Strategy
        
        Buy one call at K1, sell one call at K2, and sell one call at K3.
        Limited profit potential, unlimited risk on the upside.
        
        Payoff: max(0, S_T - K1) - max(0, S_T - K2) - max(0, S_T - K3)
        Profit: Payoff - (call1_premium - call2_premium - call3_premium)
        
        Note: K1 < K2 < K3
        
        Parameters:
        -----------
        K1 : float
            Strike price of the long call
        K2 : float
            Strike price of the first short call
        K3 : float
            Strike price of the second short call
        call1_premium : float
            Premium paid for the long call at K1
        call2_premium : float
            Premium received for the short call at K2
        call3_premium : float
            Premium received for the short call at K3
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        long_call_payoff = self._call_payoff(self.S, K1)
        short_call2_payoff = -self._call_payoff(self.S, K2)
        short_call3_payoff = -self._call_payoff(self.S, K3)
        payoff = long_call_payoff + short_call2_payoff + short_call3_payoff
        net_premium = call1_premium - call2_premium - call3_premium
        profit = payoff - net_premium
        self._plot_strategy(self.S, payoff, profit, 
                          f"Call Ladder (CALL K1={K1} - CALL K2={K2} - CALL K3={K3})", 
                          [K1, K2, K3], net_premium)
        return payoff, profit
    
    def put_ladder(self, K1: float, K2: float, K3: float,
                  put1_premium: float, put2_premium: float, put3_premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Put Ladder Strategy
        
        Buy one put at K1, sell one put at K2, and sell one put at K3.
        Limited profit potential, unlimited risk on the downside.
        
        Payoff: max(0, K1 - S_T) - max(0, K2 - S_T) - max(0, K3 - S_T)
        Profit: Payoff - (put1_premium - put2_premium - put3_premium)
        
        Note: K3 < K2 < K1 (for puts, we buy higher strike, sell lower strikes)
        
        Parameters:
        -----------
        K1 : float
            Strike price of the long put (highest)
        K2 : float
            Strike price of the first short put (middle)
        K3 : float
            Strike price of the second short put (lowest)
        put1_premium : float
            Premium paid for the long put at K1
        put2_premium : float
            Premium received for the short put at K2
        put3_premium : float
            Premium received for the short put at K3
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        long_put_payoff = self._put_payoff(self.S, K1)
        short_put2_payoff = -self._put_payoff(self.S, K2)
        short_put3_payoff = -self._put_payoff(self.S, K3)
        payoff = long_put_payoff + short_put2_payoff + short_put3_payoff
        net_premium = put1_premium - put2_premium - put3_premium
        profit = payoff - net_premium
        self._plot_strategy(self.S, payoff, profit,
                          f"Put Ladder (PUT K1={K1} - PUT K2={K2} - PUT K3={K3})",
                          [K1, K2, K3], net_premium)
        return payoff, profit
    
    # ========================================================================
    # ADVANCED STRATEGIES
    # ========================================================================
    
    def fly(self, K1: float, K2: float, K3: float,
           call1_premium: float, call2_premium: float, call3_premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Butterfly Spread (Fly) Strategy - Using Calls
        
        Buy one call at K1, sell two calls at K2, and buy one call at K3.
        Creates a profit zone around the middle strike K2.
        
        Payoff: max(0, S_T - K1) - 2*max(0, S_T - K2) + max(0, S_T - K3)
        Profit: Payoff - (call1_premium - 2*call2_premium + call3_premium)
        
        Note: K1 < K2 < K3, typically K2 is at-the-money
        
        Parameters:
        -----------
        K1 : float
            Strike price of the first long call (lowest)
        K2 : float
            Strike price of the short calls (middle, typically ATM)
        K3 : float
            Strike price of the second long call (highest)
        call1_premium : float
            Premium paid for the long call at K1
        call2_premium : float
            Premium received per short call at K2 (total received = 2 * call2_premium)
        call3_premium : float
            Premium paid for the long call at K3
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        long_call1_payoff = self._call_payoff(self.S, K1)
        short_calls_payoff = -2 * self._call_payoff(self.S, K2)
        long_call3_payoff = self._call_payoff(self.S, K3)
        payoff = long_call1_payoff + short_calls_payoff + long_call3_payoff
        net_premium = call1_premium - 2 * call2_premium + call3_premium
        profit = payoff - net_premium
        self._plot_strategy(self.S, payoff, profit,
                          f"Fly (CALL K1={K1} - 2×CALL K2={K2} + CALL K3={K3})",
                          [K1, K2, K3], net_premium)
        return payoff, profit
    
    def condor(self, K1: float, K2: float, K3: float, K4: float,
              call1_premium: float, call2_premium: float, call3_premium: float, call4_premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Condor Spread Strategy - Using Calls
        
        Buy one call at K1, sell one call at K2, sell one call at K3, and buy one call at K4.
        Creates a profit zone between K2 and K3.
        
        Payoff: max(0, S_T - K1) - max(0, S_T - K2) - max(0, S_T - K3) + max(0, S_T - K4)
        Profit: Payoff - (call1_premium - call2_premium - call3_premium + call4_premium)
        
        Note: K1 < K2 < K3 < K4
        
        Parameters:
        -----------
        K1 : float
            Strike price of the first long call (lowest)
        K2 : float
            Strike price of the first short call
        K3 : float
            Strike price of the second short call
        K4 : float
            Strike price of the second long call (highest)
        call1_premium : float
            Premium paid for the long call at K1
        call2_premium : float
            Premium received for the short call at K2
        call3_premium : float
            Premium received for the short call at K3
        call4_premium : float
            Premium paid for the long call at K4
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        long_call1_payoff = self._call_payoff(self.S, K1)
        short_call2_payoff = -self._call_payoff(self.S, K2)
        short_call3_payoff = -self._call_payoff(self.S, K3)
        long_call4_payoff = self._call_payoff(self.S, K4)
        payoff = long_call1_payoff + short_call2_payoff + short_call3_payoff + long_call4_payoff
        net_premium = call1_premium - call2_premium - call3_premium + call4_premium
        profit = payoff - net_premium
        self._plot_strategy(self.S, payoff, profit,
                          f"Condor (CALL K1={K1} - CALL K2={K2} - CALL K3={K3} + CALL K4={K4})",
                          [K1, K2, K3, K4], net_premium)
        return payoff, profit
    
    def risk_reversal(self, K1: float, K2: float, put_premium: float, call_premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Risk Reversal Strategy
        
        Buy a put at K1 and sell a call at K2. Creates a synthetic position that
        limits downside risk while maintaining upside potential (or vice versa).
        
        Payoff: max(0, K1 - S_T) - max(0, S_T - K2)
        Profit: Payoff - (put_premium - call_premium)
        
        Note: Typically K1 < K2 (protective put with covered call)
        
        Parameters:
        -----------
        K1 : float
            Strike price of the long put
        K2 : float
            Strike price of the short call
        put_premium : float
            Premium paid for the put option
        call_premium : float
            Premium received for the call option
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        put_payoff = self._put_payoff(self.S, K1)
        call_payoff = -self._call_payoff(self.S, K2)
        payoff = put_payoff + call_payoff
        net_premium = put_premium - call_premium
        profit = payoff - net_premium
        self._plot_strategy(self.S, payoff, profit,
                          f"Risk Reversal (PUT K1={K1} - CALL K2={K2})",
                          [K1, K2], net_premium)
        return payoff, profit
    
    def synthetic_forward(self, K: float, call_premium: float, put_premium: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthetic Forward Strategy
        
        Buy a call and sell a put with the same strike price K.
        Replicates the payoff of a long position in the underlying asset.
        
        Payoff: max(0, S_T - K) - max(0, K - S_T) = S_T - K
        Profit: Payoff - (call_premium - put_premium)
        
        Parameters:
        -----------
        K : float
            Strike price (same for both call and put)
        call_premium : float
            Premium paid for the call option
        put_premium : float
            Premium received for the put option
            
        Returns:
        --------
        tuple of np.ndarray
            (payoff, profit) arrays
        """
        call_payoff = self._call_payoff(self.S, K)
        put_payoff = -self._put_payoff(self.S, K)
        payoff = call_payoff + put_payoff  # This simplifies to S_T - K
        net_premium = call_premium - put_premium
        profit = payoff - net_premium
        self._plot_strategy(self.S, payoff, profit,
                          f"Synthetic Forward (CALL - PUT, K={K})",
                          [K], net_premium)
        return payoff, profit


# ========================================================================
# EXAMPLE USAGE
# ========================================================================

if __name__ == "__main__":
    """
    Example usage of the OptionStrategies class.
    
    This demonstrates how to create an instance and use various strategies.
    Premium values are examples - in practice, these would be calculated using
    option pricing models (e.g., Black-Scholes) or obtained from market data.
    """
    
    # Initialize the strategies class
    strategies = OptionStrategies(S_range=(0, 200), n_points=1000)
    
    # Example: Long Call
    print("\n=== Long Call Example ===")
    strategies.long_call(K=100, premium=5.0)
    
    # Example: Long Put
    print("\n=== Long Put Example ===")
    strategies.long_put(K=100, premium=4.5)
    
    # Example: Short Call
    print("\n=== Short Call Example ===")
    strategies.short_call(K=100, premium=5.0)
    
    # Example: Short Put
    print("\n=== Short Put Example ===")
    strategies.short_put(K=100, premium=4.5)
    
    # Example: Straddle
    print("\n=== Straddle Example ===")
    strategies.straddle(K=100, call_premium=5.0, put_premium=4.5)
    
    # Example: Strangle
    print("\n=== Strangle Example ===")
    strategies.strangle(K1=110, K2=90, call_premium=3.0, put_premium=2.5)
    
    # Example: Call Spread
    print("\n=== Call Spread Example ===")
    strategies.call_spread(K1=95, K2=105, call1_premium=7.0, call2_premium=3.0)
    
    # Example: Put Spread
    print("\n=== Put Spread Example ===")
    strategies.put_spread(K1=105, K2=95, put1_premium=6.0, put2_premium=2.5)
    
    # Example: Call Ratio
    print("\n=== Call Ratio Example ===")
    strategies.call_ratio(K1=95, K2=110, call1_premium=7.0, call2_premium=2.0)
    
    # Example: Put Ratio
    print("\n=== Put Ratio Example ===")
    strategies.put_ratio(K1=105, K2=90, put1_premium=6.0, put2_premium=1.5)
    
    # Example: Call Ladder
    print("\n=== Call Ladder Example ===")
    strategies.call_ladder(K1=90, K2=100, K3=110, 
                         call1_premium=12.0, call2_premium=5.0, call3_premium=2.0)
    
    # Example: Put Ladder
    print("\n=== Put Ladder Example ===")
    strategies.put_ladder(K1=110, K2=100, K3=90,
                        put1_premium=10.0, put2_premium=4.5, put3_premium=1.5)
    
    # Example: Fly (Butterfly)
    print("\n=== Fly Example ===")
    strategies.fly(K1=90, K2=100, K3=110,
                 call1_premium=12.0, call2_premium=5.0, call3_premium=2.0)
    
    # Example: Condor
    print("\n=== Condor Example ===")
    strategies.condor(K1=85, K2=95, K3=105, K4=115,
                    call1_premium=18.0, call2_premium=10.0, call3_premium=5.0, call4_premium=2.0)
    
    # Example: Risk Reversal
    print("\n=== Risk Reversal Example ===")
    strategies.risk_reversal(K1=90, K2=110, put_premium=4.5, call_premium=3.0)
    
    # Example: Synthetic Forward
    print("\n=== Synthetic Forward Example ===")
    strategies.synthetic_forward(K=100, call_premium=5.0, put_premium=4.5)
    
    print("\n=== All strategies plotted successfully! ===")

