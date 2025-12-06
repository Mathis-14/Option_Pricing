import numpy as np
import matplotlib.pyplot as plt
import os
import math

# ============================================================
# Academic-style plot of a European put:
#   - f(S): Black–Scholes put price (convex in S)
#   - g(S): intrinsic value max(0, K - S)
#   - shaded area between f and g: time value f - g
# ============================================================

# --- Model parameters (you can change these) -----------------
K = 100        # Strike price
T = 1.0        # Time to maturity in years
r = 0.02       # Risk-free interest rate
sigma = 0.3    # Volatility of the underlying

# --- Spot price grid -----------------------------------------
# We avoid S = 0 to keep log(S/K) well-defined in Black–Scholes.
S_min, S_max = 1e-3, 200
S = np.linspace(S_min, S_max, 400)

# --- Intrinsic value: g(S) = max(0, K - S) -------------------
g = np.maximum(0, K - S)

# --- Normal CDF using math.erf, applied elementwise ----------
def N_array(x: np.ndarray) -> np.ndarray:
    """
    Standard normal cumulative distribution function Φ(x),
    applied elementwise to a NumPy array, using math.erf.
    """
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))

# --- Black–Scholes put price: f(S) --------------------------
# d1 and d2 in the Black–Scholes formula
d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

# Put price f(S) = K e^{-rT} Φ(-d2) − S Φ(-d1)
f = K * np.exp(-r * T) * N_array(-d2) - S * N_array(-d1)

# ============================================================
# Plotting
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Intrinsic value: piecewise linear payoff
ax.plot(S, g, linewidth=2, color="orange", label="g: Intrinsic value")

# Total put value: Black–Scholes price (smooth, convex)
ax.plot(S, f, linewidth=2, color="steelblue", label="f: Put value (Black–Scholes)")

# Shade the time value region (f - g)
ax.fill_between(S, g, f, where=f >= g,
                alpha=0.25, hatch='//', edgecolor='black')

# Axis labels and title
ax.set_xlabel("Spot Price $S_t$")
ax.set_ylabel("Put Value $P_t$")
ax.set_title("Relationship between Put Value and Spot Price")

# Mark the strike K with a vertical dashed line
ax.axvline(K, linestyle="--", linewidth=1.2, color="orange")
ax.text(K, -2, "K (Strike)", ha="center", va="top")

# Annotate the time value near S ≈ K
S_annot = K - 15
g_annot = max(0, K - S_annot)

# Recompute f(S_annot) with scalar math for the annotation point
d1_a = (math.log(S_annot / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
d2_a = d1_a - sigma * math.sqrt(T)

def N_scalar(x: float) -> float:
    """Scalar version of the standard normal CDF Φ(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

f_annot = K * math.exp(-r * T) * N_scalar(-d2_a) - S_annot * N_scalar(-d1_a)

ax.annotate("Time Value (f - g)",
            xy=(S_annot, (g_annot + f_annot) / 2.0),
            xytext=(S_annot - 25, (g_annot + f_annot) / 2.0 + 15),
            arrowprops=dict(arrowstyle="->"))

# Legend and axis limits
ax.legend(loc="upper left")
ax.set_xlim(0, S_max)
ax.set_ylim(0, max(f) + 20)

plt.tight_layout()

# ============================================================
# Save the figure in folder "plot"
# ============================================================
os.makedirs("plot", exist_ok=True)  # create folder if needed
output_file = "plot/put_value_decomposition_convex.png"
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Plot saved to: {output_file}")

