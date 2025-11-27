import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------------
# Create an academic-style 2D plot of a European call option value
# decomposed into intrinsic value and time value
# -------------------------------------------------------------

# Spot price range (horizontal axis)
S_min, S_max = 0, 200
S = np.linspace(S_min, S_max, 400)

# Strike price
K = 100

# Intrinsic value g = max(0, S − K)
g = np.maximum(0, S - K)

# Total option value f (convex curve, always above intrinsic value)
# We add a smooth "time value hump" around the strike
time_value = 15 * np.exp(-0.5 * ((S - K) / 25) ** 2)
f = g + time_value

# -------------------------------------------------------------
# Plot configuration
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

# Plot intrinsic value (piecewise linear)
ax.plot(S, g, linewidth=2, color="orange", label="g: Intrinsic value")

# Plot total call value (convex)
ax.plot(S, f, linewidth=2, color="steelblue", label="f: Call value")

# Shade (f - g), the time value of the option
ax.fill_between(S, g, f, where=f >= g,
                alpha=0.25, hatch='//', edgecolor='black')

# Axis labels and title
ax.set_xlabel("Spot Price $S_t$")
ax.set_ylabel("Call Value $C_t$")
ax.set_title("Relationship between Call Value and Spot Price")

# Draw a vertical dashed line at strike K and annotate
ax.axvline(K, linestyle="--", linewidth=1.2, color="orange")
ax.text(K, -4, "K (Strike)", ha="center", va="top")

# Annotate the time value zone
S_annot = K + 25  # a spot price slightly to the right of K for annotation
g_annot = np.maximum(0, S_annot - K)
f_annot = g_annot + 15 * np.exp(-0.5 * ((S_annot - K) / 25) ** 2)

ax.annotate("Time Value (f - g)",
            xy=(S_annot, (g_annot + f_annot) / 2),
            xytext=(S_annot + 30, (g_annot + f_annot) / 2 + 20),
            arrowprops=dict(arrowstyle="->"))

# Legend and limits
ax.legend(loc="upper left")
ax.set_xlim(S_min, S_max)
ax.set_ylim(0, max(f) + 20)

plt.tight_layout()

# -------------------------------------------------------------
# Save the plot inside folder "plot"
# -------------------------------------------------------------
os.makedirs("plot", exist_ok=True)     # create folder if it does not exist
output_file = "plot/call_value_decomposition.png"
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Plot saved successfully to: {output_file}")
