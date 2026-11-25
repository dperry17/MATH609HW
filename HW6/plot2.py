import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("stationary_results.csv")

# Make h increase from left to right: 1/16, 1/8, 1/4
df = df.sort_values("h")

# Set a different color and marker for every method
styles = {
    "Jacobi": {"color": "blue", "marker": "o"},
    "Gauss-Seidel": {"color": "green", "marker": "s"},
    "Optimal SOR": {"color": "red", "marker": "^"}
}

plt.figure(figsize=(8, 5))

for method, group in df.groupby("method"):
    style = styles[method]

    # Measured final residual ratio
    plt.plot(
        group["h"],
        group["final_consecutive_ratio_rk_over_rk_minus_1"],
        linestyle="-",
        linewidth=2,
        marker=style["marker"],
        markersize=7,
        color=style["color"],
        label=f"{method}: measured ratio"
    )

    # Theoretical spectral radius
    plt.plot(
        group["h"],
        group["theoretical_spectral_radius"],
        linestyle="--",
        linewidth=2,
        marker=style["marker"],
        markersize=7,
        color=style["color"],
        alpha=0.75,
        label=f"{method}: theoretical $\\rho(B)$"
    )

# Label the h-values as fractions
plt.xticks(
    [1/16, 1/8, 1/4],
    [r"$1/16$", r"$1/8$", r"$1/4$"]
)

plt.xlabel(r"Grid spacing $h$")
plt.ylabel("Residual ratio / spectral radius")
plt.title("Measured Residual Ratios versus Theoretical Spectral Radii")
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(fontsize=9)
plt.tight_layout()

# Save and display
plt.savefig("residual_ratios.png", dpi=300)
plt.show()