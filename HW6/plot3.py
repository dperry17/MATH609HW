import pandas as pd
import matplotlib.pyplot as plt

# Read Krylov-method results
df = pd.read_csv("krylov_results.csv")

# Sort from h = 1/4 to h = 1/16
df = df.sort_values("h", ascending=False)

plt.figure(figsize=(8, 5))

for method, group in df.groupby("method"):
    plt.plot(
        group["h"],
        group["final_consecutive_ratio_rk_over_rk_minus_1"],
        marker="o",
        linewidth=2,
        markersize=7,
        label=method
    )

plt.xticks(
    [1/4, 1/8, 1/16],
    [r"$1/4$", r"$1/8$", r"$1/16$"]
)

plt.xlabel(r"Grid spacing $h$")
plt.ylabel(r"$\|r^{(k)}\|_\infty / \|r^{(k-1)}\|_\infty$")
plt.title("Final Consecutive Residual Ratios: GD and CG")
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend()
plt.tight_layout()

plt.savefig("gd_cg_residual_ratios.png", dpi=300)
plt.show()