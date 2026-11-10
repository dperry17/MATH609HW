import pandas as pd
import matplotlib.pyplot as plt

files = {
    "Thomas Algorithm": "thomas.csv",
    "Conjugate Gradient": "cg.csv",
    "Gradient Descent": "gd.csv"
}

for algorithm, filename in files.items():
    df = pd.read_csv(filename)

    plt.figure(figsize=(8, 5))

    # Unscaled max error
    plt.loglog(
        df["h"],
        df["error"],
        marker="o",
        label=r"$\|u_h-\phi_h\|_\infty$"
    )

    # Error divided by h
    plt.loglog(
        df["h"],
        df["first_order_err"],
        marker="s",
        label=r"$\|u_h-\phi_h\|_\infty/h$"
    )

    # Error divided by h^2
    plt.loglog(
        df["h"],
        df["second_order_err"],
        marker="^",
        label=r"$\|u_h-\phi_h\|_\infty/h^2$"
    )

    # Error divided by h^3
    plt.loglog(
        df["h"],
        df["third_ordder_err"],
        marker="d",
        label=r"$\|u_h-\phi_h\|_\infty/h^3$"
    )

    plt.xlabel(r"Step size $h$")
    plt.ylabel("Error")
    plt.title(f"Errors: {algorithm}")

    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    # Optional: display large h on the left and small h on the right
    plt.gca().invert_xaxis()

    output_name = filename.replace(".csv", "_errors.png")
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.show()