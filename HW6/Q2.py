import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import diags, eye, kron

def make_system(N):
    """
    N = number of subintervals in each direction.
    h = 1/N.
    There are m = N-1 interior nodes per direction.
    """

    h = 1.0 / N
    m = N - 1

    T = diags(
        diagonals=[
            -np.ones(m - 1),
            2 * np.ones(m),
            -np.ones(m - 1)
        ],
        offsets=[-1, 0, 1],
        format="csr"
    )

    I = eye(m, format="csr")

    A = kron(I, T) + kron(T, I)
    A = A.tocsr()

    f = np.zeros(m * m)
    f[(m - 1) * m: m * m] = 1.0

    return h, A, f


def inf_norm(v):
    """Infinity norm."""
    return np.max(np.abs(v))


def residual(A, x, f):
    """r = f - A x."""
    return f - A @ x


def jacobi(A, f, tol=1e-2, max_iter=1_000_000):
    n = len(f)
    x = np.zeros(n)

    D = A.diagonal()
    r = residual(A, x, f)
    r0_norm = inf_norm(r)

    ratios = []

    for k in range(1, max_iter + 1):
        # x^(k+1) = D^{-1} [f - (A-D)x^(k)]
        x_new = x + r / D

        r_new = residual(A, x_new, f)
        r_new_norm = inf_norm(r_new)
        r_old_norm = inf_norm(r)

        ratio = r_new_norm / r_old_norm
        ratios.append(ratio)

        x = x_new
        r = r_new

        if r_new_norm / r0_norm <= tol:
            return x, k, ratio, r_new_norm / r0_norm, ratios


def gauss_seidel(A, f, tol=1e-2, max_iter=1_000_000):
    n = len(f)
    x = np.zeros(n)

    r = residual(A, x, f)
    r0_norm = inf_norm(r)
    ratios = []

    for k in range(1, max_iter + 1):
        r_old_norm = inf_norm(r)

        # Update components one at a time.
        # Newly computed x[j] values are used immediately.
        for i in range(n):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]

            sigma = 0.0
            diagonal = 0.0

            for index in range(row_start, row_end):
                j = A.indices[index]
                aij = A.data[index]

                if i == j:
                    diagonal = aij
                else:
                    sigma += aij * x[j]

            x[i] = (f[i] - sigma) / diagonal

        r = residual(A, x, f)
        r_new_norm = inf_norm(r)

        ratio = r_new_norm / r_old_norm
        ratios.append(ratio)

        if r_new_norm / r0_norm <= tol:
            return x, k, ratio, r_new_norm / r0_norm, ratios

    
def sor(A, f, omega, tol=1e-2, max_iter=1_000_000):
    n = len(f)
    x = np.zeros(n)

    r = residual(A, x, f)
    r0_norm = inf_norm(r)
    ratios = []

    for k in range(1, max_iter + 1):
        r_old_norm = inf_norm(r)

        for i in range(n):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]

            sigma = 0.0
            diagonal = 0.0

            for index in range(row_start, row_end):
                j = A.indices[index]
                aij = A.data[index]

                if i == j:
                    diagonal = aij
                else:
                    sigma += aij * x[j]

            # Ordinary Gauss-Seidel value
            x_gs = (f[i] - sigma) / diagonal

            # SOR relaxation
            x[i] = (1.0 - omega) * x[i] + omega * x_gs

        r = residual(A, x, f)
        r_new_norm = inf_norm(r)

        ratio = r_new_norm / r_old_norm
        ratios.append(ratio)

        if r_new_norm / r0_norm <= tol:
            return x, k, ratio, r_new_norm / r0_norm, ratios

    
def gradient_descent(A, f, tol=1e-8, max_iter=1_000_000):
    n = len(f)
    x = np.zeros(n)

    r = residual(A, x, f)
    r0_norm = inf_norm(r)
    ratios = []

    for k in range(1, max_iter + 1):
        Ar = A @ r

        alpha = np.dot(r, r) / np.dot(r, Ar)
        x = x + alpha * r

        r_new = residual(A, x, f)
        r_new_norm = inf_norm(r_new)
        r_old_norm = inf_norm(r)

        ratio = r_new_norm / r_old_norm
        ratios.append(ratio)

        r = r_new

        if r_new_norm / r0_norm <= tol:
            return x, k, ratio, r_new_norm / r0_norm, ratios

    
def conjugate_gradient(A, f, tol=1e-8, max_iter=1_000_000):
    n = len(f)
    x = np.zeros(n)

    r = residual(A, x, f)
    p = r.copy()

    r0_norm = inf_norm(r)
    rs_old = np.dot(r, r)
    ratios = []

    for k in range(1, max_iter + 1):
        Ap = A @ p

        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p

        r_new = r - alpha * Ap

        r_new_norm = inf_norm(r_new)
        r_old_norm = inf_norm(r)

        ratio = r_new_norm / r_old_norm
        ratios.append(ratio)

        if r_new_norm / r0_norm <= tol:
            return x, k, ratio, r_new_norm / r0_norm, ratios

        rs_new = np.dot(r_new, r_new)
        beta = rs_new / rs_old

        p = r_new + beta * p
        r = r_new
        rs_old = rs_new

    raise RuntimeError("Conjugate gradient did not converge before max_iter.")


if __name__ == "__main__":

    stationary_rows = []
    krylov_rows = []

    # Store one solution to plot later
    solution_to_plot = None
    N_to_plot = 16

    for N in [4, 8, 16]:
        h, A, f = make_system(N)

        # Theoretical spectral radii from the homework
        rho_jacobi = np.cos(np.pi * h)
        rho_gs = rho_jacobi ** 2

        # Optimal SOR parameter
        omega_star = 2.0 / (1.0 + np.sqrt(1.0 - rho_jacobi ** 2))

        # Equivalent formula:
        # omega_star = 2.0 / (1.0 + np.sin(np.pi * h))

        rho_sor = (1.0 - np.sin(np.pi * h)) / (1.0 + np.sin(np.pi * h))

        x_j, k_j, ratio_j, final_j, _ = jacobi(A, f, tol=1e-2)

        stationary_rows.append({
            "method": "Jacobi",
            "h": h,
            "N": N,
            "iterations_k": k_j,
            "final_residual_ratio_rk_over_r0": final_j,
            "final_consecutive_ratio_rk_over_rk_minus_1": ratio_j,
            "theoretical_spectral_radius": rho_jacobi,
            "omega": np.nan
        })

        x_gs, k_gs, ratio_gs, final_gs, _ = gauss_seidel(A, f, tol=1e-2)

        stationary_rows.append({
            "method": "Gauss-Seidel",
            "h": h,
            "N": N,
            "iterations_k": k_gs,
            "final_residual_ratio_rk_over_r0": final_gs,
            "final_consecutive_ratio_rk_over_rk_minus_1": ratio_gs,
            "theoretical_spectral_radius": rho_gs,
            "omega": 1.0
        })

        x_sor, k_sor, ratio_sor, final_sor, _ = sor(
            A, f, omega=omega_star, tol=1e-2
        )

        stationary_rows.append({
            "method": "Optimal SOR",
            "h": h,
            "N": N,
            "iterations_k": k_sor,
            "final_residual_ratio_rk_over_r0": final_sor,
            "final_consecutive_ratio_rk_over_rk_minus_1": ratio_sor,
            "theoretical_spectral_radius": rho_sor,
            "omega": omega_star
        })

        x_gd, k_gd, ratio_gd, final_gd, _ = gradient_descent(
            A, f, tol=1e-8
        )

        krylov_rows.append({
            "method": "Gradient Descent",
            "h": h,
            "N": N,
            "iterations_k": k_gd,
            "final_residual_ratio_rk_over_r0": final_gd,
            "final_consecutive_ratio_rk_over_rk_minus_1": ratio_gd
        })

        x_cg, k_cg, ratio_cg, final_cg, _ = conjugate_gradient(
            A, f, tol=1e-8
        )

        krylov_rows.append({
            "method": "Conjugate Gradient",
            "h": h,
            "N": N,
            "iterations_k": k_cg,
            "final_residual_ratio_rk_over_r0": final_cg,
            "final_consecutive_ratio_rk_over_rk_minus_1": ratio_cg
        })

        # Keep CG solution at h=1/16 for the plot
        if N == N_to_plot:
            solution_to_plot = x_cg

    stationary_table = pd.DataFrame(stationary_rows)
    stationary_table.to_csv("stationary_results.csv", index=False)

    krylov_table = pd.DataFrame(krylov_rows)
    krylov_table.to_csv("krylov_results.csv", index=False)

    print("\nStationary-method results:")
    print(stationary_table.to_string(index=False))

    print("\nGradient descent and CG results:")
    print(krylov_table.to_string(index=False))

    print("\nCreated files:")
    print("  stationary_results.csv")
    print("  krylov_results.csv")