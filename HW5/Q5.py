from math import e
import numpy as np
from collections import defaultdict
import pandas as pd

def thomas(lower, diag, upper, rhs):
    n = len(diag)
    a = lower.astype(float).copy()
    b = diag.astype(float).copy()
    c = upper.astype(float).copy()
    d = rhs.astype(float).copy()

    for i in range(1, n):
        factor = a[i-1]/b[i-1]
        b[i] -= factor * c[i-1]
        d[i] -= factor * d[i-1]

    x = np.zeros(n)
    x[-1] = d[-1]/b[-1]

    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x

def collect_data(data, phi, u, h):
    data['h'].append(h)
    data['error'].append(np.linalg.norm(u - phi, ord=np.inf))
    data['first_order_err'].append(np.linalg.norm((u - phi)/h, ord=np.inf))
    data['second_order_err'].append(np.linalg.norm((u-phi)/h**2, ord=np.inf))
    data['third_ordder_err'].append(np.linalg.norm((u-phi)/h**3, ord=np.inf))

def theta(x):
    return np.exp(x**2)

def cg(A, b, tol=1e-9):
    n = len(b)
    rn = b.copy()
    xn = np.zeros(n)
    x_next = xn.copy()
    d = rn.copy()
    r_next = rn.copy()
    for k in range(n):
        z = A @ d
        alpha = (rn@rn)/(d@z)
        x_next = xn + alpha * d
        r_next = rn - alpha * z
        beta = (r_next@r_next)/(rn@rn)
        d = r_next + beta * d
        if np.linalg.norm(rn - r_next) < tol:
            return x_next
        xn = x_next
        rn = r_next
    return xn

def gd(A, b, tol=1e-9, max_iter=100):
    n = len(b)
    x = np.zeros(n)
    r = b - A @ x
    for k in range(max_iter):
        if np.linalg.norm(r) < tol:
            return x
        alpha = (r.T @ r) / (r.T @ A @ r)
        x = x + alpha * r
        r = r - alpha * A @ r
    return x

def main():
    thomas_data = defaultdict(list)
    gd_data = defaultdict(list)
    cg_data = defaultdict(list)
    a = 0
    b = 1
    p_list = [i for i in range(1, 15)]
    for p in p_list:
        h = 2**-p
        N = 2 ** p
        x = np.linspace(a, b, N+1)
        x_interior = x[1:N]
        n = N-1
        diag = 2.0 + h**2 *(4*x_interior**2 + 2)
        lower = -np.ones(n-1)
        upper = -np.ones(n-1)
        rhs = np.zeros(n)
        rhs[0] += 1
        rhs[-1] += e
        phi_interior = thomas(lower, diag, upper, rhs)
        phi = np.zeros(N+1)
        phi[0] = 1.0
        phi[N] = e
        phi[1:N] = phi_interior
        u = theta(x)
        collect_data(thomas_data, phi, u, h)
        A = (
            np.diag(diag)
            + np.diag(upper, k=1)
            + np.diag(lower, k=-1)
        )
        phi_interior = cg(A, rhs)
        phi = np.zeros(N+1)
        phi[0] = 1.0
        phi[N] = e
        phi[1:N] = phi_interior
        collect_data(cg_data, phi, u, h)
        phi_interior = gd(A, rhs)
        phi = np.zeros(N+1)
        phi[0] = 1.0
        phi[N] = e
        phi[1:N] = phi_interior
        collect_data(gd_data, phi, u, h)
    thomas_df = pd.DataFrame(thomas_data)
    cg_df = pd.DataFrame(cg_data)
    gd_df = pd.DataFrame(gd_data)
    thomas_df.to_csv('thomas.csv')
    cg_df.to_csv('cg.csv')
    gd_df.to_csv('gd.csv')
        


if __name__ == '__main__':
    main()
