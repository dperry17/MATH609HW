from math import e
import numpy as np
from collections import defaultdict
import pandas as pd

def thomas(l2, l1, d, u1, u2, b):
    n = len(d)
    d, l1, l2 = d.astype(float).copy(), l1.astype(float).copy(), l2.astype(float).copy()
    u1, u2, b = u1.astype(float).copy(), u2.astype(float).copy(), b.astype(float).copy()

    for i in range(1, n):
        if abs(d[i - 1]) < 1e-15:
            raise ZeroDivisionError(f"zero pivot at row {i-1}")

        if l1[i - 1] != 0.0:                 # skip if nothing to eliminate
            w = l1[i - 1] / d[i - 1]
            d[i] -= w * u1[i - 1]
            if i - 1 < len(u2):
                u1[i] -= w * u2[i - 1]
            b[i] -= w * b[i - 1]

        if i < n - 1 and (i - 1) < len(l2) and l2[i - 1] != 0.0:
            w2 = l2[i - 1] / d[i - 1]
            l1[i] -= w2 * u1[i - 1]
            d[i + 1] -= w2 * u2[i - 1]
            b[i + 1] -= w2 * b[i - 1]

    x = np.zeros(n)
    x[-1] = b[-1] / d[-1]
    if n >= 2:
        x[-2] = (b[-2] - u1[-1] * x[-1]) / d[-2]
    for i in range(n - 3, -1, -1):
        x[i] = (b[i] - u1[i] * x[i + 1] - u2[i] * x[i + 2]) / d[i]
    return x

def collect_data(data, phi, u, h):
    data['h'].append(h)
    data['error'].append(np.linalg.norm(u - phi, ord=np.inf))
    data['first_order_err'].append(np.linalg.norm((u - phi), ord=np.inf)/h)
    data['third_order_err'].append(np.linalg.norm((u-phi), ord=np.inf)/h**3)
    data['fourth_order_err'].append(np.linalg.norm((u-phi), ord=np.inf)/h**4)
    data['fifth_order_err'].append(np.linalg.norm((u-phi), ord=np.inf)/h**5)

def theta(x):
    return np.exp(x**2)

def cg(A, b, tol=1e-9, max_iter=1000000):
    n = len(b)
    rn = b.copy()
    xn = np.zeros(n)
    x_next = xn.copy()
    d = rn.copy()
    r_next = rn.copy()
    for _ in range(max_iter):
        z = A @ d
        if np.linalg.norm(z) == 0:
            print("z is zero")
            return xn
        alpha = (rn@rn)/(d @ z)
        x_next = xn + alpha * d
        r_next = rn - alpha * z
        beta = (r_next@r_next)/(rn@rn)
        d = r_next + beta * d
        if np.linalg.norm(rn - r_next) < tol:
            return x_next
        xn = x_next
        rn = r_next
    return xn

def gd(A, b, tol=1e-9, max_iter=1000000):
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

def build_system(x, h):
    N = len(x)
    m = N-2
    A = np.zeros((m, m))
    rhs = np.zeros(m)
    q = (4*x**2 + 2)
    phi_left = 1.0
    phi_right = np.e

    for j in range(m):
        i = j + 1
        if i in (1, N - 2):

            diag = 24.0 + 12.0 * h**2 * q[i]
            left = -12.0
            right = -12.0
            A[j, j] = diag
            if i - 1 == 0:
                rhs[j] -= left * phi_left
            else:
                A[j, j - 1] = left
            if i + 1 == N - 1:
                rhs[j] -= right * phi_right
            else:
                A[j, j + 1] = right
        else:
            A[j, j] = 30.0 + 12.0 * h**2 * q[i]

            if i - 1 == 0:
                rhs[j] -= (-16.0) * phi_left
            else:
                A[j, j - 1] = -16.0
            if i + 1 == N - 1:
                rhs[j] -= (-16.0) * phi_right
            else:
                A[j, j + 1] = -16.0

            if i - 2 == 0:
                rhs[j] -= (1.0) * phi_left
            elif i - 2 > 0:
                A[j, j - 2] = 1.0
            if i + 2 == N - 1:
                rhs[j] -= (1.0) * phi_right
            elif i + 2 < N - 1:
                A[j, j + 2] = 1.0
    return A, rhs, phi_left, phi_right

def symmetric_scale(A, rhs):
    m = len(rhs)

    # Special small-grid cases
    if m <= 2:
        weights = np.ones(m)
    else:
        weights = np.full(m, 0.75)
        weights[0] = 1.0
        weights[-1] = 1.0

    A_sym = weights[:, None] * A
    rhs_sym = weights * rhs

    return A_sym, rhs_sym

def main():
    thomas_data = defaultdict(list)
    gd_data = defaultdict(list)
    cg_data = defaultdict(list)
    a = 0
    b = 1
    p_list = [i for i in range(1, 7)]
    for p in p_list:
        h = 2**-p
        N = 2 ** p + 1
        x = np.linspace(a, b, num=N)

        A, rhs, phi_left, phi_right = build_system(x, h)
        A, rhs = symmetric_scale(A, rhs)
        d = np.diag(A).copy()
        u1 = np.diag(A, k=1).copy()
        u2 = np.diag(A, k=2).copy()
        l1 = np.diag(A, k=-1).copy()
        l2 = np.diag(A, k=-2).copy()
        phi = thomas(l2, l1, d, u1, u2, rhs.copy())
        phi = np.concatenate(([phi_left], phi, [phi_right]))
        u = theta(x)
        collect_data(thomas_data, phi, u, h)
        phi = np.concatenate(([phi_left], cg(A, rhs), [phi_right]))
        collect_data(cg_data, phi, u, h)
        phi = np.concatenate(([phi_left], gd(A, rhs), [phi_right]))
        collect_data(gd_data, phi, u, h)
    thomas_df = pd.DataFrame(thomas_data)
    cg_df = pd.DataFrame(cg_data)
    gd_df = pd.DataFrame(gd_data)
    thomas_df.to_csv('thomas.csv')
    cg_df.to_csv('cg.csv')
    gd_df.to_csv('gd.csv')
        


if __name__ == '__main__':
    main()
