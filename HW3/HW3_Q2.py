import numpy as np
import pandas as pd
from math import sqrt
from collections import defaultdict

def SOR(c, T, n, x_actual):
    x_new = np.zeros(n)
    x_old = x_new.copy()
    e_prev, e, e_suc = 0, 0, 0
    table = defaultdict(list)
    for ii in range(10):
        x_new = T @ x_old + c
        table['k'] += [ii+1]
        table['x[1]'] += [x_new[0]]
        table['x[2]'] += [x_new[1]]
        table['x[3]'] += [x_new[2]]
        e = np.linalg.norm(x_new - x_actual, np.inf)
        table['e'] += [e]
        if e_prev > 0:
            e_suc = e / e_prev
        table['e_suc'] += [e_suc]
        e_prev = e
        x_old = x_new.copy()
    return x_new, pd.DataFrame(table)

def jacobi(c, T, n, x_actual):
    x_new = np.zeros(n)
    x_old = x_new.copy()
    e_prev, e, e_suc = 0, 0, 0
    table = defaultdict(list)
    for ii in range(10):
        x_new = T @ x_old + c
        table['k'] += [ii+1]
        table['x[1]'] += [x_new[0]]
        table['x[2]'] += [x_new[1]]
        table['x[3]'] += [x_new[2]]
        e = np.linalg.norm(x_new - x_actual, np.inf)
        table['e'] += [e]
        if e_prev > 0:
            e_suc = e / e_prev
        table['e_suc'] += [e_suc]
        e_prev = e
        x_old = x_new.copy()
    return x_new, pd.DataFrame(table)

def gauss_seidel(c, T, n, x_actual):
    x_new = np.zeros(n)
    x_old = x_new.copy()
    e_prev, e, e_suc = 0, 0, 0
    table = defaultdict(list)
    for ii in range(10):
        x_new = T @ x_old + c
        table['k'] += [ii+1]
        table['x[1]'] += [x_new[0]]
        table['x[2]'] += [x_new[1]]
        table['x[3]'] += [x_new[2]]
        e = np.linalg.norm(x_new - x_actual, np.inf)
        table['e'] += [e]
        if e_prev > 0:
            e_suc = e / e_prev
        table['e_suc'] += [e_suc]
        e_prev = e
        x_old = x_new.copy()

    return x_new, pd.DataFrame(table)

def main():
    A = np.array([[3,1,0], [1,3,1], [0,1,3]])
    b = np.array([4,5,4])
    n = 3
    x_act = np.array([1,1,1])

    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    D = np.diag(np.diag(A))
    omega = 9 - 3 * sqrt(7)
    c_j = np.linalg.inv(D) @ b
    c_gs = np.linalg.inv(D+L) @ b
    c_sor = omega * np.linalg.inv((D + omega * L)) @ b

    T_gauss_seidel = -1 * np.linalg.inv(D+L) @ U
    T_jacobi = -1 * np.linalg.inv(D) @ (L+U)
    T_SOR = np.linalg.inv(D+omega*L) @ ((1-omega)*D - omega * U)

    x_approx_gs, df_gs = gauss_seidel(c_gs, T_gauss_seidel, n, x_act)
    print(f'The approx solution for Gauss Siedel is {x_approx_gs}')
    df_gs.to_csv('gauss-siedel-table.csv')

    x_approx_j, df_j = jacobi(c_j, T_jacobi, n, x_act)
    print(f'The approx solution for the jacobi method is {x_approx_j}')
    df_j.to_csv('jacobi-table.csv')

    x_approx_sor, df_sor = SOR(c_sor, T_SOR, n, x_act)
    print(f'The approx solution for the sor method is {x_approx_sor}')
    df_sor.to_csv('sor-table.csv')

if __name__ == '__main__':
    main()