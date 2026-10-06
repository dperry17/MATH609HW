import numpy as np
import pandas as pd
from collections import defaultdict

def run(T, c, b, A):
    table = defaultdict(list)
    x_old = np.zeros(9)
    x_new = x_old.copy()
    epsilon = 1e-2
    r = 0
    r_prev = 0
    r_suc = 0
    max_iter = 100
    final_iter = 0

    for i in range(max_iter):
        x_new = T @ x_old + c
        r = np.linalg.norm(b - (A@x_new), np.inf)
        if r_prev > 0:
            r_suc = r/r_prev
        if r <=epsilon:
            break
        x_old = x_new.copy()
        final_iter = i
        r_prev = r
    table['k'].append(final_iter)
    table['r_norm'].append(r)
    table['r_successive'].append(r_suc)
    return pd.DataFrame(table), x_new
        

def main():
    B = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
    I = -1 * np.eye(3)
    Z = np.zeros((3,3))
    A = np.block([[B, I, Z], [I, B, I], [Z, I, B]])
    b = np.array([0,0,1,0,0,1,0,0,1])

    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    D = np.diag(np.diag(A))

    T_j = -1 * np.linalg.inv(D) @ (L+U)
    c_j = np.linalg.inv(D) @ b
    T_gs = -1 * np.linalg.inv(D+L) @ U
    c_gs = np.linalg.inv(D+L) @ b

    df_j, approx_j = run(T_j, c_j, b, A)
    print(f'The approximate solution from jacobi is {approx_j}')
    df_j.to_csv('jacobi-q4.csv')

    df_gs, approx_gs = run(T_gs, c_gs, b, A)
    print(f'The approximate solution from Gauss-siedel is {approx_gs}')
    df_gs.to_csv('gauss-seidel-q4.csv')

if __name__ == '__main__':
    main()