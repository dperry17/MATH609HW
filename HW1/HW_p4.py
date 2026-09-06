import numpy as np
import matplotlib.pyplot as plt

def thomas(A_main, A_sub, A_sup, f):
    N = A_main.shape[0]
    u = np.zeros((N,))
    temp = np.empty((N,))
    temp[0] = A_sup[0]/A_main[0]
    u[0] = f[0] / A_main[0]
    for i in range(1, N): #using 0 index so thomas goes from 1 tp n-1
        if i < N-1:
            temp[i] = A_sup[i] / (A_main[i] - A_sub[i] * temp[i-1])
        u[i] = (f[i] - A_sub[i] * u[i-1]) / (A_main[i] - A_sub[i] * temp[i-1])
    for i in range(N-2, 0, -1):
        u[i] -= temp[i] * u[i+1]
    return u


def main(epsilon=1e-3):
    f = []

    for n in range(1, 9):
        plt.figure()
        N = pow(2, n)
        h = pow(2, -n)
        f = np.array([2*i*h + 1 for i in range(N)])
        A_main = np.array([2*epsilon + pow(h, 2) for i in range(N)])
        A_sub = np.array([0] + [-1*epsilon for i in range(1, N)])
        A_sup = np.array([-1*epsilon for i in range(N-1)] + [0])
        u = thomas(A_main, A_sub, A_sup, f)
        x = np.array([i*h for i in range(N)])
        plt.title(f"Solution plots at size {n}")
        plt.ylabel("Solution to system")
        plt.xlabel("xi")
        plt.plot(x, u)
        plt.savefig(f"figs/p4_sol_size_{n}")

if __name__ == '__main__':
    main()