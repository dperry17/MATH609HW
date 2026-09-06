import numpy as np
import pandas as pd

def GE(A, b, n):
    for i in range(n):
        pivot = i + np.argmax(np.abs(A[i:, i]))

        if pivot != i:
            A[[i, pivot]] = A[[pivot, i]]
            b[[i, pivot]] = b[[pivot, i]]

        for j in range(i+1, n):
            factor = A[i, j] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
        x = np.zeros(n)
        for i in range(n-1,-1, -1):
            x[i] = (b[i] - A[i, i+1:] @ x[i + 1:]) / A[i, i]
    return x 

def main():
    data = []
    runs = [i for i in range(4, 21)]
    for n in runs:
        A = np.ndarray((n, n))
        b = np.array([1/i * (pow(1 + i, n) - 1) for i in range(1, n+1)])
        for i in range(n):
            for j in range(n):
                A[i, j] = pow(2 + i, j)
        x = GE(A, b, n)
        data.append({'n': n, 'Solutions': x.tolist()})
    print(data)
    pd.DataFrame(data, columns = ['n', 'Solutions']).to_csv("problem5_results.csv")

if __name__ == '__main__':
    main()
