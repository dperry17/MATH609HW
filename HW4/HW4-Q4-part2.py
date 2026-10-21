import numpy as np


def conjugate_gradient(A, b, tol=1e-10):
    n = len(b)
    x = np.zeros(n)

    r = b - A @ x
    d = r.copy()

    for _ in range(n):
        alpha = (r @ r) / (d @ A @ d)

        x = x + alpha * d
        r_new = r - alpha * (A @ d)

        if np.linalg.norm(r_new) < tol:
            return x

        beta = (r_new @ r_new) / (r @ r)
        d = r_new + beta * d
        r = r_new

    return x

def gradient_descent(A, b, tol=1e-10, max_iter=10000):
    x = np.zeros_like(b)

    for _ in range(max_iter):
        r = b - A @ x

        if np.linalg.norm(r) < tol:
            return x

        alpha = (r @ r) / (r @ A @ r)
        x = x + alpha * r

    return x



def gauss_seidel(A, b, tol=1e-10, max_iter=10000):
    n = len(b)
    x = np.zeros(n)

    for _ in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            sum_before = A[i, :i] @ x[:i]
            sum_after = A[i, i+1:] @ x_old[i+1:]

            x[i] = (b[i] - sum_before - sum_after) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x

    return x

def jacobi(A, b, tol=1e-10, max_iter=10000):
    n = len(b)
    x = np.zeros(n)

    D = np.diag(A)
    R = A - np.diagflat(D)

    for _ in range(max_iter):
        x_new = (b - R @ x) / D

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new

        x = x_new

    return x

def main():
    A = np.array([
    [10,  1,  2,  3,  4],
    [ 1,  9, -1,  2, -3],
    [ 2, -1,  7,  3, -5],
    [ 3,  2,  3, 12, -1],
    [ 4, -3, -5, -1, 15]])

    b = np.array([12, -27, 14, -17, 12])

    x = jacobi(A, b)
    print(f'The result from Jacobi method is {x}')
    x = gauss_seidel(A, b)
    print(f'The result from Gauss-siedel method is {x}')
    x = gradient_descent(A, b)
    print(f'The result from Gradient Descent method is {x}')
    x = conjugate_gradient(A, b)
    print(f'The result from Conjugate Gradient method is {x}')

if __name__ == '__main__':
    main()