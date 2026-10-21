import numpy as np

def CG(A, b):
    n = A.shape[0]
    x0 = np.zeros(n)
    r0 = b.copy()
    d0 = r0.copy()
    for i in range(n):
         r = r0.copy()
         d = d0.copy()
         alpha = (r.T @ r) / (d.T @ A @ d)
         x = x0 + alpha * d
         r0 = b - A @ x
         beta = (r0.T @ r0) / (r.T @ r)
         d0 = r0 + beta * d
         x0 = x.copy()
    return x0

def GD(A, b, epsilon=1e-6):
     n = A.shape[0]
     x0 = np.zeros(n)
     r = b
     while np.linalg.norm(r) > epsilon:
          r = b - A @ x0
          nu = (r.T @ r)/(r.T @ A @ r)
          x = x0 + nu * r
          x0 = x.copy()
     return x0

def main():
    A1 = np.zeros((16,16))
    A2 = np.zeros((32,32))
    for i in range(16):
        for j in range(16):
            A1[i, j] = 1/(1+i+j)

    for i in range(32):
            for j in range(32):
                A2[i, j] = 1/(1+i+j)

    b1 = 1/3 * np.sum(A1, axis=1)
    b2 = 1/3 * np.sum(A2, axis=1)

    x = CG(A1, b1)
    print(f'The result for A1 from CG is {x}')
    x = CG(A2, b2)
    print(f'The result for A2 from CG is {x}')

    x = GD(A1, b1)
    print(f'The result for A1 from GD is {x}')
    x = GD(A2, b2)
    print(f'The result for A2 from GD is {x}')

if __name__ == '__main__':
     main()