
def FMIP(init, max_iter=100, tol=1e-9):
    x1_n = init[0]
    x2_n = init[1]
    x1_new = 0
    x2_new = 0
    for k in range(max_iter):
        print(f'iter: {k}')
        x1_new = (pow(x1_n, 2) + pow(x2_n, 2) + 8)/10
        x2_new = (x1_n * pow(x2_n, 2) + x1_n + 8)/10
        if abs(x1_new - x1_n) < tol or abs(x2_n - x2_new) < tol:
            return x1_new, x2_new
        x1_n = x1_new
        x2_n = x2_new
    return x1_n, x2_n

def main():
    init = (1.5, 1.5)
    x1, x2 = FMIP(init)
    print(f'The roots of the system are [{x1:.3f}, {x2:.3f}]')

if __name__ == '__main__':
    main()