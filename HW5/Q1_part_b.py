from math import cbrt

def fpm(init_guess, tol=1e-9, max_iter=200):
    tn = init_guess
    t_next = 0
    for i in range(max_iter):
        print(f'iter: {i}')
        t_next = cbrt(tn - 1)
        if abs(tn-t_next) < tol:
            return t_next
        tn = t_next
    return tn

def main():
    init = 0
    root = fpm(init)
    print(f'The approx root of the function is {root}')

if __name__ == '__main__':
    main()
        