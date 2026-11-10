def newton_root(a, init_guess, root, tol=1e-9, max_iter=1000):
    tn = init_guess
    t_next = 0

    for i in range(max_iter):
        print(f'iter: {i}')
        t_next = (1/root) * ((root-1) * tn + a/pow(tn, root-1))
        if abs(t_next - tn) < tol:
            return t_next
        tn = t_next
    return tn

def main():
    a = 5
    root =2
    init_guess = 2
    approx_root = newton_root(a, init_guess, root)
    print(f'The approximate square root of {a} is {approx_root}')

if __name__ == '__main__':
    main()