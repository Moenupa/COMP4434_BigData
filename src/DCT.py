import numpy as np

D = np.arange(0, 8, 1)
g = lambda i, u: np.cos((2 * i + 1) / 16 * u * np.pi)
C = lambda u: 1 if u != 0 else 2 ** -.5
def F(f, u):
    # print(g(D, u))
    return C(u) / 2 * np.sum(g(D, u) * f(D))

if __name__ == '__main__':
    f1 = lambda i: 100 * np.cos((2 * i + 1) / 6 * np.pi)
    f3 = lambda _: 100
    
    print(
        f"""
        F_1(0) = {F(f1, 0):<8}
        F_1(1) = {F(f1, 1):<8}
        F_3(0) = {F(f3, 0):<8}
        F_3(1) = {F(f3, 1):<8}
        """
    )
    