import numpy as np

def compute_solution_and_inverse_numpy(A, b, xLU):
    """
    1. Calc sol sistemului Ax = b
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    2. Calc inversa matricei A
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
    3. Calculez normele
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    """
    #1.
    if np.linalg.det(A) == 0:
        raise ValueError("Matricea A nu este inversabila, det(A) = 0")
    else:
        x_lib = np.linalg.solve(A, b)

    #2.
    A_inv_lib = np.linalg.inv(A)

    #3.
    norm1 = np.linalg.norm(xLU - x_lib, 2)
    norm2 = np.linalg.norm(xLU - np.dot(A_inv_lib, b), 2)

    #Afiare
    print(f"Solutia sistemului (x_lib):\n{x_lib}")
    print(f"Inversa matricei A (A_inv_lib):\n{A_inv_lib}")
    print(f"Norma ||xLU - x_lib||_2 = {norm1}")
    print(f"Norma ||xLU - A^{-1} b||_2 = {norm2:.15f}")

    eps = 10**(-15)
    if abs(norm2) < eps:
        print(norm2, "-->", True)
    else:
        print(norm2, "-->", False)

    return x_lib, A_inv_lib

# Ex
n = 3
A = np.array([[4, 2, 3],
              [2, 7, 5.5],
              [6, 3, 12.5]])

b = np.array([21.6, 33.6, 51.6])
xLU = np.array([2.5, 2.2, 2.4])


compute_solution_and_inverse_numpy(A, b, xLU)