import numpy as np
## Ax = b
## A = LU
## LUx=b => { Ly=b, Ux=y }

def lu_inplace_with_fixed_U_diag(A, U_diag):
    n = A.shape[0]
    for p in range(n):
        # Acc elementele L pe linia p (pentru coloanele 0 ... p)

        for j in range(0, p + 1):
            if j == 0:
                # consideram suma = 0 când j==0
                s = 0.0
            else:
                s = A[p, :j] @ A[:j, j]  # Ex : A[1,0] * A[0,1] + A[1,1] * A[1,1]
            A[p, j] = (A[p, j] - s) / U_diag[j]

        # Acc elementele U pe linia p (pentru coloanele p+1 ... n-1)
        for j in range(p + 1, n):
            s = A[p, :p] @ A[:p, j] # Ex : A[1,0] * A[0,2] + A[1,1] * A[1,2] + A[1,2] * A[2,2]
            A[p, j] = (A[p, j] - s) / A[p, p]
    return A


# Example usage:
A_init = np.array([[4, 2, 3],
                       [2, 7, 5.5],
                       [6, 3, 12.5]], dtype=float)
U_diag = np.array([2, 3, 4], dtype=float)

# Compute the in-place LU decomposition (combined matrix)
A_combined = lu_inplace_with_fixed_U_diag(A_init.copy(), U_diag)


'''Verify if the solution is correct'''
# Extract L and U from A_combined:
L = np.tril(A_combined)  # Lower triangular part (including diagonal)
U = np.triu(A_combined, k=1)  # Strictly upper triangular part

# Reconstruct the full U by inserting the fixed diagonal U_diag
n = A_init.shape[0]
U_full = np.zeros_like(A_init)
for i in range(n):
    U_full[i, i] = U_diag[i]
U_full += U

# Reconstruct A from L and U to verify the factorization:
A_reconstructed = L @ U_full

print("Combined matrix (in-place LU):")
print(A_combined)
print("\nMatrix L:")
print(L)
print("\nMatrix U:")
print(U_full)
print("\nProduct L * U:")
print(A_reconstructed)
print("\nOriginal matrix A:")
print(A_init)

''''''''
'''Compute the determinant of A'''
def compute_detA(L,U):
    return np.prod(np.diag(L)) * np.prod(U)

print("Det A = ", compute_detA(A_combined,U_diag))
'''Compute Ax=b '''
# 1: Direct methode
def substitution(A_combined,U_diag,b_values):
    # Ax = b => LUx = b => {Ly = b, Ux = y}
    #extract lower triangular part
    # if b_values.shape[0] != A_combined.shape[0]:
    #     raise ValueError("The dimensions of the matrix A and the vector b do not match.")
    #
    # # Solve Ly = b
    # y = np.linalg.solve(np.tril(A_combined), b_values)
    #
    # # Replace the diagonal of U in-place
    # np.fill_diagonal(A_combined, U_diag)
    #
    # # Solve Ux = y
    # x = np.linalg.solve(np.triu(A_combined), y)
    #
    # return x

    if len(b_values) != len(A_combined):
        raise ValueError("The dimensions of the matrix A and the vector b do not match.")

    # Solve Ly = b (forward substitution)
    y = [0] * len(b_values)
    for i in range(len(b_values)):
        sum_val = 0
        for j in range(i):
            sum_val += A_combined[i][j] * y[j]
        y[i] = (b_values[i] - sum_val) / A_combined[i][i]

    # Replace the diagonal of U in-place
    for i in range(len(U_diag)):
        A_combined[i][i] = U_diag[i]

    # Solve Ux = y (backward substitution)
    x = [0] * len(y)
    for i in range(len(y) - 1, -1, -1):
        sum_val = 0
        for j in range(i + 1, len(y)):
            sum_val += A_combined[i][j] * x[j]
        x[i] = (y[i] - sum_val) / A_combined[i][i]

    return x


b = np.array([21.6,33.6,51.6])
print("Direct method:" , substitution(A_combined.copy(),U_diag,b))
x_LU = substitution(A_combined.copy(),U_diag,b)
''''''
'''Verify solution through norm calculation'''
def compute_norm(A_init,x_LU,b,epsilon):
    norm = np.linalg.norm(np.dot(A_init,x_LU) - b)
    print(f"{norm:.15f}")
    if abs(norm) <= epsilon:
        return (norm,True)
    return (norm,False)

epsilon  = 10**(-14)

print("Norm of the solution:", compute_norm(A_init,x_LU,b,epsilon))
print("-"*50)
""" Testing """

A2 = np.array([[2.5,2,2],[-5,-2,-3],[5,6,6.5]])
U_diag2 = np.array([1,1,1],dtype=float)

A_combined2 = lu_inplace_with_fixed_U_diag(A2.copy(), U_diag2)

# Extract L and U from A_combined:
L2 = np.tril(A_combined2)  # Lower triangular part (including diagonal)

U2 = np.triu(A_combined2, k=1)  # Strictly upper triangular part
np.fill_diagonal(U2, U_diag2)
print("A combined 2 : \n",A_combined2)
print("L2 : \n",L2)
print("U2 : \n",U2)
print("Product L2 * U2 : \n",L2 @ U2)
print("Original matrix A2 : \n",A2)
b2 = np.array([2,-6,2])
print("Det A2 = ", compute_detA(A_combined2,U_diag2))
print("Substitution method:" , substitution(A_combined2.copy(),U_diag2,b2))
print("Norm of the solution A2:", compute_norm(A2,substitution(A_combined2.copy(),U_diag2,b2),b2,epsilon))
print("-"*50)

#lib
import numpy as np

def compute_solution_and_inverse_numpy(A, b, xLU):
    """
    1. Calculate the solution of the system Ax = b
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    2. Calculate the inverse of matrix A
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
    3. Calculate the norms
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    """
    # 1.
    if np.linalg.det(A) == 0:
        raise ValueError("Matrix A is not invertible, det(A) = 0")
    else:
        x_lib = np.linalg.solve(A, b)

    # 2.
    A_inv_lib = np.linalg.inv(A)

    # 3.
    norm1 = np.linalg.norm(xLU - x_lib, 2)
    norm2 = np.linalg.norm(xLU - np.dot(A_inv_lib, b), 2)

    # Display
    print(f"Solution of the system (x_lib):\n{x_lib}")
    print(f"Inverse of matrix A (A_inv_lib):\n{A_inv_lib}")
    print(f"Norm ||xLU - x_lib||_2 = {norm1}")
    print(f"Norm ||xLU - A^{{-1}} b||_2 = {norm2:.10f}")

    eps = 10**(-9)
    if abs(norm2) < eps:
        print(norm2, "-->", True)
    else:
        print(norm2, "-->", False)

    return x_lib, A_inv_lib


#ex1
xLU = substitution(A_combined2.copy(), U_diag2, b2)
xLU_arr = np.array(xLU)
compute_solution_and_inverse_numpy(A2, b2, xLU_arr)
print("-"*50)
#ex2
n = 101
A_ex = np.random.rand(n, n)
Ud_ex = np.random.rand(n)
A_ex_combined = lu_inplace_with_fixed_U_diag(A_ex.copy(), Ud_ex)
L_ex = np.tril(A_ex_combined)
U_ex = np.triu(A_ex_combined, k=1)
np.fill_diagonal(U_ex, Ud_ex)
b_ex = np.random.rand(n)
xLU_ex = np.array(substitution(A_ex_combined.copy(), Ud_ex, b_ex))
compute_solution_and_inverse_numpy(A_ex, b_ex, xLU_ex)
