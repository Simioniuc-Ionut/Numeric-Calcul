import numpy as np
## Ax = b
## A = LU
## LUx=b => { Ly=b, Ux=y }
def lu_inplace_with_fixed_U_diag_optimized(A, U_diag):
    n = A.shape[0]
    for p in range(n):
        A[p:n, p] = (A[p:n, p] - A[p:n, :p] @ A[:p, p]) / U_diag[p]
        if p + 1 < n:
            A[p, p+1:n] = (A[p, p+1:n] - A[p, :p] @ A[:p, p+1:n]) / A[p, p]
    return

def lu_inplace_with_fixed_U_diag(A, U_diag):
    """
    Computes the in-place LU decomposition of matrix A using a fixed diagonal for U.

    This function overwrites A with:
      - The lower triangular matrix L (including its diagonal) in the lower part.
      - The strictly upper triangular part of U in the upper part.
      The diagonal of U is provided externally by U_diag.

    The formulas used are:
      For each column p from 0 to n-1:
        For i from p to n-1:
          L[i, p] = (A[i, p] - sum_{k=0}^{p-1} (L[i, k] * U[k, p])) / U_diag[p]
        For j from p+1 to n-1:
          U[p, j] = (A[p, j] - sum_{k=0}^{p-1} (L[p, k] * U[k, j])) / L[p, p]

    Note:
      - L[i,k] is stored in A[i, k] for k ≤ i.
      - U[k,j] is stored in A[k, j] for k < j.

    :param A: The square matrix to decompose (modified in-place).
    :param U_diag: A 1D array containing the diagonal elements of U.
    :return: The modified matrix A containing L in its lower part and U in its upper part.
    """
    n = A.shape[0]
    for p in range(n):
        # Compute L[i, p] for i = p, ..., n-1
        for i in range(p, n):
            sum_LU = 0.0
            for k in range(p):
                sum_LU += A[i, k] * A[k, p]
            A[i, p] = (A[i, p] - sum_LU) / U_diag[p]

        # Compute U[p, j] for j = p+1, ..., n-1
        for j in range(p + 1, n):
            sum_LU = 0.0
            for k in range(p):
                sum_LU += A[p, k] * A[k, j]
            # A[p, p] holds L[p, p]
            A[p, j] = (A[p, j] - sum_LU) / A[p, p]

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
def direct_meth(A_combined,U_diag,b_values):
    # Ax = b => LUx = b => {Ly = b, Ux = y}
    #extract lower triangular part
    if b_values.shape[0] != A_combined.shape[0]:
        raise ValueError("The dimensions of the matrix A and the vector b do not match.")

    # Solve Ly = b
    y = np.linalg.solve(np.tril(A_combined), b_values)

    # Replace the diagonal of U in-place
    np.fill_diagonal(A_combined, U_diag)

    # Solve Ux = y
    x = np.linalg.solve(np.triu(A_combined), y)

    return x
def indirect_meth(A_combined,U_diag,b_values):
        # Ax = b => LUx = b => {Ly = b, Ux = y}
        #extract lower triangular part
        if b_values.shape[0] != A_combined.shape[0]:
            raise ValueError("The dimensions of the matrix A and the vector b do not match.")

        # Solve Ly = b
        A_inv = np.linalg.inv(np.tril(A_combined))
        y = np.dot(A_inv, b_values)

        # Replace the diagonal of U in-place
        np.fill_diagonal(A_combined, U_diag)

        # Solve Ux = y
        x = np.linalg.solve(np.triu(A_combined), y)

        # To print first 13 decimals
        # np.set_printoptions(precision=13, suppress=True)

        return x

b = np.array([21.6,33.6,51.6])
print("Direct method:" , direct_meth(A_combined.copy(),U_diag,b))
print("Indirect method:" , indirect_meth(A_combined.copy(),U_diag,b))
x_LU = direct_meth(A_combined.copy(),U_diag,b)
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
np.fill_diagonal(U2, U_diag2,)
print("A combined 2 : \n",A_combined2)
print("L2 : \n",L2)
print("U2 : \n",U2)
print("Product L2 * U2 : \n",L2 @ U2)
print("Original matrix A2 : \n",A2)
b2 = np.array([2,-6,2])
print("Det A2 = ", compute_detA(A_combined2,U_diag2))
print("Direct method:" , direct_meth(A_combined2.copy(),U_diag2,b2))
print("Indirect method:" , indirect_meth(A_combined2.copy(),U_diag2,b2))
print("Norm of the solution A2:", compute_norm(A2,direct_meth(A_combined2.copy(),U_diag2,b2),b2,epsilon))
print("-"*50)