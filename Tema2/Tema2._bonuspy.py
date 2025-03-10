import numpy as np


def idxL(i, j):
    return i * (i + 1) // 2 + j

def idxU(i, j, n):
    return i * n - (i * (i - 1)) // 2 + (j - i)


def lu_decomposition_vector_storage(A, dU):

    n = A.shape[0]
    size = n * (n + 1) // 2
    L_vec = np.zeros(size)
    U_vec = np.zeros(size)

    # Initialize U diagonal with the provided dU values.
    for i in range(n):
        U_vec[idxU(i, i, n)] = dU[i]

    for p in range(n):
        # Compute L[i, p] for i = p,...,n-1.
        for i in range(p, n):
            sum_LU = sum(L_vec[idxL(i, k)] * U_vec[idxU(k, p, n)] for k in range(p))
            # L[i, p] = (A[i, p] - sum) / dU[p]
            L_vec[idxL(i, p)] = (A[i, p] - sum_LU) / dU[p]

        # Compute U[p, j] for j = p+1,...,n-1.
        for j in range(p + 1, n):
            sum_LU = sum(L_vec[idxL(p, k)] * U_vec[idxU(k, j, n)] for k in range(p))
            # U[p, j] = (A[p, j] - sum) / L[p, p]
            U_vec[idxU(p, j, n)] = (A[p, j] - sum_LU) / L_vec[idxL(p, p)]

    return L_vec, U_vec


def direct_substitution(L_vec, b, n):

    y = np.zeros(n)
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L_vec[idxL(i, j)] * y[j]
        # Note: L[i,i] is stored at idxL(i,i).
        y[i] = (b[i] - sum_val) / L_vec[idxL(i, i)]
    return y


def back_substitution(U_vec, y, n):
    """
    Solves the upper-triangular system U * x = y via back substitution.
    U is stored in compact form in U_vec.

    :param U_vec: Compact storage vector for U.
    :param y: Right-hand side vector (result from forward substitution).
    :param n: Dimension of the system.
    :return: Solution vector x satisfying U*x = y.
    """
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += U_vec[idxU(i, j, n)] * x[j]
        # U[i,i] is dU[i] and stored at idxU(i,i,n).
        x[i] = (y[i] - sum_val) / U_vec[idxU(i, i, n)]
    return x


def reconstruct_LU(L_vec, U_vec, n):
    """
    Reconstructs the full matrix product LU from the compact storage vectors.

    For each (i, j), we compute:
       (LU)[i,j] = sum_{k=0}^{min(i,j)} L[i,k]*U[k,j]

    :param L_vec: Compact storage vector for L.
    :param U_vec: Compact storage vector for U.
    :param n: Dimension of the matrix.
    :return: The reconstructed matrix LU.
    """
    LU = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(min(i, j) + 1):
                s += L_vec[idxL(i, k)] * U_vec[idxU(k, j, n)]
            LU[i, j] = s
    return LU


# Exemplu de utilizare
if __name__ == '__main__':
    # Define the matrix A (remains unchanged) and vector dU for U's diagonal.
    A = np.array([[4, 2, 3],
                  [2, 7, 5.5],
                  [6, 3, 12.5]], dtype=float)
    dU = np.array([2, 3, 4], dtype=float)

    # Define the right-hand side vector b for the system A*x = b.
    b = np.array([21.6,33.6,51.6], dtype=float)
    n = A.shape[0]

    # Compute the LU decomposition using compact storage.
    L_vec, U_vec = lu_decomposition_vector_storage(A, dU)

    # Solve the system A*x = b via the LU factors:
    # 1. Solve L*y = b (forward substitution).
    y = forward_substitution(L_vec, b, n)
    # 2. Solve U*x = y (back substitution).
    x = back_substitution(U_vec, y, n)

    # Reconstruct the product LU for verification.
    LU_product = reconstruct_LU(L_vec, U_vec, n)

    # Display results.
    print("Original matrix A:")
    print(A)

    print("\nLU decomposition (reconstructed product LU):")
    print(LU_product)

    print("\nReconstruction error (Frobenius norm):", np.linalg.norm(LU_product - A))

    print("\nSolution x for A*x = b:")
    print(x)
