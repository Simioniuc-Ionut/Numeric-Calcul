import pandas as pd

# Ex 1 
EPS = 10**(-10)
def read_sparse_matrix_b(file_path):
    """
    Citește un vector din fișierul file_path.
    Formatul fișierului:
      - Prima linie: dimensiunea n
      - Următoarele n linii: fiecare linie conține o valoare (float)
    Returnează: (vector, n)
    """
    res = {}
    
    with open(file_path, 'r') as f:
        n = int(f.readline().strip())
        for i in range(n):
          res[(i,0)] = float(f.readline().strip())
    return res, n
def read_sparse_matrix(file_path):
  """
  Reads a sparse matrix from the file at file_path.
  Assumes the file has the following format:
    - First line: dimension n
    - Remaining lines: each line contains "value,i,j"
  Returns a dictionary with keys (i,j) and the value of the element,
  as well as n.
  """
  
  A = {}
  with open(file_path, 'r') as f:
    n = int(f.readline().strip())
    for line in f:
      parts = line.strip().split(',')
      if len(parts) != 3:
        continue
      try:
        value = float(parts[0])
        i = int(parts[1])
        j = int(parts[2])
      except ValueError:
        continue
      if abs(value) < EPS:
        continue
      A[(i, j)] = value
  return A, n

# Old version
# def read_sparse_matrix_list_of_list(file_A_path, file_b_path):
#   eps = 1e-10 

#   with open(file_A_path, 'r') as file_A, open(file_b_path, 'r') as file_b:
    
#     n = int(file_A.readline().strip())
#     n_b = int(file_b.readline().strip())
#     if n != n_b:
#       print(f"Error: Different dimensions: {n} != {n_b}")
#       return None, None, None
#     print(f"Dimensions: n = {n}")

#     # Initialize structures: list of lists for the matrix, diagonal vector, and b vector
#     A_matrix = [[] for _ in range(n)]
#     d_diagonal = [None] * n
#     b_vector = [float(file_b.readline().strip()) for _ in range(n)]

#     # Iterate through each line in file A (format: value, row, column)
#     for line in file_A:
#       parts = line.strip().split(',')
#       if len(parts) != 3:
#         continue  # Skip invalid lines
#       try:
#         value = float(parts[0])
#         row = int(parts[1])
#         col = int(parts[2])
#       except ValueError:
#         print("Invalid data:", line)
#         continue

#       # Check if indices are within allowed limits
#       if row < 0 or row >= n or col < 0 or col >= n:
#         print("Invalid index in line:", line)
#         continue

#       # If the element is on the diagonal, check that it is not zero
#       if row == col:
#         if abs(value) < eps:
#           print(f"Diagonal element at row {row} is zero!")
#           return None, None, None
#         d_diagonal[row] = value
#       else:
#         # Check if the element is already stored in the corresponding row list
#         found = False
#         for idx, (v, c) in enumerate(A_matrix[row]):
#             if c == col:
#                 print(f"Duplicate element at row {row} column {col}!")
#                 A_matrix[row][idx] = (v + value,col)
#                 found = True
#                 break
#         if not found:
#             A_matrix[row].append((value, col))
              
              
#     # Check that each row has the diagonal element
#     for i in range(n):
#       if d_diagonal[i] is None:
#         print(f"Missing diagonal element at row {i}")
#         return None, None, None

#   return A_matrix, b_vector, d_diagonal

def read_sparse_matrix_list(file_A_path, eps=1e-10,isAplusB= None):
    """
    Reads a sparse matrix from the file at file_A_path.
    Returns the matrix as a list of lists of tuples and the diagonal vector.
    """
    with open(file_A_path, 'r') as file_A:
        n = int(file_A.readline().strip())
        # print(f"Dimensions of A: n = {n}")

        # Initialize structures: list of lists for the matrix and diagonal vector
        A_matrix = [[] for _ in range(n)]
        d_diagonal = [None] * n

        # Iterate through each line in file A (format: value, row, column)
        for line in file_A:
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue  # Skip invalid lines
            try:
                value = float(parts[0])
                row = int(parts[1])
                col = int(parts[2])
            except ValueError:
                print("Invalid data:", line)
                continue

            # Check if indices are within allowed limits
            if row < 0 or row >= n or col < 0 or col >= n:
                print("Invalid index in line:", line)
                continue

            # If the element is on the diagonal, check that it is not zero
            if row == col:
                if abs(value) < eps:
                    print(f"Diagonal element at row {row} is zero!")
                    return None, None
                d_diagonal[row] = value
            else:
                # Check if the element is already stored in the corresponding row list
                found = False
                for idx, (v, c) in enumerate(A_matrix[row]):
                    if c == col:
                        print(f"Duplicate element at row {row} column {col}!")
                        A_matrix[row][idx] = (v + value, col)
                        found = True
                        break
                if not found:
                    A_matrix[row].append((value, col))

        if isAplusB == None:
          # Check that each row has the diagonal element
          for i in range(n):
              if d_diagonal[i] is None:
                  print(f"Missing diagonal element at row {i}")
                  return None, None
        else:
          for i in range(n):
              if d_diagonal[i] is None:
                  d_diagonal[i] =0.0

    return A_matrix, d_diagonal
def read_vector_b_list_of_list(file_b_path):
    """
    Reads the vector b from the file at file_b_path.
    Returns the vector as a list of floats.
    """
    with open(file_b_path, 'r') as file_b:
        n = int(file_b.readline().strip())
        print(f"Dimensions of b: n = {n}")

        # Read the vector b
        b_vector = [float(file_b.readline().strip()) for _ in range(n)]

    return b_vector


def ex1_call():
  for i in range(1,6):
    A_file,d_diagonal = read_sparse_matrix_list(f"C:/Users/Asus/Documents/Facultate/Anul3/Sem2/CN/Tema3/files/a_{i}.txt")
    b_file = read_vector_b_list_of_list(f"C:/Users/Asus/Documents/Facultate/Anul3/Sem2/CN/Tema3/files/b_{i}.txt")
    print("-"*20)
    print(A_file[0:10])
    print("---diag---")
    print(d_diagonal[0:10])
    print(" -- b --")
    print(b_file[0:10])

# ex1_call()

# Ex 2
def gauss_seidel_inplace_sparse_dict(A_dict, n_A, b, eps=10**(-10), kmax=10000):
  # Group elements from A_dict by rows
  row_dict = {}
  for (i, j), value in A_dict.items():
    if i not in row_dict:
      row_dict[i] = {}
    row_dict[i][j] = value

  n = n_A
  x = [0.0] * n  # solution vector, initially zero
  k = 0

  while True:
    delta = 0.0
    for i in range(n):
      old = x[i]
      s = 0.0
      # Get the diagonal element; it must be non-zero
      diag = row_dict[i].get(i)
      if diag is None or abs(diag) < eps:
        raise ValueError(f"Missing or zero diagonal element at row {i}")
      # Iterate only over elements in row i, for j != i
      for j, aij in row_dict[i].items():
        if j != i:
          s += aij * x[j]
      # Update x[i]
      x[i] = (b[(i,0)] - s) / diag
      delta = max(delta, abs(x[i] - old))
    k += 1
    if delta < eps:
      return x, k
    if k >= kmax or delta > 1e8:
      raise ValueError("Gauss-Seidel diverged or did not converge within kmax iterations.")

def gauss_seidel_inplace_sparse(A_sparse, diag, b, eps=10**(-10), kmax=10000):
  """
  Solves Ax = b using in-place Gauss-Seidel,
  where A_sparse is a list (of length n) where each element is a list of tuples (aij, j)
  representing the non-zero elements in row i (excluding the diagonal element, which is stored separately in diag).
  
  b is the vector of free terms.
  
  The x_cp vector is updated in-place.
  """
  n = len(b)
  x_cp = [0] * n
  k = 0
  
  while True:
    Delta_x = 0.0
    for i in range(n):
      old = x_cp[i]
      s = 0.0
      # If there are non-zero elements in row i, use them for the sum s,
      # otherwise s remains 0.0
      if A_sparse[i]:
        for (aij, j) in A_sparse[i]:
          s += aij * x_cp[j] 
          
      # Update x[i] using the diagonal element from diag
      x_cp[i] = (b[i] - s) / diag[i]
      Delta_x = max(Delta_x, abs(x_cp[i] - old))
      
    k += 1
    if Delta_x < eps:
      # Convergence: the solution is x_cp
      return x_cp, k
    if k >= kmax or Delta_x > 1e8:
      # Convergence criterion not met
      raise ValueError("The Gauss-Seidel method diverged or did not converge within kmax iterations.")


# exemplul de pe fisa:
A_sparse = [
    [(2.5, 2)],
    [(3.5, 0), (0.33, 4),(1.05, 2)],
    [],
    [(1.3, 1)],
    [ (1.5, 3), (0.73, 0)]   
]
d_diag =(102.5, 104.88, 100.0, 101.3, 102.23)
b = [6.0,7.0,8.0,9.0,1.0]
x_cp = [1.0,2.0,3.0,4.0,5.0]

'''dict'''
# A_file,n_A = read_sparse_matrix(f"C:/Users/Asus/Documents/Facultate/Anul3/Sem2/CN/Tema3/files/a_1.txt")
# b_file ,n_b= read_sparse_matrix_b(f"C:/Users/Asus/Documents/Facultate/Anul3/Sem2/CN/Tema3/files/b_1.txt")

'''lists'''
# A_file,d_diagonal = read_sparse_matrix_list(f"C:/Users/Asus/Documents/Facultate/Anul3/Sem2/CN/Tema3/files/a_1.txt")
# b_file=read_vector_b_list_of_list(f"C:/Users/Asus/Documents/Facultate/Anul3/Sem2/CN/Tema3/files/b_1.txt")

'''dict'''
# x_solution, iterations = gauss_seidel_inplace_sparse_dict(A_file,n_A, b_file)


'''list'''
# x_solution, iterations = gauss_seidel_inplace_sparse(A_file,d_diagonal, b_file)


# Exemplu cu solutia din fisa de tema
# x_solution, iterations = gauss_seidel_inplace_sparse(A_sparse,d_diag, b)


# print("Soluția aproximată:", x_solution)
# print("Numărul de iterații:", iterations)

## Ex 3

import numpy as np
def verify_solution_list_of_list(A_sparse, diag, b, x_cp):
  """
  Verifies the solution calculated with Gauss-Seidel by computing the norm ||Ax - b||∞.
  """
  n = len(b)
  Ax = np.zeros(n)

  # Calculate the product Ax
  for i in range(n):
    Ax[i] = diag[i] * x_cp[i]  # Diagonal element
    if A_sparse[i]:  # Non-zero elements outside the diagonal
      for (aij, j) in A_sparse[i]:
        Ax[i] += aij * x_cp[j]

  # Calculate the norm ||Ax - b||∞
  norm_inf = np.max(np.abs(Ax - b))
  
  if abs(norm_inf) <= 10**(-8):
    return 0.0  # Solution is exact
  return norm_inf

# Function call after calculating x_solution with Gauss-Seidel
# error_norm = verify_solution_list_of_list(A_file,d_diagonal,b_file,x_solution)


def verify_solution_dict(A_dict, b, x_GS, n):
    """
    Verifies the solution calculated with Gauss-Seidel by computing the norm ||Ax - b||∞.
    A_dict is a dictionary where keys are (i, j) and values are the non-zero elements of A.
    """
    Ax = np.zeros(n)

    # Calculate the product Ax
    for (i, j), a_ij in A_dict.items():
        Ax[i] += a_ij * x_GS[j]

    # Calculate the norm ||Ax - b||∞
    norm_inf = np.max(np.abs(Ax - np.array([b[(i, 0)] for i in range(n)])))
    
    if abs(norm_inf) <= 10**(-8):
      return 0.0  # Solution is exact
    return norm_inf
    
# error_norm = verify_solution_dict(A_file, b_file, x_solution,n_A)

# print("Norm ||Ax - b||∞ =", error_norm)


# Bonus

EPS = 10**(-10)

'''dict'''
def sum_sparse_matrices_dict(A1, A2):
  """
  Computes the sum of two sparse matrices represented as dictionaries.
  Assumes the matrices have the same dimensions.
  Returns a dictionary with key (i,j) -> (A1[i,j] + A2[i,j]).
  """
  result = {}
  keys = set(A1.keys()).union(set(A2.keys()))
  for key in keys:
    val = A1.get(key, 0.0) + A2.get(key, 0.0)
    if abs(val) >= EPS:
      result[key] = val
  return result

def compare_sparse_matrices_dict(A_calc, A_expected, eps=EPS):
  """
  Compares two sparse matrices (represented as dictionaries) element by element.
  Returns True if for any key (i,j) the absolute difference is less than eps.
  """
  keys = set(A_calc.keys()).union(set(A_expected.keys()))
  for key in keys:
    v1 = A_calc.get(key, 0.0)
    v2 = A_expected.get(key, 0.0)
    if abs(v1 - v2) >= eps:
      print(f"Difference at {key}: calculated {v1} vs expected {v2}")
      return False
  return True
'''list'''
def sum_sparse_matrices_list(A1, A2, diag1, diag2, eps=10**(-9)):
  """
  Computes the sum of two sparse matrices stored as list-of-lists + diagonal vector.
  
  Parameters:
    - A1, A2: list-of-lists for the off-diagonal elements.
    For each row i, A1[i] is a list of tuples (value, col).
    - diag1, diag2: vectors of diagonal elements (length n).
    - eps: threshold to eliminate very small values.
  
  Returns:
    - A_sum: list-of-lists for the off-diagonal elements of the sum.
    - diag_sum: vector of diagonal elements of the sum.
  """
  n = len(diag1)
  diag_sum = [diag1[i] + diag2[i] for i in range(n)]
  A_sum = []
  for i in range(n):
    row_dict = {}
    # Add elements from A1:
    for (val, col) in A1[i]:
      row_dict[col] = row_dict.get(col, 0.0) + val
    # Add elements from A2:
    for (val, col) in A2[i]:
      row_dict[col] = row_dict.get(col, 0.0) + val
    # Build the list of tuples for row i, removing very small values.
    row_list = [(row_dict[col], col) for col in row_dict if abs(row_dict[col]) >= eps]
    A_sum.append(row_list)
  return A_sum, diag_sum

  
def compare_sparse_matrices_list(A_calc, A_expected, diag_calc, diag_expected, eps=EPS):
  """
  Compares two sparse matrices stored as list-of-lists + diagonal vector.
  
  Parameters:
    - A_calc and A_expected: list-of-lists for the off-diagonal elements.
    - diag_calc and diag_expected: vectors of diagonal elements.
    - eps: tolerance threshold.
  
  Returns True if for every element (diagonal and off-diagonal)
  the absolute difference is less than eps, otherwise False.
  """
  n = len(diag_calc)
  # Compare diagonal elements:
  for i in range(n):
    if abs(diag_calc[i] - diag_expected[i]) >= eps:
      print(f"Diagonal difference at row {i}: calculated {diag_calc[i]} vs expected {diag_expected[i]}")
      return False
  # Compare off-diagonal elements, row by row:
  for i in range(n):
    # Convert the list of tuples into a dictionary for row i
    row_calc = {col: val for (val, col) in A_calc[i]}
    row_expected = {col: val for (val, col) in A_expected[i]}
    keys = set(row_calc.keys()).union(row_expected.keys())
    for col in keys:
      v_calc = row_calc.get(col, 0.0)
      v_expected = row_expected.get(col, 0.0)
      if abs(v_calc - v_expected) >= eps:
        print(f"Off-diagonal difference at row {i}, col {col}: calculated {v_calc} vs expected {v_expected}")
        return False
  return True


class SparseMatrix():

  def __init__(self, A_path, B_path, sparse_type , d_diagonal = None):
    self.eps = 10**(-9)
    self.type=sparse_type
    if sparse_type == "list-of-dicts":
      self.A, self.n_A = read_sparse_matrix(A_path)
      self.B, self.n_B = read_sparse_matrix_b(B_path)
    else:
      # list_of_list + diagonal
      self.A, self.d_diagonal = read_sparse_matrix_list(A_path)
      self.B=read_vector_b_list_of_list(B_path)
    
  def gauss_siedel(self):
    if self.type == "list-of-dicts":
      x_solution,iterration =gauss_seidel_inplace_sparse_dict(self.A,self.n_A,self.B,self.eps)
    else:
      x_solution,iterration = gauss_seidel_inplace_sparse(self.A,self.d_diagonal,self.B,self.eps)
    return x_solution,iterration
  
  def verify_solution(self,solution):
    if self.type == "list-of-dicts":
      error_norm=verify_solution_dict(self.A,self.B,solution,self.n_A)
    else:
      error_norm=verify_solution_list_of_list(self.A,self.d_diagonal,self.B,solution)
    print("Norm ||Ax - b||∞ =", error_norm)
  
  def sum_compare_solution(self):
    if self.type == "list-of-dicts":
      A , B ,C_expected,n_A,n_B,n_C= self.__read_expected_dict()
      
      sum_A_B = sum_sparse_matrices_dict(A,B)
      
      if compare_sparse_matrices_dict(sum_A_B, C_expected):
          print("The sum of the matrices is correct (differences are below epsilon).")
      else:
          print("The sum of the matrices is NOT correct.")
      
    else:
      A, B, C_expected, a_diagonal,b_diagonal, c_diag = self.__read_expected_list()
      
      sum_A_B,sum_diag = sum_sparse_matrices_list(A,B,a_diagonal,b_diagonal)
      
      if compare_sparse_matrices_list(sum_A_B, C_expected,sum_diag,c_diag):
          print("The sum of the matrices is correct (differences are below epsilon).")
      else:
          print("The sum of the matrices is NOT correct.")
      
    
  def __read_expected_dict(self):
    # Read matrices from files (a.txt, b.txt, aplusb.txt)
    A, n_A = read_sparse_matrix("C:/Users/Asus/Documents/Facultate/Anul3/Sem2/CN/Tema3/files/a.txt")
    B, n_B = read_sparse_matrix("C:/Users/Asus/Documents/Facultate/Anul3/Sem2/CN/Tema3/files/b.txt")
    C_expected, n_C = read_sparse_matrix("C:/Users/Asus/Documents/Facultate/Anul3/Sem2/CN/Tema3/files/aplusb.txt")
    return A, B, C_expected, n_A, n_B, n_C
  def __read_expected_list(self):
    # Read matrices from files (a.txt, b.txt, aplusb.txt)
    A, a_diagonal = read_sparse_matrix_list("C:/Users/Asus/Documents/Facultate/Anul3/Sem2/CN/Tema3/files/a.txt")
    B,b_diagonal = read_sparse_matrix_list("C:/Users/Asus/Documents/Facultate/Anul3/Sem2/CN/Tema3/files/b.txt")
    C_expected,c_diag = read_sparse_matrix_list("C:/Users/Asus/Documents/Facultate/Anul3/Sem2/CN/Tema3/files/aplusb.txt",10**(-9),True)
    return A, B, C_expected, a_diagonal,b_diagonal, c_diag
  
  
def test_sparse_matrix():
    # Testare pentru toate exemplele (1-5)
    for i in range(1, 6):
        print(f"\n--- Test pentru exemplul {i} ---\n")

        # Căi către fișierele de intrare
        A_path = f"C:/Users/Asus/Documents/Facultate/Anul3/Sem2/CN/Tema3/files/a_{i}.txt"
        B_path = f"C:/Users/Asus/Documents/Facultate/Anul3/Sem2/CN/Tema3/files/b_{i}.txt"

        # Testare pentru reprezentarea "list-of-dicts"
        print(f"Reprezentare: list-of-dicts (Exemplul {i})")
        try:
            sparse_matrix_dict = SparseMatrix(A_path, B_path, sparse_type="list-of-dicts")
            x_solution_dict, iterations_dict = sparse_matrix_dict.gauss_siedel()
            print(f"Soluție Gauss-Seidel (list-of-dicts): {x_solution_dict[0:10]}")
            print(f"Număr de iterații: {iterations_dict}")
            sparse_matrix_dict.verify_solution(x_solution_dict)
        except ValueError as e:
            print(f"Eroare la exemplul {i} (list-of-dicts): {e}")

        # Testare pentru reprezentarea "list-of-lists"
        print(f"\nReprezentare: list-of-lists (Exemplul {i})")
        try:
            sparse_matrix_list = SparseMatrix(A_path, B_path, sparse_type="list-of-lists")
            x_solution_list, iterations_list = sparse_matrix_list.gauss_siedel()
            print(f"Soluție Gauss-Seidel (list-of-lists): {x_solution_list[0:10]}")
            print(f"Număr de iterații: {iterations_list}")
            sparse_matrix_list.verify_solution(x_solution_list)
        except ValueError as e:
            print(f"Eroare la exemplul {i} (list-of-lists): {e}")
            
    # Testare pentru bonus (compararea sumelor)
    print("\n--- Test pentru bonus (compararea sumelor) ---\n")

    # Testare pentru reprezentarea "list-of-dicts"
    print("Reprezentare: list-of-dicts")
    sparse_matrix_dict.sum_compare_solution()

    # Testare pentru reprezentarea "list-of-lists"
    print("\nReprezentare: list-of-lists")
    sparse_matrix_list.sum_compare_solution()


if __name__ == "__main__":
    # Apelarea funcției de test
    test_sparse_matrix()