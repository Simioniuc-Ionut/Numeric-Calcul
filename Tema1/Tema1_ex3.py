import numpy as np
import math
import time

c1 = 1/math.factorial(3)
c2 = 1/math.factorial(5)
c3 = 1/math.factorial(7)
c4 = 1/math.factorial(9)
c5 = 1/math.factorial(11)
c6 = 1/math.factorial(13)
c7 = 1/math.factorial(15)

n = 10000
x_values = np.random.uniform(-np.pi/2, np.pi/2, n) # x in [-pi/2, pi/2]

def horner(c: list, y: float) -> float:
    '''
    Horner's method for polynomial evaluation.

    This method evaluates a polynomial at a given point using Horner's method,
    which is an efficient algorithm to compute polynomial values.

    :param c: List of coefficients of the polynomial, where c[0] is the leading coefficient.
    :param y: The point at which the polynomial is to be evaluated.
    :return: The value of the polynomial at the given point y.
    '''
    b = c[0]
    for i in range(1, len(c)):
        b = b * y + c[i]
    return b

polynomials = [
    ("p1", lambda x: x * horner([c2, -c1, 1], x**2)),
    ("p2", lambda x: x * horner([-c3, c2, -c1, 1], x**2)),
    ("p3", lambda x: x * horner([c4, -c3, c2, -c1, 1], x**2)),
    ("p4", lambda x: x * horner([c4, -c3, 0.00833, -0.166, 1], x**2)),
    ("p5", lambda x: x * horner([c4, -c3, 0.008333, -0.1666, 1], x**2)),
    ("p6", lambda x: x * horner([c4, -c3, 0.0083333, -0.16666, 1], x**2)),
    ("p7", lambda x: x * horner([-c5, c4, -c3, c2, -c1, 1], x**2)),
    ("p8", lambda x: x * horner([c6, -c5, c4, -c3, c2, -c1, 1], x**2))
]


sin_exact = np.sin(x_values)
errors = []

for name, poly in polynomials:
  # print("p:", pol, "sin:", sin_exact)
    errors.append(np.abs(poly(x_values) - sin_exact))

# print("Errors shape:", np.array(errors).shape) # shape: (8, 10000)

errors = np.array(errors)
best_3_per_x = np.argsort(errors, axis=0)[:3]  # sort the errors for each x ( axis=0 for columns )

for i, best_indices in enumerate(best_3_per_x, start=1):
    print(f"top {i} pol -> {best_indices + 1}") # index starts from 0, so we add 1

print("\n")

# the sum of errors for each polynomial
sum_errors = np.sum(errors, axis=1) # axis=1 for rows
sorted_indices = np.argsort(sum_errors)

for place, index in enumerate(sorted_indices, start=1):
    print(f"{place}: Polynomial {index + 1} -> Sum of errors: {sum_errors[index]}")

# Bonus
# calculate the time for each polynomial
timing_results = []
for index, (name, poly) in enumerate(polynomials, start=1):
    start_time = time.perf_counter()
    poly(x_values)  # calculate the polynomial
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    timing_results.append((index, elapsed_time))

# sort the results by time
timing_results.sort(key=lambda x: x[1])

print("\nTiming results:")
for poly_number, computation_time in timing_results:
    print(f"Polynomial {poly_number} -> Computation time: {computation_time:.6f} seconds")