import numpy as np

print("Extended matrix form of the system of equations")
extended_matrix_form = np.array([[3, 7, 3], [-5, 2, 2]])
print(extended_matrix_form)

print()
print("Coefficient matrix")
A = extended_matrix_form[:, 0:2]
print(A)

print()
print("Right hand side")
b = extended_matrix_form[:, 2:3]
print(b)

print()
print("Solution to the linear system of equations")
solution = np.linalg.solve(A, b)
print(solution)

print()
print("Check the solution by inserting it in the original equations left side")
check = A @ solution
print(check)

print()
print("Subtract the left over parts")
print(check - b)
