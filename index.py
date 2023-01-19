import numpy as np
import matplotlib.pyplot as plt
import math

np.set_printoptions(suppress=True)


h = 0.01
tb = 1
ta = 0
n = (tb - ta) / h

A = np.zeros((int(n - 1), int(n - 1)))
B = np.zeros((int(n - 1), 1))

def f(x):
    return math.exp(x) * (x**2 + 1)

aux = 2 / (h**2)

a_i = round(1/(h**2) - 1/(2*h)) 
c_i = round(1/(h**2) + 1/(2*h)) 

b_i = []
for i in range(1, int(n)):
    b_i.append(i / n - aux)
    B[i - 1, 0] = f(i / n)

B[-1,0] += - c_i * math.exp(1)

for row in range(len(A)):
    for column in range(len(A)):
        # b_i
        if row == column:
            A[row, column] = b_i[row]
        
        # a_i
        if row == column + 1:
            A[row, column] = a_i

        # c_i
        if row == column - 1:
            A[row, column] = c_i

# A.Y = B => Y = A^-1.B
Y = np.linalg.inv(A).dot(B)

print(Y)

y_axis = [0]
for i in Y:
    y_axis.append(i[0])
y_axis.append(math.exp(1))

x_axis = []
for i in range(1 * int(n) + 1):
    x_axis.append(i / (1 * int(n)))

plt.plot(x_axis, y_axis)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

        