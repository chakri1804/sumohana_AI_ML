import numpy as np
import matplotlib.pyplot as plt
import scipy

def polybasisfunc(x,degree,number_of_samples):
    phi = np.zeros((number_of_samples,degree+1))
    deg = list(range(0,degree+1))
    k = list(range(0,number_of_samples))
    for i in k:
        for j in deg:
            # print(x[i])
            phi[i][j] = x[i]**j
    return phi

# Number of training samples
N = 10
# Generate equispaced floats in the interval [0, 2pi]
x = np.linspace(0, 2*np.pi, N)
# Generate noise
mean = 0
std = 0.05
# Generate some numbers from the sine function
y = np.sin(x)
# Add noise
y += np.random.normal(mean, std, N)
print("Enter degree of polynomial:")
degree = input()
degree = int(degree)
print("Enter testing samples:")
test_samples = input()
test_samples = int(test_samples)
print("Enter Lagrangian Multiplier:")
L = input()
L = float(L)
phi = polybasisfunc(x,degree,N)
# print(phi)
# W = np.zeros((N,N))
# DDD = (phiT*phi + lambda*I)^-1 * phiT * b
W = np.matmul(np.matmul(np.linalg.inv(np.matmul(phi.T,phi)+(L*np.identity(degree+1))),phi.T),y)
# W = np.matmul(DDD,y)
# print(W)
x1 = np.linspace(0, 2*np.pi, test_samples)
phi1 = polybasisfunc(x1,degree,test_samples)
y1 = np.matmul(phi1,W)
# print(y1)
# X = np.reshape(x,(10,1))
# y1 = np.matmul(X,W)
# y1 = np.polynomial.polynomial(W)
# print(y1)
# error = (y1-y)
# print(error)
plt.plot(x,y,'*')
plt.plot(x1,y1,'.')
plt.show()
