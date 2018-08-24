import numpy as np
import matplotlib.pyplot as plt

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
N = 30
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
phi = polybasisfunc(x,degree,N)
# print(phi)
W = np.zeros((N,N))
W = np.matmul(np.linalg.pinv(phi),y)
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
plt.plot(x,y,'*',label='True Values')
plt.plot(x1,y1,'.',label='Estimated')
plt.legend()
plt.show()

x1 = np.linspace(0, 2*np.pi, N)
phi1 = polybasisfunc(x1,degree,N)
y1 = np.matmul(phi1,W)

std = np.var(y-y1)**0.5
y2 = np.random.normal(y1,std,N)
# labels = np.random.normal(y1,std1,N)
# error = y1 - labels
# # plt.plot(x,error,'b')
# # plt.show()
# variance = np.var(error)
print("Most likelihood variance =",np.var(y-y1))
print("Most likelihood labels by adding gaussian noise to estimated labels=")
print(y2)
print("Actual labels =")
print(y1)
error = y1-y2
print("error for each label :")
print(error)
print("Mean and variance of error =",np.mean(error),np.var(error))

plt.title("Plot with all possible labels")
plt.plot(x,y,'*',label='True Values')
plt.plot(x,y1,'r',label='Estimated')
plt.plot(x,y2,'.',label='Generated labels from Y_estim')
plt.legend()
plt.show()