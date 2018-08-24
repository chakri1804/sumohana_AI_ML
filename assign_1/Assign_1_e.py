import numpy as np
import matplotlib.pyplot as plt

# Polynomial basis function and its matrix generator
# This function has a hardcoded bias included. So for removal of it in further steps,
# np.delete() is used

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


# Ask user for parameters
print("Enter degree of polynomial:")
degree = input()
degree = int(degree)
print("Enter testing samples:")
test_samples = input()
test_samples = int(test_samples)
print("Enter alpha (standard deviation of weights)")
alpha = input()
alpha = float(alpha)
print("Enter sigma (standard deviation of labels)")
sigma = input()
sigma = float(sigma)
L = ((sigma**2)/(alpha**2))

# Ridge regression
## Not zero centred data
phi = polybasisfunc(x,degree,N)
W = np.matmul(np.matmul(np.linalg.inv(np.matmul(phi.T,phi)+(L*np.identity(degree+1))),phi.T),y)
x1 = np.linspace(0, 2*np.pi, test_samples)
phi1 = polybasisfunc(x1,degree,test_samples)
y1 = np.matmul(phi1,W)
# Plotting
plt.title("Regression on raw data")
plt.plot(x,y,'*',label='True Values')
plt.plot(x1,y1,'.',label='Estimated')
plt.legend()
plt.show()


## Not zero centred data and bias ommited
zeroy = y - np.mean(y)
# print(np.mean(y),np.mean(zeroy))
phi = np.delete(phi,0,1)
W = np.matmul(np.matmul(np.linalg.inv(np.matmul(phi.T,phi)+(L*np.identity(degree))),phi.T),zeroy)
# print(np.shape(W))
x1 = np.linspace(0, 2*np.pi, test_samples)
phi1 = polybasisfunc(x1,degree,test_samples)
phi1 = np.delete(phi1,0,1)
y1 = np.matmul(phi1,W) + np.mean(y)
plt.title("Regression on zero-centred data")
plt.plot(x,y,'*',label='True Values')
plt.plot(x1,y1,'.',label='Estimated')
plt.legend()
plt.show()
