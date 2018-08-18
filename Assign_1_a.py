import numpy as np
import matplotlib.pyplot as plt
import scipy

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
x = np.transpose(x)
temp = np.full(np.shape(x),1)
# W = np.zeros((N,N))
X = np.column_stack((temp,x))
W = np.matmul(np.linalg.pinv(X),y)
print(W)
y1 = np.matmul(X,W)
plt.plot(x,y,'*',label='True values')
plt.plot(x,y1,'r',label='estimated')
plt.legend()
plt.show()

# Error as variance of error in prediction
error = (y1-y)
print("Variance of error =",np.var(error))
