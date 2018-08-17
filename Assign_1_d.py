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
W = np.zeros((N,N))
X = np.column_stack((temp,x))
# Most likelihood weights
W = np.matmul(np.linalg.pinv(X),y)
print(W)
y1 = np.matmul(X,W)
# error = (y1-y)
plt.plot(x,y,'*')
plt.plot(x,y1,'r')
plt.show()

# Most likelihood case variance 1/{beta}
labels = np.random.normal(y1,std,N)
error = y1 - y
# plt.plot(x,error,'b')
# plt.show()
variance = np.var(error)
print("variance in labels =",variance)
