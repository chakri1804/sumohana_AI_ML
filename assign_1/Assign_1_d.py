import numpy as np
import matplotlib.pyplot as plt

# Number of training samples
N = 50
# Generate equispaced floats in the interval [0, 2pi]
x = np.linspace(0, 2*np.pi, N)
# Generate noise
mean = 0
std = 0.01
# Generate some numbers from the sine function
y = np.sin(x)
# Add noise
y += np.random.normal(mean, std, N)
x = np.transpose(x)
temp = np.full(np.shape(x),1)
# W = np.zeros((N,N))
X = np.column_stack((temp,x))
# Most likelihood weights
W = np.matmul(np.linalg.pinv(X),y)
print(W)
y1 = np.matmul(X,W)
# error = (y1-y)
plt.plot(x,y,'*',label='True Values')
plt.plot(x,y1,'r',label='Estimated')
plt.legend()
plt.show()

# Most likelihood case variance 1/{beta}
# beta = np.var(y-y1)
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
