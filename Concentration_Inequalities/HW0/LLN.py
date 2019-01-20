import numpy as np
import matplotlib.pyplot as plt

## LLN basically states as we take more and more samples from a single distribution and take the mean of them
## The mean tends to the mean of the original distribution

## Here I am plotting a graph of the mean of the samples drawn from a given distribution vs
## the number of samples chosen

# Uniform Random Variable
n1 = 10000
a = []
for i in range(n1):
    a.append(np.mean(np.random.uniform(-10,10,(i,))))

# Log-Normal Random Variable
n2 = 10000
b = []
mean = 0.0
sigma = 1.0
for i in range(n2):
    b.append(np.mean(np.random.lognormal(mean,sigma,(i,))))

act_mean = np.exp(mean + (sigma**2)/2)
# Gaussian Random Variable
n3 = 10000
c = []
for i in range(n3):
    c.append(np.mean(np.random.normal(0,1,(i,))))

# Laplace Random Variable
n4 = 10000
d = []
for i in range(n4):
    d.append(np.mean(np.random.laplace(0,0.1,(i,))))

llns = [a,b,c,d]
names = ["Uniform","Log-Normal","Gaussian", "Laplace"]
fig, ax = plt.subplots(4, 1)

for i in range(len(llns)):
    ax[i].set_title(names[i])
    ax[i].plot(llns[i])
    if (i != 1):
        ax[i].plot(np.zeros(np.shape(llns[i])), 'r')
    else:
        ax[i].plot(np.zeros(np.shape(llns[i])) + act_mean, 'r')
plt.show()
