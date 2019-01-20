import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Determine the number of IIDs taken for CLT approximation
n = 10000
# Number of samples being called for each IIDs
k = 100000
# Uniform Random Variable
a = np.zeros((k,))
for i in range(n):
    a += np.random.uniform(-100,100,(k,))
a = a/n

# Rayleigh RV with scale 1
b = np.zeros((k,))
for i in range(n):
    b += np.random.rayleigh(1, (k,))
b = b/n

# Poisson RV with Lambda = 1
c = np.zeros((k,))
for i in range(n):
    c += np.random.poisson(1,(k,))
c = c/n

# Log-Normal RV
d = np.zeros((k,))
for i in range(n):
    d += np.random.lognormal(0,0.1,(k,))
d = d/n

clts = [a,b,c,d]
names = ["Uniform","Rayleigh","Poisson","Log-Normal"]

fig, ax = plt.subplots(4, 1)

for i in range(len(clts)):
    hist, bins = np.histogram(clts[i], bins=50, normed=True)
    bin_centers = (bins[1:]+bins[:-1])*0.5
    ax[i].set_title(names[i])
    ax[i].plot(bin_centers, hist)
plt.show()
