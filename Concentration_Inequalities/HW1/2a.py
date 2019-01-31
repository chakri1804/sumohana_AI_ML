import numpy as np
import matplotlib.pyplot as plt

def tail_prob(data,t):
    probs = []
    for i in range(0,len(t)):
        norm = data - np.mean(data)
        temp = np.count_nonzero(norm > t[i])
        temp = temp*1.0/len(data)
        probs.append(temp)
    return probs

# Samples per distribution
k = 100000
# Reference for plots
t = np.linspace(0,100,k)

# Data generation
std = 10
std_ref = 2*std
data = np.random.normal(0,std,k)
ref_gaussian = np.random.normal(0,std_ref,k)

tail = tail_prob(data,t)
tail1 = tail_prob(ref_gaussian,t)
plt.plot(t,tail,label="Gaussian")
plt.plot(t,tail1,label="Reference Gaussian")
plt.legend()
plt.show()
