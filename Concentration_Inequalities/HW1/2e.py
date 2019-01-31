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

# Number of bounded distros to be summed up
n = 20
# Samples per distribution
k = 10000
# Reference for plots
t = np.linspace(0,100,k)

data = np.zeros((k,))

# Generating Chi-Squared samples
for i in range(n):
    x = np.random.uniform(-1,1,(k,))
    data = data + x

std_ref = 2*np.std(data)
ref_gaussian = np.random.normal(0,std_ref,k)

tail = tail_prob(data,t)
tail1 = tail_prob(ref_gaussian,t)
plt.plot(t,tail,label="Bounded RVs summed (Uniform)")
plt.plot(t,tail1,label="Reference Gaussian")
plt.legend()
plt.show()
