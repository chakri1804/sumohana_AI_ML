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
data = np.random.lognormal(0,1,k)
std_ref = np.std(data)
ref_gaussian = np.random.normal(0,std_ref,k)

tail = tail_prob(data,t)
tail1 = tail_prob(ref_gaussian,t)
plt.plot(t,tail,label="Heavy Tailed distribution (lognormal)")
plt.plot(t,tail1,label="Reference Gaussian")
plt.legend()
plt.show()
