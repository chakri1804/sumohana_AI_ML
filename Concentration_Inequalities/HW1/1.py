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

# Degrees of freedom
n = 100
# Samples per distribution selected
k = 100000
# Reference for plots
t = np.linspace(0,100,k)
data = np.zeros((k,))

# Generating Chi-Squared samples
for i in range(n):
    x = np.random.normal(0,1,(k,))
    data = data + x**2
# Reference Gaussian generation. Variance is chosen as the variance of the chi-squared samples generated before
ref_gaussian = np.random.normal(0,np.std(data),(k,))

tail = np.array(tail_prob(data,t))
plt.plot(t,tail,label='Chi-square')

tail1 = np.array(tail_prob(ref_gaussian,t))
plt.plot(t,tail1,label='Gaussian')
plt.ylabel("Tail probabilities")
plt.legend()
plt.show()
