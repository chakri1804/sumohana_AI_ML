import numpy as np
import matplotlib.pyplot as plt

## Comparing tightness among Chernoff, Hoeffding and Bennets inequalities is basically checking out
## Which of the inequalities model the tail probabilities the best

## For convinience, Uniform random variable has been chosen with 0 mean and bounded between -10 and 10
## n such uniforms are added up to get make Hoeffding and Bennets relevant

samples = 10000
## Lambdas for Bennets, Chernoff and Hoeffding
t = np.linspace(0,100,1e4)
# Fix the number of distributions to add up
k = 10
# Fix bounds on uniform random variable
a = -5.0
b = 5.0

def tail_prob(data,t):
    probs = []
    for i in range(0,len(t)):
        norm = data - np.mean(data)
        temp = np.count_nonzero(norm > t[i])
        temp = temp*1.0/len(data)
        probs.append(temp)
    return probs

data = np.zeros((samples,))
for i in range(k):
    data += np.random.uniform(a,b,samples)

def h(x):
    return (1+x)*np.log(1+x)-x

# Variance formula for Uniform RV
v = (k*a**2)/3.0

## the bounds
bennet = np.exp(-v*h(b*t*1.0/v)/b**2) ## The expression from problem 4
hoeff = np.exp(-2*np.square(t)/(k*np.square(a*2)))
tail = np.array(tail_prob(data,t))

## Chernoff bound must hold for any s, so fixing s to prevent multivariable plotting dilemma
s = 1e-1
chernoff = np.power((np.exp(s*a)-np.exp(-s*a))/(2*s*a),k) * np.exp(-s*t)

plt.plot(tail,label='Tail probabilities')
plt.plot(bennet,label='Bennet Bound')
plt.plot(hoeff,label='Hoeffding Bound')
plt.plot(chernoff,label='Chernoff Bound')
plt.legend()
plt.show()
