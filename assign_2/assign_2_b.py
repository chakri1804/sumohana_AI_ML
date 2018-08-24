import numpy as np
import pandas as pd

def euclidean_distances(X,X_train):
	dist = np.zeros((1000,1))
	for j in np.arange(0,1000):
		dist[j] = np.linalg.norm(X-X_train[j])
	dist = np.reshape(dist,1000)
	return dist

df_X = pd.read_csv('X.csv',header=None)
df_Y = pd.read_csv('Y.csv',header=None)
X = np.array(df_X,dtype=np.float64)
X = X.T
Y = np.array(df_Y,dtype=np.float64)
X_test = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])

print("Give K for the K nearest neighbors \n")
K = input()
K = int(K)

for i in range(0,len(X_test)):
	dist = euclidean_distances(X_test[i],X)
	idx = np.argpartition(dist, K)
	b = np.take(Y, idx[:K])
	estimate = float(np.sum(b)/K)
	if estimate>0 :
		print(X[i],":1")
	else:
		print(X[i],":-1")
