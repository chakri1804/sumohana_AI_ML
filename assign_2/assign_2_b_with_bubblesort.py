import numpy as np
import pandas as pd

def euclidean_distances(X,X_train):
	dist = np.zeros((1000,1))
	for j in np.arange(0,1000):
		dist[j] = ((X[0]-X_train[j][0])**2)+((X[1]-X_train[j][1])**2)
	dist = np.sqrt(dist)
	dist = np.reshape(dist,1000)
	return dist

def bubbleSort(arr):
	n = len(arr)
	for i in range(0,n):
		for j in range(0, i):
			if arr[j][0] > arr[j+1][0]:
				temp1 = arr[j][0]
				temp2 = arr[j][1]
				arr[j][0] = arr[j+1][0]
				arr[j][1] = arr[j+1][1]
				arr[j+1][0] = temp1
				arr[j+1][1] = temp2
	return arr

df_X = pd.read_csv('X.csv',header=None)
df_Y = pd.read_csv('Y.csv',header=None)
X = np.array(df_X,dtype=np.float64)
X = X.T
Y = np.array(df_Y,dtype=np.float64)
X_test = np.array([[1,1],[1,-1],[-1,1],[-1,-1],[1e-5,0.1]])

print("Give K for the K nearest neighbors \n")
K = input()
K = int(K)

for i in range(0,len(X_test)):
	dist = euclidean_distances(X_test[i],X)
	temp = np.array((dist,Y.reshape(1000,)))
	temp = temp.T
	temp = bubbleSort(temp)
	temp = temp[:K]
	temp1 = [x[1] for x in temp]
	estimate = float(np.sum(temp1)/K)
	if estimate>0 :
		print(X_test[i],":1")
	else:
		print(X_test[i],":-1")
