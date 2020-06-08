import numpy as np
import cvxpy as cp
import pandas as pd

df_X = pd.read_csv('Xsvm.csv',header=None)
df_Y = pd.read_csv('ysvm.csv',header=None)
X = np.array(df_X,dtype=np.float64)
Y = np.array(df_Y,dtype=np.float64)
print(X.shape,Y.shape)

# Convex Optimization
a = cp.Variable(len(Y))
R1 = cp.matmul(cp.diag(a),Y)
R2 = cp.matmul(X.T,R1)
R4 = cp.norm(R2)**2
R4.shape

P1 = cp.sum(a)
Const1 = P1 - 0.5*R4
Const2 = cp.matmul(a.T,Y)
Const3 = [0<=a,Const2 == 0]
obj = cp.Maximize(Const1)
prob = cp.Problem(obj, Const3)
prob.solve(verbose=True)
# print(a.value)

A = (np.array(a.value)).reshape(500,1)
W = np.zeros((2,))
for i in range(len(Y)):
    W += A[i]*Y[i]*(X[i].T)
    if(A[i]>1e-4):
        print(i)

print(W)
W0 = (1/Y[281]) - np.dot(W,X[281])
print(W0)

Test = np.array([[1.9, 0.4],[0.9, 0.9],[1.4, 1.5],[0.01, 0.005]])
for i in range(len(Test)):
    est = np.sign(np.dot(W,Test[i])+W0)
    print(Test[i],est)

# Verification
from sklearn import svm

clf = svm.SVC()
clf.fit(X,Y)
clf.predict(Test)
