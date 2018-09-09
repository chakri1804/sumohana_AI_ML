
# coding: utf-8

# In[3]:


import numpy as np
import cvxpy as cp
import pandas as pd

df_X = pd.read_csv('Xsvm.csv',header=None)
df_Y = pd.read_csv('ysvm.csv',header=None)
X = np.array(df_X,dtype=np.float64)
Y = np.array(df_Y,dtype=np.float64)
print(X.shape,Y.shape)


# In[4]:


# Convex Optimization
a = cp.Variable(len(Y))
R1 = cp.matmul(cp.diag(a),Y)
R2 = cp.matmul(X.T,R1)
R4 = cp.norm(R2)**2
R4.shape


# In[5]:


P1 = cp.sum(a)
Const1 = P1 - 0.5*R4
# Const1 = np.reshape(Const1,(1,))


# In[6]:


Const2 = cp.matmul(a.T,Y)
Const3 = [0<=a,Const2 == 0]
obj = cp.Maximize(Const1)
prob = cp.Problem(obj, Const3)
prob.solve(verbose=True)


# In[7]:


print(a.value)


# In[8]:


A = (np.array(a.value)).reshape(500,1)


# In[9]:


W = np.zeros((2,))
for i in range(len(Y)):
    W += A[i]*Y[i]*(X[i].T)
    if(A[i]>1e-4):
        print(i)


# In[10]:


print(W)


# In[11]:


W0 = (1/Y[281]) - np.dot(W,X[281])
print(W0)


# In[12]:


Test = np.array([[2,0.5],[0.8,0.7],[1.58,1.33],[0.008, 0.001]])

for i in range(len(Test)):
    est = np.sign(np.dot(W,Test[i])+W0)
    print(Test[i],est)


# In[14]:


# Verification
from sklearn import svm

clf = svm.SVC()
clf.fit(X,Y)


# In[15]:


clf.predict(Test)

