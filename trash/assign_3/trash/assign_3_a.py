import numpy as np
import cvxpy as cvx
import pandas as pd

df_X = pd.read_csv('Xsvm.csv',header=None)
df_Y = pd.read_csv('ysvm.csv',header=None)

X = np.array(df_X,dtype=np.float64)
Y = np.array(df_Y,dtype=np.float64)
temp = np.hstack((X,Y))
