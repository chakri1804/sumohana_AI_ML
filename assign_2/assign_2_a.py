import numpy as np
import pandas as pd

# Import Data
df_X = pd.read_csv('X.csv',header=None)
df_Y = pd.read_csv('Y.csv',header=None)
X = np.array(df_X,dtype=np.float64)
# X = X.T
Y = np.array(df_Y,dtype=np.float64)
X_test = np.array([[1,1],[1,-1],[-1,1],[-1,-1],[2,0]])

# Get number of labels in both classes for class occurance
# probability
unique, counts = np.unique(Y, return_counts=True)
print(unique[0],counts[0],len(Y))
pr0 = float(counts[0])/len(Y)
pr1 = float(counts[1])/len(Y)

############################################
# Various means and variances for given data
############################################

# Consider labels with 1 and -1 and store their indices
y = Y
y_0 = np.where(y==-1.)[0]
y_1 = np.where(y==1.)[0]
# Calculate label specific mean and variances with all combinations
m00 = np.mean(X[0][y_0])
m01 = np.mean(X[0][y_1])
m10 = np.mean(X[1][y_0])
m11 = np.mean(X[1][y_1])
v00 = np.var(X[0][y_0])
v01 = np.var(X[0][y_1])
v10 = np.var(X[1][y_0])
v11 = np.var(X[1][y_1])
s00 = np.std(X[0][y_0])
s01 = np.std(X[0][y_1])
s10 = np.std(X[1][y_0])
s11 = np.std(X[1][y_1])
######################
#  Calculate Pr(y=k|x)
######################

for i in range(0,len(X_test)):
    pr_1 = (1.0/(2*np.pi*s01*s11))*(np.exp(-((X_test[i][0]-m01)**2)/(2.0*v01)))*(np.exp(-((X_test[i][1]-m11)**2)/(2.0*v11)))*pr1
    pr_0 = (1.0/(2*np.pi*s00*s10))*(np.exp(-((X_test[i][0]-m00)**2)/(2.0*v00)))*(np.exp(-((X_test[i][1]-m10)**2)/(2.0*v10)))*pr0
    print(pr_1,pr_0,X_test[i])
    if pr_1 > pr_0:
        print(X_test[i],'1')
    else:
        print(X_test[i],'-1')
