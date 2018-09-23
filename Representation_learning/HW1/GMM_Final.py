#!/usr/bin/env python
# coding: utf-8

# In[238]:


import numpy as np


# In[239]:


print("For the sake of verification, data was generated from a Gaussian Mixture")
print("Give degree of data for generation (in positive integers)")
deg = int(input())
print("Number of modes for generation")
Nm = int(input())
print("Number of data points")
Nspm = int(input())


# In[240]:


Ns = Nspm

Mean = []
Cov = []
data = []
Mix = np.linspace(10,100,Nm)
Mix = Mix/np.sum(Mix)
Mix = Mix.tolist()
for i in range(Nm):
    Mean.append(np.random.randint(1,10,deg))
    temp = np.random.randint(2,9,(deg,deg))
    Cov.append(np.matmul(temp,temp.T))
for i in range(Nspm):
    temp1 = 0
    for j in range(Nm):
        temp1 += Mix[j] * np.random.multivariate_normal(Mean[j],Cov[j],1)
    data.append(temp1)
data = np.array(data)
data = data.reshape(Ns,deg)
print(Mix)


# In[241]:


def gaus_mul_pdf(x,mean,cov,degree):
    x = x.reshape((degree,1))
    mean = mean.reshape((degree,1))
    ph = np.matmul(np.matmul(((x-mean).T),np.linalg.inv(cov)),(x-mean))
    temp = np.exp((-0.5)*ph)/np.sqrt((2*np.pi*np.linalg.det(cov)))
    temp = np.asscalar(temp)
#     print(temp.shape)
    return temp


# In[242]:


def gam(mix,data_matrix,mean_matrix,cov_matrix,Nm,i,k,degree):
    temp = 0
    for w in range(Nm):
        temp = temp + mix[w]*gaus_mul_pdf(data_matrix[i],mean_matrix[w],cov_matrix[w],degree)
    temp1 = mix[k]*gaus_mul_pdf(data_matrix[i],mean_matrix[k],cov_matrix[k],degree)/temp
    return temp1


# In[243]:


def Nk(mix,data_matrix,mean_matrix,cov_matrix,Nm,k,Ns,degree):
    temp = 0
    for w in range(Ns):
        temp = temp + gam(mix,data_matrix,mean_matrix,cov_matrix,Nm,w,k,degree)
    return temp


# In[244]:


def log_likelihood(data,mean_matrix,cov_matrix,mix,deg,Ns):
    temp = 0
    temp1 = 0
    for i in range(Ns):
        for j in range(deg):
            temp += mix[j]*gaus_mul_pdf(data[i],mean_matrix[j],cov_matrix[j],deg)
        temp1 += np.log(np.array(temp))
#     print(temp1)
    return temp1


# In[245]:


### Initial Parameter setting
# mix = np.full((1,Nm),1/Nm)
mix = []
cov_matrix = []
mean_matrix = []
for i in range(Nm):
    mean_matrix.append(np.random.randint(1,10,deg))
    temp = (np.random.randint(1,10,(deg,deg)))
    cov_matrix.append(np.matmul(temp,temp.T))
    mix.append(float(1.0/Nm))


# In[246]:


# cov_matrix[0]
log_likelihood(data,mean_matrix,cov_matrix,mix,deg,Ns)


# In[247]:


print("Give log error delta")
error = float(input())
itera = 0
err_delta = 100.0
while(itera<100):
    new_mean_matrix = []
    new_cov_matrix = []
    new_mix = []
    log_o = log_likelihood(data,mean_matrix,cov_matrix,mix,deg,Ns)
#     print(mix)
    for k in range(Nm):
        meh = Nk(mix,data,mean_matrix,cov_matrix,Nm,k,Ns,deg)
        temp1 = 0
        temp2 = 0
        temp3 = 0
        for i in range(Ns):
            temp1 = np.outer((data[i]-mean_matrix[k]),(data[i]-mean_matrix[k]))
            temp2 += gam(mix,data,mean_matrix,cov_matrix,Nm,i,k,deg)*temp1
            temp3 += gam(mix,data,mean_matrix,cov_matrix,Nm,i,k,deg)*data[i]
        new_cov_matrix.append(temp2/meh)
        new_mean_matrix.append(temp3/meh)
        new_mix.append(meh/Ns)
    mean_matrix = new_mean_matrix
    cov_matrix = new_cov_matrix
    mix = new_mix
    log_n = log_likelihood(data,mean_matrix,cov_matrix,mix,deg,Ns)
    err_delta = np.abs(log_o - log_n)
#     print("### ",err_delta)
    print(log_o)
    itera += 1
    if err_delta<error:
        break


# In[248]:


print(mix)
print(mean_matrix)
print(cov_matrix)

