#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# In[3]:


def Mean_centering(X):
    X = X.astype('float')
    for i in range(len(X)):
        X[i] = X[i] - np.mean(X[i])
    print(np.mean(X[0]))
    return X


# In[4]:


pic = Image.open('wildlife-bears2.jpg')
img = (np.array(pic))


# In[5]:


a,b,c = img.shape
print(a,b,c)


# In[6]:


Flat = img.reshape((a*b,c))
Flat = Flat.astype('float32')


# In[7]:


Flat = Flat.T
Flat.shape


# In[8]:


Flat = Mean_centering(Flat)


# In[9]:


Cx = (np.matmul(Flat,Flat.T)/(len(Flat[0])))


# In[10]:


Lx,Ex = np.linalg.eig(Cx)


# In[11]:


Y = np.matmul(Ex.T,Flat)
Y = Y.T
Y = Y.reshape(img.shape)


# In[12]:


Y = Y.astype(int)


# In[13]:


plt.imshow(Y)


# In[14]:


plt.imshow(pic)

