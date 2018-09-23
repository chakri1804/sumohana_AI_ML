# In[238]:
import numpy as np

###
print("For the sake of verification, data was generated from a Gaussian Mixture")
print("Give degree of data for generation (in positive integers)")
deg = int(input())
print("Number of modes for generation")
Nm = int(input())
print("Number of data points")
Ns = int(input())

### Data Generation
Mean = []
Cov = []
data = []
Mix = np.random.uniform(10,100,Nm)
Mix = Mix/np.sum(Mix)
Mix = Mix.tolist()
for i in range(Nm):
    Mean.append(np.random.randint(1,10,deg))
    temp = np.diag(np.random.randint(2,9,deg))
    Cov.append(temp)
for i in range(Ns):
    temp1 = 0
    for j in range(Nm):
        temp1 += Mix[j] * np.random.multivariate_normal(Mean[j],Cov[j],1)
    data.append(temp1)
data = np.array(data)
data = data.reshape(Ns,deg)

# In[241]:
def gaus_mul_pdf(x,mean,cov,degree):
    x = x.reshape((degree,1))
    mean = mean.reshape((degree,1))
    ph = np.matmul(np.matmul(((x-mean).T),np.linalg.inv(cov)),(x-mean))
    num = np.exp((-0.5)*ph)
    den = np.sqrt(np.linalg.det(2*np.pi*cov))
    temp = num/den
    temp = np.asscalar(temp)
    return temp

def gam(mix,data_matrix,mean_matrix,cov_matrix,Nm,i,k,degree):
    temp = 0
    for w in range(Nm):
        temp +=  mix[w]*gaus_mul_pdf(data_matrix[i],mean_matrix[w],cov_matrix[w],degree)
    temp1 = mix[k]*gaus_mul_pdf(data_matrix[i],mean_matrix[k],cov_matrix[k],degree)/temp
    return temp1

def Nk(mix,data_matrix,mean_matrix,cov_matrix,Nm,k,Ns,degree):
    temp = 0
    for w in range(Ns):
        temp = temp + gam(mix,data_matrix,mean_matrix,cov_matrix,Nm,w,k,degree)
    return temp

def log_likelihood(data,mean_matrix,cov_matrix,mix,Nm,Ns,degree):
    temp = 0
    temp1 = 0
    for i in range(Ns):
        for j in range(Nm):
            temp += mix[j]*gaus_mul_pdf(data[i],mean_matrix[j],cov_matrix[j],degree)
        temp1 += np.log(temp)
    return temp1

mix = []
cov_matrix = []
mean_matrix = []
for i in range(Nm):
    mean_matrix.append(np.random.randint(1,10,deg))
    temp = np.diag(np.random.randint(1,10,deg))
    cov_matrix.append(temp)
    mix.append(float(1.0/Nm))

log_likelihood(data,mean_matrix,cov_matrix,mix,Nm,Ns,deg)

print("Give log error delta")
error = float(input())
itera = 0
err_delta = 100.0
log_o = log_likelihood(data,mean_matrix,cov_matrix,mix,Nm,Ns,deg)

while(itera<100):
    new_mean_matrix = []
    new_cov_matrix = []
    new_mix = []
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
    log_n = log_likelihood(data,mean_matrix,cov_matrix,mix,Nm,Ns,deg)
    err_delta = np.abs(log_o - log_n)
#     print("### ",err_delta)
    print(log_o)
    itera += 1
    log_o = log_n
    if err_delta<error:
        break
# In[248]:
print(mix)
print(Mix)
print(mean_matrix)
print(Mean)
print(cov_matrix)
print(Cov)
