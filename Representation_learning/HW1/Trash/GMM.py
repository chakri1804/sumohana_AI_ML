import numpy as np
from math import log
### Dataset generation
#

# np.random.seed(42)

print("A note to the user\n This code has a function of generating data on it's own \n from a given number of multivariate_normal functions. \n This makes the covariance matrices go to Singular sometimes even though they are ensured to be positive semi-definite.")

print("In SUCH a case, please execute the code again \n")
print("\n")

print("Give the degree for the data to be generated")
D = int(input())
print("Give the number of modes the data must have")
Nm = int(input())
print("Give the number of samples per mode")
Nspm = int(input())
print("Give log likelihood error tolerance")
err = float(input())

# D = 5
# Nm = 2
# Nspm = 100
Ns = Nspm*Nm

Mean = []
Cov = []
data = []
for i in range(Nm):
    Mean.append(np.linspace(1,2,D))
    Cov.append(np.identity(D)*(i+1))
    for j in range(Nspm):
        data.append(np.random.multivariate_normal(Mean[i],Cov[i]))

def MulGaus_pdf(x,mean,cov):
    # Reshaping and converting for simpler steps
    x = np.array(x)
    mean = np.array(mean)
    cov = np.array(cov)
    d = len(mean)
    x = x.reshape((d,1))
    mean = mean.reshape((d,1))
    # Actual Functions
    temp = np.matmul((x-mean).T,np.linalg.inv(cov))
    temp = np.matmul(temp,(x-mean))
    num = np.exp(-0.5*temp)
    den = np.sqrt(np.linalg.det(2*np.pi*cov))
    pdf = num/den
    return pdf

'''
# Debug for PDF Function
# print(MulGaus_pdf(Mean[0],Mean[0],Cov[0]))
# den = np.sqrt(np.linalg.det(2*np.pi*Cov[0]))
# print(1/den)
'''

def posterior(data,mean_list,cov_list,mix_list,i,k):
    post = mix_list[k]*MulGaus_pdf(data[i],mean_list[k],cov_list[k])
    post_sum = 0
    for w in range(len(mix_list)):
        post_sum += mix_list[w]*MulGaus_pdf(data[i],mean_list[w],cov_list[w])
    gamma = post/post_sum
    return (np.asscalar(post), np.asscalar(post_sum), np.asscalar(gamma))

def posterior_sum(data,mean_list,cov_list,mix_list,i):
    _,temp,_ = posterior(data,mean_list,cov_list,mix_list,i,0)
    return temp

'''
# Debug for posterior function
mix = np.random.uniform(1,10,len(Mean))
mix = mix/np.sum(mix)
print(posterior(data,Mean,Cov,mix,1,0))
print(posterior_sum(data,Mean,Cov,mix,1))
'''

def log_likelihood(data,mean_list,cov_list,mix_list):
    log_like = 0
    for i in range(len(data)):
        post_sum = posterior_sum(data,mean_list,cov_list,mix_list,i)
        log_post_sum = np.log(post_sum)
        log_like += log_post_sum
    # print(log_like)
    return np.asscalar(log_like)

'''
# print(log_likelihood(data,Mean,Cov,mix))
'''


#####################################################################
### Random initialisation for the mean, covariance and mixing factors
#####################################################################

mix = np.random.uniform(2,3,len(Mean))
mix_list = mix/np.sum(mix)
mean_list = []
cov_list = []
for i in range(Nm):
    mean_list.append(np.random.uniform(1,5,D))
    temp1 = np.random.uniform(1,5,(D,D))
    cov_list.append(np.matmul(temp1,temp1.T))
print('old log likelihood =', log_likelihood(data,mean_list,cov_list,mix_list))
#############################
### Expeectation Maximization
#############################
itera = 0
while(itera<100):
    mean_list_temp = []
    cov_list_temp = []
    mix_list_temp = []
    log_o = log_likelihood(data,mean_list,cov_list,mix_list)
    for w in range(Nm):
        Nk = 0
        mean_num = 0
        cov_num = 0
        for i in range(Ns):
            _,_,gam = posterior(data,mean_list,cov_list,mix_list,i,w)
            Nk += gam
        for x in range(Ns):
            _,_,gam = posterior(data,mean_list,cov_list,mix_list,x,w)
            mean_num += gam*data[x]
            cov_num += gam*np.outer((data[x]-mean_list[w]),(data[x]-mean_list[w]))
        mean_list_temp.append(mean_num/Nk)
        cov_list_temp.append(cov_num/Nk)
        mix_list_temp.append(Nk/Ns)
    itera += 1
    mean_list = mean_list_temp
    cov_list = cov_list_temp
    mix_list = mix_list_temp
    log_n = log_likelihood(data,mean_list_temp,cov_list_temp,mix_list_temp)
    error = log_n - log_o
    print('new log likelihood = ',log_n,'Log likelihood error = ',error)
    if(error<err):
        break
###
print(mix_list)
print("###############")
print(mean_list)
print(Mean)
print("###############")
print(cov_list)
print(Cov)
###
