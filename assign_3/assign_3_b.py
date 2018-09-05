import numpy as np

# Parameters
print("Give Number of Hidden Layer nodes :")
M = int(input())
print("Give number of training samples per bit pair:")
n = int(input())
print("Give std.dev for noise :")
noise = float(input())
print("Give number of Epochs :")
epochs = int(input())
print("Give learning rate :")
lr = float(input())
print("XOR/OR/AND? eg.Enter XOR for XOR")
option = input()

# Generate Datasets

Input = np.array([[0,0],[0,1],[1,0],[1,1]])
if(option == "OR"):
    Output = np.array([[0],[1],[1],[1]])
if(option == "AND"):
    Output = np.array([[0],[0],[0],[1]])
if(option == "XOR"):
    Output = np.array([[0],[1],[1],[0]])

X = []
Y = []

for i in range(len(Input)):
	for j in range(n):
		X.append(Input[i]+np.random.normal(0,noise,(1,2)))
		Y.append(Output[i]+np.random.normal(0,noise))

X = np.array(X)
X = X.reshape((len(Y),2))
Y = np.array(Y)

# defining functions

def sigm(x):
	return 1/(1+np.exp(-x))

def diff_sigm(x):
	return (sigm(x)-sigm(x)**2)

def layer(x,W,b):
	return (np.matmul(W.T,x.reshape(len(x),1)) + b)

def sq_err(y,Y):
	return (y-Y)**2
# Initializing weights

W1 = np.random.normal(0,1,(2,M))
Bi1 = np.random.normal(0,1,(M,1))
W2 = np.random.normal(0,1,(M,1))
Bi2 = np.random.normal(0,1,(1,1))

# training

for i in range(epochs):
	w1 = np.zeros(W1.shape)
	b1 = np.zeros(Bi1.shape)
	w2 = np.zeros(W2.shape)
	b2 = np.zeros(Bi2.shape)
	for j in range(len(Y)):
		#forward path
		out1 = layer(X[j],W1,Bi1)
		# print(out1)
		z = sigm(out1)
		# print(z)
		out2 = layer(z,W2,Bi2)
		y = sigm(out2)
		# print(y)
		#backpropagation
		b2 += 2*(y-Y[j])*diff_sigm(out2)
		w2 += 2*(y-Y[j])*diff_sigm(out2)*z
		loss = sq_err(y,Y[j])
		for k in range(M):
			b1[k] += (2*(y-Y[j])*diff_sigm(out2)*diff_sigm(out1[k])*W2[k]).reshape(1,)
			w1[:,k] += (2*(y-Y[j])*diff_sigm(out2)*diff_sigm(out1[k])*W2[k]*X[j]).reshape(2,)
	print(loss)
	W1 -= lr*w1
	W2 -= lr*w2
	Bi1 -= lr*b1
	Bi2 -= lr*b2

while(1):
    a=input("Enter test sample: ").split(',')
    for i in range(0,2):
        a[i]=float(a[i])
    a=np.array(a)
    print(a.shape)
    out_1 = layer(a,W1,Bi1)
    z = sigm(out_1)
    out_2 = layer(z,W2,Bi2)
    y_pred= sigm(out_2)
    if y_pred > 0.5:
        y_pred = 1
    else :
        y_pred = 0
    print(y_pred)
