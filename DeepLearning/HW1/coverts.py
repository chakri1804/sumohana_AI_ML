class conv2D():
    def __init__(self):
        inp = np.random.uniform(0,1,(28,28,1))
        ker_dim = (5,5)
        ker_val = np.random.normal(0,1,self.ker_dim)
        stride = 1
        padding = 'True'
        activation = 'relu'

    def operate(self):

        h,w,d = np.shape(inp)
        ker_list = list(ker_dim)
        ker_list.append(d)
        ker_dim = tuple(ker_list)

        print(ker_val)

        if(ker_dim != ker_val.shape):
            print("Enter proper shaped kernel values")
            return

        ini_w = 0
        ini_h = 0

        pad_w = (((stride-1)*w) + ker_list[1] - stride)/2
        pad_w = int(pad_w)

        pad_h = (((stride-1)*h) + ker_list[0] - stride)/2
        pad_h = int(pad_h)

        if padding == "True":
            pad_inp = np.zeros(((h+2*pad_h),(w+2*pad_w),d))
            for i in range(d):
                pad_inp[:,:,i] = np.pad(inp[:,:,i], ((pad_h,pad_h),(pad_w,pad_w)), 'constant')
            pad_inp = pad_inp.astype(int)
            out_w = int(((w + 2*pad_w - ker_list[1])/stride) + 1)
            out_h = int(((h + 2*pad_h - ker_list[0])/stride) + 1)

        else :
            pad_inp = inp
            out_w = int(((w - ker_list[1])/stride) + 1)
            out_h = int(((h - ker_list[0])/stride) + 1)

        pad_inp = pad_inp.astype(int)

        conv_out = np.zeros((out_h,out_w))
        for i in range(1,out_h+1):
            for j in range(1,out_w+1):
                temp = pad_inp[ini_h:ini_h + ker_list[0] , ini_w:ini_w + ker_list[1]]
                temp = temp.astype(float)
                ker_val = ker_val.astype(float)
                if temp.shape == ker_val.shape :
                    conv_out[i-1][j-1] = np.sum(temp*ker_val)
                ini_w += stride
            ini_h += stride
            ini_w = 0

        if activation == 'sigm' :
            conv_out = sigmoid(conv_out)

        elif activation == 'relu' :
            conv_out = relu(conv_out)

        elif activation == 'tanh' :
            conv_out = tanh(conv_out)

        return conv_out, ker_val
