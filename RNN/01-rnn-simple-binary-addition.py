#-*- coding: utf-8 -*-

# based on
# https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/
# https://github.com/jaejun-yoo/RNN-implementation-using-Numpy-binary-digit-addition/blob/master/RNN%20implementation%20with%20Numpy%20(binary%20addition).ipynb
import matplotlib.pyplot as plt
import os
def exit():
    os._exit(0)

import copy, numpy as np
np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


# training dataset generation
int2binary = {}
binary_dim = 8
print(range(binary_dim))

largest_number = pow(2,binary_dim)
binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)

for i in range(largest_number):
    int2binary[i] = binary[i]

print(int2binary[0])
print(int2binary[3])

# input variables
alpha = 0.1
input_dim = 2   # 각 자리수끼리 더할 것이므로 서로 더할 두 이진수의 n번째 자리에 해당하는 digit 두 개가 input dimension이 됨
hidden_dim = 16
output_dim = 1

# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

verr=[]
vindex=[]
# training logic
for j in range(10000):
    
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    
    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        
        # generate input and output
        # LSB 부터 시작
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer (input ~+ prev_hidden)
        # h_t = sigmoid(X * W_{hx} + h_{t - 1} * W_{hh})
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1)) #  dim: (1, 1) e.g., [[ 0.52385887]]

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
    
        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))

    verr.append(overallError)
    vindex.append(j)
    future_layer_1_delta = np.zeros(hidden_dim)


    for position in range(binary_dim):
        # from MSB
        X = np.array([[a[position],b[position]]])

        # list에서 append로 넣었으니 역으로 접근하여 MSB쪽을 꺼냄
        # 진행을 LSB 부터 시작했으니 LSB 방향이 previous 방향임
        layer_1 = layer_1_values[-position-1]       # Selecting the current hidden layer from the list.
        prev_layer_1 = layer_1_values[-position-2]  # Selecting the previous hidden layer from the list
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]

        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print "Error:" + str(overallError)
        print "Pred:" + str(d)
        print "True:" + str(c)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print str(a_int) + " + " + str(b_int) + " = " + str(out)
        print "------------"


plt.plot(vindex, verr)
plt.show()