from cmath import exp
from signal import valid_signals
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#import data
data = pd.read_csv('./train.csv')

#display first rows
print(data.head())

#in order to apply complex math use np
data = np.array(data)

#Important to seperate the data
# one set is to train your neural network
# the other is to test your validation

m, n =  data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T #transpose of the matrix as seen in readme
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

#first row len
print(X_train[:,0].shape)

def InitParams():
    W1 = np.random.randn(10, 784) - 0.5 #will generate values randomly betweeen -0.5 and 0.5
    #activation 1
    b1 = np.random.randn(10, 1) - 0.5

    W2 = np.random.randn(10, 10) - 0.5
    b2 = np.random.randn(10, 1) - 0.5
    return W1, b1, W2, b2

def relU(val):
    return np.maximum(0, val)

def softmax(val):
    return np.exp(val)/np.sum(np.exp(val))

def ForwardPropagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = relU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def onehot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() +1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_relU(val):
    #derivative of relU is 1 or zero so we can use true or false
    return val > 0

def BackPropagation(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = onehot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, 2)
    dZ1 = W2.T.dot(dZ2) * deriv_relU(Z1)
    dW1 = 1/m * dZ2.dot(X.T)
    db1 = 1/m * np.sum(dZ1, 2)

    return dW1, db1, dW2, db2

def UpdateParams(W1, b1, w2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

