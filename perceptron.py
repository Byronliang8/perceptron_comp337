import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def readData(testData, trainData):
    train_data = pd.read_table(trainData, delimiter=",")
    test_data = pd.read_table(testData, delimiter=",")
    x_train = train_data.iloc[:, 0:4]
    y_train = train_data.iloc[:, 4]

    x_test = test_data.iloc[:, 0:4]
    y_test = test_data.iloc[:, 4]
    return (x_train,y_train,x_test,y_test)

def label_num(y):
    return len(y.unique())


def linearly_separable(x,w,b):
    z=np.dot(w.T, x) + b
    if z<0 :
        return -1
    if z>=0 :
       return 1

# initialize_weights randomly
def initialize_weights(dim):
    w = np.zeros(dim)
    b = 0
    return w, b

def perceptron_model(x,w,b,y,r):
    for i in range(100):
        output=linearly_separable(x,w,b)
        error=err(y,output)
        w=update(error,x,w,r)

    return 0

def err(label,output):
    return label-output

def update(err,x,w,r):
    w=w+err*x*r
    return w

# Prediction function
def predict(W, b, X):
    m = X.shape[1]
    Y_pred = np.zeros((1, m))
    W = W.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(W.T, X) + b)

    for i in range(m):
        Y_pred[0, i] = 1 if A[0, i] > 0.5 else 0

    assert (Y_pred.shape == (1, m))

    return Y_pred

trainPath='data/train.data'
testPath='data/test.data'
x_train,y_train,x_test,y_test=readData(trainPath,testPath)

w,b=initialize_weights(4)

#optimize(w,b,x_train,y_train,0.1)
print(initialize_weights(4))