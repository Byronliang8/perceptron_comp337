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

# Sigmoid function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


# initialize_weights randomly
def initialize_weights(dim):
    W = np.random.normal((dim, 1))
    b = 0
    return W, b

    assert (W.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))


# Propagation Function
def propagate_func(W, b, X, Y):
    # Forward Prop
    m = X.shape[1]
    A = sigmoid(np.dot(W.T, X) + b)
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    # Forward Prop

    # Backward Prop
    dW = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    # Backward Prop

    assert (dW.shape == W.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grad_dict = {"dW": dW, "db": db}
    return grad_dict, cost


# Optimization using gradient descent
def optimize(W, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):

        grad_dict, cost = propagate_func(W, b, X, Y)

        dW = grad_dict["dW"]
        db = grad_dict["db"]

        # Gradient Descent
        W = W - learning_rate * dW
        b = b - learning_rate * db
        # Gradient Descent

        if i % 1000 == 0:
            costs.append(cost)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    param_dict = {"W": W, "b": b}
    grad_dict = {"dW": dW, "db": db}

    return param_dict, grad_dict, costs


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


# Perceptron Model
def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost=False):
    W, b = initialize_weights(X_train.shape[0])
    param_dict, grad_dict, costs = optimize(W, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    W = param_dict["W"]
    b = param_dict["b"]

    Y_pred_test = predict(W, b, X_test)
    Y_pred_train = predict(W, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100))

    dict = {"costs": costs,
            "Y_pred_test": Y_pred_test,
            "Y_pred_train": Y_pred_train,
            "W": W,
            "b": b,
            "learning_rate": learning_rate,
            "num_iterations": num_iterations}

    return dict


trainPath='data/train.data'
testPath='data/test.data'

