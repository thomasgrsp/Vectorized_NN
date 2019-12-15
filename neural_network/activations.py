import numpy as np

def relu(x, derivative=False):
    if derivative:
        return np.maximum(0, np.greater(X, 0))
    else:
        return np.maximum(0, X)

def sigmoid(x, derivative=False):
    x = np.clip(x, -500, 500)
    if derivative:
        return np.exp(x) * (1 - np.exp(-x))
    else:
        return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)