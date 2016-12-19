import numpy as np

def identity(x):
    """
    An identity function.
    x: a numpy array
    
    returns the same array
    """
    return x

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

sigmoid_vectorized = np.vectorize(sigmoid)


def sigmoid_derv(x):
    return sigmoid(x)*(1-sigmoid(x))

sigmoid_derv_vectorized = np.vectorize(sigmoid_derv)

def error_func(target, output):
    assert target.shape == output.shape
    return 0.5*(target - output)*(target - output)

def error_derv(target, output):
    assert target.shape == output.shape
    return output - target

if __name__ == "__main__":
    import numpy as np
    print(sigmoid(0.3775))
    print(sigmoid_derv(0.3775))
    print(sigmoid_derv_vectorized(np.array([1,2])))
