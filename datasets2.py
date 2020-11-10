import numpy as np
def load_nonlinear_example1():
    X = np.array([[1,0.0],[1,2.0],[1,3.9],[1,4.0]])
    Y = np.array([7,10,11,14])
    return X,Y
    
def polynomial2_features(input):
    poly2 = input[:,1:]**2
    return np.c_[input, poly2]