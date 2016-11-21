import numpy as np
import math

def f_quad(X):

    return .02*np.power(X[0],2) + .005*np.power(X[1],2)

def grad_f_quad(X):

    return np.array([.04*X[0], .01*X[1]])
