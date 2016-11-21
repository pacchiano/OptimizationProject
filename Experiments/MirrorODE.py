import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#### GLOBALS ####
rv_a = np.random.normal(size=1000)
rv_b = np.random.normal(size=100)
rv_q = np.random.normal(size=100)
A = np.reshape(rv_a,(100,10)) # I x n matrix of iid normal rv 
b = np.reshape(rv_b, (100,1))# I length matrix of iid normal rv
Q = np.reshape(rv_q, (10,10)) # Random matrix
xstar = np.array([5,5,5,5,5,5,5,5,5,5])
#################

# Takes in z_0 a
def mirror_ode(z_0, nabla_f, t = 300.0):
    assert z_0.shape[1] == 1
    def dzdt(z,t):
        return -nabla_f(grad_potential(z))

    # Define time steps
    t_min = 0.0001
    t_max = t_min + t
    num_pts = 10000
    ts = np.linspace(t_min, t_max, num_pts)
    zs = odeint(dzdt, np.ravel(z_0), ts)

    xs = grad_potential(zs)

    return ts, xs, zs


# def f_quadratic(X):
#     assert X.shape[1] == 1
#     return (x - xstar).dot(Q.dot(x - xstar))

# def f_logsumexp(X):
#     assert X.shape[1] == 1
#     return np.log(np.sum(np.exp(A.dot(X) + b)))

def gradf_quadratic(X):
    X = X.reshape((len(X),1))
    xstar_vec = xstar.reshape((len(xstar),1))
    return np.ravel(2*Q.dot(X) - 2*Q.dot(xstar_vec))

def gradf_logsumexp(X):
    X = X.reshape((len(X),1))
    first_scalar = 1./np.sum(np.exp(A.dot(X) + b))
    second_term = A.T.dot(np.exp(A.dot(X) + b))
    return np.ravel(first_scalar * second_term)


def grad_potential(z):
    # Softmax function is \nabla \psi^*(z)
    e = np.exp(z - np.max(z))  # prevent overflow
    return e / np.sum(e)


def run_mirror_ode():
    z_0 = np.ones((10,1)) # Dummy variable for now
    ts, xs, zs  = mirror_ode(z_0, gradf_quadratic)
    print xs

run_mirror_ode()

