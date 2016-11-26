import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#### GLOBALS ####
I = 100# num_logsumexp_terms
n = 2# dimension of x
xstar = np.array([0.75,0.25])
assert len(xstar) == n
rv_a = np.random.normal(size=I*n)
rv_b = np.random.normal(size=I)
rv_q = np.random.normal(size=n*n)
A = np.reshape(rv_a,(I,n)) # I x n matrix of iid normal rv 
b = np.reshape(rv_b, (I,1))# I length matrix of iid normal rv
Q = np.reshape(rv_q, (n,n)) # Random matrix
Q = Q.T.dot(Q)
#################


def amd_ode(z_0, nabla_f, t = 300.0, r = 3.0):
    z_0 = np.ravel(z_0)
    x_0 = np.ravel(grad_potential(z_0))
    dim = x_0.shape[0]
    s_0 = np.hstack((x_0,z_0))
    print x_0.shape
    print z_0.shape
    print s_0.shape
    def dsdt(s, t):
        x = s[:dim]
        z = s[dim:]

        ds = np.zeros(len(s))

        ds[:dim] = (r/t)*(grad_potential(z) - x)
        ds[dim:] = (-t/r)*nabla_f(x)

        return ds

    epsilon = 0.0001
    t_min = epsilon
    t_max = t_min + t
    num_pts = 10000
    ts = np.linspace(t_min, t_max, num_pts)
    ss = odeint(dsdt, s_0, ts)

    return ts, ss[:,:dim], ss[:,dim:]



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

    # Convert to xs
    print zs.shape
    xs = np.apply_along_axis(grad_potential, 1 ,zs)

    return ts, xs, zs


def f_quadratic(X):
    X = X.reshape((len(X),1))
    xstar_vec = xstar.reshape((len(xstar),1))
    return np.ravel((X - xstar_vec).T.dot(Q.dot(X - xstar_vec)))

def f_logsumexp(X):
    X = X.reshape((len(X),1))
    return np.log(np.sum(np.exp(A.dot(X) + b)))

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

def run_logsum_mirror_ode():
    z_0 = np.ones((n,1)) # Dummy variable for now

    # Run for a long time to numerically compute optimum
    ts, xs, zs  = amd_ode(z_0, gradf_logsumexp, t=100000)
    x_max = xs[-1,:]

    ts, xs, zs  = amd_ode(z_0, gradf_logsumexp)
    fs = np.apply_along_axis(f_logsumexp, 1, xs)
    plt.plot(ts, np.log(fs - f_logsumexp(x_max)), linestyle="solid", linewidth = 1.0, color="red")
    plt.xlabel("$t$", fontsize=24)
    plt.ylabel("Log Error", fontsize=24)
    plt.title("Logsumexp Mirror Descent ODE", fontsize = 24)
    plt.show()


def run_quad_mirror_ode():
    z_0 = np.ones((n,1)) # Dummy variable for now
    ts, xs, zs  = amd_ode(z_0, gradf_quadratic)
    fs = np.apply_along_axis(f_quadratic, 1, xs)
    plt.plot(ts, np.log(fs - f_quadratic(xstar)), linestyle="solid", linewidth = 1.0, color="red")
    plt.xlabel("$t$", fontsize=24)
    plt.ylabel("Log Error", fontsize=24)
    plt.title("Quadratic Mirror Descent ODE", fontsize = 24)
    plt.show()

run_logsum_mirror_ode()

