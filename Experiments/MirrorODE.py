import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#### GLOBALS ####
seed = 775 #246223
np.random.seed(seed)
I = 100# num_logsumexp_terms
d = 3# dimension of x
xstar = np.random.randn(d,1)
xstar = xstar/np.sum(xstar)

t_step = 1.0/20.0/10.0


assert len(xstar) == d
D = np.random.randn(d,d)
Q = np.dot(D,D.transpose())
A = np.random.randn(100,d)
b = np.random.randn(100,1)
#################


def amd_ode(z_0, nabla_f, t = 2500.0, r = 3.0):
    z_0 = np.ravel(z_0)
    x_0 = np.ravel(grad_potential(z_0))
    dim = x_0.shape[0]
    s_0 = np.hstack((x_0,z_0))
    def dsdt(s, t):
        x = s[:dim]
        z = s[dim:]

        ds = np.zeros(len(s))

        ds[:dim] = (r/t)*(grad_potential(z) - x)
        ds[dim:] = (-t/r)*nabla_f(x)

        return ds

    epsilon = np.sqrt(1.0/20.0)/10.0 #s/10
    num_pts = t*10
    t_min = epsilon
    t_max = t_min + num_pts * epsilon
    ts = np.linspace(t_min, t_max, num_pts)
    ss = odeint(dsdt, s_0, ts)

    return ts, ss[:,:dim], ss[:,dim:]



# Takes in z_0 a
def mirror_ode(z_0, nabla_f, t = 2500.0):
    assert z_0.shape[1] == 1
    def dzdt(z,t):
        return -nabla_f(grad_potential(z))

    # Define time steps
    epsilon = np.sqrt(1.0/20.0)/10.0 #s/10
    num_pts = t*10
    t_min = epsilon
    t_max = t_min + num_pts * epsilon
    ts = np.linspace(t_min, t_max, num_pts)
    zs = odeint(dzdt, np.ravel(z_0), ts)

    # Convert to xs
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

def run_logsum_mirror_ode(z_0, horizon=2500.0):
    # Run for a long time to numerically compute optimum
    ts, xs, zs  = amd_ode(z_0, gradf_logsumexp, t=100000)
    x_star = xs[-1,:]
    fmin = f_logsumexp(x_star)
    ts, xs, zs  = amd_ode(z_0, gradf_logsumexp, t=horizon)
    fs = np.apply_along_axis(f_logsumexp, 1, xs)
    plt.plot(ts, np.log(fs - fmin), linestyle="solid", linewidth = 1.0, color="red")
    plt.xlabel("$t$", fontsize=24)
    plt.ylabel("Log Error", fontsize=24)
    plt.title("Logsumexp Mirror Descent ODE", fontsize = 24)
    plt.show()
    return ts, xs, zs, fs, x_star


def run_quad_mirror_ode(z_0, horizon=2500.0):
    ts, xs, zs  = amd_ode(z_0, gradf_quadratic, t=horizon)
    fs = np.apply_along_axis(f_quadratic, 1, xs)
    plt.plot(ts, np.log(fs), linestyle="solid", linewidth = 1.0, color="red")
    plt.xlabel("$t$", fontsize=24)
    plt.ylabel("Log Error", fontsize=24)
    plt.title("Quadratic Mirror Descent ODE", fontsize = 24)
    plt.show()
    fmin = 0
    return ts, xs, zs, fs, np.ravel(xstar)

#z0 = np.random.rand(d,1)
#x0 = grad_potential(z0)
#run_logsum_mirror_ode(z0)

