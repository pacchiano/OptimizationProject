__author__ = 'nileshtrip'
import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def Nesterov_ODE(X_0, t=300.0, nabla_f=None):
    # We order such \dot{X} is first and \dot{Y} is second

    dim = X_0.shape[0]
    S_0 = np.hstack((X_0,  np.zeros(dim)))

    def dS_dt(S, t, r=3.0):

        X = S[:dim]
        Y = S[dim:]

        grad = nabla_f(X)

        dS = np.zeros(2*dim)

        dS[:dim] = Y #X update
        dS[dim:] = -r/t*Y-grad #Y update

        return dS

    t_min = 0.0001
    t_max = t_min + t
    num_pts = 10000
    t = np.linspace(t_min, t_max, num_pts)
    Ss = odeint(dS_dt, S_0, t)

    return t, Ss

def Nesterov_GD(X_0, s=0.001, t=300.0, nabla_f=None):

    k = int(t/np.power(s, .5)) # t=k \sqrt{s}
    print k
    dim = X_0.shape[0]

    X = np.zeros((k, dim))
    Y = np.zeros((k, dim))

    Y_0 = X_0
    X[0, :] = X_0
    Y[0, :] = Y_0

    for k in xrange(1, k):
        X[k, :] = Y[k-1, :] - s*nabla_f(Y[k-1, :])
        Y[k, :] = X[k, :] + (k-1)/(k+2)*(X[k, :] - X[k-1, :])

    return X, Y

def plot_traj(Xs, show=False, save=False, path=None):

    L = Xs[:,0]
    R = Xs[:,1]

    fig = plt.figure(figsize=(5,5))
    plt.plot(L, R, linestyle="solid", linewidth = 1.0, color="red")
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
    if save:
        plt.savefig(path, format="eps")
    if show:
        plt.show()

def plot_errors(t, Xs, show=False, save=False, path=None):

    fig = plt.figure(figsize=(5,5))
    errors = np.log(map(f_quad, Xs))
    plt.plot(t, errors, linestyle="solid", linewidth = 1.0, color="black")
    plt.xlabel("$t$")
    plt.ylabel("Log Error")
    if save:
        plt.savefig(path, format="eps")
    if show:
        plt.show()

def f_quad(X):

    return .02*np.power(X[0],2) + .005*np.power(X[1],2)

def grad_f_quad(X):

    return np.array([.04*X[0], .01*X[1]])

def run_ODE():

    X_0 = np.array([1.0, 1.0])
    dim = X_0.shape[0]
    t, Ss = Nesterov_ODE(X_0, nabla_f=grad_f_quad)
    Xs = Ss[:,:dim]
    plot_errors(t, Xs, show=True, save=True, path="./quadratic_errors.eps")
    plot_traj(Xs, show=True, save=True, path="./quadratic_traj.eps")

def run_GD():

    X_0 = np.array([1.0, 1.0])
    dim = X_0.shape[0]
    X, Y = Nesterov_GD(X_0, s=10.0, t=3000.0, nabla_f=grad_f_quad)
    #plot_errors(t, Y, show=True, save=False, path="./quadratic_errors.eps")
    plot_traj(Y, show=True, save=False, path="./quadratic_traj.eps")

def main():

    run_GD()

if __name__ == "__main__":
    main()
