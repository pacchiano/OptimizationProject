__author__ = 'nileshtrip'
import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def Nesterov_ODE(X_0, nabla_f=None):
    # We order such \dot{X} is first and \dot{Y} is second

    dim = X_0.shape[0]
    S_0 = np.hstack((X_0,  np.zeros(dim)))

    def dS_dt(S, t, r=1.0):

        X = S[:dim]
        Y = S[dim:]

        grad = nabla_f(X)

        dS = np.zeros(2*dim)

        dS[:dim] = Y #X update
        dS[dim:] = -r/t*Y-grad #Y update

        return dS

    t_min = 0.00001
    t_max = 300
    num_pts = 10000
    t = np.linspace(t_min, t_max, num_pts)
    Ss = odeint(dS_dt, S_0, t)

    return t, Ss

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

def main():

    X_0 = np.array([1.0, 1.0])
    dim = X_0.shape[0]
    t, Ss = Nesterov_ODE(X_0, nabla_f=grad_f_quad)

    Xs = Ss[:,:dim]
    plot_errors(t, Xs, show=True, save=True, path="./quadratic_errors.eps")
    plot_traj(Xs, show=True, save=True, path="./quadratic_traj.eps")
    #plt.plot(t, errors)
    #plt.show()

    #plot(t, Ss)

if __name__ == "__main__":
    main()
