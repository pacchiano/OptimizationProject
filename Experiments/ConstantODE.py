__author__ = 'nileshtrip'
import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from target_funcs import f_quad, grad_f_quad
from utils import plot_mult_traj, plot_traj, plot_errors

def Constant_ODE(X_0, t=500.0, nabla_f=None, rt_kappa=3.0):
    # We order such \dot{X} is first and \dot{Y} is second

    dim = X_0.shape[0]

    def dX_dt(X, t, rt_kappa=rt_kappa):

        grad = 1.0*(1+rt_kappa)/2*nabla_f(X) #setting \root{\kappa} = 3

        dX = -grad #X update

        return dX

    epsilon = 0.0001
    t_min = epsilon
    t_max = t_min + t
    num_pts = 10000
    t = np.linspace(t_min, t_max, num_pts)
    Xs = odeint(dX_dt, X_0, t)

    return t, Xs

def Nesterov_GD(X_0, s=2.0, t=500.0, nabla_f=None, rt_kappa=3.0):

    K = int((1.0*t)/s) # t = k s
    dim = X_0.shape[0]

    X = np.zeros((K, dim))
    Y = np.zeros((K, dim))

    Y_0 = X_0
    X[0, :] = X_0
    Y[0, :] = Y_0

    for k in xrange(1, K):
        X[k, :] = Y[k-1, :] - s*nabla_f(Y[k-1, :])
        Y[k, :] = X[k, :] + 1.0*(rt_kappa-1)/(rt_kappa+1)*(X[k, :] - X[k-1, :])

    t = np.array(range(1, K+1))
    return t, X, Y

def f_quad(X):

    return .02*np.power(X[0],2) + .005*np.power(X[1],2)

def grad_f_quad(X):

    return np.array([.04*X[0], .01*X[1]])

def run_ODE_and_GD():

    T=500.0
    rt_kappa=10.0

    X_i = np.array([1.0, 1.0])
    dim = X_i.shape[0]
    t1, Ss = Constant_ODE(X_i, t=T, rt_kappa=rt_kappa, nabla_f=grad_f_quad)
    X1 = Ss[:,:dim]

    t2, X2, Y2 = Nesterov_GD(X_i, s=1.0, t=T, rt_kappa=rt_kappa, nabla_f=grad_f_quad)
    t3, X3, Y3 = Nesterov_GD(X_i, s=0.2, t=T, rt_kappa=rt_kappa, nabla_f=grad_f_quad)
    titles = ["ODE", "s=1.0", "s=0.2"]
    plot_mult_traj(X1, X2, X3, titles, show=True, save=True, path="./quadratic_traj_compare_constant.eps")
    plot_errors(t1, X1, t2, X2, t3, X3, titles, show=True, save=True, path="./quadratic_errors_compare_constant.eps")

def main():

    run_ODE_and_GD()


if __name__ == "__main__":
    main()
