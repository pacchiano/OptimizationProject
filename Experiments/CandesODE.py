__author__ = 'nileshtrip'
import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from target_funcs import f_quad, grad_f_quad, grad_least_squares, step_least_squares, lasso_fat_design, lasso_fat_design_gradient, grad_logistic_regression, logistic_regression, least_squares
from utils import plot_mult_traj, plot_errors, plot_errors_raw


def Nesterov_ODE(X_0, t=300.0, r=3.0, nabla_f=None):
    # We order such \dot{X} is first and \dot{Y} is second

    dim = X_0.shape[0]
    S_0 = np.hstack((X_0,  np.zeros(dim)))

    def dS_dt(S, t, r=r):

        X = S[:dim]
        Y = S[dim:]

        grad = nabla_f(X)

        dS = np.zeros(2*dim)

        dS[:dim] = Y #X update
        dS[dim:] = -r/t*Y-grad #Y update

        return dS

    epsilon = 0.0001
    t_min = epsilon
    t_max = t_min + t
    num_pts = 10000
    t = np.linspace(t_min, t_max, num_pts)
    Ss = odeint(dS_dt, S_0, t)

    return t, Ss

def Nesterov_GD(X_0, s=0.001, t=300.0, r=3.0, nabla_f=None):

    K = int(t/np.power(s, .5)) # t=k \sqrt{s}
    dim = X_0.shape[0]

    X = np.zeros((K, dim))
    Y = np.zeros((K, dim))

    Y_0 = X_0
    X[0, :] = X_0
    Y[0, :] = Y_0

    for k in xrange(1, K):
        X[k, :] = Y[k-1, :] - 1.0*s*nabla_f(Y[k-1, :])
        Y[k, :] = X[k, :] + 1.0*(k-1)/(k+r-1)*(X[k, :] - X[k-1, :])

    t = np.array(range(1, K+1))*math.pow(s,.5)
    return t, X, Y

def run_ODE_and_GD():

    X_i = np.array([1.0, 1.0])
    dim = X_i.shape[0]
    t1, Ss = Nesterov_ODE(X_i, r=3.0, nabla_f=grad_f_quad)
    X1 = Ss[:,:dim]

    t2, X2, Y2 = Nesterov_GD(X_i, s=2.0, t=300.0, nabla_f=grad_f_quad)
    t3, X3, Y3 = Nesterov_GD(X_i, s=0.25, t=300.0, nabla_f=grad_f_quad)
    titles = ["ODE", "s=2.0", "s=0.25"]
    plot_mult_traj(X1, X2, X3, titles, show=True, save=True, path="./quadratic_traj_compare_annealed.eps")
    plot_errors(t1, X1, t2, X2, t3, X3, titles, show=True, save=True, path="./quadratic_errors_compare_annealed.eps")

def run_generalizedNesterov():
    time = 10
    time = 5

    #target_nabla = grad_least_squares
    #target_nabla = lasso_fat_design_gradient
    target_nabla = grad_logistic_regression

    #target_function = least_squares
    #target_function = lasso_fat_design
    target_function = logistic_regression

    #plot_name = "least_squares_errors_compare"
    #plot_name = "lasso_fat_design_errors_compare"
    plot_name = "logistic_regression_errors_compare"

    if target_nabla == lasso_fat_design_gradient or target_nabla == grad_least_squares:
        X_i = np.zeros((500)) ## for lasso_fat_design

    else:
        X_i = np.zeros((100))
    dim = X_i.shape[0]   
    t1, X1, Y1 = Nesterov_GD(X_i, s=step_least_squares, t=time, r = 3.0, nabla_f=target_nabla)
    print "First series done"
    t2, X2, Y2 = Nesterov_GD(X_i, s=step_least_squares, t=time, r = 4.0, nabla_f=target_nabla)
    print "Second series done"
    t3, X3, Y3 = Nesterov_GD(X_i, s=step_least_squares, t=time, r = 5.0, nabla_f=target_nabla)
    print "Third series done"
    titles = ["r=3.0", "r=4.0", "r=5.0"]
    #plot_errors_raw( t1, X1, t2, X2, t3, X3, titles, show=True, save=True, path="./least_squares_errors_compare_annealed.eps")
    plot_errors_raw( t1, X1, t2, X2, t3, X3, titles, show=True, save=True, path="./new_plots/" + plot_name + ".eps", target_function = target_function)



def main():

    run_ODE_and_GD()

if __name__ == "__main__":
    main()
