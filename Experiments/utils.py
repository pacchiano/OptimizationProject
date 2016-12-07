import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show, legend, xlabel, ylabel, errorbar, xlim, ylim, savefig, fill, fill_between

from target_funcs import f_quad, grad_f_quad

def plot_mult_traj(X, Y, Z, titles=None, show=False, save=False, path=None):

    L1 = X[:,0]
    R1 = X[:,1]

    L2 = Y[:,0]
    R2 = Y[:,1]

    L3 = Z[:,0]
    R3 = Z[:,1]

    fig = plt.figure(figsize=(5,5))
    plt.plot(L1, R1, linestyle="solid", linewidth = 1.0, color="red")
    plt.scatter(L2, R2, marker='+', color="blue", s=20.0)
    plt.scatter(L3, R3, marker='+', color="green", s=20.0)
    plt.xlabel("$X_1$", fontsize=16)
    plt.ylabel("$X_2$", fontsize=16)
    legend(titles, loc='lower right')
    if save:
        plt.savefig(path, format="eps", bbox_inches='tight')
    if show:
        plt.show()

def plot_errors(t1, X, t2, Y, t3, Z, titles=None, show=False, save=False, path=None):

    fig = plt.figure(figsize=(5,5))
    error_X = np.log(map(f_quad, X))
    error_Y = np.log(map(f_quad, Y))
    error_Z = np.log(map(f_quad, Z))

    plt.plot(t1, error_X, linestyle="solid", linewidth = 1.0, color="red")

    plt.scatter(t2, error_Y, marker='+', color="blue", s=20.0)

    plt.scatter(t3, error_Z , marker='+', color="green", s=20.0)
    plt.xlabel("$t$", fontsize=24)
    plt.ylabel("Log Error", fontsize=24)
    legend(titles, loc='upper right')
    if save:
        plt.savefig(path, format="eps", bbox_inches='tight')
    if show:
        plt.show()


def plot_errors_raw(t1, X, t2, Y, t3, Z, titles=None, show=False, save=False, path=None, target_function = None):


    fig = plt.figure(figsize=(5,5))

    min_func_value =min( np.min(map(target_function, X)[-1]), np.min(map(target_function, Y)[-1]), np.min(map(target_function, Z)[-1])) - np.exp(-10)

    pre_error_X = map(target_function, X) - min_func_value

    for i in range(len(pre_error_X)):
        print pre_error_X[i]

    error_X = np.log(map(target_function, X)-min_func_value)
    print "error X", sum(error_X) 
    error_Y = np.log(map(target_function, Y)-min_func_value)
    print "error Y", sum(error_Y)
    error_Z = np.log(map(target_function, Z)-min_func_value)
    print "error Z", sum(error_Z)

    plt.plot(t1, error_X, linestyle="solid", linewidth = 1.0, color="red")

    #plt.scatter(t2, error_Y, marker='+', color="blue", s=20.0)
    plt.plot(t2, error_Y, linestyle = "solid", linewidth = 1.0, color = "blue")

    print t1.shape
    print  t2.shape
    print t3.shape

    print error_X.shape
    print error_Y.shape
    print error_Z.shape

    print error_Z

    #plt.scatter(t3, error_Z , marker='+', color="green", s=20.0)
    plt.plot(t3, error_Z, linestyle = "solid", linewidth = 1.0, color = "green")
    plt.xlabel("$t$", fontsize=24)
    plt.ylabel("Log Error", fontsize=24)
    legend(titles, loc='upper right')
    if save:
        plt.savefig(path, format="eps", bbox_inches='tight')
    if show:
        plt.show()

