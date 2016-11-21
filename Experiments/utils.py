import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show, legend, xlabel, ylabel, errorbar, xlim, ylim, savefig, fill, fill_between

from target_funcs import f_quad, grad_f_quad

def plot_traj_instab(X, show=False, save=False, path=None):

    L = X[:,0]
    R = Xs[:,1]

    fig = plt.figure(figsize=(5,5))
    plt.plot(L, R, linestyle="solid", linewidth = 1.0, color="red")
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
    if save:
        plt.savefig(path, format="eps")
    if show:
        plt.show()

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
