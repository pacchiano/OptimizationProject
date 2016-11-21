__author__ = 'nileshtrip'
import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from target_funcs import f_quad, grad_f_quad
from utils import plot_mult_traj, plot_errors
from CandesODE import Nesterov_ODE, Nesterov_GD

def run():

    X_i = np.array([1.0, 1.0])
    dim = X_i.shape[0]
    t1, Ss = Nesterov_ODE(X_i, r=1.0, nabla_f=grad_f_quad)
    X1 = Ss[:,:dim]

    t2, X2, Y2 = Nesterov_GD(X_i, s=2.0, t=300.0, r=1.0, nabla_f=grad_f_quad)
    t3, X3, Y3 = Nesterov_GD(X_i, s=0.25, t=300.0, r=1.0, nabla_f=grad_f_quad)
    titles = ["ODE", "s=2.0", "s=0.25"]
    plot_mult_traj(X1, X2, X3, titles, show=True, save=True, path="./quadratic_traj_compare_annealed_r1.eps")
    plot_errors(t1, X1, t2, X2, t3, X3, titles, show=True, save=True, path="./quadratic_errors_compare_annealed_r1.eps")


def main():

    run()

if __name__ == "__main__":
    main()
