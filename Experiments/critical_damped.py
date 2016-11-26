import numpy as np
import math
from math import exp, sqrt, pow
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show, legend, xlabel, ylabel, errorbar, xlim, ylim, savefig, fill, fill_between

# We specify initial conditions as x(0) = x_0 and \dot{x}_0 = 0.

def overD(t, x0=5, z=2):

    rt = sqrt(pow(z,2)-1)
    c2 = x0/(1+(z+rt)/(rt-z))
    c1 = x0 - c2

    return np.exp(-z*t)*(c1*np.exp(t*rt)+c2*np.exp(-t*rt))

def criticalD(t, x0=5,  z=1):

    c1 = x0
    c2 = z*c1

    return np.exp(-z*t)*(c1+c2*t)

def underD(t, x0=5, z=0.5):

    rt = sqrt(1-pow(z, 2))
    c1 = x0
    c2 = z*c1/(rt)

    return np.exp(-z*t)*(c1*np.cos(rt*t)+c2*np.sin(rt*t))

def plot_1D_traj(x0, titles=None, show=False, save=False, path=None):

    t = np.arange(0.0, 10.0, 0.01)
    over = overD(t, x0=x0, z=3)
    critical = criticalD(t, x0=x0, z=1)
    under = underD(t, x0=x0, z=0.3)
    zero = 0*t

    fig = plt.figure(figsize=(5,5))
    plt.plot(t, over, linestyle="solid", linewidth = 1.0, color="red")
    plt.plot(t, critical, linestyle="solid", linewidth = 1.0, color="blue")
    plt.plot(t, under, linestyle="solid", linewidth = 1.0, color="green")
    plt.plot(t, zero, linestyle="dashed", linewidth = 2.0, color="black")
    plt.xlabel("$t$", fontsize=16)
    plt.ylabel("$X(t)$", fontsize=16)
    legend(titles, loc='lower right')
    if save:
        plt.savefig(path, format="eps", bbox_inches='tight')
    if show:
        plt.show()


def main():
    titles = ["over", "critical", "under"]
    plot_1D_traj(x0=3, titles=titles, show=True, save=False, path=None)

if __name__=="__main__":
    main()
