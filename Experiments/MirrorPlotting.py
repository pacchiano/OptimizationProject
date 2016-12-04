import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick
import math
import sys
sys.path.insert(0, '../AcceleratedMirrorDescent/amd/')
import AcceleratedMethods as ac
import Projections as proj
import SimulationSimplex as sim
import MirrorODE as mir

d = mir.d
xstar = mir.xstar

def f(x):
    fn = mir.f_quadratic(x) # mir.f_logsumexp(x) #
    return fn

def gradf(x):
    g = mir.gradf_quadratic(x) # mir.gradf_logsumexp(x) #
    return g.reshape(d,1)


# inital points
z0 = np.random.rand(d,1)
x0 = mir.grad_potential(z0)
horizon = 2500

# continuous time ODE
ts, xs, zs, fs, xstar  = mir.run_quad_mirror_ode(z0, horizon) # mir.run_logsum_mirror_ode(z0, horizon) #
fstar = f(xstar)
conts_time = fs[0::10]-fstar
x1x2 = xs[0::10,:]

def fMinusFStar(x):
    return f(x) - fstar

vertices = np.array([[1, 0], [np.cos(2*np.pi/3), np.sin(2*np.pi/3)], [np.cos(4*np.pi/3), np.sin(4*np.pi/3)]]).T
def toSimplex(x):
    return np.dot(vertices, x-xstar)


# Parameters, simplex constrained projections
lmax = 20
s = 1/lmax
r = 3
p1 = proj.SimplexProjectionExpSort(dimension = d, epsilon = 0.3)
p2 = proj.SimplexProjectionExpSort(dimension = d, epsilon = 0)
s1 = s*p1.epsilon/(1+d*p1.epsilon)
s2 = s

# descent methods
amd = ac.AcceleratedMethod(f, gradf, p1, p2, s1, s2, r, x0, 'accelerated descent')
amddiv = ac.AcceleratedMethod(f, gradf, p1, p2, s1, s2, r, x0, 'accelerated divergent descent', divergent=True)
md = ac.MDMethod(f, gradf, p2, s2, x0, 'mirror descent')
methods = [md,
           amd,
           amddiv,
           ]
ms = range(len(methods))

# run the descent
values = {} # objective value
xs = {} # primal variable

for m in ms:
    values[m] = np.zeros((horizon, 1))
    xs[m] = np.zeros((2, horizon))

for m in ms:
    method = methods[m]
    r = method.r if hasattr(method, 'r') else 3
    for k in range(horizon):
        values[m][k] = fMinusFStar(method.x)
        xMat = method.x
        x = np.hstack(xMat)
        xs[m][:, k] = toSimplex(x)
        method.step()

# plotting
figsize = (12, 8)
min_value = max(1e-11, min([np.nanmin(values[m]) for m in ms]))
max_value = max([np.nanmax(values[m]) for m in ms])
colors = ['b', 'g', 'r', 'c', 'm']
n1 = 30

def setAxisZoom(ax, points, combineDeltas):
    # take a box around the the last few values
    delta = {0: {1: 1},
             1: {1: 1},
             2: {1: 1},
             3: {1: 1}}
    delta = delta[0]
    delta_coeff = 1
    deltaXs = [np.max(np.abs(points[m][0,:])) for m in ms[0:]]
    deltaYs = [np.max(np.abs(points[m][1,:])) for m in ms[0:]]
    delta[1] = min(delta[1], delta_coeff*max(combineDeltas(deltaXs), combineDeltas(deltaYs)))
    lim = delta[1]

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])

fig1 = plt.figure(figsize=figsize)
ax = fig1.add_subplot(1,1,1)
ax.plot(range(0,horizon),conts_time,color='purple',label='ODE')
for m in ms:
    ax.scatter(range(0,values[m].size),values[m], marker='+', color=colors[m], label=methods[m].name)
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(loc=3)
ax.set_xlim([1, horizon])
ax.set_ylim([min_value, max_value])
ax.set_xlabel('k')
ax.set_ylabel('f(x(k))-f^*')
plt.show()

n2 = n1+50
fig = plt.figure(figsize=figsize)

gs = gridspec.GridSpec(2, 3)
# top plot contains function values in log log scale
ax = fig.add_subplot(gs[0,:])
ax.plot(range(0,horizon),conts_time,color='purple',label='ODE')
for m in ms:
    ax.scatter(range(0,values[m].size),values[m], marker='+', color=colors[m], label=methods[m].name)
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(loc=3)
ax.set_xlim([1, horizon])
ax.set_ylim([min_value, max_value])
ax.set_xlabel('k')
ax.set_ylabel('f(x(k))-f*')

# Plot 3 and 4: plot a subset of the trajectory
points = [xs[m][:, n1:n2] for m in ms]
vals = [values[m][n1:n2] for m in ms]

ax2 = fig.add_subplot(gs[1,0])
ax2.plot(x1x2[0,:], x1x2[1,:], color='purple')
for m in ms:
    ax2.scatter(xs[m][0,:], xs[m][1,:], marker='+', color=colors[m])
setAxisZoom(ax2, points, np.mean)
ax2.set_xlabel('x0(k)-x0*')
ax2.set_ylabel('x1(k)-x1*')
ax2.set_xticklabels([])

n1 = 200
n2 = n1+50
points = [xs[m][:, n1:n2] for m in ms]
ax3 = fig.add_subplot(gs[1,1])
ax3.plot(x1x2[0,:], x1x2[1,:], color='purple')
for m in ms:
    ax3.scatter(xs[m][0,:], xs[m][1,:], marker='+', color=colors[m])
    # ax.plot(zs[m][0,:], zs[m][1,:], colors[m]+'--')
setAxisZoom(ax3, points, np.mean)
ax3.set_xlabel('x0(k)-x0*')
ax3.set_ylabel('x1(k)-x1*')
ax3.set_xticklabels([])

n1 = 300
n2 = n1+50
points = [xs[m][:, n1:n2] for m in ms]
ax4 = fig.add_subplot(gs[1,2])
ax4.plot(x1x2[0,:], x1x2[1,:], color='purple')
for m in ms:
    ax4.scatter(xs[m][0,:], xs[m][1,:], marker='+', color=colors[m])
    # ax.plot(zs[m][0,:], zs[m][1,:], colors[m]+'--')
setAxisZoom(ax4, points, np.mean)
ax4.set_xlabel('x0(k)-x0*')
ax4.set_ylabel('x1(k)-x1*')
ax4.set_xticklabels([])
gs.update(wspace=0.5, hspace=0.25)

plt.show()



