from sympy import *
from scipy import sparse
from numpy import empty
from scipy.sparse.linalg import dsolve

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
import matplotlib.animation as animation
import time


def phi(xi, yi, di):

    return Matrix([xi**2+yi**2-di**2])


def cm_rhs(ci, bi, xi):

    """Coefficient Matrix Right Hand Side: The function takes the coefficient
    matrix of the Lagrangian equations of motion of first kind, the second
    time derivative of the geometrical constraints, and the vectors of the
    unknowns. It provides the coefficient matrix of the linear algebraic
    equation system, and the right hand side of the equation."""

    z = zeros(ci.shape[0], ci.shape[1])
    n_b = zeros(bi.shape[0], bi.shape[1])
    s = 0
    for i, bi_val in enumerate(bi):
        for j, xi_val in enumerate(xi):
            z[-bi.shape[0]+i, j] = bi_val.coeff(xi_val)
            s += bi_val.coeff(xi_val)*xi_val
        n_b[i] = bi_val - s
        s = 0

    return ci - z, n_b


def n_sys_eq(ai, bi, ri, ri_t, r0i, r0i_t):

    """The function replaces the symbolic variables with their numerical
    values and returns with sparse matrices."""

    for j in range(len(ri)):
        ai = ai.subs(ri_t[j], r0i_t[j])
        ai = ai.subs(ri[j], r0i[j])
        bi = bi.subs(ri_t[j], r0i_t[j])
        bi = bi.subs(ri[j], r0i[j])
    return matrix2sparse(N(ai)), matrix2sparse(N(bi))


def st_space(ri, ri_t):

    """The function interweaves the position and velocity vectors into a
    state space vector"""

    w = zeros(2*len(ri), 1)
    for i in range(len(w)):
        if i % 2 is 0:
            w[i] = ri[int(i/2)]
        else:
            w[i] = ri_t[int((i-1)/2)]
    return w


def cauchy_form(ai, bi, ri_t):

    """The function rewrites the differential equation system into its
    Cauchy-form"""

    eye_m = eye(len(ri_t))
    z_m = zeros(ai.shape[0], len(ri_t))
    coeff_m = eye_m.row_join(zeros(len(ri_t), ai.shape[1]))
    a_c = coeff_m.col_join(z_m.row_join(ai))
    b_c = ri_t.col_join(bi)
    return a_c, b_c


def sys_rk4(ai, qi, r, r_t, ic, ic_t, h):

    nai, nqi = n_sys_eq(ai, qi, r, r_t, ic, ic_t)

    len_ict = len(ic_t)
    len_ic = len(ic)
    xi_1 = dsolve.spsolve(nai, nqi, use_umfpack=False)
    lbd = xi_1[len_ic+len_ict:]
    k_1 = h*xi_1

    ictk_1 = ic_t + 0.5 * k_1[len_ict:len_ic + len_ict]
    ick_1 = ic + 0.5*k_1[0:len_ict]
    nai_2, nqi_2 = n_sys_eq(ai, qi, r, r_t, ick_1, ictk_1)
    xi_2 = dsolve.spsolve(nai_2, nqi_2, use_umfpack=False)
    k_2 = h*xi_2

    ictk_2 = ic_t + 0.5 * k_2[len_ict:len_ic + len_ict]
    ick_2 = ic + 0.5 * k_2[0:len_ict]
    nai_3, nqi_3 = n_sys_eq(ai, qi, r, r_t, ick_2, ictk_2)
    xi_3 = dsolve.spsolve(nai_3, nqi_3, use_umfpack=False)
    k_3 = h*xi_3

    ictk_3 = ic_t + k_3[len_ict:len_ic + len_ict]
    ick_3 = ic + k_3[0:len_ict]
    nai_4, nqi_4 = n_sys_eq(ai, qi, r, r_t, ick_3, ictk_3)
    xi_4 = dsolve.spsolve(nai_4, nqi_4, use_umfpack=False)
    k_4 = h*xi_4

    y_sol = ic + (k_1[0:len_ict] + 2*(k_2[0:len_ict] + k_3[0:len_ict]) +
                  k_4[0:len_ict])/6

    yt_sol = ic_t + (k_1[len_ict:len_ic + len_ict] +
                  2*(k_2[len_ict:len_ic + len_ict] +
                  k_3[len_ict:len_ic + len_ict]) +
                  k_4[len_ict:len_ic + len_ict])/6
    lbd_sol = lbd + (k_1[len_ic+len_ict:] + 2*(k_2[len_ic+len_ict:] +
                                               k_3[len_ic+len_ict:]) +
                     k_4[len_ic+len_ict:])/6
    return y_sol, yt_sol, lbd_sol


def matrix2sparse(mi):
    """Converts SymPy's matrix to a SciPy sparse matrix."""
    a = empty(mi.shape, dtype=float)
    for i in range(mi.rows):
        for j in range(mi.cols):
            a[i, j] = mi[i, j]
    return sparse.csr_matrix(a)


t = Symbol('t')
lbd = Symbol('lbd')
x = Function('x')(t)
y = Function('y')(t)
m = 1
g = -9.81
d_l = 2
alpha = 10
h = 0.01
R = 0.5
beta = 10

phi_r = Matrix([phi(x, y, d_l).diff(x), phi(x, y, d_l).diff(y)])
phi_t = Matrix([phi(x, y, d_l).diff(t)])
unknowns = Matrix([x.diff(t, t), y.diff(t, t), lbd])
r = Matrix([x, y])
r_t = r.diff(t)

M = Matrix([[m, 0], [0, m]])
F = Matrix([0, m*g])
b = Matrix([- (phi_r.diff(t)).T*r.diff(t) - phi_t.diff(t) -
            2*alpha*(phi_r.T*r.diff(t) + phi_t) -
            (beta**2)*phi(x, y, d_l)])
Z = zeros(phi_r.shape[1])
A = M.row_join(phi_r).col_join(phi_r.T.row_join(Z))

C, Nb = cm_rhs(A, b, unknowns)
Q = F.col_join(Nb)

C_c, Q_c = cauchy_form(C, Q, r_t)

simulation_time = 4
steps = int(simulation_time/h)
ic = [N(sqrt(2)/2), -N(sqrt(2)/2)]
ic_t = [0, 0]

y = [None]*int(steps)
y_t = [None]*int(steps)
t = [None]*int(steps)

y[0] = ic
y_t[0] = ic_t
t[0] = 0
start = time.clock()
y_test = sys_rk4(C_c, Q_c, r, r_t, y[0], y_t[0], h)[0]
end = time.clock()
#for i in range(steps-1):
#    y[i+1] = sys_rk4(C_c, Q_c, r, r_t, y[i], y_t[i], h)[0]
#    y_t[i+1] = sys_rk4(C_c, Q_c, r, r_t, y[i], y_t[i], h)[1]
#    t[i+1] = i*h

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, projection='3d')
plt.gca().set_aspect('equal', adjustable='box')
ax.grid()
ax.set_xlim3d(-1.5, 1.5)
ax.set_ylim3d(-1.5, 1.5)
ax.set_zlim3d(-1.5, 1.5)
ax.view_init(30,60)

line, = ax.plot([], [], [], 'o-', lw=2)

x1 = [None]*int(steps)
y1 = [None]*int(steps)
z1 = [0]*int(steps)

for i in range(len(y)):
    x1[i] = y[i][0]
    y1[i] = y[i][1]


def init():

    line.set_data([], [])
    line.set_3d_properties([])
    return line,


def animate(i):

    thisx = [0, x1[i]]
    thisy = [0, y1[i]]
    thisz = [0, z1[i]]


    line.set_data(thisx, thisz)
    line.set_3d_properties(thisy)

    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                              interval=25, blit=True, init_func=init)

plt.show()
