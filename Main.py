from sympy import *
from scipy import sparse
from numpy import empty
from scipy.sparse.linalg import dsolve
from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation


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
    values."""

    for i in range(len(ri)):
        ai = ai.subs(ri_t[i], r0i_t[i])
        ai = ai.subs(ri[i], r0i[i])
        bi = bi.subs(ri_t[i], r0i_t[i])
        bi = bi.subs(ri[i], r0i[i])
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

    x = dsolve.spsolve(nai, nqi, use_umfpack=False)
    lbd = x[len_ic+len_ict:]
    k_1 = h*x
    ick = ic + 0.5*k_1[0:len_ict]
    ictk = ic_t + 0.5*k_1[len_ict:len_ic + len_ict]
    nai, nqi = n_sys_eq(ai, qi, r, r_t, ick, ictk)

    x = dsolve.spsolve(nai, nqi, use_umfpack=False)
    k_2 = h*x
    ick = ic + 0.5 * k_2[0:len_ict]
    ictk = ic_t + 0.5 * k_2[len_ict:len_ic + len_ict]
    nai, nqi = n_sys_eq(ai, qi, r, r_t, ick, ictk)

    x = dsolve.spsolve(nai, nqi, use_umfpack=False)
    k_3 = h*x
    ick = ic + k_3[0:len(ic_t)]
    ictk = ic_t + k_3[len_ict:len_ic + len_ict]
    nai, nqi = n_sys_eq(ai, qi, r, r_t, ick, ictk)

    x = dsolve.spsolve(nai, nqi, use_umfpack=False)
    k_4 = h*x

    yt_sol = ic_t + (k_1[0:len_ict] + 2*(k_2[0:len_ict] + k_3[0:len_ict])
                     +k_4[0:len_ict])/6
    y_sol = ic + (k_1[len_ict:len_ic + len_ict] +
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


def init():
    line.set_data([], [])
    return line


def animate(i,y):
    thisx = [0, 0, y[i][0]]
    thisy = [0, 0, y[i][1]]

    line.set_data(thisx, thisy)
    return line

m = 1
g = 9.81
d = 1

r_ic = Matrix([sqrt(2)/2, sqrt(2)/2])
r_t_ic = Matrix([0, 0])

alpha = 5
h = 0.01
beta = 2
ic = [N(sqrt(2)/2), N(sqrt(2)/2)]
ic_t = [0, 0]
t = Symbol('t')
lbd = Symbol('lbd')
x = Function('x')(t)
y = Function('y')(t)

phi_r = Matrix([phi(x, y, d).diff(x), phi(x, y, d).diff(y)])
phi_t = Matrix([phi(x, y, d).diff(t)])
unknowns = Matrix([x.diff(t, t), y.diff(t, t), lbd])
r = Matrix([x, y])
r_t = r.diff(t)

M = Matrix([[m, 0], [0, m]])
F = Matrix([0, m*g])
b = Matrix([- phi_r.T*r.diff(t) - phi_t.diff(t)-2*alpha*(phi_r.T*r.diff(t) +
                                                         phi_t.diff(t)) -
            beta**2*phi(x, y, d)])
Z = zeros(phi_r.shape[1])
A = M.row_join(phi_r).col_join(phi_r.T.row_join(Z))

C, Nb = cm_rhs(A, b, unknowns)
Q = F.col_join(Nb)

C_c, Q_c = cauchy_form(C, Q, r_t)

# y = sys_rk4(C_c, Q_c, r, r_t, ic, ic_t, h)[0]

simulation_time = 10
steps = 100

y = []
y_t = []
time = []

y.append(ic)
y_t.append(ic_t)
time.append(0)

for i in range(steps-1):
    y.append(sys_rk4(C_c, Q_c, r, r_t, y[i], y_t[i], h)[0])
    y_t.append(sys_rk4(C_c, Q_c, r, r_t, y[i], y_t[i], h)[1])
    time.append(i*h)

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                              interval=25, blit=True, init_func=init)


plt.show()
