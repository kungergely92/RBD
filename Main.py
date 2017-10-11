from sympy import *
from scipy import sparse
from numpy import empty
from scipy.sparse.linalg import dsolve


def phi(xi, yi, di):

    return xi**2+yi**2-di**2


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


def cauchy_form(ai, bi, ri_t):
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
                     +k_4[0:len_ict])
    y_sol = ic + (k_1[len_ict:len_ic + len_ict] +
                  2*(k_2[len_ict:len_ic + len_ict] +
                  k_3[len_ict:len_ic + len_ict]) +
                  k_4[len_ict:len_ic + len_ict])
    lbd_sol = lbd + (k_1[len_ic+len_ict:] + 2*(k_2[len_ic+len_ict:] +
                                               k_3[len_ic+len_ict:])+
                     k_4[len_ic+len_ict:])
    return y_sol, yt_sol, lbd_sol






def n_sys_eq(ai, bi, ri, ri_t, r0i, r0i_t):
    for i in range(len(ri)):
        ai = ai.subs(ri_t[i], r0i_t[i])
        ai = ai.subs(ri[i], r0i[i])
        bi = bi.subs(ri_t[i], r0i_t[i])
        bi = bi.subs(ri[i], r0i[i])
    return matrix2sparse(N(ai)), matrix2sparse(N(bi))


def st_space(ri, ri_t):
    w = zeros(2*len(ri), 1)
    for i in range(len(w)):
        if i % 2 is 0:
            w[i] = ri[int(i/2)]
        else:
            w[i] = ri_t[int((i-1)/2)]
    return w


def matrix2sparse(m):
    """Converts SymPy's matrix to a SciPy sparse matrix."""
    a = empty(m.shape, dtype=float)
    for i in range(m.rows):
        for j in range(m.cols):
            a[i, j] = m[i, j]
    return sparse.csr_matrix(a)


m = 1
g = 9.81
d = 1
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
b = Matrix([- phi_r.T*r.diff(t) - phi_t.diff(t)])
b_baumgarte = Matrix([- phi_r.T*r.diff(t) - phi_t.diff(t) -
                      2*alpha*(phi_r.T*r.diff(t) + phi_t.diff(t)) -
                      beta*beta*Matrix([phi(x, y, d)])])
Z = zeros(phi_r.shape[1])
A = M.row_join(phi_r).col_join(phi_r.T.row_join(Z))

C, Nb = cm_rhs(A, b_baumgarte, unknowns)
Q = F.col_join(Nb)

C_c, Q_c = cauchy_form(C, Q, r_t)

y, y_t, lamb = sys_rk4(C_c, Q_c, r, r_t, ic, ic_t, h)

pprint(y)
pprint(y_t)
pprint(lamb)
