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


def n_sys_eq(ai, bi, ri, ri_t, r0i, r0i_t):
    for i in range(len(ri)):
        ai = ai.replace(ri_t[i], r0i_t[i])
        ai = ai.replace(ri[i], r0i[i])
        bi = bi.replace(ri_t[i], r0i_t[i])
        bi = bi.replace(ri[i], r0i[i])
    return N(ai), N(bi)


def st_space(ri, ri_t):
    w = zeros(2*len(ri), 1)
    for i in range(len(w)):
        if i % 2 is 0:
            w[i] = ri[int(i/2)]
        else:
            w[i] = ri_t[int((i-1)/2)]
    return w


def matrix2sparse(m):
    """Converts SymPy's matrix to a NumPy array."""
    a = empty(m.shape, dtype=float)
    for i in range(m.rows):
        for j in range(m.cols):
            a[i, j] = m[i, j]
    return sparse.csr_matrix(a)


m = 1
g = 9.81
d = 1
ic = Matrix([sqrt(2)/2, sqrt(2)/2])
ic_t = Matrix([0, 0])
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
Z = zeros(phi_r.shape[1])
A = M.row_join(phi_r).col_join(phi_r.T.row_join(Z))

C, Nb = cm_rhs(A, b, unknowns)
Q = F.col_join(Nb)

N_C, N_Q = n_sys_eq(C, Q, r, r_t, ic, ic_t)

C_np = matrix2sparse(N_C)
Q_np = matrix2sparse(N_Q)

x = dsolve.spsolve(C_np, Q_np, use_umfpack=False)

pprint(x)
