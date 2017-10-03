from sympy import *


def phi(xi, yi, li):

    return xi**2+yi**2-li**2

m = Symbol('m')
g = Symbol('g')
t = Symbol('t')
l = Symbol('l')
x = Function('x')(t)
y = Function('y')(t)

phi_r = Matrix([phi(x, y, l).diff(x), phi(x, y, l).diff(y)])
M = Matrix([[m, 0], [0, m]])
F = Matrix([0, -m*g])
A = Matrix([[M, phi_r], [phi_r.T, 0]])
preview(A)






