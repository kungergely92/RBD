import sympy as sym
import math
import numpy as np
from numpy.linalg import inv

pi = math.pi
Tr = np.trace


def jsa_cylinder(radius, height, density):
    """The function takes the cylinder's radius, length, and density
    and calculates the mass, and moment of inertia."""
    mass = radius*radius*pi*height*density
    thetax = 0.25 * mass * radius * radius + \
        (1 / 12) * mass * height**2
    thetaz = 0.5 * mass * radius * radius

    js = np.array([[thetax, 0, 0], [0, thetax, 0], [0, 0, thetaz]])

    ksi = 0.5 * Tr(js) - js[0, 0]
    eta = 0.5 * Tr(js) - js[1, 1]
    zeta = 0.5 * Tr(js) - js[2, 2]

    return mass, np.array([[ksi, 0, 0], [0, eta, 0], [0, 0, zeta]])


def flatten(hypermatrix):
    """Removes the brackets from matrix of matrices."""
    rows, cols = hypermatrix.shape[0:2]
    for i in range(rows):
        for j in range(0, cols):
            row = np.append(row, hypermatrix[i, j], 1) if j > 0 else hypermatrix[i, 0]
        hm = np.append(hm, row, 0) if i > 0 else row
    return hm


class RigidBody(object):
    """RigidBody object with its mass matrix, symbolic state variables,
    and constraints."""
    def __init__(self, radius=1, height=1, density=1):
        """Initialize radius, length and density of a
        cylinder(under construction)"""
        RigidBody.counter += 1
        self.ID = RigidBody.counter
        self.density = density
        self.radius = radius
        self.length = height
        self.mass_matrix = None     # Initializing inside __init__ method
        self.position = {}
        self.r_i_0 = np.array([0, 0, 0])
        self.r_j_0 = np.array([0, 0, height])
        self.dr_i_0 = np.array([0, 0, 0])
        self.dr_j_0 = np.array([0, 0, 0])
        self.u = np.array([1, 0, 0])
        self.v = np.array([0, 1, 0])
        self.du = np.array([0, 0, 0])
        self.dv = np.array([0, 0, 0])
        self.r_i_loc = np.array([0, 0, 0])
        self.r_j_loc = np.array([0, 0, height])
        self.r_g_loc = np.array([0, 0, 0.5 * height])
        self.calculate_mass_matrix(radius, height, density)

    def calculate_mass_matrix(self, r, l, rho):
        """Calculates the 12x12 mass matrix of the rigid body object."""
        m, jsa = jsa_cylinder(r, l, rho)

        rho_i = self.r_i_loc - self.r_g_loc
        tr_rho_i = rho_i.transpose()
        x = np.array([self.r_j_loc - self.r_i_loc, self.u, self.v])
        a = np.matmul(inv(x), self.r_g_loc - self.r_i_loc)
        j_i = jsa + m * np.outer(rho_i, tr_rho_i)
        z = inv(x).dot(j_i).dot(inv(x.transpose()))

        m_11 = np.array((m - 2 * m * a[0] + z[0, 0]) * np.eye(3))
        m_12 = np.array((m * a[0] - z[0, 0]) * np.eye(3))
        m_13 = np.array((m * a[1] - z[0, 1]) * np.eye(3))
        m_14 = np.array((m * a[2] - z[0, 2]) * np.eye(3))

        m_22 = np.array((z[0, 0]) * np.eye(3))
        m_23 = np.array((z[0, 1]) * np.eye(3))
        m_24 = np.array((z[0, 2]) * np.eye(3))
        m_33 = np.array((z[1, 1]) * np.eye(3))
        m_34 = np.array((z[1, 2]) * np.eye(3))
        m_44 = np.array((z[2, 2]) * np.eye(3))

        mass_hm = np.array([[m_11, m_12, m_13, m_14],
                            [m_12, m_22, m_23, m_24],
                            [m_13, m_23, m_33, m_34],
                            [m_14, m_24, m_34, m_44]])

        mass_matrix = flatten(mass_hm)

        self.mass_matrix = mass_matrix

    def state_variables(self):

        t = sym.Symbol('t')

        u_x = sym.Function('u_{}_x'.format(self.ID))(t)
        u_y = sym.Function('u_{}_y'.format(self.ID))(t)
        u_z = sym.Function('u_{}_z'.format(self.ID))(t)

        v_x = sym.Function('v_{}_x'.format(self.ID))(t)
        v_y = sym.Function('v_{}_y'.format(self.ID))(t)
        v_z = sym.Function('v_{}_z'.format(self.ID))(t)

        r_i_x = sym.Function('r_i_{}_x'.format(self.ID))(t)
        r_i_y = sym.Function('r_i_{}_y'.format(self.ID))(t)
        r_i_z = sym.Function('r_i_{}_z'.format(self.ID))(t)

        r_j_x = sym.Function('r_j_{}_x'.format(self.ID))(t)
        r_j_y = sym.Function('r_j_{}_y'.format(self.ID))(t)
        r_j_z = sym.Function('r_j_{}_z'.format(self.ID))(t)

        position = {r_i_x: self.r_i[0], r_i_y: self.r_i[1],
                    r_i_z: self.r_i[2],
                    r_j_x: self.r_j[0], r_j_y: self.r_j[1],
                    r_j_z: self.r_j[2],
                    u_x: self.u[0], u_y: self.u[1], u_z: self.u[2],
                    v_x: self.v[0], v_y: self.v[1], v_z: self.v[2]}

        self.position = position





