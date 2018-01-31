import sympy as sym
from sympy import Matrix
import moment_of_inertia
import utilities
import numpy as np
from numpy.linalg import inv
from mechanism import Mechanism
from utilities import symbolic_state_variables


mass_matrix_assembly = utilities.mass_matrix_assembly
perpendicular = utilities.perpendicular
constant_distance = utilities.constant_distance


class RigidBody(object):
    """RigidBody object with its mass matrix, symbolic state variables,
    and constraints."""
    counter = 0

    def __init__(self, mass, jsa, length):
        """Initialize RigidBody object"""
        RigidBody.counter += 1
        self.ID = RigidBody.counter
        self.mass = mass
        self.jsa = jsa
        self.mass_matrix = None
        self.r_i = BasePoint('r_i', RigidBody.counter)
        self.r_j = BasePoint('r_j', RigidBody.counter)
        self.velocity = []
        self.r_i_0 = np.array([0, 0, 0])
        self.r_j_0 = np.array([0, 0, length])
        self.dt_r_i_0 = np.array([0, 0, 0])
        self.dt_r_j_0 = np.array([0, 0, 0])
        self.u = np.array([1, 0, 0])
        self.v = np.array([0, 1, 0])
        self.dt_u = np.array([0, 0, 0])
        self.dt_v = np.array([0, 0, 0])
        self.r_i_loc = np.array([0, 0, 0])
        self.r_j_loc = np.array([0, 0, length])
        self.r_g_loc = np.array([0, 0, 0.5 * length])
        self.constraints = []
        self.calculate_mass_matrix(self.mass, self.jsa)
        self.symbolic_state_variables()
        self.rigid_body_constraints()
        Mechanism.rigid_body_list.append(self)

    def calculate_mass_matrix(self, m, jsa):
        """Calculates the 12x12 mass matrix of the rigid body object."""

        rho_i = self.r_i_loc - self.r_g_loc
        tr_rho_i = rho_i.transpose()
        x = np.array([self.r_j_loc - self.r_i_loc, self.u, self.v])
        a = np.matmul(inv(x), self.r_g_loc - self.r_i_loc)
        j_i = jsa + m * np.outer(rho_i, tr_rho_i)
        z = inv(x).dot(j_i).dot(inv(x.transpose()))

        self.mass_matrix = mass_matrix_assembly(m, z, a)

    def symbolic_state_variables(self):
        """Defines symbolic state variables of the rigid body."""
        t = sym.Symbol('t')


        v_x = sym.Function('v_{},1'.format(str(self.ID)))(t)
        v_y = sym.Function('v_{},2'.format(str(self.ID)))(t)
        v_z = sym.Function('v_{},3'.format(str(self.ID)))(t)

        r_i_x = sym.Function('r_i,{},1'.format(str(self.ID)))(t)
        r_i_y = sym.Function('r_i,{},2'.format(str(self.ID)))(t)
        r_i_z = sym.Function('r_i,{},3'.format(str(self.ID)))(t)

        r_j_x = sym.Function('r_j,{},1'.format(str(self.ID)))(t)
        r_j_y = sym.Function('r_j,{},2'.format(str(self.ID)))(t)
        r_j_z = sym.Function('r_j,{},3'.format(str(self.ID)))(t)

        u = Matrix([u_x, u_y, u_z])
        v = Matrix([v_x, v_y, v_z])
        r_i = Matrix([r_i_x, r_i_y, r_i_z])
        r_j = Matrix([r_j_x, r_j_y, r_j_z])

        self.u_sym = u
        self.v_sym = v
        self.r_i_sym = r_i
        self.r_j_sym = r_j

    def rigid_body_constraints(self):
        constant_distance(self.r_i_sym-self.r_j_sym, self.length)
        constant_distance(self.u_sym, 1)
        constant_distance(self.v_sym, 1)
        perpendicular(self.u_sym, self.v_sym)
        perpendicular(self.r_i_sym-self.r_j_sym, self.u_sym)
        perpendicular(self.r_i_sym-self.r_j_sym, self.v_sym)


class BasePoint(object):
    """BasePoint object with symbolic and numeric coordinates"""
    def __init__(self, name, ID):
        self.symbolic_velocity = []
        self.local_coordinates = np.array([0, 0, 0])
        self.local_velocities = np.array([0, 0, 0])
        self.global_coordinates = np.array([0, 0, 0])
        self.global_velocities = np.array([0, 0, 0])
        self.name = name
        self.ID = ID
        self.symbolic_coordinates = symbolic_state_variables(self.name, self.ID)


class BaseVector(object):
    """BaseVector object with symbolic and numeric coordinates"""
    def __init__(self, name, ID):
        self.symbolic_coordinates = []
        self.symbolic_velocity = []
        self.global_coordinates = np.array([0, 0, 0])
        self.global_velocities = np.array([0, 0, 0])
        self.name = name
        self.ID = ID
        self.symbolic_coordinates = symbolic_state_variables(self.name, self.ID)