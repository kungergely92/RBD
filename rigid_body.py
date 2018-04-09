import sympy as sym
import numpy as np
from numpy.linalg import inv
from utilities import symbolic_state_variables, mass_matrix_assembly, \
    perpendicular, constant_distance


class RigidBody(object):
    """RigidBody object with mass matrix, symbolic state variables,
    and constraints."""
    counter = 0

    def __init__(self, mass, jsa, length):
        """
        Initializes RigidBody object, with mass "jsa" matrix, and length.
        :param mass: mass of RigidBody object
        :param jsa: jsa matrix
        :param length:
        """
        RigidBody.counter += 1
        self.length = length
        self.ID = RigidBody.counter
        self.mass_matrix = None
        self.constraints = []
        self.symbolic_variables = []
        self.r_i = BaseObject('r_i', self.ID)
        self.r_j = BaseObject('r_j', self.ID)
        self.u = BaseObject('u', self.ID)
        self.v = BaseObject('v', self.ID)
        self.base_points = [self.r_i,
                            self.r_j]
        self.base_vectors = [self.u,
                             self.v]
        self.r_i.local_coordinates = np.array([0, 0, 0])
        self.r_j.local_coordinates = np.array([0, 0, length])
        self.r_i.global_coordinates = np.array([0, 0, 0])
        self.r_j.global_coordinates = np.array([0, 0, length])
        self.u.local_coordinates = np.array([1, 0, 0])
        self.v.local_coordinates = np.array([0, 1, 0])
        self.u.global_coordinates = np.array([1, 0, 0])
        self.v.global_coordinates = np.array([0, 1, 0])
        self.r_g_loc = np.array([0, 0, 0.5 * length])
        self.make_symbolic_variables()
        self.calculate_mass_matrix(mass, jsa)
        self.rigid_body_constraints()

    def calculate_mass_matrix(self, m, jsa):
        """Calculates the 12x12 mass matrix of the rigid body object."""

        rho_i = self.r_i.local_coordinates - self.r_g_loc
        tr_rho_i = rho_i.transpose()
        x = np.array([self.r_j.local_coordinates - self.r_i.local_coordinates,
                      self.u.local_coordinates, self.v.local_coordinates])
        a = np.matmul(inv(x), self.r_g_loc - self.r_i.local_coordinates)
        j_i = jsa + m * np.outer(rho_i, tr_rho_i)
        z = inv(x).dot(j_i).dot(inv(x.transpose()))

        self.mass_matrix = mass_matrix_assembly(m, z, a)

    def rigid_body_constraints(self):
        """Make constraints of RigidBody constraints"""
        phi_1 = constant_distance(self.r_i.symbolic_coordinates -
                                  self.r_j.symbolic_coordinates, self.length)
        phi_2 = constant_distance(self.u.symbolic_coordinates, 1)
        phi_3 = constant_distance(self.v.symbolic_coordinates, 1)
        phi_4 = perpendicular(self.u.symbolic_coordinates,
                              self.v.symbolic_coordinates)
        phi_5 = perpendicular(self.r_i.symbolic_coordinates -
                              self.r_j.symbolic_coordinates,
                              self.u.symbolic_coordinates)
        phi_6 = perpendicular(self.r_i.symbolic_coordinates -
                              self.r_j.symbolic_coordinates,
                              self.v.symbolic_coordinates)

        self.constraints = [phi_1, phi_2, phi_3, phi_4, phi_5, phi_6]

    def make_symbolic_variables(self):

        base_object_list = [self.r_i, self.r_j, self.u, self.v]
        symbolic_variable_list = []
        for base_object in base_object_list:
            for i in range(3):
                symbolic_variable_list.append(base_object.symbolic_coordinates[i])

        self.symbolic_variables = symbolic_variable_list

    def move(self, translation_vector):

        for i in range(2):
            self.base_points[i].global_coordinates = \
                self.base_points[i].global_coordinates + translation_vector


class BaseObject(object):
    """BaseObject with symbolic and numeric coordinates. Can be used to define
    base points, and base vectors."""
    def __init__(self, name, ID, sym_pos=sym.Matrix([0, 0, 0])):

        t = sym.Symbol('t')

        self.local_coordinates = np.array([0, 0, 0])
        self.local_velocities = np.array([0, 0, 0])
        self.global_coordinates = np.array([0, 0, 0])
        self.global_velocities = np.array([0, 0, 0])
        self.sym_pos = sym_pos
        self.name = name
        self.ID = ID
        self.symbolic_coordinates = symbolic_state_variables(self.name, self.ID)
        self.symbolic_velocity = self.symbolic_coordinates.diff(t)

    def move(self, translation_vector):

        self.global_coordinates = self.global_coordinates + translation_vector





