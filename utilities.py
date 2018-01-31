import sympy as sym
import numpy as np
from mechanism import Mechanism
from numpy import linalg as la


def flatten(hypermatrix):
    """Removes the brackets from matrix of matrices."""
    rows, cols = hypermatrix.shape[0:2]
    for i in range(rows):
        for j in range(0, cols):
            row = np.append(row, hypermatrix[i, j], 1) if j > 0 else hypermatrix[i, 0]
        hm = np.append(hm, row, 0) if i > 0 else row
    return hm


def mass_matrix_assembly(mass, z_matrix, a_vector):
    """Calculates and assembles the mass matrix from the matrix Z, and
    vector a."""
    m_11 = np.array((mass - 2 * mass * a_vector[0] + z_matrix[0, 0]) * np.eye(3))
    m_12 = np.array((mass * a_vector[0] - z_matrix[0, 0]) * np.eye(3))
    m_13 = np.array((mass * a_vector[1] - z_matrix[0, 1]) * np.eye(3))
    m_14 = np.array((mass * a_vector[2] - z_matrix[0, 2]) * np.eye(3))

    m_22 = np.array((z_matrix[0, 0]) * np.eye(3))
    m_23 = np.array((z_matrix[0, 1]) * np.eye(3))
    m_24 = np.array((z_matrix[0, 2]) * np.eye(3))
    m_33 = np.array((z_matrix[1, 1]) * np.eye(3))
    m_34 = np.array((z_matrix[1, 2]) * np.eye(3))
    m_44 = np.array((z_matrix[2, 2]) * np.eye(3))

    mass_hm = np.array([[m_11, m_12, m_13, m_14],
                        [m_12, m_22, m_23, m_24],
                        [m_13, m_23, m_33, m_34],
                        [m_14, m_24, m_34, m_44]])

    return flatten(mass_hm)


def constant_distance(sym_matrix, length):
    """Constant distance constraint, defined as
    x^2+y^2+z^2-length^2"""
    dist_length = sym_matrix.norm()**2 - length**2
    Mechanism.constraint_list.append(dist_length)


def perpendicular(sym_vector_a, sym_vector_b):
    """Dot product of the input vectors of the rigid body objects."""
    a_dot_b = sym_vector_a.dot(sym_vector_b)
    Mechanism.constraint_list.append(a_dot_b)


def parallel(sym_matrix_a, sym_matrix_b):
    """Cross product of the input vectors of the rigid body objects."""
    a_cross_b = sym.cross(sym_matrix_a, sym_matrix_b)
    Mechanism.constraint_list.append(a_cross_b)


def symbolic_state_variables(name, ID):
    """Generates symbolic vector in function of 't' time, with 'name'
    and 'ID'."""

    t = sym.Symbol('t')

    x = sym.Function('{}_{},1'.format(name, str(ID)))(t)
    y = sym.Function('{}_{},2'.format(name, str(ID)))(t)
    z = sym.Function('{}_{},3'.format(name, str(ID)))(t)

    return sym.Matrix([x, y, z])
