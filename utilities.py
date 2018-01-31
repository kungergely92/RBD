
import numpy as np
from mechanism import Mechanism


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


def perpendicular(sym_matrix_a, sym_matrix_b):
    """Returns with the dot product of the input vectors"""
    return sym_matrix_a.dot(sym_matrix_b)


def constant_distance(sym_matrix, length):
    """Constant distance constraint, defined as
    x^2+y^2+z^2-length^2"""
    return sym_matrix.norm()**2 - length**2

def create_constraint(constraint_list):
    Mechanism.constraint_list.extend(constraint_list)

