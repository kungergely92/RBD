import sympy as sym
import numpy as np


def flatten(hypermatrix):
    """Removes the brackets from matrix of matrices."""
    rows, cols = hypermatrix.shape[0:2]
    for i in range(rows):
        for j in range(0, cols):
            row = np.append(row, hypermatrix[i, j], 1) if j > 0 else hypermatrix[i, 0]
        hm = np.append(hm, row, 0) if i > 0 else row
    return hm


def flatten_mass_matrix(mass_matrix):
    """Makes 12x12 symbolic mass matrix from matrix of matrices"""
    hm = sym.zeros(12)

    for i in range(4):
        for j in range(4):
            for k in range(3):
                for l in range(3):
                    hm[3*i + k, 3*j + l] = mass_matrix[i, j][k, l]

    return hm


def mass_matrix_assembly(mass, z_matrix, a_vector):
    """Calculates and assembles the mass matrix from the matrix Z, and
    vector a."""
    m_11 = (mass - 2 * mass * a_vector[0] + z_matrix[0, 0]) * sym.eye(3)
    m_12 = (mass * a_vector[0] - z_matrix[0, 0]) * sym.eye(3)
    m_13 = (mass * a_vector[1] - z_matrix[0, 1]) * sym.eye(3)
    m_14 = (mass * a_vector[2] - z_matrix[0, 2]) * sym.eye(3)

    m_22 = z_matrix[0, 0] * sym.eye(3)
    m_23 = z_matrix[0, 1] * sym.eye(3)
    m_24 = z_matrix[0, 2] * sym.eye(3)
    m_33 = z_matrix[1, 1] * sym.eye(3)
    m_34 = z_matrix[1, 2] * sym.eye(3)
    m_44 = z_matrix[2, 2] * sym.eye(3)

    mass_hm = sym.Matrix([[m_11, m_12, m_13, m_14],
                         [m_12, m_22, m_23, m_24],
                         [m_13, m_23, m_33, m_34],
                         [m_14, m_24, m_34, m_44]])

    return flatten_mass_matrix(mass_hm)


def constant_distance(symbolic_matrix, length):
    """Constant distance constraint, defined as
    x^2+y^2+z^2-length^2"""
    dist_length = symbolic_matrix[0]**2 + symbolic_matrix[1]**2 + \
                  symbolic_matrix[2]**2 - length**2

    return dist_length


def perpendicular(sym_vector_a, sym_vector_b):
    """Dot product of the input vectors of the rigid body objects."""
    a_dot_b = sym_vector_a.dot(sym_vector_b)

    return a_dot_b


def parallel(sym_matrix_a, sym_matrix_b):
    """Cross product of the input vectors of the rigid body objects."""

    a_cross_b = sym.Matrix([sym_matrix_a[1] * sym_matrix_b[2] -
                            sym_matrix_a[2] * sym_matrix_b[1],
                            sym_matrix_a[2] * sym_matrix_b[0] -
                            sym_matrix_a[0] * sym_matrix_b[2],
                            sym_matrix_a[0] * sym_matrix_b[1] -
                            sym_matrix_a[1] * sym_matrix_b[0]])

    return a_cross_b


def symbolic_state_variables(name, ID):
    """Generates symbolic vector in function of 't' time, with 'name'
    and 'ID'."""

    t = sym.Symbol('t')

    x = sym.Function('{}_{},1'.format(name, str(ID)))(t)
    y = sym.Function('{}_{},2'.format(name, str(ID)))(t)
    z = sym.Function('{}_{},3'.format(name, str(ID)))(t)

    return sym.Matrix([x, y, z])


def sym_matrix_to_list(sym_mtx):
    """Converts a one dimensional symbolic matrix into a python list."""
    sym_list = []
    for elem in range(sym_mtx.shape[0]):
        sym_list.append(sym_mtx[elem])

    return sym_list


def base_object_select(rigid_body, idx):

    if idx < 2:
        return [rigid_body.base_points[idx], rigid_body.base_points[idx-1]]
    else:
        idx = idx - 2
        return [rigid_body.base_vectors[idx], rigid_body.base_vectors[idx-1]]



