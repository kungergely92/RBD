import sympy as sym


class Mechanism(object):
    """Mechanism class, consisting RigidBody, and Constraint objects"""
    rigid_body_list = []
    constraint_list = []

    def __init__(self):
        self.global_mass_matrix = sym.zeros(1)

    def build_global_mass_matrix(self):
        """Constructs global mass matrix"""

        number_of_rigid_bodies = len(Mechanism.rigid_body_list)

        mass_matrix = sym.zeros(number_of_rigid_bodies*12)
        for n, rigid_body in enumerate(Mechanism.rigid_body_list):
            rows = rigid_body.mass_matrix.shape[0]
            columns = rigid_body.mass_matrix.shape[1]
            for i in range(rows):
                for j in range(columns):
                    mass_matrix[i + n * rows, j + n * columns] = rigid_body.mass_matrix[i, j]

        self.global_mass_matrix = mass_matrix



