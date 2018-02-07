import sympy as sym


class Mechanism(object):
    """Mechanism class, consisting RigidBody, and Constraint objects"""
    rigid_body_list, constraint_list, symbolic_variable_list = [], [], []

    def __init__(self):
        self.phi_r = sym.zeros(1)
        self.global_mass_matrix = sym.zeros(1)
        self.build_global_mass_matrix()
        self.make_phi_r(Mechanism.constraint_list,
                        Mechanism.symbolic_variable_list)

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

    def make_phi_r(self, phi, sym_vars):
        """Generates the gradient of the symbolic constraints. The function takes
        a constraint list (phi) of length "n" and takes the gradient of it by
        differentiating every element by the variables in list (sym_vars) of
        length "m". The result is an n by m matrix"""
        rows = len(sym_vars)
        columns = len(phi)

        phi_r = sym.zeros(rows, columns)

        for j, constraint in enumerate(phi):
            for i, var in enumerate(sym_vars):
                phi_r[i, j] = constraint.diff(var)

        self.phi_r = phi_r
