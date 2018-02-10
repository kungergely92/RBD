import sympy as sym


class Mechanism(object):
    """Mechanism class, consisting RigidBody, and Constraint objects"""

    def __init__(self, rigid_body_list):

        self.rigid_body_list = rigid_body_list
        self.phi_r = sym.zeros(1)
        self.global_mass_matrix = sym.zeros(1)
        self.build_global_mass_matrix()
        self.make_phi_r()

    def build_global_mass_matrix(self):
        """Constructs global mass matrix"""

        number_of_rigid_bodies = len(self.rigid_body_list)

        mass_matrix = sym.zeros(number_of_rigid_bodies*12)
        for n, rigid_body in enumerate(self.rigid_body_list):
            rows = rigid_body.mass_matrix.shape[0]
            columns = rigid_body.mass_matrix.shape[1]
            for i in range(rows):
                for j in range(columns):
                    mass_matrix[i + n * rows, j + n * columns] = rigid_body.mass_matrix[i, j]

        self.global_mass_matrix = mass_matrix

    def make_phi_r(self):
        """Generates the gradient of the symbolic constraints. The function takes
        a RigidBody list, extracts the constraint list (phi) of length "n" and
        takes the gradient of it by differentiating every element by the
        variables in list (sym_vars) of length "m". The result is an n by m
        matrix"""

        sym_vars = []
        phi = []

        for rigid_body in self.rigid_body_list:
            sym_vars.extend(rigid_body.symbolic_variables)
            phi.extend(rigid_body.constraints)

        rows = len(sym_vars)
        columns = len(phi)

        phi_r = sym.zeros(rows, columns)

        for j, constraint in enumerate(phi):
            for i, var in enumerate(sym_vars):
                phi_r[i, j] = constraint.diff(var)

        self.phi_r = phi_r

    def make_coefficient_matrix(self):

        z_mtx = sym.zeros(self.phi_r.shape[1])
        top = self.global_mass_matrix.row_join(self.phi_r)
        bottom = self.phi_r.T.row_join(z_mtx)

        return top.col_join(bottom)
