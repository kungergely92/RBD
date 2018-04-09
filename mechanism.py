import sympy as sym
from utilities import cm_rhs

t = sym.Symbol('t')


class Mechanism(object):
    """Mechanism class, consisting RigidBody, and Constraint objects"""

    def __init__(self, rigid_body_list, constraint_list, baumgarte_params):

        self.rigid_body_list = rigid_body_list
        self.constraint_list = constraint_list
        self.alpha = baumgarte_params[0]
        self.beta = baumgarte_params[1] # A rigid body has 12 symbolic coordinates
        self.r = []
        self.r_t = []
        self.phi = []
        self.phi_t = []
        self.phi_r = []
        self.b = []
        self.coefficient_mtx = []
        self.global_mass_matrix = sym.zeros(1)
        self.make_sym()
        self.build_global_mass_matrix()
        self.make_equation()

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

    def make_sym(self):
        """Generates the gradient of the symbolic constraints. The function takes
        a RigidBody list, extracts the constraint list (phi) of length "n" and
        takes the gradient of it by differentiating every element by the
        variables in list (sym_vars) of length "m". The result is an n by m
        matrix"""

        phi_list = []
        sym_vars = sym.zeros(len(self.rigid_body_list) * 12, 1)

        for i, rigid_body in enumerate(self.rigid_body_list):  # Global symbolic variable matrix (column vector)
            for j, sim_var in enumerate(rigid_body.symbolic_variables):
                sym_vars[i * 12 + j, 0] = rigid_body.symbolic_variables[j]
            phi_list.extend(rigid_body.constraints)

        self.r = sym_vars

        for i in self.constraint_list:
            phi_list.append(i.symbolic)

        phi = sym.zeros(len(phi_list), 1)

        for i, constraint in enumerate(phi_list):
            phi[i] = constraint

        self.phi = phi
        self.phi_t = self.phi.diff(t)

        rows = len(self.r)
        columns = len(phi)

        phi_r = sym.zeros(rows, columns)

        for j, constraint in enumerate(phi):
            for i, var in enumerate(sym_vars):
                phi_r[i, j] = constraint.diff(var)

        self.phi_r = phi_r

        self.b = sym.Matrix([- (self.phi_r.diff(t)).T*self.r.diff(t) - self.phi_t.diff(t) -
                             2*self.alpha*(self.phi_r.T*self.r.diff(t) + self.phi_t) -
                             (self.beta**2)*self.phi])

    def make_equation(self):

        z_mtx = sym.zeros(self.phi_r.shape[1])
        top = self.global_mass_matrix.row_join(self.phi_r)
        bottom = self.phi_r.T.row_join(z_mtx)
        A = top.col_join(bottom)
        C, Nb = cm_rhs(A, self.b, self.r)
        self.coefficient_mtx = top.col_join(bottom)


