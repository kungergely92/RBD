from sympy.physics.vector.functions import cross
from mechanism import Mechanism


class Constraint(Mechanism):
    """Constraint class, with rigid body IDs."""
    constraint_list = []

    def __init__(self, rigid_body_1, rigid_body_2):
        self.rigid_body_ids = '{}-{}'.format(rigid_body_1.ID, rigid_body_2.ID)
        Constraint.constraint_list.extend(self.equation)


class Perpendicular(Constraint):

    def __init__(self, rigid_body_1, rigid_body_2):
        rb1_axis = rigid_body_1.r_j_sym - rigid_body_1.r_i_sym
        rb2_axis = rigid_body_2.r_j_sym - rigid_body_2.r_i_sym
        self.equation = rb1_axis.dot(rb2_axis)
        Constraint.__init__(rigid_body_1, rigid_body_2)


class Parallel(Constraint):

    def __init__(self, rigid_body_1, rigid_body_2):
        rb1_axis = rigid_body_1.r_j_sym - rigid_body_1.r_i_sym
        rb2_axis = rigid_body_2.r_j_sym - rigid_body_2.r_i_sym
        self.equation = cross(rb1_axis, rb2_axis)
        Constraint.__init__(rigid_body_1, rigid_body_2)



