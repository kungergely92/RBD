from utilities import perpendicular, constant_distance


class Coincident(object):
    """
    """
    def __init__(self, rigid_body_1, rigid_body_2, idx_1, idx_2):

        """

        :param rigid_body_1: rigidBody object to be constrained
        :param rigid_body_2: rigidBody object being constrained to
        :param idx_1: baseObject on first
        :param idx_2:
        """

        self.symbolic = constant_distance(rigid_body_1.base_points[idx_1].symbolic_coordinates -
                                          rigid_body_2.base_points[idx_2].symbolic_coordinates, 0)

        global_crs_1 = rigid_body_1.base_points[idx_1].global_coordinates
        global_crs_2 = rigid_body_2.base_points[idx_2].global_coordinates

        translation_vector = global_crs_2 - global_crs_1

        rigid_body_1.move(translation_vector)


class Fixed(object):
    """
    """
    def __init__(self, rigid_body, global_position, idx):

        """

        :param rigid_body_1: rigidBody object to be constrained
        :param rigid_body_2: rigidBody object being constrained to
        :param idx_1: baseObject on first
        :param idx_2:
        """

        self.symbolic = constant_distance(rigid_body.base_points[idx].symbolic_coordinates -
                                          global_position.sym_pos, 0)

        global_crs = rigid_body.base_points[idx].global_coordinates

        translation_vector = global_position.global_coordinates - global_crs

        rigid_body.move(translation_vector)


class ConstantDistance(object):
    """"""
    def __init__(self, base_object_1, base_object_2, distance):

        self.symbolic = constant_distance(base_object_1.symbolic_coordinates -
                                          base_object_2.symbolic_coordinates,
                                          distance)


class Perpendicular(object):
    """"""
    def __init__(self, base_object_1, base_object_2):

        self.symbolic = perpendicular(base_object_1.symbolic_coordinates,
                                      base_object_2.symbolic_coordinates)
