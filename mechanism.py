

class Mechanism(object):
    """Mechanism class, consisting RigidBody, and Constraint objects"""
    rigid_body_list = []

    def __init__(self):
        Mechanism.rigid_body_list.append(self)
