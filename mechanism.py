

class Mechanism(object):
    """Mechanism class, consisting RigidBody, and Constraint objects"""
    rigid_bodies = []
    constraint_list_global = []

    def __init__(self):
        Mechanism.rigid_bodies.append(self)
        Mechanism.constraint_list_global.extend(self.constraints)
        Mechanism.constraint_list_global.extend(self.constraint_list)
