from rigid_body import RigidBody
from moment_of_inertia import jsa_cylinder
from mechanism import Mechanism


class Cylinder(Mechanism, RigidBody):
    """Makes cylinder object."""
    def __init__(self, radius=1, length=1, density=1):
        self.radius = radius
        self.length = length
        self.density = density
        self.mass, self.jsa = jsa_cylinder(radius, length, density)
        RigidBody.__init__(self, self.mass, self.jsa, self.length)
        Mechanism.__init__(self)
