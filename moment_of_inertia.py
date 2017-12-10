import numpy as np
import math

pi = math.pi
Tr = np.trace


def jsa_cylinder(radius, height, density):
    """The function takes the cylinder's radius, length, and density
    and calculates the mass, and special moment of inertia matrix."""
    mass = radius*radius*pi*height*density
    thetax = 0.25 * mass * radius * radius + \
        (1 / 12) * mass * height**2
    thetaz = 0.5 * mass * radius * radius

    js = np.array([[thetax, 0, 0], [0, thetax, 0], [0, 0, thetaz]])

    ksi = 0.5 * Tr(js) - js[0, 0]
    eta = 0.5 * Tr(js) - js[1, 1]
    zeta = 0.5 * Tr(js) - js[2, 2]

    return mass, np.array([[ksi, 0, 0], [0, eta, 0], [0, 0, zeta]])