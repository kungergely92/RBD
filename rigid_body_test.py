from cylinder import Cylinder
from mechanism import Mechanism
from sympy import pprint

d_l = 0.5
R = 0.01
rho = 2700

segment = Cylinder(R, d_l, rho)

pendulum = Mechanism()

pprint(pendulum.global_mass_matrix)

