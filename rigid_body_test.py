from cylinder import Cylinder
from mechanism import Mechanism
from sympy import pprint
import time

start = time.time()
d_l = 0.5
R = 0.01
rho = 2700


segment_1 = Cylinder(R, d_l, rho)

pendulum = Mechanism([segment_1])

pprint(pendulum.make_coefficient_matrix().shape)
end = time.time()

pprint(end - start)

