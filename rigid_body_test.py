import sympy as sym
import numpy as np
import time
from cylinder import Cylinder
from mechanism import Mechanism
from rigid_body import BaseObject
from constrains import Coincident, Fixed

t = sym.Symbol('t')

start = time.time()
d_l = 0.5
R = 0.01
rho = 2700
transl = np.array([0, 0, 1])

symbolic_origin = sym.Matrix([0, 0, 0])

origin = BaseObject('origin', 0, symbolic_origin)

segment_1 = Cylinder(R, d_l, rho)
segment_2 = Cylinder(R, d_l, rho)

segment_2.move(transl)

joint_1 = Fixed(segment_1, origin, 0)
joint_2 = Coincident(segment_2, segment_1, 0, 1)

baumgarte_params = [0, 0, 0]

pendulum = Mechanism([segment_1, segment_2], [joint_1, joint_2], baumgarte_params)

coeff = pendulum.global_mass_matrix.row_join(pendulum.phi_r)

sym.pprint(pendulum.coefficient_mtx.shape)

end = time.time()

sym.pprint(end - start)

