from experiments import sawtooth_linregs, union_on_circle, probability_of_increase, half_circle_one_random_layer, sawtooth_one_layer, sawtooth_linregs_randlayer, sawtooth_transitions, circle_distribution, example_function_dualrep, plot_affine_regions, plot_setsum_dualrep, plot_affine_regions, plot_union_dualrep
import numpy as np
from matplotlib import pyplot as plt

from util import count_transitions, generate_points_on_upper_hemisphere, plot_function, plot_points, plot_sawtooths, upper_convex_hull, plot_blocks


# depth
L = 6
# repetitions
R = 100
plot = True

# union_on_circle(4)
# half_circle_one_random_layer(R, plot=plot, Nmax=16)
# sawtooth_one_layer(R, plot=plot, Lmax=L)
#probability_of_increase(R, plot=plot, Nmax=16)

# sawtooth_linregs(L, R)
# sawtooth_transitions(L, R)

#sawtooth_linregs_randlayer(L, R)

# plot_sawtooths(1, 0)
# circle_distribution(64)

# plot_affine_regions()

plot_affine_regions(db=True)
# example_function_dualrep()
# plot_union_dualrep()
# plot_setsum_dualrep()