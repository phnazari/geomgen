from experiments import sawtooth_linregs, union_on_circle, probability_of_increase, half_circle_one_random_layer, sawtooth_transitions, circle_distribution, example_function_dualrep, plot_affine_regions, plot_setsum_dualrep, plot_affine_regions, plot_union_dualrep, plot_circle_function
import numpy as np
from matplotlib import pyplot as plt


# depth
L = 5
# repetitions
R = 10


union_on_circle(4)
half_circle_one_random_layer(R, plot=True, Nmax=16)
probability_of_increase(R, plot=True, Nmax=16)

sawtooth_linregs(L, R)
sawtooth_transitions(L, R)
circle_distribution(64)
plot_affine_regions()
plot_affine_regions(db=True)
plot_circle_function(4)
example_function_dualrep()
plot_union_dualrep()
plot_setsum_dualrep()