from experiments import det_rand_net, rand_net
import numpy as np

from nn import RandNN
from util import count_transitions, generate_points_on_upper_hemisphere, plot_points, upper_convex_hull


# depth
L = 1
# repetitions
R = 2
# width of the network
W = 3
plot = True

# det_rand_net(L, R, plot)
# rand_net(L, plot)

P, N = generate_points_on_upper_hemisphere(20)

nn = RandNN(L, W=W, P=P, N=N)
nn.evaluate()
nn.plot()

