from experiments import det_rand_net, half_circle, rand_net, half_circle_one_layer, sawtooth_plots, sawtooth_histograms, sawtooth_one_layer, sawtooth, expected_good_edges
import numpy as np
from matplotlib import pyplot as plt
import math


from nn import RandNN
from util import count_transitions, generate_points_on_upper_hemisphere, plot_function, plot_points, plot_sawtooths, upper_convex_hull, plot_blocks


# depth
L = 7
# repetitions
R = 100
plot = True

if False:
    W1 = np.random.randn(2, 1)
    W2 = np.random.randn(1, 2)
    b1 = np.random.randn(2)
    b2 = np.random.randn(1)

    while -b1[0]/W1[0,0] > 1 or -b1[0]/W1[0,0] < 0 or -b1[1]/W1[1,0] > 1 or -b1[1]/W1[1,0] < 0:
        W1 = np.random.randn(2, 1)
        W2 = np.random.randn(1, 2)
        b1 = np.random.randn(2)
        b2 = np.random.randn(1)

    plot_blocks(W1, W2, b1, b2, "test")

# sawtooth(L, R, plot=plot)

# sawtooth_plots(L, R, plot, randomblocks=True)

# half_circle(R, plot=plot, L=L, size=14)

# half_circle_one_layer(R, plot=plot, size=15)

# det_rand_net_one_layer(R, plot)

if False:
    # Initialize an empty list to store the expected number of good edges
    expected_edges = []
    test = []
    test2 = []
    test3 = []

    N = 1001
    # Iterate over all even n from 2 to 100 (inclusive)
    for n in range(2, N, 2):
        a = n // 2
        E = expected_good_edges(n, a)
        expected_edges.append(E)
        #test.append(math.comb(n, 2))
        #test2.append(math.factorial(a)/math.factorial(2*a-n))
        #test3.append(math.factorial(a))

    # Plot the results
    plt.plot(range(2, N, 2), expected_edges, marker='o', label='Expected', alpha=0.5)
    # plt.plot(range(2, N, 2), test, marker='o', label='Test', alpha=0.5)
    # plt.plot(range(2, N, 2), test2, marker='o', label='Test2', alpha=0.5)
    # plt.plot(range(2, N, 2), test3, marker='o', label='Test3', alpha=0.5)
    plt.xlabel('n')
    plt.ylabel('Expected Number of Good Edges')
    plt.title('Expected Number of Good Edges vs n')
    plt.legend()
    plt.grid(True)
    plt.show()

