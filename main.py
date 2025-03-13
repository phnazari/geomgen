from experiments import sawtooth_linregs, union_on_circle, probability_of_increase, half_circle_one_random_layer, sawtooth_one_layer, sawtooth_linregs_randlayer, sawtooth_transitions, circle_distribution
import numpy as np
from matplotlib import pyplot as plt
import math


from nn import ReLuNet
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
circle_distribution(64)


if False:
    def rho(x):
        return np.maximum(0, x)  # ReLU function

    def f(x, random=False):
        if not random:
            W1 = np.array([1, 1])
            W2 = np.array([2, -4])
            b1 = np.array([0, -1/2])
            b2 = np.array([0])
        else:
            np.random.seed(1)
            W1 = np.array([-1, 1])  # np.random.randn(2)
            W2 = np.array([-4, -4])  # np.random.randn(2)
            b1 = np.array([1/4, -3/4])  # np.random.randn(2)
            b2 = np.array([1])
            print(W1, W2, b1, b2)

        input_vector = W1*x + b1
        result = np.dot(W2, rho(input_vector)) + b2
        return result

        input_vector = np.array([1, 1]) * x + np.array([0, -1/2])
        print(input_vector)
        print("\n")
        return np.dot(np.array([2, -4]), rho(input_vector))

    def f_n(x, n):
        for i in range(n):
            if i == 0:
                x = f(x, random=False)
            else:
                x = f(x, random=False)
        return x


    # Generate values in the domain [0,1]
    n=2
    x_values = np.linspace(-1, 2, 400)
    y_values = np.array([f_n(x, n) for x in x_values])


    # Fix the random seed for reproducibility
    # Plot the function
    plt.plot(x_values, y_values, label="$f(x)$")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Plot of f(x)")
    plt.legend()
    plt.grid()
    plt.show()