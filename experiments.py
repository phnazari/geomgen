import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Arc

from util import (
    compute_upper_simplices_3d,
    generate_points_on_upper_hemisphere,
    plot_grid,
    upper_convex_hull,
    Q,
)
from tropical import add_bias, setmult, setsum
from nn import Layer, ReLuNet, SawtoothNetwork
import os

from scipy.stats import norm
from scipy.integrate import quad
import math


######################################### SAWTOOTH #########################################


def detrand_sawtooth(L, R):
    """
    Generate R instances of (partially randomized) sawtooth networks and log their complexity

    Args:
        L (int): Total number of layers
        R (int): Number or repetitions

    Returns:
        tuple: Contains:
            - results_transitions (ndarray): Transition counts for each configuration, shape (L,R,L)
            - results_linregs (ndarray): Linear region counts for each configuration, shape (L,R,L)
            - nn_det (SawtoothNetwork): Reference deterministic network with L layers
    """

    results_transitions = np.zeros((L, R, L))
    results_linregs = np.zeros((L, R, L))

    for r in range(R):
        for l in range(L):
            s = L - l
            print(f"iteration r={r}, l={l}, s={s}")
            # initialize a sawtooth network with l deterministic and s random blocks
            nn = SawtoothNetwork(l, s)
            # do a forward pass, logging the complexity after each intermediate layer
            nn(all_layers=True)

            # save complexities
            results_transitions[l, r] = nn.transitions(all_layers=True)
            results_transitions[l, r, :l] = 2 ** np.arange(
                1, l + 1
            )  # this is known theoretically
            results_linregs[l, r] = nn.linregs(all_layers=True)

    nn_det = SawtoothNetwork(L, 0)
    nn_det(all_layers=True)

    return results_transitions, results_linregs, nn_det


def sawtooth_linregs(L, R):
    """Generate and plot linear region counts for sawtooth networks with varying deterministic layers.

    Creates multiple sawtooth networks with different numbers of deterministic and random layers,
    computes their linear region counts, and visualizes the results in a grid plot.

    Args:
        L (int): Maximum number of total layers
        R (int): Number of random repetitions per configuration

    Returns:
        None: Results are saved as plots to disk
    """

    _, results_linregs, nn_det = detrand_sawtooth(L, R)

    elements_per_row = 3
    fig = plot_grid(
        results_linregs,
        nn_det.linregs(all_layers=True),
        L,
        elements_per_row,
        "#Linear Regions",
    )
    output_dir = f"~/Desktop/plots/{L}_{R}/linregs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, f"combined_{L}_{R}.png"))

    plt.show()


def sawtooth_transitions(L, R):
    """Generate and plot transition counts for sawtooth networks with varying deterministic layers.

    Creates multiple sawtooth networks with different numbers of deterministic and random layers,
    computes their transition counts, and visualizes the results in a grid plot.

    Args:
        L (int): Maximum number of total layers
        R (int): Number of random repetitions per configuration

    Returns:
        None: Results are saved as plots to disk
    """

    results_transitions, _, _ = detrand_sawtooth(L, R)

    elements_per_row = 3
    fig = plot_grid(
        results_transitions,
        np.array(2 ** np.arange(1, L + 1)),
        L,
        elements_per_row,
        "Transitions",
    )
    output_dir = f"~/Desktop/plots/{L}_{R}/trans"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, f"combined_{L}_{R}.png"))

    plt.show()


######################################### CIRCLES #########################################


def union_on_circle(num_points=5):
    """Generate and visualize point sets and their unions on a circle.

    Creates two sets of points on the upper hemisphere, computes their set sums and unions
    after a one-dimensional ReLU, and visualizes the results with scatter plots and arcs.

    Args:
        num_points (int, optional): Number of points to generate on the hemisphere. Defaults to 5.

    Returns:
        None: Results are displayed as a matplotlib plot
    """

    P, N = generate_points_on_upper_hemisphere(num_points)

    # compute the sum and its upper convex hull
    S = setsum(P[0], N[0])
    uch = upper_convex_hull([S])[0]

    W = 1
    b = u(num_points, 0)

    # compute the points after the ReLU layer
    if W > 0:
        # multiply N by W
        L = setmult(W, N[0])
        M = add_bias(L, b)
        O = setmult(W, P[0])

        A = setsum(L, L)
        B = setsum(M, O)
        C = A.union(B)
    else:
        L = setmult(-W, P[0])
        M = add_bias(L, b)
        O = setmult(-W, N[0])

        A = setsum(L, L)
        B = setsum(M, O)
        C = A.union(B)

    fig = plt.figure()
    plt.scatter(*zip(*A), alpha=0.4, label="A", color="r")
    plt.scatter(*zip(*B), alpha=0.4, label="B", color="g")

    # add the circular arcs
    for l in L:
        if W > 0:
            plt.gca().add_patch(Arc(l, 2 * W, 2 * W, theta1=0, theta2=90, color="r"))
        else:
            plt.gca().add_patch(Arc(l, -2 * W, -2 * W, theta1=0, theta2=90, color="r"))

    for m in M:
        if W > 0:
            plt.gca().add_patch(Arc(m, 2 * W, 2 * W, theta1=0, theta2=90, color="g"))
        else:
            plt.gca().add_patch(Arc(m, -2 * W, -2 * W, theta1=0, theta2=90, color="g"))

    # plot upper convex hull using circles around points
    uch = upper_convex_hull([C])[0]
    plt.gca().axis("off")
    plt.gca().scatter(*zip(*uch), alpha=0.4, label="UCH", color="b")
    plt.axis("equal")
    plt.legend()
    plt.show()


def u(n, i):
    """Compute an upper bound \mathfrak u.

    This function calculates an \mathfrak u. For example, if w > 0, this is the upper bound making sure the
    i'th point of wN_0 + wN_0 is contained in the upper convex hull of P_1 + N_1.

    Args:
        n (int): Number of points to generate on the hemisphere.
        i (int): Number indexing the point of wN_0 + wN_0

    Returns:
        \mathfrak u(i, n) (int): the upper bound
    """

    norm = 2 * (2 * n + 1)
    pihalf = np.pi / 2
    Cn = 2 * np.sin(pihalf * 1 / norm) ** 2
    Dn = np.sin(pihalf * (4 * i) / norm)
    return Cn / Dn


def l(n, i):
    """Compute a lower bound \mathfrak l.

    This function calculates an \mathfrak l. For example, if w > 0, this is the lower bound making sure the
    i'th point of wN_0 + wP_0 \boxplus b is contained in the upper convex hull of P_1 + N_1.

    Args:
        n (int): Number of points to generate on the hemisphere.
        i (int): Number indexing the point of wN_0 + wP_0 \boxplus b
    Returns:
        \mathfrak l(i, n) (int): the lower bound

    """

    norm = 2 * (2 * n + 1)
    pihalf = np.pi / 2
    Cn = 2 * np.sin(pihalf * 1 / norm) ** 2
    Dn = np.sin(pihalf * (2 * (2 * i + 1)) / norm)
    return -Cn / Dn


def probability_of_increase(R, plot=False, Nmax=32):
    """Compute the empirical probability of complexity increase and compare with theoretical bound.

    This function generates random one-layer ReLU networks with n points on the hemisphere
    and computes the probability that their complexity (number of linear regions) increases
    beyond the minimal 2n+1. It compares the empirical probability with a theoretical
    upper bound.

    Args:
        R (int): Number of random trials per n
        plot (bool, optional): Whether to display the plot. Defaults to False.
        Nmax (int, optional): Maximum number of points to test. Defaults to 32.

    Returns:
        None. Displays a plot comparing empirical probabilities with theoretical bounds.
    """

    n = 2
    max = 15
    sigma_values = np.linspace(1, 5, 9)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sigma_values)))

    results = np.zeros(Nmax)

    # empirically compute the average probability of increase
    for n in range(2, Nmax):
        for r in range(R):
            P, N = generate_points_on_upper_hemisphere(n)

            nn = ReLuNet(1, P=P, N=N)
            nn.evaluate(all_layers=False)

            if nn.linregs > 2 * n + 1:
                results[n] += 1

    results /= R

    # theoretical probability
    upper_bound_values = [
        1 / 2
        + 1
        / (4 * 2 * np.pi)
        * (
            np.arctan(
                np.abs(u(n, np.ceil(n / 2)))
                + np.arctan(np.abs(l(n, np.ceil(n / 2) - 1)))
            )
        )
        for n in range(2, Nmax)
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(2, Nmax),
        results[2:],
        marker="o",
        linestyle="-",
        color="b",
        label="Empirical",
    )

    plt.plot(
        range(2, Nmax),
        upper_bound_values,
        linestyle="--",
        color="r",
        label="Theoretical",
    )

    plt.axhline(y=0.5, color="gray", linestyle="--", label="Y=0.5")

    plt.xlabel("n", fontsize=14)
    plt.ylabel(r"$\mathbb{P}(\uparrow)$", fontsize=14)
    plt.ylim(0.25, 0.75)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if plot:
        plt.show()

    return


def _as(n):
    """
    Asymptotic approximation for the expected complexity gain.

    This function calculates an asymptotic approximation for the expected
    complexity gain when adding a random layer to a half-circle configuration
    with n points.

    Args:
        n (int): Number of points on the upper hemisphere

    Returns:
        float: Approximated expected complexity gain
    """

    t = 3 / (8 * (2 * n + 1))
    gamma = 0.577
    return -1 / 2 * n + 1 + t * np.log((4 * (2 * n + 1)) / np.pi) + t * gamma


def compute_P(s, n, atan_l_values, atan_u_values, delta_l_values, delta_u_values):
    """Compute the probability of complexity increase for a specific value of s.

    This function calculates the probability that a random one-layer ReLU network
    with n points on the hemisphere will have exactly s linear regions. The calculation
    depends on the value of s relative to n, and uses precomputed arctangent values
    and their differences.

    Args:
        s (int): The number of linear regions to compute the probability for
        n (int): Number of points on the upper hemisphere
        atan_l_values (list): Precomputed arctangent values for lower bounds
        atan_u_values (list): Precomputed arctangent values for upper bounds
        delta_l_values (list): Differences between consecutive atan_l values
        delta_u_values (list): Differences between consecutive atan_u values

    Returns:
        float: The probability that a random network with n points has exactly s linear regions
    """

    if s == n + 2:
        return (1 / 4) * (2 - (2 / np.pi) * (atan_l_values[0] + atan_u_values[0]))

    elif s == 2 * n + 2:
        if n % 2 == 0:
            m = n // 2
            return (1 / 4) * (
                2
                - (2 / np.pi)
                * (
                    atan_u_values[0]
                    + atan_l_values[0]
                    - delta_l_values[m - 1]
                    - delta_u_values[m - 1]
                )
            )
        else:
            return (1 / 4) * (2 - (2 / np.pi) * (atan_u_values[0] + atan_l_values[0]))

    elif s == 3 * n + 2:
        return (
            (1 / 4)
            * (2 / np.pi)
            * (2 * atan_u_values[n - 1] + 2 * atan_l_values[n - 1])
        )

    else:
        i = s - (n + 2)
        if 0 < i < n and i % 2 == 0:
            m = i // 2
            return (
                (1 / 4) * (2 / np.pi) * (delta_l_values[m - 1] + delta_u_values[m - 1])
            )

        elif n < i < 2 * n:
            if i % 2 == 0:
                m = i // 2
                return (
                    (1 / 4)
                    * (2 / np.pi)
                    * (
                        delta_u_values[i - n - 1]
                        + delta_l_values[m - 1]
                        + delta_u_values[m - 1]
                        + delta_l_values[i - n - 1]
                    )
                )
            else:
                return (
                    (1 / 4)
                    * (2 / np.pi)
                    * (delta_u_values[i - n - 1] + delta_l_values[i - n - 1])
                )

        return 0  # Should not occur for s in s_values


def compute_distribution(n):
    # Compute l(i, n) and u(i, n)
    l_values = [l(n, k) for k in range(n)]  # i = 0 to n-1
    u_values = [u(n, k) for k in range(1, n + 1)]  # i = 1 to n

    # Compute arctangents
    atan_l_values = [np.arctan(np.abs(l_k)) for l_k in l_values]
    atan_u_values = [np.arctan(np.abs(u_k)) for u_k in u_values]

    # Compute delta values
    delta_l_values = [
        atan_l_values[k] - atan_l_values[k + 1] for k in range(n - 1)
    ]  # k = 0 to n-2
    delta_u_values = [
        atan_u_values[k] - atan_u_values[k + 1] for k in range(n - 1)
    ]  # k = 0 to n-2, for Delta u_1 to u_{n-1}

    # Generate support (s values)
    s_values = (
        [n + 2]
        + [n + 2 + i for i in range(2, n, 2)]
        + [2 * n + 2]
        + [n + 2 + i for i in range(n + 1, 2 * n)]
        + [3 * n + 2]
    )

    # Compute probabilities
    P_s = [
        compute_P(s, n, atan_l_values, atan_u_values, delta_l_values, delta_u_values)
        for s in s_values
    ]

    return s_values, P_s


def circle_distribution(n):
    """
    Compute and visualize the theoretical distribution of complexity for a circle configuration.

    This function calculates the probability distribution of the number of linear regions
    when adding a random layer to a network with points arranged on a circle.
    It computes the distribution using the theoretical formulas, calculates the
    expected value, and visualizes the results.

    Args:
        n (int): Number of points on the circle, which determines the initial complexity
                 and the support of the distribution.

    Returns:
        tuple: A tuple containing:
            - s_values (list): Support points of the distribution (possible complexity values)
            - P_s (list): Probability values corresponding to each support point
    """

    s_values, P_s = compute_distribution(n)

    # Compute the expectation
    expectation = sum(s * P for s, P in zip(s_values, P_s))

    print(f"Expectation: {expectation:.10f}")
    print(f"starting complexity: {2*n+1}")
    print(f"marginal gain: {expectation - (2*n+1)}")

    # Verify the sum (should be close to 1)
    print(f"Sum of probabilities: {sum(P_s):.10f}")

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.stem(
        s_values,
        P_s,
        basefmt=" ",
        use_line_collection=True,
        linefmt="navy",
        markerfmt="navy",
    )
    plt.axvline(x=2 * n + 1, color="crimson", linestyle="--", label=f"s=2n+1")
    plt.xlabel("s", fontsize=14)
    plt.ylabel(r"$\mathbb{P}(S = s)$", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def half_circle_one_random_layer(R, plot=False, Nmax=32):
    """
    Analyze the expected complexity gain of adding one random layer to a half-circle configuration.

    Args:
        R (int): Number of random trials to run
        plot (bool, optional): Whether to display plots. Defaults to False.
        Nmax (int, optional): Maximum number of points to analyze. Defaults to 32.

    The function:
    1. Generates points on upper hemisphere for n=2 to Nmax
    2. Creates neural networks with one random layer
    3. Computes empirical complexity gain vs theoretical predictions
    4. Plots empirical mean, theoretical approximation, and exact computation
    """

    results = np.zeros((R, Nmax))
    results_2 = np.zeros((Nmax))

    for n in range(2, Nmax):
        for r in range(R):
            P, N = generate_points_on_upper_hemisphere(n)

            nn = ReLuNet(1, P=P, N=N)
            nn.evaluate(all_layers=False)

            results[r, n] = nn.linregs - (2 * n + 1)

        s_values, P_s = compute_distribution(n)
        expectation = sum(s * P for s, P in zip(s_values, P_s))
        results_2[n] = expectation - (2 * n + 1)

    mean_linreg = np.mean(results[:, 2:Nmax], axis=0)

    plt.figure()

    # Plot the empirical mean
    plt.plot(range(2, Nmax), mean_linreg, label="Empirical Mean", color="blue")

    # plot approximation
    plt.plot(
        range(2, Nmax),
        _as(np.arange(2, Nmax)),
        label="Approx",
        linestyle="--",
        color="red",
    )

    # plot computations
    plt.plot(
        range(2, Nmax),
        results_2[2:],
        label="Sum",
        color="green",
        linestyle=":",
        marker="o",
        markersize=4,
    )

    plt.xlabel("n")
    plt.ylabel("S-(2n+1)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.show()


def plot_circle_function(n):
    """
    Plots a function defined by points on a circle and its evaluation.

    This function creates two plots:
    1. A visualization of points on the upper hemisphere, with positive points (P) in red
       and negative points (N) in blue.
    2. A plot of the resulting function defined by these points, evaluated over a range
       of x values.

    The function is defined as Q(P, x) - Q(N, x), where Q is a tropical operation
    that computes the maximum of dot products between x and points in the set.

    Args:
        n (int): The number of points to generate on the upper hemisphere.
            This controls the complexity of the resulting function.

    Returns:
        None: The function displays the plots but does not return any values.
    """

    # Generate points on the upper hemisphere
    P, N = generate_points_on_upper_hemisphere(n)

    # Create first figure for points
    fig1 = plt.figure(figsize=(6, 5))
    ax1 = fig1.add_subplot(111)

    # Plot the points in the first figure
    Ps = np.array(list(P[0]))
    Ns = np.array(list(N[0]))

    ax1.scatter(Ps[:, 0], Ps[:, 1], c="r", marker="o", label="P", alpha=0.5)
    ax1.scatter(Ns[:, 0], Ns[:, 1], c="b", marker="o", label="N", alpha=0.5)
    ax1.legend()
    ax1.axis("equal")

    # Create second figure for function
    fig2 = plt.figure(figsize=(6, 5))
    ax2 = fig2.add_subplot(111)

    # Plot the function in the second figure
    x = np.linspace(-1, 7, 2000)
    y = np.array([Q(P[0], xi) for xi in x]) - np.array([Q(N[0], xi) for xi in x])

    ax2.plot(x, y)

    plt.show()


############################# Example Function #############################


def plot_affine_regions(db=False):
    """
    Plot the affine regions of a neural network function.

    This function visualizes the affine regions (linear pieces) of a piecewise linear
    function represented by a neural network. It creates a grid of points and determines
    which affine region each point belongs to by evaluating the dual representation.

    Args:
        db (bool, optional): If True, only distinguishes between positive and negative
                            regions (decision boundary mode). If False, uniquely identifies
                            each affine region. Defaults to False.

    Returns:
        None: The function displays the plot but does not return any values.
    """

    # Create a grid of points
    border = 3
    numpoints = 500
    x = np.linspace(-border, border, numpoints)
    y = np.linspace(-border, border, numpoints)
    X, Y = np.meshgrid(x, y)

    # Get dual representation
    P, N = compute_dual_representation()

    # For each point, compute which affine piece is active by checking which
    # tropical polynomials are maximal
    regions = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            point = np.array([X[i, j], Y[i, j]])

            # Find max index for P set
            max_val_p = float("-inf")
            max_idx_p = 0
            for idx, p in enumerate(P[0]):
                val_p = point[0] * p[0] + point[1] * p[1] + p[2]
                if val_p > max_val_p:
                    max_val_p = val_p
                    max_idx_p = idx

            # Find max index for N set
            max_val_n = float("-inf")
            max_idx_n = 0
            for idx, n in enumerate(N[0]):
                val_n = point[0] * n[0] + point[1] * n[1] + n[2]
                if val_n > max_val_n:
                    max_val_n = val_n
                    max_idx_n = idx

            # Encode both indices and which value was larger into regions array
            # Use positive values when P is larger, negative when N is larger
            if not db:
                regions[i, j] = max_idx_p * len(N[0]) + max_idx_n + 1
            else:
                if max_val_p >= max_val_n:
                    regions[i, j] = 1
                else:
                    regions[i, j] = -1

    # Plot the regions
    plt.figure(figsize=(10, 10))
    if db:
        plt.imshow(
            regions, extent=[x[0], x[-1], y[0], y[-1]], origin="lower", cmap="RdBu"
        )
    else:
        plt.imshow(
            regions, extent=[x[0], x[-1], y[0], y[-1]], origin="lower", cmap="tab20"
        )

    plt.show()


def compute_dual_representation():
    """
    Computes the dual representation of a neural network function.

    This function constructs a three-layer neural network with specific weights and biases,
    and computes its dual representation. The dual representation consists of sets P and N,
    which represent the positive and negative parts of the function respectively.

    The network architecture is as follows:
    - Layer 1: 2 inputs, 2 outputs with weights W1 and biases b1
    - Layer 2: 2 inputs, 2 outputs with weights W2 and biases b2
    - Layer 3: 2 inputs, 1 output with weights W3 and biases b3 (linear activation)

    Returns:
        tuple: A tuple containing:
            - P (list): List of sets representing the positive part of the function
            - N (list): List of sets representing the negative part of the function
    """

    P = [{(1, 0, 0)}, {(0, 1, 0)}]
    N = [set()]

    W1 = np.array([[-1, -1], [1, -2]])
    b1 = np.array([1, -1])

    W2 = np.array([[-1, 2], [2, -1]])
    b2 = np.array([1, 2])

    W3 = np.array([[3, -1]])
    b3 = np.array([2])

    layer1 = Layer(input_dim=2, output_dim=2, reduce=False, W=W1, b=b1)
    layer2 = Layer(input_dim=2, output_dim=2, reduce=False, W=W2, b=b2)
    layer3 = Layer(input_dim=2, output_dim=1, reduce=False, W=W3, b=b3, lin=True)

    P, N = layer1(P, N)
    P, N = layer2(P, N)
    P, N = layer3(P, N)

    return P, N


def example_function_dualrep():
    """
    Visualizes the dual representation of a function in 3D space.

    This function computes the dual representation of a specific function using
    three layers of transformations and plots the resulting positive (P) and
    negative (N) sets in 3D space. Each set is visualized with different colors
    and markers, and their upper convex hulls are displayed as transparent surfaces.

    Returns:
        None: The function displays a 3D plot but does not return any values.
    """

    P, N = compute_dual_representation()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Define plotting parameters for P and N sets
    plot_params = [
        {"points": P, "color": "blue", "marker": "o"},
        {"points": N, "color": "red", "marker": "x"},
    ]

    # Plot points and hulls for both sets
    for params in plot_params:
        # Plot points
        for point_set in params["points"]:
            for point in point_set:
                ax.scatter(
                    point[0],
                    point[1],
                    point[2],
                    c=params["color"],
                    marker=params["marker"],
                )

        # Compute and plot upper hulls
        upper_hulls = [
            compute_upper_simplices_3d(point_set) for point_set in params["points"]
        ]
        for hull_points, point_set in zip(upper_hulls, params["points"]):
            for simplex in hull_points:
                triangle = np.array([list(point_set)[vertex] for vertex in simplex])
                ax.plot_trisurf(
                    triangle[:, 0],
                    triangle[:, 1],
                    triangle[:, 2],
                    color=params["color"],
                    alpha=0.5,
                )

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show(block=True)  # Display the plot and block execution


def plot_union_dualrep():
    """
    Plots the union of the dual representation of the example function.

    This function computes the dual representation the example function,
    plots the points from both sets in 3D space, and visualizes the upper
    convex hull of their union. Points from set P are shown as blue circles,
    while points from set N are shown as red x-marks. The upper convex hull
    of their union is displayed as a blue semi-transparent surface.

    The function also prints the vertices that form the upper convex hull.

    Returns:
        None: The function displays a 3D plot but does not return any values.
    """

    P, N = compute_dual_representation()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Define plotting parameters for P and N sets
    plot_params = [
        {"points": P, "color": "blue", "marker": "o"},
        {"points": N, "color": "red", "marker": "x"},
    ]

    # Plot points and hulls for both sets
    for params in plot_params:
        # Plot points
        for point_set in params["points"]:
            for point in point_set:
                ax.scatter(
                    point[0],
                    point[1],
                    point[2],
                    c=params["color"],
                    marker=params["marker"],
                )

    # Compute and plot upper hull
    S = P[0].union(N[0])
    upper_hulls = compute_upper_simplices_3d(S)

    # Count vertices in the upper convex hull
    vertices = set()
    for simplex in upper_hulls:
        vertices.update(list(S)[i] for i in simplex)

    # Plot upper hull triangles
    for simplex in upper_hulls:
        triangle = np.array([list(S)[vertex] for vertex in simplex])
        ax.plot_trisurf(
            triangle[:, 0], triangle[:, 1], triangle[:, 2], color="blue", alpha=0.5
        )

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show(block=True)  # Display the plot and block execution


def plot_setsum_dualrep():
    """
    Plots the setsum of the dual representation of the example function.

    This function computes the dual representation of the example function, performs
    the setsum operation between the positive and negative sets, and visualizes
    the resulting set and its upper convex hull in 3D space.

    The function displays:
    - Points from the setsum operation
    - The upper convex hull of these points

    Returns:
        None: The function displays the plot but does not return any values.
    """

    P, N = compute_dual_representation()

    # Compute setsum and upper convex hull
    S = setsum(P[0], N[0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot points from setsum
    for point in S:
        ax.scatter(point[0], point[1], point[2], c="blue", marker="o")

    # Compute and plot upper hull
    upper_hulls = compute_upper_simplices_3d(S)

    # Count vertices in the upper convex hull
    vertices = set()
    for simplex in upper_hulls:
        vertices.update(simplex)
    num_vertices = len(vertices)

    for simplex in upper_hulls:
        triangle = np.array([list(S)[vertex] for vertex in simplex])
        ax.plot_trisurf(
            triangle[:, 0], triangle[:, 1], triangle[:, 2], color="blue", alpha=0.5
        )

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show(block=True)
