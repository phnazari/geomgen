import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull, QhullError


def upper_convex_hull(S):
    """Compute the upper convex hull for a list of point sets.

    Takes a list of point sets and computes the upper convex hull for each set.

    Args:
        S (list[set]): List of sets, where each set contains points as (x,y) tuples

    Returns:
        list[set]: List of sets containing only the points on the upper convex hull
                  of each input set
    """

    S_upper = []
    # iterate over dimensions
    for s in S:
        s_upper = _upper_convex_hull(s)
        S_upper.append(s_upper)

    return S_upper


def double_uch(P, N):
    """Compute upper convex hull for both positive and negative point sets.

    A convenience function that applies upper_convex_hull() to both P and N point sets.

    Args:
        P (list[set]): List of positive point sets for each dimension
        N (list[set]): List of negative point sets for each dimension

    Returns:
        tuple: Contains:
            - P_upper (list[set]): Upper convex hull of positive point sets
            - N_upper (list[set]): Upper convex hull of negative point sets
    """

    return upper_convex_hull(P), upper_convex_hull(N)


def _upper_convex_hull(S):
    """Compute the upper convex hull of a set of points in R^2.

    Takes a set of points in 2D space and returns only those points that lie on the
    upper convex hull. The upper convex hull is the portion of the convex hull from
    the leftmost to rightmost point when traversing counterclockwise.

    Args:
        S (set): Set of points, where each point is a tuple of (x,y) coordinates

    Returns:
        set: Set containing only the points that lie on the upper convex hull.
             Returns input set if it has 2 or fewer points or if hull computation fails.
    """

    points = np.array(list(S))

    # if points is too short, then it is already its own upper convex hull
    if len(points) <= 2:
        return S

    # use the quickhull algorithm to compute the complete convex hull.
    # Might fail if the intrinsic dimension of points is too small.
    try:
        hull = ConvexHull(points)
        vertices = points[hull.vertices]
    except QhullError as e:
        return S

    vertices = points[hull.vertices]

    # Extract the upper convex hull:
    # 1. identify the left- and right-most points
    # 2. filter for the one with max y value because at the boundary
    # there could be two on top of each other
    minx = np.argmin(vertices[:, 0])
    minx_candidates = np.where(vertices[:, 0] == vertices[minx, 0])[0]
    if len(minx_candidates) > 1:
        minx = minx_candidates[np.argmax(vertices[minx_candidates, 1])]
    maxx = np.argmax(vertices[:, 0])
    maxx_candidates = np.where(vertices[:, 0] == vertices[maxx, 0])[0]
    if len(maxx_candidates) > 1:
        maxx = maxx_candidates[np.argmax(vertices[maxx_candidates, 1])]

    # concatenate the two parts making up the upper convex hull
    # (might have started in the middle)
    if maxx >= minx:
        upper_hull = np.concatenate([vertices[: minx + 1], vertices[maxx:]])
    else:
        upper_hull = vertices[maxx : minx + 1]

    return set(map(tuple, upper_hull))


def count_transitions(points, P, N):
    """Count transitions between positive and negative regions in a set of points.

    Counts the number of times consecutive points (when sorted by x-coordinate) switch
    between belonging to the positive set P and negative set N.

    Args:
        points (set): Set of points to analyze
        P (list[set]): List of positive point sets for each dimension
        N (list[set]): List of negative point sets for each dimension

    Returns:
        int: Number of transitions between positive and negative regions
    """

    transitions = 0
    points = np.array(list(points))

    # sort points, first by x-coordinate and then by y-coordinate
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    for i in range(1, len(points)):
        if (tuple(points[i - 1]) in P[0] and tuple(points[i]) in N[0]) or (
            tuple(points[i - 1]) in N[0] and tuple(points[i]) in P[0]
        ):
            transitions += 1

    return transitions


def plot_points(P, N, l, UCHp=[]):
    """Plot points from positive and negative sets with optional upper convex hull.

    Creates a scatter plot showing points from positive and negative sets in different colors,
    with an optional overlay of the upper convex hull points.

    Args:
        P (set): Set of positive points
        N (set): Set of negative points
        l (int): Label used in output filename
        UCHp (list, optional): Points on upper convex hull to highlight. Defaults to empty list.
    """

    Ps = np.array(list(P))
    Ns = np.array(list(N))

    plt.scatter(Ps[:, 0], Ps[:, 1], c="r", marker="o", label="P", alpha=0.5)
    plt.scatter(Ns[:, 0], Ns[:, 1], c="b", marker="o", label="N", alpha=0.5)

    # if an upper convex hull is passed, plot them in blue
    if len(UCHp) > 0:
        UCHp = np.array(sorted(UCHp, key=lambda x: (x[0], x[1])))

        plt.scatter(
            UCHp[:, 0],
            UCHp[:, 1],
            facecolors="none",
            edgecolors="g",
            label="UCH",
            s=250,
        )

    plt.legend()
    plt.savefig(f"~/Desktop/{l}.png")
    plt.show()


def plot_grid(data, benchmark, L, elements_per_row, name):
    """Plot a grid of graphs comparing data against a benchmark.

    Creates a grid of plots showing median values with shaded percentile ranges, comparing
    experimental data against a deterministic benchmark.

    Args:
        data (ndarray): Array of shape (L,R,L) containing experimental results
        benchmark (ndarray): Array of length L containing benchmark values
        L (int): Number of layers/plots
        elements_per_row (int): Number of plots to show per row
        name (str): Label for the y-axis and legend

    Returns:
        Figure: The matplotlib figure containing the grid of plots
    """

    fig, axs = plt.subplots(
        L // elements_per_row + L % elements_per_row,
        elements_per_row,
        figsize=(15, 5 * (L // elements_per_row + L % elements_per_row)),
    )

    for l in range(L):
        s = L - l

        row = l // elements_per_row
        col = l % elements_per_row

        # Plot mean and interquartile range of linear regions
        median_linregs = np.median(data[l, :, :], axis=0)

        axs[row, col].plot(range(1, L + 1), median_linregs, label=f"Mean {name}")
        axs[row, col].plot(
            range(1, L + 1),
            benchmark,
            color="r",
            linestyle="dashed",
            linewidth=2,
            label="Deterministic",
        )

        step_size = 5
        start = 30

        for delta in range(step_size, start, step_size):
            lower_percentile = np.percentile(data[l, :, :], start - delta, axis=0)
            upper_percentile = np.percentile(data[l, :, :], 100 - start + delta, axis=0)
            alpha = 0.1 + 0.3 * (start - delta) / start

            axs[row, col].fill_between(
                range(1, L + 1),
                lower_percentile,
                upper_percentile,
                alpha=alpha,
                color="crimson",
            )

        if row == 0 and col == 0:
            axs[row, col].set_xlabel("Blocks")
            axs[row, col].set_ylabel(name)
            axs[row, col].legend(loc="upper left")

        axs[row, col].set_yscale("log")

    plt.tight_layout()

    return fig


def relu(x):
    return np.maximum(0, x)


def generate_points_on_upper_hemisphere(num_points):
    """
    Generates points on the upper hemisphere of a unit circle.

    This function creates two sets of points P and N by generating points along
    the upper hemisphere (from 0 to π/2) in an alternating fashion. The points
    are evenly spaced along the arc.

    Args:
        num_points (int): Number of points to generate for each set P and N.
                         The total number of points generated will be 2*num_points + 2.

    Returns:
        tuple: A pair (P, N) where:
            - P (list of sets): Contains one set of points representing positive points
            - N (list of sets): Contains one set of points representing negative points
            Each point is represented as a tuple (x,y) of coordinates on the unit circle.
    """

    n = 2 * num_points + 2

    angles = np.linspace(0, np.pi / 2, n, endpoint=True)

    x = np.cos(angles)
    y = np.sin(angles)
    points = np.vstack((x, y)).T

    N = [{tuple(p) for p in points[0::2]}]
    P = [{tuple(n) for n in points[1::2]}]

    return P, N


def Q(S, x):
    """
    Computes the CPA function defined a set S at x.

    For a set S of points (a,b), computes max{ax + b} over all points in S.

    Args:
        S (set): Set of points, where each point is a tuple (a,b) representing
                coefficients of a linear function ax + b
        x (float): Value at which to evaluate the maximum

    Returns:
        float: Maximum value of ax + b over all points (a,b) in S
    """

    # handle empty set case
    if not S:
        return float("-inf")

    return max([s[0] * x + s[1] for s in S])


def compute_upper_simplices_3d(points):
    """
    Computes the simplices forming the upper convex hull of a set of points in 3D space.

    Args:
        points (set): Set of points in 3D space, where each point is a tuple (x,y,z)
                     representing coordinates.

    Returns:
        numpy.ndarray: Array of simplices (triangles) that form the upper convex hull.
                      Each simplex is represented by indices into the points array.
                      Only includes facets visible from above (positive z direction).
    """

    points = np.array(list(points))
    # Compute the full convex hull.
    hull = ConvexHull(points)

    # The equations for the hull facets have the form:
    #     a*x + b*y + c*z + d = 0
    # where the normal vector is (a, b, c). By convention in scipy.spatial.ConvexHull,
    # the interior of the hull lies on the negative side of the hyperplane.
    # For the upper convex hull (visible from +z), we select facets
    # where the z-component of the normal is positive.
    upper_facets = []
    for eq, simplex in zip(hull.equations, hull.simplices):
        # eq[2] is the z-component of the facet's outward normal.
        if eq[2] > 0:
            upper_facets.append(simplex)

    return np.array(upper_facets)
