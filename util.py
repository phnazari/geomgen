import os
import numpy as np
from matplotlib import pyplot as plt


def upper_convex_hull(S):
    S_upper = []
    for s in S:
        s_upper = _upper_convex_hull(s)
        S_upper.append(s_upper)

    return S_upper

from scipy.spatial import ConvexHull


def _upper_convex_hull(S):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """    

    points = np.array(list(S))

    if len(points) <= 1:
        return S

    hull = ConvexHull(points)
    vertices = points[hull.vertices]

    # we have to filter for the one with max y value because at the boundary there could be two on top of each other
    minx = np.argmin(vertices[:, 0])
    minx_candidates = np.where(vertices[:, 0] == vertices[minx, 0])[0]
    if len(minx_candidates) > 1:
        minx = minx_candidates[np.argmax(vertices[minx_candidates, 1])]
    maxx = np.argmax(vertices[:, 0])
    maxx_candidates = np.where(vertices[:, 0] == vertices[maxx, 0])[0]
    if len(maxx_candidates) > 1:
        maxx = maxx_candidates[np.argmax(vertices[maxx_candidates, 1])]


    # TODO: should make sure that, if there are two points with the same x-coordinate, the one with the higher y-coordinate is chosen
    

    if maxx >= minx:
        upper_hull = np.concatenate([vertices[:minx+1], vertices[maxx:]])
        # upper_hull = vertices[minx:maxx]  # np.delete(vertices, np.s_[maxx:minx + 1], axis=0)
    else:
        upper_hull = vertices[maxx:minx + 1]

    if len(upper_hull) == 0:
        print(minx, maxx)

    # Filter vertices to get only the upper hull
    #upper_hull = []
    #for i in range(len(vertices)):
    #    if i == 0 or i == len(vertices) - 1 or vertices[i][1] > vertices[i-1][1]:
    #        upper_hull.append(vertices[i])

    return set(map(tuple, upper_hull))


def count_transitions(points, P, N):
    transitions = 0
    points = np.array(list(points)) # sorted(UCH_det, key=lambda x: x[0])  # Sort points by x-coordinate
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    for i in range(1, len(points)):
        #if tuple(points[i]) in P[0] and tuple(points[i]) in N[0]:
        #    print("P/N")
        #elif tuple(points[i]) in P[0]:
        #    print("P")
        #elif tuple(points[i]) in N[0]:
        #    print("N")
        #else:
        #    print("Unknown")
        if (tuple(points[i-1]) in P[0] and tuple(points[i]) in N[0]) or (tuple(points[i-1]) in N[0] and tuple(points[i]) in P[0]):
            transitions += 1

    return transitions


def plot_points(P, N, l, UCHp=[]):
    Ps = np.array(list(P))
    Ns = np.array(list(N))

    plt.scatter(Ps[:, 0], Ps[:, 1], c='r', marker='o', label="P", alpha=0.5)
    plt.scatter(Ns[:, 0], Ns[:, 1], c='b', marker='o', label="N", alpha=0.5)
    
    if len(UCHp) > 0:
        # short UCHp by x coordinate
        UCHp = np.array(sorted(UCHp, key=lambda x: (x[0], x[1])))
        # plt.plot(UCHp[:, 0], UCHp[:, 1], 'g-')

        plt.scatter(UCHp[:, 0], UCHp[:, 1], facecolors='none', edgecolors='g', label="UCH", s=250)

    plt.legend()

    #plt.xlim(-1, 40)
    #plt.ylim(-30, 1)


    plt.savefig(f"/home/philipp/Desktop/{l}.png")

    plt.show()


def relu(x):
    return np.maximum(0, x)


def sawtooth(x, W1, W2, b1, b2):
    return (W2 @ relu(np.squeeze(W1 * x) + b1) + b2)[0]


def sawtooths(xs, W1, W2, b1, b2):
    #for _ in range(l):
    #    x = sawtooth(x)
    ys = []
    for x in xs:
        ys.append(sawtooth(x, W1, W2, b1, b2))
    return ys


def block(x, W1, W2, b1, b2):
    return W2 @ relu(np.squeeze(W1 * x) + b1) + b2


def blocks(xs, W1, W2, b1, b2):
    ys = []
    for x in xs:
        ys.append(block(x, W1, W2, b1, b2))
    return ys


def layer(xs, W, b):
    ys = []
    for x in xs:
        ys.append(W @ x + b)
    return ys


def plot_layers(W, b, name):
    xs = np.linspace(-1, 1, 1000)
    ys = layer(xs, W, b)
    plt.plot(xs, ys)
    plt.savefig(f"/home/philipp/Desktop/{name}.png")
    plt.show()


def plot_blocks(W1, W2, b1, b2, name):
    xs = np.linspace(0, 1, 1000)
    ys = blocks(xs, W1, W2, b1, b2)
    plt.plot(xs, ys)
    plt.xlim(-0.25, 1.25)
    plt.savefig(f"/home/philipp/Desktop/{name}.png")
    plt.show()


def plot_grid(data, benchmark, L, elements_per_row, name):
    fig, axs = plt.subplots(L // elements_per_row + L % elements_per_row, elements_per_row, figsize=(15, 5 * (L // elements_per_row + L % elements_per_row)))

    for l in range(L):
        s = L - l

        row = l // elements_per_row
        col = l % elements_per_row

        # Plot mean and std of linear regions
        mean_linregs = np.mean(data[l, :, :], axis=0)
        std_linregs = np.std(data[l, :, :], axis=0)

        axs[row, col].plot(range(1, L+1), mean_linregs, label=f'Mean {name}')
        axs[row, col].fill_between(range(1, L+1), mean_linregs - std_linregs, mean_linregs + std_linregs, alpha=0.2, label='Uncertainty')
        axs[row, col].plot(range(1, L+1), benchmark, color='r', linestyle='dashed', linewidth=2, label='Deterministic')
        # axs[row, col].set_title(f'Linear Regions: {l} deterministic blocks, {s} random blocks')
        if row == 0 and col == 0:
            axs[row, col].set_xlabel('Blocks')
            axs[row, col].set_ylabel(name)
        axs[row, col].set_yscale('log')
        axs[row, col].legend()

    plt.tight_layout()

    return fig


def plot_sawtooths(L, R):
    # ys = []
    xs = np.linspace(-1, 1, 500)
    ys = xs

    W1 = np.array([[1], [1]])
    W2 = np.array([[2, -4]])
    b1 = np.array([0., -1/2])
    b2 = np.array([0.])

    W1 = np.array([[-3], [3]])
    W2 = np.array([[-1, -1]])
    b1 = np.array([-1, -2])
    b2 = np.array([1])

    ## random matrices
    #W1 = np.random.randn(2, 1)
    #W2 = np.random.randn(1, 2)
    #b1 = np.random.randn(2)

    for _ in range(L):
        ys = sawtooths(ys, W1, W2, b1, b2)

    for _ in range(R):
        W1 = np.random.randn(2, 1)
        W2 = np.random.randn(1, 2)
        b1 = np.random.randn(2)

        ys = sawtooths(ys, W1, W2, b1)

    plt.plot(xs, ys)
    plt.show()


def generate_points_on_upper_hemisphere(num_points):
    n = 2*num_points + 2
    # angles = np.linspace(0, np.pi/2, num_points, endpoint=True)
    angles = np.linspace(0, np.pi/2, n, endpoint=True)
    # angles = np.random.uniform(0, np.pi/2, num_points)
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.vstack((x, y)).T

    N = [{tuple(p) for p in points[0::2]}]
    P = [{tuple(n) for n in points[1::2]}]

    return P, N


def plot_function(P, N, name):
    x = np.linspace(-0.5, 1.5, 2000)
    y = np.array([Q(P, xi) for xi in x]) - np.array([Q(N, xi) for xi in x])
    plt.plot(x, y)
    plt.savefig(f"/home/philipp/Desktop/{name}.png")
    plt.show()


def Q(S, x):
    """
    S is a set of points (a,b) where each such point represents a function ax + b. Q(S) returns the max over all of those functions evaluated at x.
    """

    return max([s[0] * x + s[1] for s in S])



