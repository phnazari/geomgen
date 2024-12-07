import numpy as np
from matplotlib import pyplot as plt


def tt_upper_convex_hull(P, N):
    P_upper = []
    N_upper = []
    transitions = []
    for p,n in zip(P, N):
        p_upper, n_upper, t =_upper_convex_hull(p | n)
        P_upper.append(p_upper)
        N_upper.append(n_upper)
        transitions.append(t)

    return P_upper, N_upper, transitions

def upper_convex_hull(S):
    S_upper = []
    for s in S:
        s_upper = _upper_convex_hull(s)
        S_upper.append(s_upper)

    return S_upper

def _upper_convex_hull(S):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    points = np.array(list(S))

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = np.unique(points, axis=0)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return S

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()

        if len(upper) < 2 or (len(upper) >= 2 and not (p[0] == upper[-1][0] and p[1] < upper[-1][1])):
            upper.append(p)

    upper = np.stack(upper)

    #lower = []
    #for p in points:
    #    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
    #        lower.pop()
    #    lower.append(p)
    #if len(lower) > 0:
    #    print("\n")
    #    print(points)
    #    print(upper)
    #    print(lower)
    #    print("\n")
    #    upper = upper[:-1]


    # TODO: we should only remove one if there is also a lower part, right? Should understand the algorithm!!!
    # upper = np.stack(upper)[:-1]

    #point_source = []
    #P_upper = []
    #N_upper = []
    #for p in upper:
    #    in_P = tuple(p) in P
    #    in_N = tuple(p) in N
    #    #in_P = any(tuple(p) in points_set for points_set in P)
    #    #in_N = any(tuple(p) in points_set for points_set in N)
    #    if in_P and in_N:
    #        point_source.append('P/N')
    #    elif in_P:
    #        point_source.append('P')
    #        P_upper.append(p)
    #    elif in_N:
    #        point_source.append('N')
    #        N_upper.append(p)
    #    else:
    #        point_source.append('Unknown')
    
    #P_upper = np.array(P_upper)
    #N_upper = np.array(N_upper)

    #transitions = count_transitions(point_source)

    # print("UCH sources:", point_source)
    #print("Number of N to P transitions:", transitions)
    #print("Size of UCH:", len(upper))
    #print("Total numer of points:", len(points))
    #print("Ratio of upper points:", len(upper) / len(points))
    #print("\n")

    # convert P_upper into a set of tuples
    #P_upper = set(map(tuple, P_upper))
    #N_upper = set(map(tuple, N_upper))
    S_upper = set(map(tuple, upper))

    return S_upper


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
    Ps = np.array(list(P[0]))
    Ns = np.array(list(N[0]))

    plt.scatter(Ps[:, 0], Ps[:, 1], c='r', marker='o', label="P")
    plt.scatter(Ns[:, 0], Ns[:, 1], c='b', marker='o', label="N")
    
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

def sawtooth(x, W1, W2, b1):
    return (relu(W2 @ relu(np.squeeze(W1 * x) + b1)))[0]

def sawtooths(x, l):
    for _ in range(l):
        x = sawtooth(x)
    return x


def plot_sawtooths(l):
    ys = []
    xs = np.linspace(0, 1, 100)
    for x in xs:
        ys.append(sawtooths(x, l))
        #exit()
        #print("\n\n")
    plt.plot(xs, ys)
    plt.show()


def generate_points_on_upper_hemisphere(num_points):
    angles = np.linspace(0, np.pi, num_points, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.vstack((x, y)).T

    P = [{tuple(p) for p in points[0::2]}]
    N = [{tuple(n) for n in points[1::2]}]

    return P, N


