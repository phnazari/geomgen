import numpy as np

from tropical import iterate_points
from util import upper_convex_hull



def detblock(P, N, random=False):
    if not random:
        W1 = np.array([[1], [1]])
        W2 = np.array([[2, -4]])
        b1 = np.array([0., -1/2])
        b2 = np.array([0.])
        t1 = 0
        t2 = 0
    else:
        W1 = np.random.randn(2, 1)
        W2 = np.random.randn(1, 2)
        b1 = np.random.randn(2)
        b2 = np.random.randn(1)
        t1 = 0
        t2 = 0

    P, N = iterate_points(P, N, W1, b1, t1)
    P, N = iterate_points(P, N, W2, b2, t2)

    P = upper_convex_hull(P)
    N = upper_convex_hull(N)

    return P, N


def shiftblock(P, N, random=False):
    if not random:
        W3 = np.array([[1.]])
        b3 = np.array([-1/2])
        t3 = -np.inf
    else:
        W3 = np.random.randn(1, 1)
        b3 = np.random.randn(1)
        t3 = -np.inf

    P, N = iterate_points(P, N, W3, b3, t3)

    P = upper_convex_hull(P)
    N = upper_convex_hull(N)

    return P, N


class NN:
    def __init__(self, l, s):
        self.P = [{(1, 0)}]
        self.N = [set()]

        self.l = l
        self.s = s

    def forward(self):
        i = 0
        for i in range(self.l):
            print(f"Iteration {i + 1} of {self.s + self.l}")
            self.P, self.N = detblock(self.P, self.N, random=False)

        self.P, self.N = shiftblock(self.P, self.N, random=False)

        j = 0
        for j in range(self.s):
            print(f"Iteration {i + j + 1} of {self.s + self.l}")
            self.P, self.N = detblock(self.P, self.N, random=True)

        if self.s > 0:
            self.P, self.N = shiftblock(self.P, self.N, random=True)

        return self.P, self.N
