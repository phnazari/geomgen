import numpy as np

from tropical import iterate_points
from util import count_transitions, plot_points, upper_convex_hull



def double_uch(P, N):
    return upper_convex_hull(P), upper_convex_hull(N)


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
    P, N = double_uch(P, N)
    P, N = iterate_points(P, N, W2, b2, t2)
    P, N = double_uch(P, N)

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
    P, N = double_uch(P, N)

    #P = upper_convex_hull(P)
    #N = upper_convex_hull(N)

    return P, N


def squareblock(P, N, W=2, random=True):
    W4 = np.random.randn(W, W)
    b4 = np.random.randn(W)
    t4 = 0
    W5 = np.random.randn(W, W)
    b5 = np.random.randn(W)
    t5 = 0

    P, N = iterate_points(P, N, W4, b4, t4)
    P, N = double_uch(P, N)
    P, N = iterate_points(P, N, W5, b5, t5)
    P, N = double_uch(P, N)

    #P = upper_convex_hull(P)
    #N = upper_convex_hull(N)

    return P, N

def scaleupblock(P, N, W=2, random=True):
    W1 = np.random.randn(W, 1)
    b1 = np.random.randn(W)
    t1 = -np.inf

    P, N = iterate_points(P, N, W1, b1, t1)
    P, N = double_uch(P, N)

    return P, N

def scaledownblock(P, N, W=2, random=True):
    W1 = np.random.randn(1, W)
    b1 = np.array([0.])# np.random.randn(1)
    t1 = -np.inf

    print(b1)

    P, N = iterate_points(P, N, W1, b1, t1)
    P, N = double_uch(P, N)

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

        if self.s > 0:
            self.P, self.N = shiftblock(self.P, self.N, random=False)

        j = 0
        for j in range(self.s):
            print(f"Iteration {i + j + 1} of {self.s + self.l}")
            self.P, self.N = detblock(self.P, self.N, random=True)

        if self.s > 0:
            self.P, self.N = shiftblock(self.P, self.N, random=True)

        return self.P, self.N
    

class RandNN:
    def __init__(self, L, W=2, P=[{(1, 0)}], N=[set()]):
        self.P = P
        self.N = N
        self.W = W

        # self.P = [{(1, 0)}]
        # self.N = [set()]

        self.L = L

    def forward(self):
        if self.L > 0:
            self.P, self.N = scaleupblock(self.P, self.N, W=self.W, random=True)

        for i in range(self.L):
            print(f"Iteration {i + 1} of {self.L}")
            self.P, self.N = squareblock(self.P, self.N, W=self.W, random=True)

        if self.L > 0:
            self.P, self.N = scaledownblock(self.P, self.N, W=self.W, random=True)

        return self.P, self.N
    
    def evaluate(self):
        self.P, self.N = self.forward()
        PN = self.P[0] | self.N[0]
        self.UCH = upper_convex_hull([PN])[0]
        self.transitions = count_transitions(self.UCH, self.P, self.N)

        print(f"Number of transitions: {self.transitions}")
        print(f"Size of UCH: {len(self.UCH)}")

    def plot(self):
        plot_points(self.P, self.N, self.L, np.array(list(self.UCH)))
