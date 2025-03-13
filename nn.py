from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np

from tropical import add_bias, iterate_points, setmult, setsum
from util import count_transitions, double_uch, plot_blocks, plot_function, plot_points, upper_convex_hull


class Block:
    def __init__(self, random=True, lin=False, reduce=True):
        self.reduce = reduce
        self.random = random
        self.lin = lin

        if not random:
            self.W1 = np.array([[1.], [1.]])
            self.W2 = np.array([[2., -4.]])
            self.b1 = np.array([0., -1/2])
            self.b2 = np.array([0.])
            self.t1 = 0
            self.t2 = 0
        else:
            self.W1 = np.random.randn(2, 1)
            self.W2 = np.random.randn(1, 2)
            self.b1 = np.random.randn(2)
            self.b2 = np.random.randn(1)
            self.t1 = 0
            self.t2 = -np.inf


    # TODO: use the closed form calculation from my notes
    def __call__(self, P, N):
        P_new, N_new = iterate_points(P, N, self.W1, self.b1, self.t1)
        if self.reduce:
            P_new, N_new = double_uch(P_new, N_new)
        
        P_new, N_new = iterate_points(P_new, N_new, self.W2, self.b2, self.t2)
        if N_new[0] in P_new[0]:
            print("N_new[0] is in P_new[0]")
        if self.reduce:
            P_new, N_new = double_uch(P_new, N_new)

        return P_new, N_new
    

class Layer:
    def __init__(self, input_dim=2, output_dim=1, lin=False, W=None, b=None, reduce=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reduce = reduce

        self.W = W if W is not None else np.random.randn(output_dim, input_dim)
        self.b = b if b is not None else np.random.randn(output_dim)
        self.t = -np.inf if lin else 0

    def __call__(self, P, N):
        P_new, N_new = iterate_points(P, N, self.W, self.b, self.t)
        if self.reduce:
            P_new, N_new = double_uch(P_new, N_new)

        return P_new, N_new


class Shift:
    def __init__(self, input_dim=2, output_dim=1, reduce=True, b=None, lin=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reduce = reduce

        self.W = np.array([[1.]])
        self.b = b if b is not None else np.random.randn(output_dim)
        self.t = -np.inf if lin else 0

    def __call__(self, P, N):
        P, N = iterate_points(P, N, self.W, self.b, self.t)

        if self.reduce:
            P, N = double_uch(P, N)

        return P, N


class SawtoothNetwork:
    def __init__(self, l, s, shift=True):
        self.P = [{(1., 0.)}]
        self.N = [set()]

        self.Ps = []
        self.Ns = []

        self.l = l
        self.s = s
        self.shift = shift

        #self.firstlayer = Layer(input_dim=1, output_dim=2, lin=False, W=np.array([[1.], [1.]]), b=np.array([0., -1/2]))
        #self.middlelayers = [Layer(input_dim=2, output_dim=2, lin=False, W=np.array([[2, -4], [2,-4]]), b=np.array([0, -1/2.])) for _ in range(l-1)]
        #self.lastlayer = Layer(input_dim=2, output_dim=1, lin=True, W=np.array([[2., -4.]]), b=np.array([0.]))
        #self.layers = [self.firstlayer] + self.middlelayers + [self.lastlayer]

        self.blocks = [Block(random=False) for _ in range(l)]

        #if shift and l > 0:
        #    self.blocks = self.blocks + [Shift(b=np.array([-1/2]))]

        if s > 0:
            self.blocks = self.blocks + [Block(random=True) for _ in range(s)]

        if s < 0:
            self.blocks = self.blocks + [Layer(input_dim=1, output_dim=1, lin=False)]

    def __call__(self, all_layers=False):
        for block in self.blocks:

            #if self.s == 0 and len(self.N[0])>0:
            #    print(self.l, self.P[0], self.N[0])
            #    plot_points(self.P[0], self.N[0], self.l)

            self.P, self.N = block(self.P, self.N)

            if all_layers and not isinstance(block, Shift):
                self.Ps.append(self.P)
                self.Ns.append(self.N)

        self.Ps = np.array(self.Ps)
        self.Ns = np.array(self.Ns)

    def linregs(self, all_layers=False):
        if not all_layers:
            return len(upper_convex_hull([setsum(self.P[0], self.N[0])])[0])
        else:
            return [len(upper_convex_hull([setsum(P[0], N[0])])[0]) for P, N in zip(self.Ps, self.Ns)]

    def transitions(self, all_layers=False):
        if not all_layers:
            return count_transitions(upper_convex_hull([self.P[0] | self.N[0]])[0], self.P, self.N)
        else:
            return [count_transitions(upper_convex_hull([P[0] | N[0]])[0], P, N) for P, N in zip(self.Ps, self.Ns)]

    """
    def evaluate(self, all_layers=False):
        self.__call__(all_layers=all_layers)

        if not all_layers:
            PN = self.P[0] | self.N[0]
            self.UCH = upper_convex_hull([PN])[0]
            self.transitions = count_transitions(self.UCH, self.P, self.N)
            self.linregs = len(upper_convex_hull([setsum(self.P[0], self.N[0])])[0])
        else:
            PN = [P[0] | N[0] for P, N in zip(self.Ps, self.Ns)]
            self.UCH = [upper_convex_hull([pn])[0] for pn in PN]
            self.transitions = [count_transitions(uch, P, N) for uch, P, N in zip(self.UCH, self.Ps, self.Ns)]
            self.linregs = [len(upper_convex_hull([setsum(P[0], N[0])])[0]) for P, N in zip(self.Ps, self.Ns)]
    """

    def plot(self):
        plot_points(self.P[0], self.N[0], self.l, upper_convex_hull([self.P[0] | self.N[0]])[0])

    @property
    def numpoints(self):
        return len(self.P[0]) + len(self.N[0])


class ReLuNet:
    def __init__(self, L, P=[{(1, 0)}], N=[set()], input_dim=1, output_dim=1):
        self.P = P
        self.L = L

        self.Ps = []
        self.Ns = []

        self.N = N

        self.layers = [Layer(input_dim=input_dim, output_dim=output_dim) for _ in range(L)]


    def __call__(self, all_layers=False):
        for layer in self.layers:
            self.P, self.N = layer(self.P, self.N)

            if all_layers:
                self.Ps.append(self.P)
                self.Ns.append(self.N)


    def evaluate(self, all_layers=False):
        self.__call__(all_layers=all_layers)

        if not all_layers:
            PN = self.P[0] | self.N[0]
            self.UCH_union = upper_convex_hull([PN])[0]
            self.transitions = count_transitions(self.UCH_union, self.P, self.N)
            self.linregs = len(upper_convex_hull([setsum(self.P[0], self.N[0])])[0])
        else:
            PN = [P[0] | N[0] for P, N in zip(self.Ps, self.Ns)]
            self.UCH_union = [upper_convex_hull([pn])[0] for pn in PN]
            self.transitions = [count_transitions(uch, P, N) for uch, P, N in zip(self.UCH_union, self.Ps, self.Ns)]
            self.linregs = [len(upper_convex_hull([setsum(P[0], N[0])])[0]) for P, N in zip(self.Ps, self.Ns)]

    def plot(self):
        plot_points(self.P, self.N, self.L, np.array(list(self.UCH_union)))
