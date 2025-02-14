from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np

from tropical import add_bias, iterate_points, setmult, setsum
from util import count_transitions, plot_blocks, plot_function, plot_points, upper_convex_hull


class Block:
    def __init__(self, random=False):
        self.random = random
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
    def __call__(self, P, N, last_block=False):
        if self.random:
            P_new, N_new = iterate_points(P, N, self.W1, self.b1, self.t1)
            P_new, N_new = double_uch(P_new, N_new)
            P_new, N_new = iterate_points(P_new, N_new, self.W2, self.b2, self.t2)
            P_new, N_new = double_uch(P_new, N_new)
        else:
            N_new = add_bias(setsum(setmult(4, N[0]), setmult(2, N[0])), -2).union(setsum(setsum(setmult(4, N[0]), setmult(2, N[0]))), {(0,0)}).union(add_bias(setsum(setmult(4, P[0].symmetric_difference(N[0])), setmult(2, N[0])), -2))

            P_new = setsum(setmult(4, N[0]), setmult(2, P[0])).union(setsum(setmult(4, N[0]), setsum(setmult(2, N[0]), {(0, 0)}))).union(N_new)

            N_new = [N_new]
            P_new = [P_new]

            P_new, N_new = double_uch(P_new, N_new)


        return P_new, N_new
    

class Layer:
    def __init__(self, input_dim=2, output_dim=1, lin=True, random=False):
        self.input_dim = input_dim
        self.output_dim = output_dim

        if random:
            self.W = np.random.randn(output_dim, input_dim)
            self.b = np.random.randn(output_dim)
            # self.b = np.array([0.])
            if lin:
                self.t = -np.inf
            else:
                self.t = 0
        else:
            self.W = np.array([[1.]])
            self.b = np.array([-1/2])
            if lin:
                self.t = -np.inf
            else:
                self.t = 0

    def __call__(self, P, N):
        P_new, N_new = iterate_points(P, N, self.W, self.b, self.t)
        P_new, N_new = double_uch(P_new, N_new)

        return P_new, N_new


class Shift:
    def __init__(self, input_dim=2, output_dim=1, random=False):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = np.array([[1.]])

        if random:
            self.b = np.random.randn(output_dim)
            self.t = -np.inf
        else:
            self.b = np.array([-1/2])
            self.t = -np.inf

    def __call__(self, P, N):
        P, N = iterate_points(P, N, self.W, self.b, self.t)
        P, N = double_uch(P, N)

        return P, N


def double_uch(P, N):
    return upper_convex_hull(P), upper_convex_hull(N)


class SawtoothNetwork:
    def __init__(self, l, s, shift=True, randomblocks=True):
        self.P = [{(1., 0.)}]
        self.N = [set()]

        self.Ps = []
        self.Ns = []

        self.l = l
        self.s = s
        self.shift = shift
        self.blocks = [Block(random=False) for _ in range(l)]

        if shift and l > 0:
            self.blocks = self.blocks + [Shift(random=False)]

        if s > 0:
            if randomblocks:
                self.blocks = self.blocks + [Block(random=True) for _ in range(s)]
            else:
                self.blocks = self.blocks + [Layer(input_dim=1, output_dim=1, random=True, lin=False) for _ in range(s)]

            #if shift:
            #    self.blocks = self.blocks + [Shift(random=True)]

    def __call__(self, all_layers=False):
        for i, block in enumerate(self.blocks):
            # last_block = i == len(self.blocks) - 1
            self.P, self.N = block(self.P, self.N)  # , last_block=last_block

            if all_layers and not isinstance(block, Shift):
                self.Ps.append(self.P)
                self.Ns.append(self.N)

        self.Ps = np.array(self.Ps)
        self.Ns = np.array(self.Ns)

        #print(f"N: {self.N[0]}")
        #print(f"N Delta P: {self.P[0].symmetric_difference(self.N[0])}")

        if False:
            from matplotlib import pyplot as plt
            sum = setsum(self.P[0], self.N[0])
            uch_sum = upper_convex_hull([sum])[0]
            plt.scatter(*zip(*sum))
            plt.scatter(*zip(*uch_sum), facecolors='none', edgecolors='g', label="UCH", s=250)
            plt.show()

        if False:


            # diff = self.N[0].symmetric_difference(self.P[0])
            diff = self.N[0] #  self.P[0].symmetric_difference(self.N[0])

            # x_values = (np.unique([x for x, y in diff]) - 2**self.l)/(2**(self.l+2))
            x_values = np.unique([x for x, y in diff])
            #x_values = np.unique([x for x, y in diff])/(2**(self.l+1))
            print(f"Unique x-values for points: {x_values}")

            #from matplotlib import pyplot as plt
            #plt.scatter(*zip(*diff))
            #plt.show()
            tests = []
            slj = []
            for x in x_values:
                y_values = np.sort(np.array([y for x_, y in diff if x_ == x]))
                test2 = (np.max(y_values) - np.min(y_values))/(y_values[1]-y_values[0])
                #test2 = np.max(y_values)
                print(f"{x}: {np.max(y_values)} ~ {np.min(y_values)} ({y_values[1]-y_values[0]}) ({(np.max(y_values) - np.min(y_values))/(y_values[1]-y_values[0]) + 1})")
                test = np.max(y_values)
                tests.append(test)
                slj.append(test2)

                #tests.append(test)
                #slj.append(test2)

            print(f"slj: {slj}")

            # test = [4*slj[a] + 2*slj[-2*a] for a in range(0, 3**(self.l-1) + 1)]

            max_tests = []
            for j in range(3**(self.l) + 1):
                test = [4*slj[a] + 2*slj[j-2*a] for a in range(max(0, int(np.ceil((j-3**(self.l-1))/2))), min(int(np.floor(j/2)), 3**(self.l-1))+1)]
                max_tests.append(np.max(test))
            print(f"max_tests: {max_tests}")
            print(f"diff max_test: {np.diff(max_tests)}")
            
            
            diff = self.P[0].symmetric_difference(self.N[0])

            x_values = np.unique([x for x, y in diff])
            print(f"Unique x-values for points: {x_values}")

            zlj = []
            for x in x_values:
                y_values = np.sort(np.array([y for x_, y in diff if x_ == x]))
                test3 = (np.max(y_values) - np.min(y_values))/(y_values[1]-y_values[0])
                zlj.append(test3)

            def f(x):
                return 6*x + 2

            def fn(n):
                x = 0
                for _ in range(n):
                    x = f(x)
                return x                

            def g(x):
                return 3*x + 1
            
            def gn(n):
                x = 0
                for _ in range(n):
                    x = g(x)
                return x

            chi = [-0, -0, -8, -72, -488]

            def generate_P():
                Ptest = set()

                for j in range(gn(self.l-1) + 1):
                    for k in range(int(zlj[j]) + 1):
                        point = (2**self.l + j*2**(self.l+2), chi[self.l-1] - j*2**(self.l+1) + 8*k)
                        Ptest.add(point)

                print(Ptest == self.P[0].symmetric_difference(self.N[0]))
                #from matplotlib import pyplot as plt
                #plt.scatter(*zip(*Ptest))
                #plt.scatter(*zip(*self.P[0]))
                #plt.show()


            def generate_N():
                Ntest = set()

                for j in range(3**(self.l-1) + 1):
                    for k in range(int(slj[j]) + 1):
                        point = (j*2**(self.l+1), -fn(self.l-1) - j*2**self.l + 2*k)
                        Ntest.add(point)


                print(Ntest == self.N[0])
                #from matplotlib import pyplot as plt
                #plt.scatter(*zip(*Ntest))
                #plt.show()

            #generate_N()
            #generate_P()


        #y_values = np.sort(np.array([y for x, y in self.N[0] if x == 64]))
        #print(f"Y-values for points in N[0] with x-value 0: {y_values}")

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

        print(f"Number of transitions: {self.transitions}")
        print(f"Size of UCH: {[len(uch) for uch in self.UCH]}")
        print(f"Number of linear pieces: {self.linregs}")

    def plot(self):
        plot_points(self.P[0], self.N[0], self.l, upper_convex_hull([self.P[0] | self.N[0]])[0])

    @property
    def numpoints(self):
        return len(self.P[0]) + len(self.N[0])



class RandNN:
    def __init__(self, L, W=2, P=[{(1, 0)}], N=[set()], input_dim=1, output_dim=1):
        self.P = P
        self.L = L

        self.Ps = []
        self.Ns = []

        self.N = N
        self.W = W

        self.layers = [Layer(input_dim=input_dim, output_dim=output_dim, lin=False, random=True) for _ in range(L)]


    def __call__(self, all_layers=False):
        if False:
            if self.W > 1:
                if self.L > 0:
                    #plot_points(self.P, self.N, -100)
                    self.P, self.N = scaleupblock(self.P, self.N, W=self.W, random=True)
                    #plot_points(self.P, self.N, -99)

                for i in range(self.L):
                    print(f"Layer {i + 1} of {self.L}")
                    self.P, self.N = squareblock(self.P, self.N, W=self.W, random=True)
                    #plot_points(self.P, self.N, i + 1)
                    #plot_points([self.P[1]], [self.N[1]], i + 1)

                if self.L > 0:
                    self.P, self.N = scaledownblock(self.P, self.N, W=self.W, random=True)
                    #plot_points(self.P, self.N, 99)
            else:
                if self.L > 0:
                    for i in range(self.L):
                        print(f"Layer {i + 1} of {self.L}")
                        self.P, self.N = squareblock(self.P, self.N, W=self.W, random=True, lin=False)  # (i+1 == self.L)
                        plot_points(self.P, self.N, i + 1)


        for i, layer in enumerate(self.layers):
            self.P, self.N = layer(self.P, self.N)

            if False:
                print(i, len(P_test[0]), len(N_test[0]))

                if len(P_test[0]) < 7 or len(N_test[0]) < 7:
                    print(len(self.P[0]), len(self.N[0]))
                    plt.scatter(*zip(*P_test[0]), c='r', marker='o', label="P", alpha=0.5)
                    plt.scatter(*zip(*N_test[0]), c='b', marker='o', label="N", alpha=0.5)
                    plt.legend()
                    plt.show()
                    
                    plt.figure()
                    plt.scatter(*zip(*setsum(P_test[0], N_test[0])))
                    plt.show()

            #self.P = P_test
            #self.N = N_test

            #if i == 80:
            #    plt.scatter(*zip(*self.P[0]), c='r', marker='o', label="P", alpha=0.5)
            #    plt.scatter(*zip(*self.N[0]), c='b', marker='o', label="N", alpha=0.5)
            #    plt.legend()
            #    plt.show()

            if all_layers:  # and (self.shift is False or (self.shift is True and i != len(self.blocks) - 2)):   # for the last block, consider also the shift
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


        #print(f"Number of transitions: {self.transitions}")
        # print(f"Size of UCH: {len(self.UCH)}")
        #print(f"Number of Linear Regions: {self.linregs}")

    def plot(self):
        plot_points(self.P, self.N, self.L, np.array(list(self.UCH_union)))
