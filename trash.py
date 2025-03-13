class NN:
    def __init__(self, l, s):
        self.P = [{(1., 0.)}]
        self.N = [set()]

        self.l = l
        self.s = s

    def forward(self):
        i = 0
        for i in range(self.l):
            print(f"Det Iteration {i + 1} of {self.s + self.l}")
            W1 = np.array([[1.], [1.]])
            W2 = np.array([[2., -4.]])
            b1 = np.array([0., -1/2])
            b2 = np.array([0.])
            t1 = 0
            t2 = -np.inf
            self.P, self.N = detblock(self.P, self.N, W1, W2, b1, b2, t1, t2)

        if self.l > 0 and self.s == 0:
            self.P, self.N = shiftblock(self.P, self.N, random=False)

        j = 0
        for j in range(self.s):
            #W1 = np.random.randn(2, 1)
            #W2 = np.random.randn(1, 2)
            #b1 = np.random.randn(2)
            W1 = np.random.randint(-10, 11, size=(2, 1))
            W2 = np.random.randint(-10, 11, size=(1, 2))
            b1 = np.random.randint(-10, 11, size=2)
            b2 = np.array([0.])  # np.random.randn(1)
            t1 = 0
            t2 = -np.inf
            P_old = deepcopy(self.P)
            N_old = deepcopy(self.N)
            print(f"Rand Iteration {i + j + 1} of {self.s + self.l}")
            self.P, self.N = detblock(self.P, self.N, W1, W2, b1, b2, t1, t2)

            linregs = len(upper_convex_hull([setsum(self.P[0], self.N[0])])[0])

            if linregs >194 and self.l == 6 or False:
                print(f"l={self.l}, s={self.s}, linregs={linregs}")
                print(upper_convex_hull([setsum(self.P[0], self.N[0])])[0])
                print(len(setsum(self.P[0], self.N[0])))
                print(len(upper_convex_hull([setsum(self.P[0], self.N[0])])[0]))
                print(W1, W2, b1, b2)
                plot_function(P_old[0], N_old[0], "one")
                plot_function(self.P[0], self.N[0], "two")
                plot_blocks(W1, W2, b1, b2, "three")

                # plot setsum(self.P[0], self.N[0]) and its UCH
                PN = setsum(self.P[0], self.N[0])
                UCH = upper_convex_hull([PN])[0]
                plot_points([PN], [PN], 0, np.array(list(UCH)))

                #for x in xs:
                #    ys.append(sawtooths(x, l))  

        if self.s > 0:
            self.P, self.N = shiftblock(self.P, self.N, random=True)

        return self.P, self.N
    
    def evaluate(self):
        self.P, self.N = self.forward()
        PN = self.P[0] | self.N[0]
        self.UCH = upper_convex_hull([PN])[0]
        self.transitions = count_transitions(self.UCH, self.P, self.N)
        self.linregs = len(upper_convex_hull([setsum(self.P[0], self.N[0])])[0])

        print(f"Number of transitions: {self.transitions}")
        print(f"Size of UCH: {len(self.UCH)}")
        print(f"Number of linear pieces: {self.linregs}")


    def plot(self):
        plot_points(self.P, self.N, self.l, np.array(list(self.UCH)))

    @property
    def numpoints(self):
        return len(self.P[0]) + len(self.N[0])

def detblock(P, N, W1, W2, b1, b2, t1, t2):
    #if not random:
    #    W1 = np.array([[1], [1]])
    #    W2 = np.array([[2, -4]])
    #    b1 = np.array([0., -1/2])
    #    b2 = np.array([0.])
    #    t1 = 0
    #    t2 = -np.inf
    #else:
    #    W1 = np.random.randn(2, 1)
    #    W2 = np.random.randn(1, 2)
    #    b1 = np.random.randn(2)
    #    b2 = np.array([0.])  # np.random.randn(1)
    #    t1 = 0
    #    t2 = -np.inf

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


    if R > 0 and False:
        for l in range(L):
            s = L - l

            plt.figure()

            plt.hist(results_UCHp[l, :], label="random")  # , bins=range(int(results_UCHp.min()), int(results_UCHp.max()) + 1))
            plt.axvline(x=nn_det.numpoints, color='r', linestyle='dashed', linewidth=2, label='det')
            plt.title(f'UCH: {l} deterministic blocks, {s} random blocks blocks and {R} repetitions')
            plt.xlabel('Number of Transitions')
            plt.ylabel('Frequency')
            plt.legend()
            output_dir = f"/home/philipp/Desktop/plots/{L}_{R}/uch"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(os.path.join(output_dir, f"histogram_{l}_{s}_{R}.png"))

            plt.figure()

            plt.hist(results_transitions[l, :], label="random")
            plt.axvline(x=nn_det.transitions, color='r', linestyle='dashed', linewidth=2, label='det')
            plt.title(f'Transitions: {l} deterministic blocks, {s} random blocks and {R} repetitions')
            plt.xlabel('Number of Transitions')
            plt.ylabel('Frequency')
            plt.legend()
            output_dir = f"/home/philipp/Desktop/plots/{L}_{R}/transitions"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(os.path.join(output_dir, f"histogram_{l}_{s}_{R}.png"))


            plt.figure()

            plt.hist(results_linregs[l, :], label="random")
            plt.axvline(x=nn_det.linregs, color='r', linestyle='dashed', linewidth=2, label='det')
            plt.title(f'Linear Regions: {l} deterministic blocks, {s} random blocks and {R} repetitions')
            plt.xlabel('Number of Linear Regions')
            plt.ylabel('Frequency')
            plt.legend()
            output_dir = f"/home/philipp/Desktop/plots/{L}_{R}/linregs"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(os.path.join(output_dir, f"histogram_{l}_{s}_{R}.png"))


    H = 0

# plot_sawtooths(L, H)

j = np.random.randint(0, 3**(L-1) + 1)

#test = [j - 2*a for a in range(3**(L-1) + 1)]
#print(test)
#exit()

points = []
for a in range(3**(L-1) + 1):
    for b in range(3**(L-1) + 1):
        if 2*a + b == j:
            points.append((a, b))
    #b = j - a
    #if 0 <= b <= 3**(L-1):
    #    points.append((a, b))

#print(j)
#print(points)

test2 = [(a, j-2*a) for a in range(max(0, int(np.ceil((j-3**(L-1))/2))), min(int(np.floor(j/2)), 3**(L-1))+1)]
#print(j)
#print(points)
#print(test2)
#exit()

A = np.array(range(3**(L-1) + 1))
B = 2*np.array(range(3**(L-1) + 1))

A = 8*np.array(range(10))
B = 4*np.array(range(10))

C = np.sort(np.array([a + b for a in A for b in B]))
##print(C)
##print(3**L)

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


# dd = -2*fn(L-1) + 4*chi[L] -2 + 2**(L+1)


# #print(fn(L-1) + chi[L-1])
##print(dd)
##print(-fn(L))
##print(dd+fn(L))


D = 4*np.array(range(gn(L-1) + 1))
E = np.array(range(3**(L-1) + 1))

F = np.sort(np.unique(np.array([d + e + 1 for d in D for e in E])))

#print(F)
#print(3**L)

A = np.array(range(3**(L-1) + 1))
B = np.array(range(gn(L-1) + 1))

C = np.unique(np.sort(np.array([a + b for a in A for b in B])))

#print(C)
#print(gn(L))

# #print(F)


def squareblock(P, N, W=2, random=True, lin=False):
    W5 = np.random.randn(W, W)
    b5 = np.random.randn(W)
    if lin == True:
        t5 = -np.inf
    else:
        t5 = 0
    #P, N = iterate_points(P, N, W4, b4, t4)
    #P, N = double_uch(P, N)
    P, N = iterate_points(P, N, W5, b5, t5)
    P, N = double_uch(P, N)

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
    b1 = np.random.randn(1)
    t1 = -np.inf

    P, N = iterate_points(P, N, W1, b1, t1)
    # P, N = double_uch(P, N)

    return P, N



def half_circle(R, plot=False, size=10, L=1):
    results_linregs = np.zeros((L, R, L))

    for r in range(R):
        for l in range(L):
            print(f"iteration r={r}, l={l}")
            P, N = generate_points_on_upper_hemisphere(size)
            nn = ReLuNet(l+1, P=P, N=N)
            nn.evaluate(all_layers=True)
            results_linregs[l, r, :l+1] = nn.linregs

    P, N = generate_points_on_upper_hemisphere(size)
    nn_zero = ReLuNet(0, P=P, N=N)
    nn_zero.evaluate()
    linregs_zero = nn_zero.linregs

    
    #linregs_mean = np.mean(results_linregs, axis=1)
    #linregs_std = np.std(results_linregs, axis=1)

    fig = plt.figure()

    for l in range(L):
        mean_linreg = np.mean(results_linregs[l, :, :l+1], axis=0)
        std_linreg = np.std(results_linregs[l, :, :l+1], axis=0)
        s = L - l
        print(l)
        print(results_linregs[l, :, :l+1].shape)
        print(mean_linreg)
        print(list(range(1, l+2)))
        plt.plot(range(1, l+2), mean_linreg, label=f'Mean {l} deterministic blocks, {s} random blocks')
        plt.fill_between(range(1, l+2), mean_linreg - std_linreg, mean_linreg + std_linreg, alpha=0.2, label='Uncertainty')
        # plt.plot(range(1, L+1), linregs_zero, color='r', linestyle='dashed', linewidth=2, label='Deterministic')
        plt.title(f'Linear Regions: {l} deterministic blocks, {s} random blocks')
        plt.xlabel('Layer')
        plt.ylabel('# Linear Regions')
        # plt.yscale('log')
        #plt.legend()

    if plot:
        plt.show()


    if last_block and False:
                from matplotlib import pyplot as plt
                P4 = add_bias(setmult(4, P[0]), -2)
                PS = setsum(P4, setmult(2, N[0]))
                N4 = setmult(4, N[0])
                N2 = setmult(2, N[0])
                NS = setsum(N4, N2)

                N1 = setsum(setmult(4, N[0]), setmult(2, N[0]))
                N2 = add_bias(setsum(setmult(4, N[0]), setmult(2, N[0])), -2)
                N3 = add_bias(setsum(setmult(4, P[0].symmetric_difference(N[0])), setmult(2, N[0])), -2)


                #S1 = setsum(setsum(setmult(4, N[0]), setmult(2, N[0])), setmult(2, N[0]))
                #S2 = setsum(setsum(setmult(4, N[0]), setmult(2, N[0])), setmult(2, N[0].symmetric_difference(P[0])))
                #test = setsum(setsum(setmult(4, N[0]), setmult(2, N[0])), setmult(2, P[0]))

                S1 = setsum(setmult(4, N[0]), setmult(2, N[0]))
                S2 = setsum(setmult(4, N[0]), setmult(2, N[0].symmetric_difference(P[0])))

                S3 = P_new[0].symmetric_difference(N_new[0])
                #print(S3 == S2)

                N_new4 = setmult(4, N_new[0])
                N_new2 = setmult(2, N_new[0])
                NS_new = setsum(N_new4, N_new2)
                P_new4 = add_bias(setmult(4, P_new[0]), -2)
                PS_new = setsum(P_new4, setmult(2, N_new[0]))

                #plt.scatter(*zip(*P[0]), alpha=0.4, label="P")
                #plt.scatter(*zip(*N[0]), alpha=0.4, label="N")
                #plt.scatter(*zip(*N1), alpha=0.4, label="N1")
                #plt.scatter(*zip(*N2), alpha=0.4, label="N2")
                #plt.scatter(*zip(*N3.difference(N1 | N2)), alpha=0.4, label="N3")
                #plt.scatter(*zip(*N3), alpha=0.4, label="N3")

                plt.scatter(*zip(*S1), alpha=0.4, label="S1")
                plt.scatter(*zip(*S2), alpha=0.4, label="S2")

                intersection = N_new[0].intersection(S1)
                #print("Intersection of N_new[0] and S1:", intersection)

                intersection2 = N_new[0].intersection(S2)
                #print("Intersection of N_new[0] and S2:", intersection2)

                # print(S1.issubset(N_new[0]))

                plt.scatter(*zip(*N_new[0]), alpha=0.4, label="N_new")
                #plt.scatter(*zip(*P_new[0]), alpha=0.4, label="P_new")
                # plt.scatter(*zip(*N_new[0]), alpha=0.4, label="N_new")

                # plt.scatter(*zip(*N3), alpha=0.4, label="N3")
                # plt.scatter(*zip(*(PS.union(NS))), alpha=0.4, label="NS_new")
                # plt.scatter(*zip(*PS), alpha=0.4)

                #plt.scatter(*zip(*NS_new), alpha=0.4)
                # plt.scatter(*zip(*PS_new), alpha=0.4)

                plt.legend()
                plt.show()
                #plt.scatter(*zip(*N4))
                #plt.scatter(*zip(*NS))
                #plt.show()

            #if not last_block:
            #    P_new, N_new = double_uch(P_new, N_new)


        # test if my calculations are correct
        if False:
            N_new_test = add_bias(setsum(setmult(4, N[0]), setmult(2, N[0])), -2).union(setsum(setsum(setmult(4, N[0]), setmult(2, N[0]))), {(0,0)}).union(add_bias(setsum(setmult(4, P[0].symmetric_difference(N[0])), setmult(2, N[0])), -2))

            P_new_test = setsum(setmult(4, N[0]), setmult(2, P[0])).union(setsum(setmult(4, N[0]), setsum(setmult(2, N[0]), {(0, 0)}))).union(N_new_test)

            N_new_test = [N_new_test]
            P_new_test = [P_new_test]

            P_new_test, N_new_test = double_uch(P_new, N_new)

            print(P_new[0] == P_new_test[0])
            print(N_new[0] == N_new_test[0])