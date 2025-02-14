import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Arc

from util import count_transitions, generate_points_on_upper_hemisphere, plot_function, plot_grid, plot_points, upper_convex_hull
from tropical import add_bias, iterate_points, setmult, setsum
from nn import RandNN, SawtoothNetwork
import os

from scipy.stats import norm
from scipy.integrate import quad
import math


def det_rand_net(L, R, plot=False, all_layers=True, randomblocks=True):
    results_transitions = np.zeros((L, R, L))
    results_UCHp = np.zeros((L, R, L))
    results_linregs = np.zeros((L, R, L))
    shift = True

    for r in range(R):
        for l in range(L):
            s = L - l
            print(f"iteration r={r}, l={l}, s={s}")
            nn = SawtoothNetwork(l, s, shift=shift, randomblocks=randomblocks)
            nn.evaluate(all_layers=all_layers)
            results_transitions[l, r] = nn.transitions
            results_transitions[l, r, :l] = 2**np.arange(1, l+1)
            results_UCHp[l, r] = [len(uch) for uch in nn.UCH]
            results_linregs[l, r] = nn.linregs

    print(f"iteration r={1}, l={L}, s={0}")
    nn_det = SawtoothNetwork(L, 0, shift=shift, randomblocks=randomblocks)
    nn_det.evaluate(all_layers=True)

    return results_transitions, results_linregs, nn_det


def sawtooth_plots(L, R, plot=False, randomblocks=True):
    results_transitions, results_linregs, nn_det = det_rand_net(L, R, plot, randomblocks=randomblocks)

    if plot:
        elements_per_row = 3
        fig = plot_grid(results_linregs, nn_det.linregs, L, elements_per_row, "Linear Regions")
        output_dir = f"/home/philipp/Desktop/plots/{L}_{R}/linregs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, f"combined_{L}_{R}.png"))

        fig = plot_grid(results_transitions, np.array(2**np.arange(1, L+1)), L, elements_per_row, "Transitions")

        output_dir = f"/home/philipp/Desktop/plots/{L}_{R}/trans"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, f"combined_{L}_{R}.png"))

        plt.show()

    # nn_det.plot()


def sawtooth_histograms(L, R, plot=False):
    results_transitions, results_linregs, nn_det = det_rand_net(L, R, plot, all_layers=True)

    if plot:
        elements_per_row = 3
        fig, axes = plt.subplots(L // elements_per_row + L % elements_per_row, elements_per_row, figsize=(15, 5 * (L // elements_per_row + L % elements_per_row)))
        fig.suptitle('Histograms of Linear Regions')

        for l in range(L):
            s = L - l
            ax = axes[l // elements_per_row, l % elements_per_row]
            ax.hist(results_linregs[l, :, -1], alpha=0.7, label=f'{l}, {s}')
            ax.axvline(nn_det.linregs[-1], color='r', linestyle='dashed', linewidth=2, label='Deterministic')
            ax.set_title(f'{l} deterministic blocks, {s} random blocks')
            ax.set_xlabel('# Linear Regions')
            ax.set_ylabel('Frequency')
            ax.legend()

        output_dir = f"/home/philipp/Desktop/plots/{L}_{R}/linregs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, f"hist_combined_{L}_{R}.png"))

        if plot:
            plt.show()


def rand_net(L, plot=False):
    nn = RandNN(L)
    nn.evaluate()

    #print(f"Number of transitions: {nn.transitions}")
    #print(f"Size of UCH: {len(nn.UCH)}")

    if plot:
        plot_points(nn.P, nn.N, L, np.array(list(nn.UCH)))


def half_circle(R, plot=False, size=10, L=1):
    results_linregs = np.zeros((R, L))

    for r in range(R):
        print(f"iteration r={r}")
        P, N = generate_points_on_upper_hemisphere(size)
        nn = RandNN(L, P=P, N=N)
        nn.evaluate(all_layers=True)
        results_linregs[r] = nn.linregs

    P, N = generate_points_on_upper_hemisphere(size)
    nn_zero = RandNN(0, P=P, N=N)
    nn_zero.evaluate()
    linregs_zero = nn_zero.linregs
    
    #linregs_mean = np.mean(results_linregs, axis=1)
    #linregs_std = np.std(results_linregs, axis=1)

    fig = plt.figure()

    mean_linreg = np.mean(results_linregs, axis=0)
    std_linreg = np.std(results_linregs, axis=0)
    plt.plot(range(1, L+1), mean_linreg)
    plt.fill_between(range(1, L+1), mean_linreg - std_linreg, mean_linreg + std_linreg, alpha=0.2, label='Uncertainty')
    # plt.plot(range(1, L+1), linregs_zero, color='r', linestyle='dashed', linewidth=2, label='Deterministic')
    plt.plot(range(1, L+1), linregs_zero*np.ones(L), color='r', linestyle='dashed', linewidth=2, label='Deterministic')
    plt.title(f'Linear Regions')
    plt.xlabel('Layer')
    plt.ylabel('# Linear Regions')
    # plt.yscale('log')
    #plt.legend()
    output_dir = f"/home/philipp/Desktop/plots/{L}_{R}/linregs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"half_circle_{size}.png"))
    if plot:
        plt.show()


def test(num_points=5):
    P, N = generate_points_on_upper_hemisphere(num_points)

    S = setsum(P[0], N[0])
    uch = upper_convex_hull([S])[0]
    print(len(uch))

    W = - np.abs(np.random.randn(1)[0])
    b = np.random.randn(1)[0]

    n = num_points

    W = 1
    b = 0.8  # lower_bound(num_points, n)   # upper_bound(num_points, n)

    print(upper_bound(num_points, n))
    print(lower_bound(num_points, n))

    # multiply N by W
    N = setmult(W, N[0])
    # M = add_bias(setmult(W, P[0]), b)
    M = add_bias(N, b)
    O = setmult(W, P[0])

    A = setsum(N, N)
    B = setsum(M, O)
    # B = setsum(add_bias(setmult(W, P[0]), b), N)
    C = A.union(B)

    fig = plt.figure()
    plt.scatter(*zip(*A), alpha=.4, label='A', color='r')
    plt.scatter(*zip(*B), alpha=.4, label='B', color='g')    

    for n in N:
        plt.gca().add_patch(Arc(n, 2*W, 2*W, theta1=0, theta2=90, color='r'))

    for m in M:
        plt.gca().add_patch(Arc(m, 2*W, 2*W, theta1=0, theta2=90, color='g'))

    # plot a circle with radius 2W
    # plt.gca().add_patch(Arc((0, b), 4*W, 4*W, theta1=0, theta2=90))
    # plt.gca().add_patch(Arc((0, 0), 4*W, 4*W, theta1=0, theta2=90))

    # plot upper convex hull using circles around points
    uch = upper_convex_hull([C])[0]
    plt.gca().scatter(*zip(*uch), alpha=.4, label='UCH', color='b')
    print(len(uch))
    plt.axis('equal')
    plt.legend()
    plt.show()


def lower_bound(n, i):
    norm = 2*(2*n+1)
    pihalf = np.pi/2
    Cn = 2*np.sin(pihalf*1/norm)**2
    Dn = np.sin(pihalf*(4*i)/norm)
    return Cn/Dn

def upper_bound(n, i):
    norm = 2*(2*n+1)
    pihalf = np.pi/2
    Cn = 2*np.sin(pihalf*1/norm)**2
    Dn = np.sin(pihalf*(4*i+2)/norm)
    return -Cn/Dn


def half_circle_one_layer(R, plot=False, size=10):
    results_linregs = np.zeros(R)

    for r in range(R):
        print(f"iteration r={r}")
        P, N = generate_points_on_upper_hemisphere(size)

        summed_points = setsum(P[0], N[0])

        uch = upper_convex_hull([summed_points])[0]
        print(f"Number of vertices in UCH: {len(uch)}")


        nn = RandNN(1, P=P, N=N)
        nn.evaluate(all_layers=False)
        results_linregs[r] = nn.linregs

        print(nn.layers[0].b)

        fig = plt.figure()
        plt.scatter(*zip(*nn.P[0]), alpha=.4, label='P')
        plt.scatter(*zip(*nn.N[0]), alpha=.4, label='N')
        plt.axis('equal')
        plt.legend()
        plt.show()

        fig = plt.figure()
        summed_points = setsum(nn.P[0], nn.N[0])
        plt.scatter(*zip(*summed_points), alpha=.4, label='P + N')

        uch = upper_convex_hull([summed_points])[0]
        print(f"Number of vertices in UCH: {len(uch)}")

        plt.scatter(*zip(*uch), alpha=.4, label='UCH')

        plt.axis('equal')
        plt.legend()
        plt.show()

    P, N = generate_points_on_upper_hemisphere(size)
    nn_zero = RandNN(0, P=P, N=N)
    nn_zero.evaluate()
    linregs_zero = nn_zero.linregs

    fig = plt.figure()

    plt.hist(results_linregs, alpha=0.7, label='Random Layer')
    plt.axvline(linregs_zero, color='r', linestyle='dashed', linewidth=2, label='Input Complexity')
    plt.title('Distribution of Linear Regions')
    plt.xlabel('# Linear Regions')
    plt.ylabel('Frequency')
    plt.legend()

    # plt.yscale('log')
    #plt.legend()
    output_dir = f"/home/philipp/Desktop/plots/{1}_{R}/linregs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"half_circle_one_layer_{size}.png"))
    if plot:
        plt.show()


def sawtooth_one_layer(R, L, plot=False):
    results_linregs = np.zeros(R)

    for r in range(R):
        print(f"iteration r={r}")
        nn = SawtoothNetwork(L, 1)
        nn.evaluate(all_layers=False)
        results_linregs[r] = nn.linregs

    nn_zero = RandNN(L +1,0)
    nn_zero.evaluate()
    linregs_zero = nn_zero.linregs

    fig = plt.figure()

    plt.hist(results_linregs, alpha=0.7, label='Random Layer')
    plt.axvline(linregs_zero, color='r', linestyle='dashed', linewidth=2, label='Input Complexity')
    plt.title('Distribution of Linear Regions')
    plt.xlabel('# Linear Regions')
    plt.ylabel('Frequency')
    plt.legend()

    # plt.yscale('log')
    #plt.legend()
    output_dir = f"/home/philipp/Desktop/plots/{1}_{R}/linregs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"half_circle_one_layer_{size}.png"))
    if plot:
        plt.show()


def sawtooth(L, R, plot=False):
    results_linregs = np.zeros((R, L))
    shift = True

    for r in range(R):
        print(f"iteration r={r}")
        nn = SawtoothNetwork(L, 1, shift=shift)
        nn.evaluate(all_layers=True)
        results_linregs[r] = nn.linregs

    nn_zero = SawtoothNetwork(L+1, 0)
    nn_zero.evaluate()
    linregs_zero = nn_zero.linregs
    
    #linregs_mean = np.mean(results_linregs, axis=1)
    #linregs_std = np.std(results_linregs, axis=1)

    fig = plt.figure()

    mean_linreg = np.mean(results_linregs, axis=0)
    std_linreg = np.std(results_linregs, axis=0)
    plt.plot(range(1, L+1), mean_linreg)
    plt.fill_between(range(1, L+1), mean_linreg - std_linreg, mean_linreg + std_linreg, alpha=0.2, label='Uncertainty')
    # plt.plot(range(1, L+1), linregs_zero, color='r', linestyle='dashed', linewidth=2, label='Deterministic')
    plt.plot(range(1, L+1), linregs_zero*np.ones(L), color='r', linestyle='dashed', linewidth=2, label='Deterministic')
    plt.title(f'Linear Regions')
    plt.xlabel('Layer')
    plt.ylabel('# Linear Regions')
    # plt.yscale('log')
    #plt.legend()
    output_dir = f"/home/philipp/Desktop/plots/{L}_{R}/linregs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"half_circle_{size}.png"))
    if plot:
        plt.show()


def theoretical_boundary_complexity(k):
    n = 100
    alpha = 50
    beta = n - alpha


def _Pn12(n):
    # Define the integrand
    def integrand(p):
        phi_p = norm.cdf(p)
        term1 = phi_p**(n - 2)
        term2 = (1 - phi_p)**(n - 2)
        return (term1 + term2) * np.exp(-p**2)
    
    # Perform numerical integration
    result, _ = quad(integrand, 0, np.inf)
    
    # Multiply by the prefactor
    result /= np.sqrt(np.pi)
    return result


def Pnu(n, u):
    # Compute P(U_12 = 1) using the previously defined function
    p_u12_1 = _Pn12(n)
    
    # Compute the binomial coefficient
    num_pairs = math.comb(n, 2)
    binomial_coeff = math.comb(num_pairs, u - 1)
    
    # Compute the probability
    prob = (
        binomial_coeff
        * (p_u12_1 ** (u - 1))
        * ((1 - p_u12_1) ** (num_pairs - (u - 1)))
    )
    return prob


def Hnu(n, u):
    # Compute the small probability factor
    p = (2 * math.sqrt(2 * math.pi * math.log(n))) / (n**2)
    
    # Compute the number of pairs
    num_pairs = math.comb(n, 2)
    
    # Compute the binomial coefficient
    binomial_coeff = math.comb(num_pairs, u - 1)
    
    # Compute the probability H(u)
    H_u = (
        binomial_coeff
        * (p ** (u - 1))
        * ((1 - p) ** (num_pairs - (u - 1)))
    )
    return H_u


def Turk(u, r, k):
    # Compute the ceiling and floor values
    ceil_term = math.ceil((k + 1) / 2) - 1
    floor_term = math.floor((k + 1) / 2) - 1

    # Compute each term in the summation
    term1 = math.comb(r - 1, ceil_term) * math.comb(u - r - 1, floor_term)
    term2 = math.comb(r - 1, floor_term) * math.comb(u - r - 1, ceil_term)
    
    # Compute T_{u,r}(k)
    T_urk = term1 + term2
    return T_urk


def expected_good_edges(n, a):
    prefactor = (n-a) * a  # math.factorial(a) / math.factorial(2*a-n)
    integral = _Pn12(n)
    print(prefactor, integral)
    return prefactor * integral

