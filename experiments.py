import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Arc

from util import compute_percentile, compute_upper_simplices, count_transitions, generate_points_on_upper_hemisphere, plot_function, plot_grid, plot_points, upper_convex_hull
from tropical import add_bias, iterate_points, setmult, setsum
from nn import Layer, ReLuNet, SawtoothNetwork
import os

from scipy.stats import norm
from scipy.integrate import quad
from scipy.spatial import ConvexHull
import math


def det_rand_net(L, R, shift=False):
    results_transitions = np.zeros((L, R, L))
    results_linregs = np.zeros((L, R, L))

    for r in range(R):
        for l in range(L):
            s = L - l
            print(f"iteration r={r}, l={l}, s={s}")
            nn = SawtoothNetwork(l, s, shift=shift)
            # nn.evaluate(all_layers=all_layers)
            nn(all_layers=True)

            results_transitions[l, r] = nn.transitions(all_layers=True)
            results_transitions[l, r, :l] = 2**np.arange(1, l+1)
            # results_UCHp[l, r] = [len(uch) for uch in nn.UCH]
            results_linregs[l, r] = nn.linregs(all_layers=True)

    # print(f"iteration r={1}, l={L}, s={0}")
    nn_det = SawtoothNetwork(L, 0, shift=shift)
    nn_det(all_layers=True)
    # nn_det.evaluate(all_layers=True)

    return results_transitions, results_linregs, nn_det


def sawtooth_linregs(L, R):
    _, results_linregs, nn_det = det_rand_net(L, R)

    elements_per_row = 3
    fig = plot_grid(results_linregs, nn_det.linregs(all_layers=True), L, elements_per_row, "#Linear Regions")
    output_dir = f"/home/philipp/Desktop/plots/{L}_{R}/linregs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, f"combined_{L}_{R}.png"))

    plt.show()


def sawtooth_linregs_randlayer(L, R):
    results_linregs = np.zeros((R, L-1))
    shift = False

    for r in range(R):
        for l in range(L):
            print(f"iteration l={l}")
            nn = SawtoothNetwork(l, -1, shift=shift)
            # nn.evaluate(all_layers=all_layers)
            nn(all_layers=False)

            results_linregs[r] = nn.linregs(all_layers=False)

    # print(f"iteration r={1}, l={L}, s={0}")
    nn_det = SawtoothNetwork(L-1, 0, shift=shift)
    nn_det(all_layers=True)
    # nn_det.evaluate(all_layers=True)

    fig = plt.figure()
    plt.plot(range(1, L), np.median(results_linregs, axis=0), label='Mean', color='blue')
    # plt.plot(range(1, L), nn_det.linregs(all_layers=True), label='Deterministic', color='red', linestyle='--')
    plt.fill_between(range(1, L), np.percentile(results_linregs, 25, axis=0), np.percentile(results_linregs, 75, axis=0), alpha=0.2, color='blue', label='Interquartile Range')
    plt.xlabel('S')
    plt.ylabel('n')
    plt.legend()
    plt.tight_layout()

    plt.show()


def sawtooth_transitions(L, R):
    results_transitions, _, nn_det = det_rand_net(L, R, shift=True)

    elements_per_row = 3
    fig = plot_grid(results_transitions, np.array(2**np.arange(1, L+1)), L, elements_per_row, "Transitions")
    output_dir = f"/home/philipp/Desktop/plots/{L}_{R}/trans"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, f"combined_{L}_{R}.png"))

    plt.show()

def union_on_circle(num_points=5):
    P, N = generate_points_on_upper_hemisphere(num_points)

    S = setsum(P[0], N[0])
    uch = upper_convex_hull([S])[0]

    #fig = plt.figure()
    #plt.scatter(*zip(*S), alpha=.4, label='P', color="red")
    #plt.scatter(*zip(*uch), alpha=.4, label='UCH', color="blue")
    #plt.axis("equal")
    #for n in N[0]:
    #    plt.gca().add_patch(Arc(n, 2, 2, theta1=0, theta2=90))

    plt.show()

    W = np.abs(np.random.randn(1)[0])
    b = np.random.randn(1)[0]

    W = 1
    b = 0 # upper_bound_2(num_points, 0) - 0.00001  # lower_bound_2(num_points)-0.05  # lower_bound(num_points, 2) + 0.005 # upper_bound_2(num_points) - 0.0001  # lower_bound_2(num_points)

    if W > 0:
        # multiply N by W
        L = setmult(W, N[0])
        M = add_bias(L, b)
        O = setmult(W, P[0])

        A = setsum(L, L)
        B = setsum(M, O)
        # B = setsum(add_bias(setmult(W, P[0]), b), N)
        C = A.union(B)
    else:
        L = setmult(-W, P[0])
        M = add_bias(L, b)
        O = setmult(-W, N[0])
        # M = add_bias(O, b)

        A = setsum(L, L)
        B = setsum(M, O)
        # B = setsum(M, L)
        C = A.union(B)

    fig = plt.figure()
    plt.scatter(*zip(*A), alpha=.4, label='A', color='r')
    plt.scatter(*zip(*B), alpha=.4, label='B', color='g')    

    for l in L:
        if W > 0:
            plt.gca().add_patch(Arc(l, 2*W, 2*W, theta1=0, theta2=90, color='r'))
        else:
            plt.gca().add_patch(Arc(l, -2*W, -2*W, theta1=0, theta2=90, color='r'))

    for m in M:
        if W > 0:
            plt.gca().add_patch(Arc(m, 2*W, 2*W, theta1=0, theta2=90, color='g'))
        else:
            plt.gca().add_patch(Arc(m, -2*W, -2*W, theta1=0, theta2=90, color='g'))



    # plot upper convex hull using circles around points
    uch = upper_convex_hull([C])[0]
    plt.gca().scatter(*zip(*uch), alpha=.4, label='UCH', color='b')
    plt.axis('equal')
    plt.legend()
    plt.show()


def upper_bound(n, i):
    norm = 2*(2*n+1)
    pihalf = np.pi/2
    Cn = 2*np.sin(pihalf*1/norm)**2
    Dn = np.sin(pihalf*(4*i)/norm)
    return Cn/Dn

def lower_bound(n, i):
    norm = 2*(2*n+1)
    pihalf = np.pi/2
    Cn = 2*np.sin(pihalf*1/norm)**2
    Dn = np.sin(pihalf*(2*(2*i+1))/norm)
    return -Cn/Dn

def lower_bound_2(n, i):
    norm = 2*(2*n+1)
    pihalf = np.pi/2
    Cn = 2*np.sin(pihalf*1/norm)**2
    Dn = np.sin(pihalf*(4*i)/norm)
    return -Cn/Dn

def upper_bound_2(n, i):
    norm = 2*(2*n+1)
    pihalf = np.pi/2
    Cn = 2*np.sin(pihalf*1/norm)**2
    Dn = np.sin(pihalf*(4*i + 2)/norm)
    return Cn/Dn


def probability_of_increase(R, plot=False, Nmax=32):
    n = 2
    max = 15
    sigma_values = np.linspace(1, 5, 9)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sigma_values)))

    results = np.zeros(Nmax)

    for n in range(2, Nmax):
        for r in range(R):
            P, N = generate_points_on_upper_hemisphere(n)

            nn = ReLuNet(1, P=P, N=N)
            nn.evaluate(all_layers=False)

            if nn.linregs > 2*n+1:
                results[n] += 1

    results /= R

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, Nmax), results[2:], marker='o', linestyle='-', color='b', label='Empirical')
    
    upper_bound_values = [1/2 + 1/(4*2*np.pi)*(np.arctan(np.abs(upper_bound(n, np.ceil(n/2))) + np.arctan(np.abs(lower_bound(n, np.ceil(n/2)-1))))) for n in range(2, Nmax)]
    plt.plot(range(2, Nmax), upper_bound_values, linestyle='--', color='r', label='Theoretical')

    plt.axhline(y=0.5, color='gray', linestyle='--', label='Y=0.5')

    plt.xlabel('n', fontsize=14)
    plt.ylabel(r'$\mathbb{P}(\uparrow)$', fontsize=14)
    plt.ylim(0.25, 0.75)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if plot:
        plt.show()
    
    return


    plt.figure(figsize=(10, 6))

    for sigma, color in zip(sigma_values, colors):
        res = []
        for n in range(2, max):
            prob = 17/32 + 1/64*sigma*(lower_bound(n, np.ceil(n/2)-1)**2 + upper_bound(n, np.ceil(n/2))**2)
            res.append(prob)

            print((lower_bound(n, np.ceil(n/2)-1)**2 + upper_bound(n, np.ceil(n/2))**2))

        plt.plot(range(2, max), res, color=color, linestyle='-', linewidth=2, marker='o', markersize=5, markerfacecolor=color, markeredgewidth=2, label=r'$\sigma_{w}^{2}=$' + str(sigma))

    plt.xlabel('n', fontsize=14)
    plt.ylabel(r'$\mathbb{P}(\uparrow)$', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def _as(n):
    t = 3/(8*(2*n+1))
    gamma = 0.577
    return -1/2*n + 1 + t*np.log((4*(2*n+1))/np.pi) + t*gamma


def sawtooth_one_layer(R, plot=False, Lmax=3):
    results = np.zeros((R, Lmax-1))

    det_results = np.zeros(Lmax)
    det_points = []

    for l in range(1, Lmax+1):
        nn_sawtooth = SawtoothNetwork(l, 0, shift=True)
        nn_sawtooth.evaluate(all_layers=False)
        det_results[l-1] = (nn_sawtooth.linregs)

        det_points.append((nn_sawtooth.P, nn_sawtooth.N))


    for l in range(1, Lmax):
        print(l, Lmax)

        print("\n")
        for r in range(R):
            P, N = det_points[l-1]
            nn = ReLuNet(1, P=P, N=N)
            # nn = SawtoothNetwork(0, 1, shift=True)
            nn.evaluate(all_layers=False)
            
            results[r, l-1] = nn.linregs - det_results[l]

    mean_linreg = np.mean(results, axis=0)
    percentiles = np.percentile(results, [25, 50, 75], axis=0)

    plt.figure()
    plt.plot(range(1, Lmax), mean_linreg, label='Gain (Mean)', color="Blue")
    plt.plot(range(1, Lmax), percentiles[1], label='Median', color="Green")
    plt.fill_between(range(1, Lmax), percentiles[0], percentiles[2], alpha=0.2, label='Interquartile Range', color="blue")
    plt.plot(range(1, Lmax+1), det_results, label='Deterministic', color="Red", linestyle='--')
    plt.xlabel('S')
    plt.ylabel('n')
    plt.legend()
    plt.tight_layout()

    #output_dir = f"/home/philipp/Desktop/plots/{1}_{R}/linregs"
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    #plt.savefig(os.path.join(output_dir, f"half_circle_one_random_layer.png"))
    if plot:
        plt.show()

def half_circle_one_random_layer(R, plot=False, Nmax=32):
    results = np.zeros((R, Nmax))
    results_2 = np.zeros((Nmax))
    percentiles = np.zeros((Nmax, 2))
    

    for n in range(2, Nmax):
        for r in range(R):
            P, N = generate_points_on_upper_hemisphere(n)

            nn = ReLuNet(1, P=P, N=N)
            nn.evaluate(all_layers=False)

            results[r, n] = nn.linregs - (2*n+1)

        s_values, P_s = compute_distribution(n)
        expectation = sum(s * P for s, P in zip(s_values, P_s))
        results_2[n] = expectation - (2*n+1)
        percentiles[n, 0] = compute_percentile(0.25, s_values, P_s) - (2*n+1)
        percentiles[n, 1] = compute_percentile(0.75, s_values, P_s) - (2*n+1)

    mean_linreg = np.mean(results[:, 2:Nmax], axis=0)
    # Compute different percentile bands
    #q5_linreg, q95_linreg = np.percentile(results[:, 2:Nmax], [45, 55], axis=0)  # 5%-95%
    #q15_linreg, q85_linreg = np.percentile(results[:, 2:Nmax], [15, 85], axis=0)  # 15%-85%
    #q25_linreg, q75_linreg = np.percentile(results[:, 2:Nmax], [25, 75], axis=0)  # 25%-75%
    #t1 = np.percentile(results[:, 2:Nmax], 45, axis=0, interpolation='nearest')
    #t2 = np.percentile(results[:, 2:Nmax], 55, axis=0, interpolation='linear')


    plt.figure()

    # Plot the empirical mean
    plt.plot(range(2, Nmax), mean_linreg, label='Empirical Mean', color='blue')

    # Fill multiple percentile bands with increasing transparency
    # plt.fill_between(range(2, Nmax), t1, t2, alpha=0.1, color='blue', label='5%-95%')   # Widest
    #plt.fill_between(range(2, Nmax), q15_linreg, q85_linreg, alpha=0.2, color='blue', label='15%-85%')  # Medium
    #plt.fill_between(range(2, Nmax), q25_linreg, q75_linreg, alpha=0.3, color='blue', label='25%-75%')  # Narrowest

    # plot approximation
    plt.plot(range(2, Nmax), _as(np.arange(2, Nmax)), label='Approx', linestyle='--', color='red')

    # plot computations
    plt.plot(range(2, Nmax), results_2[2:], label='Sum', color='green', linestyle=':', marker='o', markersize=4)
    # plt.fill_between(range(2, Nmax), percentiles[2:, 0], percentiles[2:, 1], alpha=0.2, color='green', label='Interquartile Range')

    plt.xlabel('n')
    plt.ylabel("S-(2n+1)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #output_dir = f"/home/philipp/Desktop/plots/{1}_{R}/linregs"
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    #plt.savefig(os.path.join(output_dir, f"half_circle_one_random_layer.png"))
    if plot:
        plt.show()

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
    return 4*np.sin(np.pi/2*(1/(2*(2*n+1))))**4*((2*n+1)**2-1)/3

    
############################# Example Function #############################

def plot_affine_regions(db=False):
    """Plot the affine regions of the function defined by the dual representation"""
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
            point = np.array([X[i,j], Y[i,j]])
            
            # Find max index for P set
            max_val_p = float('-inf')
            max_idx_p = 0
            for idx, p in enumerate(P[0]):
                val_p = point[0]*p[0] + point[1]*p[1] + p[2]
                if val_p > max_val_p:
                    max_val_p = val_p
                    max_idx_p = idx
                    
            # Find max index for N set
            max_val_n = float('-inf') 
            max_idx_n = 0
            for idx, n in enumerate(N[0]):
                val_n = point[0]*n[0] + point[1]*n[1] + n[2]
                if val_n > max_val_n:
                    max_val_n = val_n
                    max_idx_n = idx
                    
            # Encode both indices and which value was larger into regions array
            # Use positive values when P is larger, negative when N is larger
            if not db:
                regions[i,j] = max_idx_p * len(N[0]) + max_idx_n + 1
            else:
                if max_val_p >= max_val_n:
                    regions[i,j] = 1
                else:
                    regions[i,j] = -1

            #if max_val_p >= max_val_n:
            #    regions[i,j] = max_idx_p * len(N[0]) + max_idx_n + 1
            #else:
            #    regions[i,j] = -(max_idx_p * len(N[0]) + max_idx_n + 1)

    # Plot the regions
    plt.figure(figsize=(10,10))
    if db:
        plt.imshow(regions, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='RdBu')
    else:
        plt.imshow(regions, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='tab20')
    # plt.colorbar(label='Region Index')
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.title('Affine Regions of Dual Function')
    plt.show()


def compute_dual_representation():
    """Computes the dual representation by applying three layers of transformations"""
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
    P, N = compute_dual_representation()
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define plotting parameters for P and N sets
    plot_params = [
        {'points': P, 'color': 'blue', 'marker': 'o'},
        {'points': N, 'color': 'red', 'marker': 'x'}
    ]

    # Plot points and hulls for both sets
    for params in plot_params:
        # Plot points
        for point_set in params['points']:
            for point in point_set:
                ax.scatter(point[0], point[1], point[2], 
                          c=params['color'], marker=params['marker'])

        # Compute and plot upper hulls
        upper_hulls = [compute_upper_simplices(point_set) for point_set in params['points']]
        for hull_points, point_set in zip(upper_hulls, params['points']):
            for simplex in hull_points:
                triangle = np.array([list(point_set)[vertex] for vertex in simplex])
                ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                              color=params['color'], alpha=0.5)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show(block=True)  # Display the plot and block execution

def plot_union_dualrep():
    P, N = compute_dual_representation()
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define plotting parameters for P and N sets
    plot_params = [
        {'points': P, 'color': 'blue', 'marker': 'o'},
        {'points': N, 'color': 'red', 'marker': 'x'}
    ]

    # Plot points and hulls for both sets
    for params in plot_params:
        # Plot points
        for point_set in params['points']:
            for point in point_set:
                ax.scatter(point[0], point[1], point[2], 
                          c=params['color'], marker=params['marker'])

    # Compute and plot upper hull
    S = P[0].union(N[0])
    print(S)
    upper_hulls = compute_upper_simplices(S)
    # Count vertices in the upper convex hull
    vertices = set()
    for simplex in upper_hulls:
        vertices.update(list(S)[i] for i in simplex)

    print(vertices)

    # Plot upper hull triangles
    for simplex in upper_hulls:
        triangle = np.array([list(S)[vertex] for vertex in simplex])
        ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                      color='blue', alpha=0.5)


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show(block=True)  # Display the plot and block execution



def plot_setsum_dualrep():
    P, N = compute_dual_representation()
    
    # Compute setsum and upper convex hull
    S = setsum(P[0], N[0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points from setsum
    for point in S:
        ax.scatter(point[0], point[1], point[2], c='blue', marker='o')

    # Compute and plot upper hull
    upper_hulls = compute_upper_simplices(S)
    # Count vertices in the upper convex hull
    vertices = set()
    for simplex in upper_hulls:
        vertices.update(simplex)
    num_vertices = len(vertices)
    print(S)
    print(vertices)
    print(f"Number of vertices in upper convex hull: {num_vertices}")
    for simplex in upper_hulls:
        triangle = np.array([list(S)[vertex] for vertex in simplex])
        ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                      color='blue', alpha=0.5)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show(block=True)  # Display the plot and block execution

############################# Circle Density #############################


# Define the base functions
def l(i, n):
    numerator = 2 * np.sin(np.pi / (4 * (2 * n + 1)))**2
    denominator = np.sin(np.pi * (2 * i + 1) / (4 * n + 2))
    return -numerator / denominator

def u(i, n):
    numerator = 2 * np.sin(np.pi / (4 * (2 * n + 1)))**2
    denominator = np.sin(np.pi * i / (2 * n + 1))
    return numerator / denominator

# Function to compute P(S = s)
def compute_P(s, n, atan_l_values, atan_u_values, delta_l_values, delta_u_values):
    if s == n + 2:
        return (1/4) * (2 - (2 / np.pi) * (atan_l_values[0] + atan_u_values[0]))
    
    elif s == 2 * n + 2:
        if n % 2 == 0:
            m = n // 2
            return (1/4) * (2 - (2 / np.pi) * (atan_u_values[0] + atan_l_values[0] - 
                                                delta_l_values[m - 1] - delta_u_values[m - 1]))
        else:
            return (1/4) * (2 - (2 / np.pi) * (atan_u_values[0] + atan_l_values[0]))
    
    elif s == 3 * n + 2:
        return (1/4) * (2 / np.pi) * (2 * atan_u_values[n - 1] + 2 * atan_l_values[n - 1])
    
    else:
        i = s - (n + 2)
        if 0 < i < n and i % 2 == 0:
            m = i // 2
            return (1/4) * (2 / np.pi) * (delta_l_values[m - 1] + delta_u_values[m - 1])
        
        elif n < i < 2 * n:
            if i % 2 == 0:
                m = i // 2
                return (1/4) * (2 / np.pi) * (delta_u_values[i - n - 1] + delta_l_values[m - 1] + 
                                              delta_u_values[m - 1] + delta_l_values[i - n - 1])
            else:
                return (1/4) * (2 / np.pi) * (delta_u_values[i - n - 1] + delta_l_values[i - n - 1])
        
        return 0  # Should not occur for s in s_values


def compute_distribution(n):
    # Set the parameter n

    # Compute l(i, n) and u(i, n)
    l_values = [l(k, n) for k in range(n)]          # i = 0 to n-1
    u_values = [u(k, n) for k in range(1, n + 1)]   # i = 1 to n

    # Compute arctangents
    atan_l_values = [np.arctan(np.abs(l_k)) for l_k in l_values]
    atan_u_values = [np.arctan(np.abs(u_k)) for u_k in u_values]

    # Compute delta values
    delta_l_values = [atan_l_values[k] - atan_l_values[k + 1] for k in range(n - 1)]  # k = 0 to n-2
    delta_u_values = [atan_u_values[k] - atan_u_values[k + 1] for k in range(n - 1)]  # k = 0 to n-2, for Delta u_1 to u_{n-1}

    # Generate support (s values)
    s_values = ([n + 2] + 
                [n + 2 + i for i in range(2, n, 2)] + 
                [2 * n + 2] + 
                [n + 2 + i for i in range(n + 1, 2 * n)] + 
                [3 * n + 2])

    # Compute probabilities
    P_s = [compute_P(s, n, atan_l_values, atan_u_values, delta_l_values, delta_u_values) 
        for s in s_values]

    return s_values, P_s


def circle_distribution(n):
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
    plt.stem(s_values, P_s, basefmt=" ", use_line_collection=True, linefmt='navy', markerfmt='navy')
    plt.axvline(x=2*n+1, color='crimson', linestyle='--', label=f's=2n+1')
    plt.xlabel('s', fontsize=14)
    plt.ylabel(r'$\mathbb{P}(S = s)$', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
