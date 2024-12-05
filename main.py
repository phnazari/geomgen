import numpy as np
from matplotlib import pyplot as plt

from util import count_transitions, plot_points, upper_convex_hull
from tropical import iterate_points
from nn import NN
import os


# number of deterministic blocks
#l = 2
# number of random blocks
#s = 0
# depth
L = 6
# repetitions
R = 50
# plot?
plot = False

results_transitions = np.zeros((L, R))
results_UCHp = np.zeros((L, R))

for r in range(R):
    for l in range(L):
        print(f"iteration r={r}, l={l}")
        s = L - l
        nn = NN(l, s)
        P, N = nn.forward()
        PN = P[0] | N[0]
        UCH = upper_convex_hull([PN])[0]
        transitions = count_transitions(UCH, P, N)
        results_transitions[l, r] = transitions
        results_UCHp[l, r] = len(P[0]) + len(N[0])

        #if l == 5 and s == 1:
        #    print(len(P[0]) + len(N[0]))
        #    plot_points(P, N, L, np.array(list(UCH)))

print(results_UCHp)
print(transitions)


nn_det = NN(L, 0)
P_det, N_det = nn_det.forward()
PN_det = P_det[0] | N_det[0]
UCH_det = upper_convex_hull([PN_det])[0]
transitions = count_transitions(UCH_det, P_det, N_det)

print(f"Number of transitions: {transitions}")
print(f"Size of UCH: {len(UCH_det)}")

if R > 0:
    for l in range(L):
        s = L - l

        plt.figure()

        plt.hist(results_UCHp[l, :], label="random")  # , bins=range(int(results_UCHp.min()), int(results_UCHp.max()) + 1))
        plt.axvline(x=len(UCH_det), color='r', linestyle='dashed', linewidth=2, label='det')
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
        plt.axvline(x=transitions, color='r', linestyle='dashed', linewidth=2, label='det')
        plt.title(f'Transitions: {l} deterministic blocks, {s} random blocks and {R} repetitions')
        plt.xlabel('Number of Transitions')
        plt.ylabel('Frequency')
        plt.legend()
        output_dir = f"/home/philipp/Desktop/plots/{L}_{R}/transitions"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f"histogram_{l}_{s}_{R}.png"))


#if plot:
#    plot_points(P_det, N_det, L, np.array(list(UCH_det))) 
