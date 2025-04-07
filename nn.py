from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np

from tropical import iterate_points, setsum
from util import (
    count_transitions,
    double_uch,
    plot_points,
    upper_convex_hull,
)


class Block:
    """A block consisting of two ReLU layers, as in Telgarsky's sawtooth construction.
    Can be deterministic, or random. In the latter case, the activation function corresponding
    to the outer layer is the identity. See the thesis for an explanation why.

    This class implements a two-layer neural network block that can be either deterministic
    or random. For deterministic blocks, weights and biases are fixed to create a sawtooth
    pattern. For random blocks, weights and biases are randomly initialized.

    Args:
        random (bool, optional): If True, initialize with random weights. If False, use fixed
                               sawtooth weights. Defaults to True.
        lin (bool, optional): If True, use linear activation. If False, use ReLU.
                            Defaults to False.
        reduce (bool, optional): If True, reduce points after each layer to the upper convex hull.
                               Defaults to True.

    Attributes:
        W1 (ndarray): Weight matrix for first layer, shape (2,1)
        W2 (ndarray): Weight matrix for second layer, shape (1,2)
        b1 (ndarray): Bias vector for first layer, shape (2,)
        b2 (ndarray): Bias vector for second layer, shape (1,)
        t1 (float): Threshold for first layer activation
        t2 (float): Threshold for second layer activation
        reduce (bool): Whether to reduce points after each layer
        random (bool): Whether block uses random or fixed weights
        lin (bool): Whether to use linear or ReLU activation
    """

    def __init__(self, random=True, lin=False, reduce=True):
        self.reduce = reduce
        self.random = random
        self.lin = lin

        if not random:
            self.W1 = np.array([[1.0], [1.0]])
            self.W2 = np.array([[2.0, -4.0]])
            self.b1 = np.array([0.0, -1 / 2])
            self.b2 = np.array([0.0])
            self.t1 = 0
            self.t2 = 0
        else:
            self.W1 = np.random.randn(2, 1)
            self.W2 = np.random.randn(1, 2)
            self.b1 = np.random.randn(2)
            self.b2 = np.random.randn(1)
            self.t1 = 0

            # in the random case, the outer layer is linear. See the thesis for an explanation
            self.t2 = -np.inf

    # for the deterministic blocks, one could also use the closed form calculation
    def __call__(self, P, N):
        """Forward pass through the block.

        Transforms input points through two layers with optional ReLU activation,
        tracking positive and negative point sets.

        Args:
            P (list[set]): List of positive point sets for each dimension
            N (list[set]): List of negative point sets for each dimension

        Returns:
            tuple: Contains:
                - P_new (list[set]): Transformed positive point sets
                - N_new (list[set]): Transformed negative point sets
        """

        # apply the first layer of the block
        P_new, N_new = iterate_points(P, N, self.W1, self.b1, self.t1)

        # if reduce is true, reduce P and N to their upper convex hulls
        if self.reduce:
            P_new, N_new = double_uch(P_new, N_new)

        # apply the second layer of the block
        P_new, N_new = iterate_points(P_new, N_new, self.W2, self.b2, self.t2)
        if self.reduce:
            P_new, N_new = double_uch(P_new, N_new)

        return P_new, N_new


class Layer:
    """A single neural network layer with ReLU activation.

    Implements a layer that performs an affine transformation followed by optional ReLU activation.
    Maintains and transforms sets of positive and negative points that define the function.

    Args:
        input_dim (int, optional): Input dimension. Defaults to 2.
        output_dim (int, optional): Output dimension. Defaults to 1.
        lin (bool, optional): If True, omits ReLU activation. Defaults to False.
        W (ndarray, optional): Weight matrix. If None, initialized randomly. Defaults to None.
        b (ndarray, optional): Bias vector. If None, initialized randomly. Defaults to None.
        reduce (bool, optional): If True, reduces point sets to upper convex hull. Defaults to True.

    Attributes:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        reduce (bool): Whether to reduce point sets
        W (ndarray): Weight matrix of shape (output_dim, input_dim)
        b (ndarray): Bias vector of shape (output_dim,)
        t (float): ReLU threshold, -inf for linear layer
    """

    def __init__(
        self, input_dim=2, output_dim=1, lin=False, W=None, b=None, reduce=True
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reduce = reduce

        self.W = W if W is not None else np.random.randn(output_dim, input_dim)
        self.b = b if b is not None else np.random.randn(output_dim)
        self.t = -np.inf if lin else 0

    def __call__(self, P, N):
        """Forward pass through the layer.

        Transforms input points through an affine transformation followed by optional ReLU activation,
        tracking positive and negative point sets.

        Args:
            P (list[set]): List of positive point sets for each dimension
            N (list[set]): List of negative point sets for each dimension

        Returns:
            tuple: Contains:
                - P_new (list[set]): Transformed positive point sets
                - N_new (list[set]): Transformed negative point sets
        """
        # do the forward pass
        P_new, N_new = iterate_points(P, N, self.W, self.b, self.t)

        # if true, reduce P and N to their upper convex hulls
        if self.reduce:
            P_new, N_new = double_uch(P_new, N_new)

        return P_new, N_new


class Shift:
    """A layer that applies a constant shift to the input.

    Implements a layer that performs a simple translation by adding a bias vector.
    Maintains and transforms sets of positive and negative points that define the function.

    Args:
        input_dim (int, optional): Input dimension. Defaults to 2.
        output_dim (int, optional): Output dimension. Defaults to 1.
        reduce (bool, optional): If True, reduces point sets to upper convex hull. Defaults to True.
        b (ndarray, optional): Bias vector. If None, initialized randomly. Defaults to None.

    Attributes:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        reduce (bool): Whether to reduce point sets
        W (ndarray): Weight matrix fixed to identity, shape (1,1)
        b (ndarray): Bias vector of shape (output_dim,)
        t (float): Threshold fixed to -inf for linear layer
    """

    def __init__(self, input_dim=2, output_dim=1, reduce=True, b=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reduce = reduce

        self.W = np.array([[1.0]])
        self.b = b if b is not None else np.random.randn(output_dim)
        self.t = -np.inf

    def __call__(self, P, N):
        """Applies the shift transformation to input point sets.

        Performs a forward pass by applying the shift transformation to the input
        positive and negative point sets.

        Args:
            P (list[set]): List of positive point sets for each dimension
            N (list[set]): List of negative point sets for each dimension

        Returns:
            tuple: Contains:
                - P (list[set]): Transformed positive point sets
                - N (list[set]): Transformed negative point sets
        """
        P, N = iterate_points(P, N, self.W, self.b, self.t)

        # if true, reduce P and N to their upper convex hulls
        if self.reduce:
            P, N = double_uch(P, N)

        return P, N


class SawtoothNetwork:
    """A network that implements a sawtooth-like function through composition of blocks.

    The network consists of deterministic and/or random blocks that transform input points
    through affine transformations and ReLU operations. Each block maintains sets of positive (P)
    and negative (N) points that define the function.

    Args:
        l (int): Number of deterministic blocks
        s (int): Number of random blocks if positive, or single random layer if negative

    Attributes:
        P (list): List containing set of positive points
        N (list): List containing set of negative points
        Ps (list): History of positive point sets after each block (when all_layers=True)
        Ns (list): History of negative point sets after each block (when all_layers=True)
        l (int): Number of deterministic blocks
        s (int): Number of random blocks
        blocks (list): List of Block and Layer objects defining the network
    """

    def __init__(self, l, s):
        self.P = [{(1.0, 0.0)}]
        self.N = [set()]

        self.Ps = np.empty(l + s, dtype=object)
        self.Ns = np.empty(l + s, dtype=object)

        self.l = l
        self.blocks = [Block(random=False) for _ in range(l)]
        # a bit of a hack. Apply a -1/2 shift because the score function has threshold 0, not 1/2.
        self.blocks = self.blocks + [Shift(b=np.array([-1 / 2]))]

        if s > 0:
            self.blocks = self.blocks + [Block(random=True) for _ in range(s)]

        if s < 0:
            self.blocks = self.blocks + [Layer(input_dim=1, output_dim=1, lin=False)]

    def __call__(self, all_layers=False):
        """Forward pass through the network.

        Transforms input points through sequence of blocks, tracking positive and negative point sets.

        Args:
            all_layers (bool): If True, store intermediate point sets after each block.
                             If False, only store final point sets.

        The forward pass applies each block's transformation sequentially to the input points,
        maintaining separate positive (P) and negative (N) point sets that define the function.
        When all_layers=True, the intermediate P and N sets are stored in self.Ps and self.Ns.
        """

        block_counter = 0
        for block in self.blocks:
            self.P, self.N = block(self.P, self.N)

            # only log the representation after a proper block
            if all_layers and not isinstance(block, Shift):
                self.Ps[block_counter] = self.P
                self.Ns[block_counter] = self.N

                block_counter += 1

    def linregs(self, all_layers=False):
        """Returns the number of linear regions in the network.

        Computes the number of linear regions by finding the size of the upper convex hull
        of the sum of positive and negative point sets.

        Args:
            all_layers (bool): If True, return linear region counts for all intermediate layers.
                             If False, only return count for final layer.

        Returns:
            If all_layers=False:
                int: Number of linear regions in final layer
            If all_layers=True:
                list: Number of linear regions after each layer
        """

        if not all_layers:
            return len(upper_convex_hull([setsum(self.P[0], self.N[0])])[0])
        else:
            return [
                len(upper_convex_hull([setsum(P[0], N[0])])[0])
                for P, N in zip(self.Ps, self.Ns)
            ]

    def transitions(self, all_layers=False):
        """Returns the number of transitions between positive and negative regions. Assumes input-space to be one-dimensional.

        Computes the number of transitions by finding points on the upper convex hull
        where adjacent points switch between positive and negative sets.

        Args:
            all_layers (bool): If True, return transition counts for all intermediate layers.
                             If False, only return count for final layer.

        Returns:
            If all_layers=False:
                int: Number of transitions in final layer
            If all_layers=True:
                list: Number of transitions after each layer
        """
        if not all_layers:
            return count_transitions(
                upper_convex_hull([self.P[0] | self.N[0]])[0], self.P, self.N
            )
        else:
            return [
                count_transitions(upper_convex_hull([P[0] | N[0]])[0], P, N)
                for P, N in zip(self.Ps, self.Ns)
            ]

    def plot(self):
        """Plot the current state of the network.

        Visualizes the positive points (P), negative points (N), and their upper convex hull.
        Points are plotted in red (P) and blue (N), with the upper convex hull shown in green.
        The plot is saved to disk and displayed.

        Returns:
            None
        """

        plot_points(
            self.P[0], self.N[0], self.l, upper_convex_hull([self.P[0] | self.N[0]])[0]
        )

    @property
    def numpoints(self):
        return len(self.P[0]) + len(self.N[0])


class ReLuNet:
    """A neural network with ReLU activation functions.

    This class implements a feedforward neural network with ReLU activations.
    It tracks the positive (P) and negative (N) regions through the network layers
    and can compute complexity metrics like linear regions and decision boundary.

    Args:
        L (int): Number of layers in the network
        P (list[set], optional): Initial positive regions. Defaults to [{(1,0)}].
        N (list[set], optional): Initial negative regions. Defaults to [set()].
        input_dim (int, optional): Input dimension. Defaults to 1.
        output_dim (int, optional): Output dimension. Defaults to 1.

    Attributes:
        P (list[set]): Current positive regions
        N (list[set]): Current negative regions
        L (int): Number of layers
        Ps (ndarray): Positive regions after each layer (when all_layers=True)
        Ns (ndarray): Negative regions after each layer (when all_layers=True)
        layers (list[Layer]): The network layers
    """

    def __init__(self, L, P=[{(1, 0)}], N=[set()], input_dim=1, output_dim=1):
        """Initialize a ReLuNet instance.

        Args:
            L (int): Number of layers in the network
            P (list[set], optional): Initial positive regions. Defaults to [{(1,0)}].
            N (list[set], optional): Initial negative regions. Defaults to [set()].
            input_dim (int, optional): Input dimension. Defaults to 1.
            output_dim (int, optional): Output dimension. Defaults to 1.

        Attributes:
            P (list[set]): Current positive regions
            N (list[set]): Current negative regions
            L (int): Number of layers
            Ps (ndarray): Positive regions after each layer (when all_layers=True)
            Ns (ndarray): Negative regions after each layer (when all_layers=True)
            layers (list[Layer]): The network layers
        """
        self.P = P
        self.L = L

        self.Ps = np.empty(L, dtype=object)
        self.Ns = np.empty(L, dtype=object)

        self.N = N

        self.layers = [
            Layer(input_dim=input_dim, output_dim=output_dim) for _ in range(L)
        ]

    def __call__(self, all_layers=False):
        """Forward pass through the network.

        Transforms input points through sequence of layers, tracking positive and negative point sets.

        Args:
            all_layers (bool): If True, store intermediate point sets after each layer.
                             If False, only store final point sets.

        Returns:
            None
        """

        for i, layer in enumerate(self.layers):
            self.P, self.N = layer(self.P, self.N)

            if all_layers:
                self.Ps[i] = self.P
                self.Ns[i] = self.N

    # TODO: again, handle both complexity measure differently.
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
            self.transitions = [
                count_transitions(uch, P, N)
                for uch, P, N in zip(self.UCH_union, self.Ps, self.Ns)
            ]
            self.linregs = [
                len(upper_convex_hull([setsum(P[0], N[0])])[0])
                for P, N in zip(self.Ps, self.Ns)
            ]

    def plot(self):
        plot_points(self.P, self.N, self.L, np.array(list(self.UCH_union)))
