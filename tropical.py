import numpy as np


def scalar_multiply_set(scalar, S):
    """Multiply a set of points by a scalar.

    Multiplies each coordinate of each point in the input set by the given scalar.

    Args:
        scalar (float): The scalar value to multiply by
        S (set): Set of points, where each point is a tuple of coordinates

    Returns:
        set: New set containing the scaled points
    """
    return {tuple(scalar * element[i] for i in range(len(element))) for element in S}


def setsum(*sets):
    """Sum multiple sets of points element-wise.

    Takes multiple sets of points and returns their Minkowski sum. Each point in the result
    is formed by adding corresponding coordinates from one point from each input set.

    Args:
        *sets: Variable number of sets, where each set contains points represented as tuples
              of coordinates. Empty sets are skipped.

    Returns:
        set: The Minkowski sum of all input sets. Returns empty set if all inputs are empty.
    """

    # handle empty sets in the minkowski-sum
    first_non_empty_index = next((i for i, s in enumerate(sets) if s), None)
    if first_non_empty_index is None:
        return set()

    # iteratively build the minkowsi-sum
    result = sets[first_non_empty_index]
    for s in sets[first_non_empty_index + 1 :]:
        # if the next set is empty, continue
        if len(s) == 0:
            continue

        # if the next set is non-empty, do a Minkowsi-addition
        new_result = set()
        for b in s:
            for a in result:
                new_result.add(tuple(a[i] + b[i] for i in range(len(a))))
        result = new_result

    return result


def matrixproduct(A, S):
    """Matrix multiplication of a matrix A with a list of sets S.

    Computes the matrix product A @ S where S is a list of sets. Each element of the result
    is a set containing the Minkowski sum of scaled sets from S, where the scaling factors
    come from the corresponding row of A.

    Args:
        A (ndarray): Matrix of shape (m,n)
        S (list[set]): List of n sets, where each set contains points as coordinate tuples

    Returns:
        list[set]: List of m sets representing the matrix product A @ S. Returns empty set if S is empty.
    """

    # list of sets storing the final result
    result = []

    # if S is empty, return
    if len(S) == 0:
        return S

    # apply the multiplication
    for i in range(len(A)):

        # summands holds all of the sets which will have to be minkowski-summed
        summands = []
        for j in range(len(S)):
            summand = scalar_multiply_set(A[i][j], S[j])
            summands.append(summand)

        # save as the i'th set
        result.append(setsum(*summands))

    return result


def add_bias(S, b):
    """Add a bias to the last coordinate of each point in a set.

    Args:
        S (set): Set of points as coordinate tuples
        b (float): Bias value to add to last coordinate

    Returns:
        set: New set with bias added to last coordinate of each point
    """

    return {
        tuple(s[i] + (b if i == len(s) - 1 else 0) for i in range(len(s))) for s in S
    }


def setmult(b, S):
    """Multiply each point in a set by a scalar.

    Args:
        b (float): Scalar multiplier
        S (set): Set of points as coordinate tuples

    Returns:
        set: New set with each point multiplied by b
    """

    return {tuple(s[i] * b for i in range(len(s))) for s in S}


def iterate_points(P, N, W, b, t):
    """Transform sets of points through a linear layer with optional ReLU activation.

    Applies a weight matrix W and bias vector b to sets of positive and negative dual points P and N.

    Args:
        P (list[set]): List of positive point sets for each dimension
        N (list[set]): List of negative point sets for each dimension
        W (ndarray): Weight matrix
        b (ndarray): Bias vector
        t (float): Activation threshold, -inf for linear activation

    Returns:
        tuple: Contains:
            - P_new (list[set]): Transformed positive point sets
            - N_new (list[set]): Transformed negative point sets
    """

    # split weight matrix in positive and negative part
    Wp = np.maximum(W, 0)
    Wn = np.maximum(-W, 0)

    # apply all relevant matrix-vetor of set products
    WnP = matrixproduct(Wn, P)
    WpN = matrixproduct(Wp, N)
    WpP = matrixproduct(Wp, P)
    WnN = matrixproduct(Wn, N)

    # build the updated dual set N_new
    N_new = []
    for i in range(len(WpN)):
        N_new.append(setsum(WnP[i], WpN[i]))

    # build the updated dual set P_new
    P_new = []
    for i in range(len(WpP)):
        # P is offset by the bias
        P_new_i = add_bias(setsum(WpP[i], WnN[i]), b[i])

        # differentiate two cases: the layer is a ReLU or linear
        if t > float("-inf"):
            P_new_i = P_new_i.union(add_bias(N_new[i], t))
        P_new.append(P_new_i)

    return P_new, N_new
