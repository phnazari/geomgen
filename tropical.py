import numpy as np


def scalar_multiply_set(scalar, S):
    return {(scalar * element[0], scalar * element[1]) for element in S}


def setsum(*sets):
    # result = sets[0]
    first_non_empty_index = next((i for i, s in enumerate(sets) if s), None)
    if first_non_empty_index is None:
        return set()
    result = sets[first_non_empty_index]
    for s in sets[first_non_empty_index + 1:]:
        if len(s) == 0:
            continue
        new_result = set()
        for b in s:
            for a in result:
                new_result.add((a[0] + b[0], a[1] + b[1]))
        result = new_result

    return result

def matrixproduct(A, S):
    result = []
    if len(S) == 0:
        print("returning")
        return S


    for i in range(len(A)):
        summands = []
        for j in range(len(S)):
            summand = scalar_multiply_set(A[i][j], S[j])
            summands.append(summand)
        result.append(setsum(*summands))


    return result


def add_bias(S, b):
    return {(s[0], s[1] + b) for s in S}


def setmult(b, S):
    return {(s[0]*b, s[1] * b) for s in S}


def iterate_points(P, N, W, b, t):
    Wp = np.maximum(W, 0)
    Wn = np.maximum(-W, 0)

    WnP = matrixproduct(Wn, P)
    WpN = matrixproduct(Wp, N)
    WpP = matrixproduct(Wp, P)
    WnN = matrixproduct(Wn, N)


    N_new = []
    for i in range(len(WpN)):
        N_new.append(setsum(WnP[i], WpN[i]))

    P_new = []
    for i in range(len(WpP)):
        P_new_i = add_bias(setsum(WpP[i], WnN[i]), b[i])
        if t > float("-inf"):
            P_new_i = P_new_i.union(add_bias(N_new[i], t))
        P_new.append(P_new_i)

    return P_new, N_new

