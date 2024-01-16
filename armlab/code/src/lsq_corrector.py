import numpy as np

def getLsqCorrection(expected, actual, scheme=None):
    """
    Given an expected value E and actual value A, returns the best fit matrix C such that:
    
        E = C * A
    
    So if you want to get E, compute C * A
    
    For IK, if you want to get to E, you should send C * E
    
    Expects wide arrays, i.e. horizontal stacked row vectors
    """
    actual = scheme(actual)
    
    expected, actual = np.asarray(expected), np.asarray(actual)
    return expected @ np.linalg.pinv(actual)


def applyLsqCorrection(value, matrix, scheme):
    value = scheme(value)
    return matrix @ value


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    
    # Testing identity
    a1 = np.array([[1, 2, 3], [4, 5, 6]]).T
    a2 = np.array([[1, 2, 3], [4, 5, 6]]).T
    scheme = lambda x: np.array([1+0*x[0,:], x[0,:], x[1,:], x[2,:]])
    M = getLsqCorrection(a1, a2, scheme)
    
    print(M)
    print(applyLsqCorrection(a2, M, scheme))
    print()
    
    # Testing squared
    t = np.arange(100)
    expected = 2 * t ** 2 + (np.random.rand(1, 100) * 3)
    actual = t
    
    scheme = lambda x: np.array([x[:]**2])
    M = getLsqCorrection(expected, actual, scheme)
    print(M)
    print(applyLsqCorrection(actual, M, scheme))
    
    # Testing possible offsets
    world_mutator = np.array([
        [1, 1, 0, 0],
        [0, 0.1, 1, 0],
        [0, 0, 0, 1],
    ])
    
    expected = np.random.randint(-10, 10, (3, 15))
    scheme = lambda x: np.array([1+0*x[0,:], x[0,:], x[1,:], x[2,:]])
    actual = applyLsqCorrection(expected, world_mutator, scheme)
    print(expected)
    
    recovered_mat = getLsqCorrection(expected, actual, scheme)
    print(applyLsqCorrection(actual, recovered_mat, scheme))