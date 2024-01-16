import numpy as np

# np.set_printoptions(precision=4, suppress=True)

# From https://stackoverflow.com/a/59204638
def rotationMatrixFromVectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return R


def getExtrinsic():
    cameraDown = np.array([-0.235, 2.334, -9.346]) # Measured from IMU
    H_1_0 = np.array([
        [1, 0, 0, -30],
        [0, 1, 0, 445],
        [0, 0, 1, 990],
        [0, 0, 0, 1]
    ])
    
    R_2_1 = rotationMatrixFromVectors(cameraDown, [0, 0, -1])
    H_2_1 = np.eye(4)
    H_2_1[0:3, 0:3] = R_2_1
    
    H_3_2 = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    inv_H = H_1_0 @ H_2_1 @ H_3_2
    
    return np.linalg.inv(inv_H)

if __name__ == '__main__':
    # print(getExtrinsic().tolist())
    print(getExtrinsic())