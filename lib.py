"""02504 Computer vision courses."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# week 1
def box3d(n):
    """Generate a set of points: 3D cross in cube."""
    coordinates = np.zeros([3, 15 * n])
    half_length = 0.5
    border = half_length * np.ones(n)
    variation = np.linspace(-half_length, half_length, n)
    elems = np.array([-border, border, variation])
    indices = np.array(
        [
            [0, 0, 2],
            [0, 1, 2],
            [1, 0, 2],
            [1, 1, 2],
            [0, 2, 0],
            [0, 2, 1],
            [1, 2, 0],
            [1, 2, 1],
            [2, 0, 0],
            [2, 0, 1],
            [2, 1, 0],
            [2, 1, 1],
        ]
    )
    for i in range(12):
        coordinates[:, i * n : (i + 1) * n,] = elems[indices[i]]
    for i in range(12, 15):
        coordinates[i % 3, i * n : (i + 1) * n] = variation
    return coordinates


# week 2
def get_distorted(r, distortion_coefficients):
    """Return p(1 + dr(p))."""
    k3, k5, k7 = distortion_coefficients
    n = r[0, :] ** 2 + r[1, :] ** 2
    dr = k3 * n ** 2 + k5 * n ** 4 + k7 * n ** 6
    return r * (1 + dr)


def projectpoints(K, R, t, Q, distortion_coeff=[0, 0, 0]):
    """Return the projected points as a 2xn matrix."""
    _, n = Q.shape
    Q = np.vstack([Q, np.ones(n)])
    T = np.vstack([np.hstack([R, t])])

    r = T @ Q
    r = r[:2, :] / r[2, :]
    dist = get_distorted(r, distortion_coeff)
    P = K @ np.concatenate([dist, np.ones([1, n])], axis=0)
    return P


def hest(q1, q2):
    """
    Parameters
    ----------
    q1, q2: 3 x n numpy arrays
        sets of points
    Return
    ------
    H: 3x3 numpy array
        Estimated homography matrix using the linear algorithm.
    """
    B = get_B(q1, q2)
    U, S, VT = np.linalg.svd(B.T @ B)
    H = np.reshape(VT[-1], (3, 3), 'F')
    return H / H[2, 2]


def get_B(q1, q2):
    """
    Parameters
    ----------
    q1, q2: 3 x n numpy arrays
        sets of points
    """
    B = B_i(q1, q2, 0)
    for i in range(1, len(q1[0])):
        B = np.vstack((B, B_i(q1, q2, i)))
    return B


def B_i(q1, q2, i):
    return np.kron(
            q2[:,i],
            np.array([
                [ 0       , -1       ,  q1[1, i]],
                [ 1       ,  0       , -q1[0, i]],
                [-q1[1, i],  q1[0, i],  0       ],
            ])
        )


def normalize2d(Q):
    """Return the points normalised.

    The mean of Tq is [0, 0] and the standard deviation is [1, 1].

    Parameter
    ---------
    Q: 2 x n or 3 x n numpy array
        set of points to normalise.
    Return
    ------
    Q: 3 x n numpy array
        set of points to normalise.
    """
    d, n = Q.shape
    if d == 2:
        Q = np.vstack(Q, np.ones(n))

    mean = np.mean(Q, axis=1)
    std = np.std(Q, axis=1)

    T = np.array([
        [1 / std[0], 0         , - mean[0] / std[0]],
        [0         , 1 / std[1], - mean[1] / std[1]],
        [0         , 0,            1               ],
    ])
    TQ = T @ Q
    return TQ / TQ[2]