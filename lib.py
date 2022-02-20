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
    n = np.sqrt(r[0, :] ** 2 + r[1, :] ** 2)
    dr = k3 * n ** 2 + k5 * n ** 4 + k7 * n ** 6
    r[:2, :] = r[:2, :] * (1 + dr)
    return r


def projectpoints(K, R, t, Q, distortion_coeff=[0, 0, 0]):
    """Return the projected points as a 2xn matrix."""
    d, n = Q.shape
    if d == 3:
        Q = np.vstack([Q, np.ones(n)])
    T = np.hstack([R, t])

    r = T @ Q
    r = r[:2, :] / r[2, :]
    dist = get_distorted(r, distortion_coeff)
    P = K @ np.concatenate([dist, np.ones([1, n])], axis=0)
    return P


def undistortImage(Image, K, distortion_coeff):
    """
    Parameters
    ----------
    Image:
        image
    K:
        camera matrix
    distortion_coeff:
        distortion coefficients

    Return
    ------
    An undistorted version of the image
    """
    h, w, d = Image.shape
    
    grid = np.ones([h * w, 3], dtype=int)
    columns, rows = np.meshgrid(np.arange(0, w), np.arange(0, h))
    grid[:, 0], grid[:, 1] = rows.flatten(), columns.flatten()

    grid = np.linalg.inv(K) @ grid.T
    grid /= grid[2]
    
    grid = get_distorted(grid, distortion_coeff)
    grid = (K @ grid).T
    grid = grid.reshape(h, w, 3)[:, :, :2].astype(int)
    
    points = (np.arange(0, h), np.arange(0, w))
    I_undistorted = np.zeros(Image.shape)
    for i in range(d):
        I = Image[:, :, i]
        interpolating_function = RegularGridInterpolator(
            points,
            values=I,
            method="nearest",
            bounds_error=False,
            fill_value=0
        )
        I_undistorted[:, :, i] = interpolating_function(grid)
    return I_undistorted.astype(int)


def hest(q_before_H, q_after_H):
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
    T1, q1 = normalize2d(q_after_H)
    T2, q2 = normalize2d(q_before_H)
    B = get_B(q1, q2)
    u, s, vh = np.linalg.svd(B)
    H = np.reshape(vh[-1, :], (3, 3), 'F')
    H = np.linalg.inv(T1) @ H @ T2
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
    """Return the normalisation matrix and the normalised points.

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
    return T, TQ / TQ[2]


# week 3
def CrossOp(p):
    """Return the cross product operator.
    
    Take a vector in 3D and return the 3×3 matrix corresponding
    to taking the cross product with that vector.
    
    Parameter
    ---------
    p: 3 x 1 numpy array
    
    Return
    ------
    CrossOP: 3 x 3 numpy array
        cross product operator.
    
    """
    x, y, z = p.reshape(3)
    return np.array([
        [ 0, - z,  y],
        [ z,   0, -x],
        [-y,   x,  0],
    ])

def essential_matrix(R, t):
    """Return the essential matrix
    
    Parameters
    ----------
    R: 3 x 3 numpy array
        Rotation matrix of camera 2 in the referential of camera 1
    t: 3 x 1 numpy array
        Translation matrix of camera 2 in the referential of camera 1
    
    Return:
    E: 3 x 3 numpy array
        Essential matrix
    """
    return CrossOp(t) @ R


def fundamental_matrix(K1, K2, R2, t2, R1=np.eye(3), t1=np.zeros(3)):
    """Return the fundamental matrix.
    
    If R1 and t1 are not specified, we assume that we are in the
    coordinate system of camera 1.
    
    Parameters
    ----------
    K1, K2: 3 x 3 numpy array
        intrinsics parameters of the cameras
    R2: 3 x 3 numpy array
        Rotation matrix of camera 2.
    t2: 3 x 1 numpy array
        Translation matrix of camera 2.
    R1: 3 x 3 numpy array, optional. Default: identity matrix
        Rotation matrix of camera 1.
    t2: 3 x 1 numpy array, optional. Default: null vector
        Translation matrix of camera 1.
    
    Return:
    F: 3 x 3 numpy array
        Fundamental matrix
    """
    R = R2 @ R1.T
    t = t2.reshape(3, 1) - R2 @ R1.T @ t1.reshape(3, 1)

    E = essential_matrix(R, t)
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def epipolar_line(F, q, i=1):
    """
    Epipolar line in camera i.

    Parameters
    ----------
    F: 3 x 3 numpy array
        fundamental matrix.
    q: 3 x 1 numpy array
        homogeneous point in camera i.
    i: int, optional. default: 1
        number of camera, 1 or 2.
    """
    if i == 1:
        return F @ q.reshape(3, 1)
    else:
        return q.reshape(1, 3) @ F


def DrawLine(l, shape, ax):
    """
    Checks where the line intersects the four sides of the image
    and finds the two intersections that are within the frame.
    """
    def in_frame(l_im):
        q = np.cross(l.flatten(), l_im)
        q = q[:2] / q[2]
        if all(q >= 0) and all(q + 1 <= shape[1::-1]):
            return q
    lines = [
        [1, 0, 0           ],
        [0, 1, 0           ],
        [1, 0, 1 - shape[1]],
        [0, 1, 1 - shape[0]]
    ]
    P = [in_frame(l_im) for l_im in lines if in_frame(l_im) is not None]
    ax.plot(*np.array(P).T, 'r-')

    
def click_and_draw(im, im2, F, reverse_axis=False):
    # Turn off the matplotlib inline to use ginput.
    %matplotlib qt  
    n = 1

    plt.imshow(im)
    x = np.array(plt.ginput(n))
    plt.show()

    q1 = np.hstack(
        (x, np.ones((1, n)))
        ).T
    l2 = F @ q1
    
    %matplotlib inline
    
    i, j = 0, 1
    if reverse_axis:
        i, j = j, i

    fig, ax = plt.subplots(1, 2, figsize=(20, 20))
    ax[i].imshow(im1, 'gray')
    ax[j].imshow(im2, 'gray')
    ax[i].plot(x[:, 0], x[:, 1], 'ro')
    DrawLine(l2, im2.shape, ax[j])
    
    plt.plot()


def triangulate(q, P):
    """
    Return the traingulation.
    
    Parameters
    ----------
    q: 3 x n numpy array
        Pixel coordinates q1... qn
    P: list of 3 x 4 numpy arrays
        Projection matrices P1... Pn
    
    Return
    ------
    Q: 3 x 1 numpy array
        Triangulation of the point using the linear algorithm
    """
    _, n = q.shape

    B = np.zeros((2 * n, 4))
    for i in range(n):
        B[2 * i: 2 * i + 2] = [
            P[i][2, :] * q[0, i] - P[i][0, :],
            P[i][2, :] * q[1, i] - P[i][1, :],
        ]
    u, s, vh = np.linalg.svd(B)
    Q = vh[-1, :]
    return Q[:3].reshape(3, 1) / Q[3]