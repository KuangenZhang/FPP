import numpy as np
import scipy as sp
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
# code by https://github.com/ClayFlannigan/icp

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # #only allow the rotation around z
    # R_new = np.eye(3)
    # R_new[:2, :2] = R[:2, :2]
    # R = R_new

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    # remove outlier
    dist_A = np.linalg.norm((A - np.mean(A, axis=0,keepdims=True)), axis=-1)
    A = A[dist_A< np.mean(dist_A) + 1 * np.std(dist_A)]
    dist_B = np.linalg.norm((B - np.mean(B, axis=0, keepdims=True)), axis=-1)
    B = B[dist_B < np.mean(dist_B) + 1 * np.std(dist_B)]

    if len(A) < len(B):
        B = B[:len(A)]
    else:
        A = A[:len(B)]

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
        valid_indices = distances < 0.06
        if np.sum(valid_indices) == 0:
            break
        # compute the transformation between the current source and nearest destination points
        src_in_dst, _ ,_ = best_fit_transform(src[:m, valid_indices].T, dst[:m,indices[valid_indices]].T)
        # update the current source
        src = np.dot(src_in_dst, src)
        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)
    # print(distances)
    valid_indices = distances < 0.06
    src = src[:, valid_indices]
    A = A[valid_indices, :]
    src_in_dst,_,_ = best_fit_transform(A, src[:m,:].T)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(src.T[:, 0], src.T[:, 1], src.T[:, 2], c='g', marker='o')
    # ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='r', marker='o')
    # ax.scatter(dst[:, 0], dst[:, 1], dst[:, 2], c='b', marker='*')
    # plt.show()

    return src_in_dst, distances, i
