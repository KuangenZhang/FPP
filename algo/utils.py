import numpy as np
import cv2
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import numpy.matlib as npm
# import pytesseract
# import re
import glob
import shutil
import os
import pywt
import copy
import time
import matplotlib
import seaborn as sn
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from algo import icp
from shutil import copyfile
from scipy.stats import ttest_ind
from matplotlib.markers import TICKDOWN
from matplotlib.patches import Ellipse
from matplotlib import cm
from tqdm import tqdm
from scipy import stats


plt.rcParams["font.family"] = "Arial"
plt.rcParams['pdf.fonttype'] = 42 # avoid type 3 font error in ICRA, IROS, and RA

'''
This is for the current orbbec camera. This parameter should be changed if the camera is changed.
'''
default_cam_intri = np.asarray(
[[521.85359567,   0.        , 321.18647073],
[0.        , 521.7098714 , 233.81475134],
[0.        ,   0.        ,   1.        ]])


def uvz2xyz(u, v, d, cam_intri_inv= None):
    if cam_intri_inv is None:
        cam_intri_inv = np.linalg.inv(default_cam_intri)
    uvd_vec = np.asarray([u, v, 1]) * d
    xyz = np.matmul(cam_intri_inv, uvd_vec)
    return xyz

def uv_vec2cloud(uv_vec, depth_img, depth_scale = 1e-3):
    '''
        uv_vec: n×2,
        depth_image: rows * cols
    '''
    fx = default_cam_intri[0, 0]
    fy = default_cam_intri[1, 1]
    cx = default_cam_intri[0, 2]
    cy = default_cam_intri[1, 2]
    cloud = np.zeros((len(uv_vec), 3))
    cloud[:, 2] = depth_img[uv_vec[:, 1].astype(int), uv_vec[:, 0].astype(int)] * depth_scale
    cloud[:, 0] = (uv_vec[:, 0] - cx) * (cloud[:, 2] / fx)
    cloud[:, 1] = (uv_vec[:, 1] - cy) * (cloud[:, 2] / fy)
    return cloud


def depth2cloud(img_depth, cam_intri_inv = None):
    # default unit of depth and point cloud is mm.
    if cam_intri_inv is None:
        cam_intri_inv = np.zeros((1, 1, 3, 3))
        cam_intri_inv[0, 0] = np.linalg.inv(default_cam_intri)
    uv_vec = np.transpose(np.mgrid[0:480, 0:640], (1, 2, 0))[:, :, [1, 0]].reshape((-1, 2))
    point_cloud = uv_vec2cloud(uv_vec, img_depth, depth_scale=1)
    return point_cloud


def visualize_rgbd_point_cloud(depth_raw, color_raw):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(depth2cloud(depth_raw))
    pcd.colors = o3d.utility.Vector3dVector((color_raw.astype(np.float) / 255.0).reshape((-1, 3)))
    o3d.visualization.draw_geometries([pcd])



# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation
def averageQuaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = npm.outer(q,q) + A

    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0])


# Average multiple quaternions with specific weights
# The weight vector w must be of the same length as the number of rows in the
# quaternion maxtrix Q
def weightedAverageQuaternions(Q, w):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4,4))
    weightSum = 0

    for i in range(0,M):
        q = Q[i,:]
        A = w[i] * np.outer(q,q) + A
        weightSum += w[i]

    # scale
    A = (1.0/weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0])


def draw_circles_on_img(img_rgb_tracker, imgpoints_tracker, imgpoints_est_tracker):
    img_rgb_tracker = np.copy(img_rgb_tracker)
    for r in range(imgpoints_tracker.shape[0]):
        img_rgb_tracker = cv2.circle(img_rgb_tracker, tuple(imgpoints_tracker[r].astype(np.int)),
                                     radius=10, color= (0, 255, 0),thickness = 2)
        img_rgb_tracker = cv2.circle(img_rgb_tracker, tuple(imgpoints_est_tracker[r].astype(np.int)),
                                     radius=10, color=(0, 0, 255),thickness = 2)
    return img_rgb_tracker


def interactive_img_show(img):
    fig, ax = plt.subplots()
    im = ax.imshow(img, interpolation='none')
    ax.format_coord = Formatter(im)
    plt.show()


class Formatter(object):
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)


def read_target_distance(file_name, camera_offset = 15):
    df = pd.read_csv(file_name, sep='\t', skiprows=4).values
    data = df[0:20, :].astype(float)

    eye_pos = data[:, 20:23]
    eye_pos[:, -1] -= camera_offset
    target_pos = data[:, 11:14]
    target_distance = np.mean(np.linalg.norm(eye_pos - target_pos, axis=-1))
    return target_distance


def haar_dec(array_2d):
    coeffs = pywt.dwt2(array_2d, 'haar')
    cA, (cH, cV, cD) = coeffs
    return np.r_[np.c_[cA, cH], np.c_[cV, cD]]

def haar_rec(feature_2d):
    rows, cols = feature_2d.shape
    half_row = int(rows / 2)
    half_col = int(cols / 2)
    cA = feature_2d[:half_row, :half_col]
    cH = feature_2d[:half_row, half_col:]
    cV = feature_2d[half_row:, :half_col]
    cD = feature_2d[half_row:, half_col:]
    coeffs = tuple([cA, tuple([cH, cV, cD])])
    return pywt.idwt2(coeffs, 'haar')

def haar_dec_multi(feature_2d, ite_num = 5):
    # for i in range(ite_num):
    #     array_2d = haar_dec(array_2d)
    feature_2d = haar_dec(feature_2d)
    height, width = feature_2d.shape
    for i in range(1, ite_num):
        rows, cols = int(height / 2 ** i), int(width / 2 ** i)
        feature_2d[:rows, :cols] = haar_dec(feature_2d[:rows, :cols])
    return feature_2d

def haar_rec_multi(feature_2d, ite_num = 5):
    # for i in range(ite_num):
    #     feature_2d = haar_rec(feature_2d)
    height, width = feature_2d.shape
    for i in range(ite_num):
        idx = ite_num - 1- i
        rows, cols = int(height / 2 ** idx), int(width / 2 ** idx)
        feature_2d[:rows, :cols] = haar_rec(feature_2d[:rows, :cols])
    return feature_2d

def haar_filter(array_2d, ite_num = 9, threshold = 10):
    feature_width = 2**ite_num
    height, width = array_2d.shape
    feature_2d = np.zeros((feature_width, feature_width))
    init_row = int((feature_width - height) / 2 - 1)
    init_col = int((width - feature_width) / 2 - 1)
    feature_2d[init_row:init_row+height, :] = array_2d[:, init_col:init_col+feature_width]
    feature_2d = haar_dec_multi(feature_2d, ite_num = ite_num)
    feature_2d -= threshold
    feature_2d[0, 0] += threshold
    feature_2d = haar_rec_multi(feature_2d, ite_num=ite_num)
    array_2d[:, init_col:init_col + feature_width] = feature_2d[init_row:init_row+height, :]
    return array_2d

def moving_average_within_threshold(x_t, x_t_plus_1, new_ratio = 0.5 ,threshold = 50):
    zero_indices = np.copy(x_t_plus_1 == 0)
    valid_indices = np.copy(np.abs(x_t_plus_1-x_t) < threshold)
    x_t_plus_1[valid_indices] = new_ratio * x_t_plus_1[valid_indices] + (1-new_ratio) * x_t[valid_indices]
    x_t_plus_1[zero_indices] = x_t[zero_indices]
    return x_t_plus_1

def edge_preserving_filter(img):
    rows, cols = img.shape
    for i in range(1):
        for r in range(1, rows):
            img[r, :] = moving_average_within_threshold(img[r-1, :], img[r, :])
        for c in range(1, cols):
            img[:, c] = moving_average_within_threshold(img[:, c-1], img[:, c])
    return img

def plane_fitting(depth_mat):
    radius = int(depth_mat.shape[0] / 2)
    uv_mat = np.ones(((2 * radius) ** 2, 3))

    uv_mat[:, :2] = np.transpose(np.mgrid[0:2 * radius, 0:2 * radius], (1, 2, 0)).reshape((-1, 2))
    depth_vec = depth_mat.reshape((-1, 1))

    plane_coeff = np.matmul(np.linalg.pinv(uv_mat), depth_vec)
    depth_mat = np.matmul(uv_mat, plane_coeff).reshape((2 * radius, 2 * radius))
    return depth_mat


def tracker_to_virtual_points(uv_tracker, img_depth, rgbd_tracker_paras):
    uv_tracker = uv_tracker.reshape([1, -1])
    cam_matrix_tracker = rgbd_tracker_paras.get('cam_matrix_tracker')
    T_tracker_in_depth = np.linalg.inv(rgbd_tracker_paras.get('T_mat_depth_in_tracker'))

    p_eye_in_depth = cam1_to_cam2_xyz(uv_tracker, np.asarray([0]),
                                      cam1_matrix=cam_matrix_tracker,
                                      T_cam1_in_cam2=T_tracker_in_depth)[:3].T
    p_virtual_point_in_depth = cam1_to_cam2_xyz(uv_tracker, np.asarray([2000]),
                                      cam1_matrix=cam_matrix_tracker,
                                      T_cam1_in_cam2=T_tracker_in_depth)[:3].T
    return p_eye_in_depth, p_virtual_point_in_depth

def tracker_to_rgbd_without_depth(uv_tracker, img_depth, rgbd_tracker_paras):
    uv_tracker = uv_tracker.reshape([1, -1])
    cam_matrix_tracker = rgbd_tracker_paras.get('cam_matrix_tracker')
    T_tracker_in_depth = np.linalg.inv(rgbd_tracker_paras.get('T_mat_depth_in_tracker'))
    points_in_depth = depth2cloud(img_depth)

    start = time.time()
    p_eye_in_depth = cam1_to_cam2_xyz(uv_tracker, np.asarray([0]),
                                      cam1_matrix=cam_matrix_tracker,
                                      T_cam1_in_cam2=T_tracker_in_depth)[:3].T
    p_virtual_point_in_depth = cam1_to_cam2_xyz(uv_tracker, np.asarray([1000]),
                                                cam1_matrix=cam_matrix_tracker,
                                                T_cam1_in_cam2=T_tracker_in_depth)[:3].T

    indices = np.arange(points_in_depth.shape[0])
    np.random.shuffle(indices)

    points_in_depth = points_in_depth[indices[:10000]]
    indices = indices[:10000]

    valid_indices = np.abs(points_in_depth[:, 2]) > 100
    points_in_depth = points_in_depth[valid_indices]
    indices = indices[valid_indices]

    distance = np.linalg.norm(np.cross(points_in_depth - p_eye_in_depth,
                                       points_in_depth-p_virtual_point_in_depth), axis=-1)
    distance /= np.linalg.norm(p_eye_in_depth - p_virtual_point_in_depth, axis=-1)
    index_min = indices[np.argmin(distance)]
    v = int(np.floor(index_min / img_depth.shape[1]))
    u = index_min - v * img_depth.shape[1]
    uv_est_in_depth = np.asarray([u, v])
    print('Computing time of 2D gaze to 3D gaze: {:0f} ms'.format(1000 * (time.time() - start)))
    return uv_est_in_depth



def tracker_to_rgbd_with_depth(imgpoints_tracker, depth_points_tracker, rgbd_tracker_paras):
    T_tracker_in_depth = np.linalg.inv(rgbd_tracker_paras.get('T_mat_depth_in_tracker'))
    imgpoints_est_depth = cam1_to_cam2(
        imgpoints_tracker, depth_points_tracker,
        cam1_matrix=rgbd_tracker_paras.get('cam_matrix_tracker'),
        cam2_matrix=rgbd_tracker_paras.get('cam_matrix_depth'),
        T_cam1_in_cam2=T_tracker_in_depth)
    return imgpoints_est_depth

def tracker_to_rgbd(imgpoints_tracker, img_depth, rgbd_tracker_paras):
    imgpoints_est_depth = np.zeros(imgpoints_tracker.shape)
    for i in range(len(imgpoints_tracker)):
        imgpoints_est_depth[i] =  tracker_to_rgbd_without_depth(
            imgpoints_tracker[i], img_depth, rgbd_tracker_paras)
    return imgpoints_est_depth


def plot_rgb_cloud_from_depth(img_depth, img_rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(depth2cloud(img_depth))
    pcd.colors = o3d.utility.Vector3dVector((img_rgb.astype(np.float) / 255.0).reshape((-1, 3)))
    o3d.visualization.draw_geometries([pcd])

def plot_rgb_cloud(cloud, img_rgb = None, view = True, mesh_cylinder=None):
    pcd = o3d.geometry.PointCloud()
    z_vec = np.copy(cloud[:, 2])
    pcd.points = o3d.utility.Vector3dVector(cloud)
    if img_rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector((img_rgb.astype(np.float) / 255.0).reshape((-1, 3)))
    if view:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
        # vis = o3d.visualization.draw_geometries([pcd])
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        # if mesh_cylinder is not None:
        vis.add_geometry(mesh_cylinder)
        opt = vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])
        vis.run()
    return pcd


def plot_rgb_cloud_with_gaze_vector(img_depth, img_rgb, uv_depth, uv_tracker, rgbd_tracker_paras):
    uv_depth_new = np.copy(uv_depth).reshape((1, -1))
    img_rgb_correct = draw_circles_on_img(img_rgb, uv_depth_new, uv_depth_new)

    pcd = o3d.geometry.PointCloud()
    img_rgb_correct = cv2.cvtColor(img_rgb_correct, cv2.COLOR_BGR2RGB)

    pcd.points = o3d.utility.Vector3dVector(depth2cloud(img_depth))
    pcd.colors = o3d.utility.Vector3dVector((img_rgb_correct.astype(np.float) / 255.0).reshape((-1, 3)))

    uv_depth_est = tracker_to_rgbd_without_depth(uv_tracker, img_depth, rgbd_tracker_paras)
    center = uvz2xyz(uv_depth_est[0], uv_depth_est[1], img_depth[uv_depth_est[1],uv_depth_est[0]])
    points_in_depth = center.reshape((1, 3)) + np.random.uniform(-1, 1, (100, 3))
    pcd_depth = o3d.geometry.PointCloud()
    pcd_depth.points = o3d.utility.Vector3dVector(points_in_depth)
    color_vec_depth = np.zeros(points_in_depth.shape)
    color_vec_depth[:, 2] = 1
    pcd_depth.colors = o3d.utility.Vector3dVector(color_vec_depth)

    cam_matrix_tracker = rgbd_tracker_paras.get('cam_matrix_tracker')
    T_tracker_in_depth = np.linalg.inv(rgbd_tracker_paras.get('T_mat_depth_in_tracker'))


    p_eye_in_depth = cam1_to_cam2_xyz(uv_tracker, np.asarray([0]),
                                      cam1_matrix=cam_matrix_tracker,
                                      T_cam1_in_cam2=T_tracker_in_depth)[:3].T
    p_virtual_point_in_depth = cam1_to_cam2_xyz(uv_tracker, np.asarray([8000]),
                                                cam1_matrix=cam_matrix_tracker,
                                                T_cam1_in_cam2=T_tracker_in_depth)[:3].T

    s = np.linspace(0, 1, num = 10000)
    gaze_vector = np.matmul(s.reshape((-1, 1)), p_virtual_point_in_depth - p_eye_in_depth) + p_eye_in_depth

    pcd_gaze = o3d.geometry.PointCloud()
    pcd_gaze.points = o3d.utility.Vector3dVector(gaze_vector)
    color_vec_depth = np.zeros(gaze_vector.shape)
    color_vec_depth[:, 1] = 1
    pcd_gaze.colors = o3d.utility.Vector3dVector(color_vec_depth)
    o3d.visualization.draw_geometries([pcd, pcd_depth, pcd_gaze])


def rgbd_to_tracker(imgpoints_depth, depth_points, rgbd_tracker_paras):
    imgpoints_est_tracker = cam1_to_cam2(imgpoints_depth, depth_points,
                                         cam1_matrix = rgbd_tracker_paras.get('cam_matrix_depth'),
                                         cam2_matrix = rgbd_tracker_paras.get('cam_matrix_tracker'),
                                         T_cam1_in_cam2 = rgbd_tracker_paras.get('T_mat_depth_in_tracker'))
    return imgpoints_est_tracker

def xyz1_to_xyz2(xyz1, rgbd_tracker_paras):
    p_vec_1 = np.ones((4, 1))
    p_vec_1[:3, 0] = xyz1
    T_cam1_in_cam2 = rgbd_tracker_paras.get('T_mat_depth_in_tracker')
    p_vec_2 = np.matmul(T_cam1_in_cam2, p_vec_1)
    return p_vec_2[:3, 0]

def cam1_to_cam2(uv_vec_1, z_vec_1, cam1_matrix, cam2_matrix, T_cam1_in_cam2):
    p_vec_2 = cam1_to_cam2_xyz(uv_vec_1, z_vec_1, cam1_matrix, T_cam1_in_cam2)
    uvz_vec_2 = np.matmul(cam2_matrix, p_vec_2[:3])
    uv_vec_2 = uvz_vec_2[:2] / uvz_vec_2[[2]]
    uv_vec_2 = uv_vec_2.T
    return uv_vec_2

def cam1_to_cam2_xyz(uv_vec_1, z_vec_1, cam1_matrix, T_cam1_in_cam2):
    uv_vec_1 = uv_vec_1.reshape((-1, 2))
    num_points = uv_vec_1.shape[0]
    uv_vec_1 = np.asarray(uv_vec_1)
    uvz_vec_1 = np.ones((num_points, 3))
    uvz_vec_1[:, :2] = uv_vec_1
    uvz_vec_1 *= np.reshape(z_vec_1, (-1, 1))
    p_vec_1 = np.ones((4, num_points))
    p_vec_1[:3, :] = np.matmul(np.linalg.inv(cam1_matrix), uvz_vec_1.T)
    p_vec_2 = np.matmul(T_cam1_in_cam2, p_vec_1)
    return p_vec_2

def cam1_to_cam2_without_calibration(uv_1, u_ratio, v_ratio):
    uv_2 = np.copy(uv_1)
    uv_2[:, 0] = uv_1[:, 0] * u_ratio
    uv_2[:, 1] = uv_1[:, 1] * v_ratio
    return uv_2

# def read_numbers_from_img(img_text, is_show=False):
#     # converting image into gray scale image
#     img_text = cv2.cvtColor(img_text, cv2.COLOR_BGR2GRAY)
#     # converting it to binary image by Thresholding
#     # this step is require if you have colored image because if you skip this part
#     # then tesseract won't able to detect text correctly and this will give incorrect result
#     img_text = cv2.threshold(img_text, 254.5, 255, cv2.THRESH_BINARY)[1]
#     text = pytesseract.image_to_string(img_text)
#     float_array = np.asarray(re.findall(r"[-+]?\d*\.\d+|\d+", text), dtype=np.float32)
#     if is_show:
#         cv2.imshow('threshold_img: {}, {}'.format(float_array[1:3],float_array[-3:]), img_text)
#         cv2.waitKey(100)
#     return float_array

def obtain_file_time(file_name):
    return float(os.path.splitext(os.path.basename(file_name))[0])

def read_image_to_video(img_name_vec, video_name, fps=2):
    if os.path.exists(video_name):
        os.remove(video_name)
    img_name_vec = sorted(img_name_vec)
    img_array = []
    for filename in img_name_vec:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # H.264 encoder
    out = cv2.VideoWriter(video_name, fourcc, fps, size)


    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def fig_to_img():
    img_name = 'temp.jpg'
    plt.savefig(img_name, bbox_inches='tight')
    img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    os.remove(img_name)
    return img

def split_rgbd_data_to_folder():
    data_dir = '../data/dynamic'
    color_img_name_vec = glob.glob('{}/*.jpg'.format(data_dir))
    for r in range(len(color_img_name_vec)):
        color_img_name = color_img_name_vec[r]
        depth_img_name = color_img_name.replace(".jpg", ".npy")
        # depth_img_name = depth_img_name.replace("rgb/", "depth/")
        img_dir = '{}/test{}'.format(data_dir, r)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        copyfile(color_img_name, '{}/rgb.jpg'.format(img_dir))
        copyfile(depth_img_name, '{}/depth.npy'.format(img_dir))

def visualize_gaze_and_environment():
    val_dir = 'data/terrain_RGBD'
    exp_index = 2
    # depth_img_name = glob.glob('data/validation/img/test{}/depth/*.npy'.format(exp_index))[0]
    color_img_name = glob.glob('{}/test{}/rgb/*.jpg'.format(val_dir, exp_index))[0]
    validation_results = np.load('{}/tracker_to_depth_validation.npy'.format(val_dir), allow_pickle=True).item()
    target_uv_in_depth_vec = validation_results.get('target_uv_in_depth_vec')
    target_uv_in_tracker_vec = validation_results.get('target_uv_in_tracker_vec')

    depth_img_name_vec = glob.glob('{}/test{}/depth/*.npy'.format(val_dir, exp_index))
    img_depth_vec = np.zeros((len(depth_img_name_vec), 480, 640))

    for r in range(len(depth_img_name_vec)):
        depth_img_name = depth_img_name_vec[r]
        img_depth = np.load(depth_img_name) * 1000.0  # convert m to mm
        # img_depth = edge_preserving_filter(img_depth)
        img_depth_vec[r] = img_depth
    img_depth = np.mean(img_depth_vec, axis=0)
    # img_depth[img_depth > 3000] = 0

    interactive_img_show(img_depth)

    img_rgb = cv2.imread(color_img_name, cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    print('Max depth value: {}'.format(np.max(img_depth)))
    print(img_depth.shape, img_rgb.shape)

    rgbd_tracker_paras = np.load('../paras/rgbd_tracker_paras.npy', allow_pickle=True).item()
    plot_rgb_cloud_with_gaze_vector(img_depth, img_rgb, target_uv_in_depth_vec[exp_index - 1],
                                    target_uv_in_tracker_vec[exp_index - 1], rgbd_tracker_paras)



def read_rgb_cloud(val_dir = 'data/orbbec_terrains/test1', img_idx = 50, view_cloud = False):
    rgb_img_names = glob.glob('{}/rgb/*.jpg'.format(val_dir, img_idx))
    rgb_img_names.sort()

    cloud_names = glob.glob('{}/depth/*.npy'.format(val_dir, img_idx))
    cloud_names.sort()

    rgb_name = rgb_img_names[img_idx]
    cloud_name = cloud_names[img_idx]
    print(rgb_name, cloud_name)

    point_cloud = np.load(cloud_name) * 1000.0  # convert m to mm
    # point_cloud[point_cloud[..., 2] > 3000] = 0
    img_rgb = cv2.imread(rgb_name, cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    if view_cloud:
        plot_rgb_cloud(point_cloud.reshape((-1, 3)), img_rgb)
        interactive_img_show(point_cloud[:,:, 2])

    return img_rgb, point_cloud

def obtain_folder_number(folder_name_list, num_idx = 4):
    folder_num_vec = []
    for folder_name in folder_name_list:
        folder_num = float(os.path.basename(os.path.dirname(folder_name))[num_idx:])
        folder_num_vec.append(folder_num)
    return np.asarray(folder_num_vec)

def obtain_file_time_vec(file_name_list):
    file_time_vec = []
    for file_name in file_name_list:
        file_time = float(os.path.splitext(os.path.basename(file_name))[0])
        file_time_vec.append(file_time)
    file_time_vec = np.around(np.asarray(file_time_vec), decimals=3)
    return file_time_vec

def synchronize_for_single_file(file_name_list, reference_time):
    file_name_list = np.asarray(file_name_list)
    file_time_vec = obtain_file_time_vec(file_name_list)
    time_error_vec = np.abs(file_time_vec - reference_time).squeeze()
    return file_name_list[time_error_vec < 0.1]

def find_nearest_file(file_name_list, reference_time):
    file_time_vec = obtain_file_time_vec(file_name_list)
    time_error_vec = np.abs(file_time_vec - reference_time).squeeze()
    return file_name_list[np.argmin(time_error_vec)]

def synchronize_file_list_and_time_vec(file_name_list, reference_time_vec, is_find_nearest_file = False):
    synchronized_files = []
    for reference_time in reference_time_vec:
        if is_find_nearest_file:
            synchronized_files.append(find_nearest_file(file_name_list, reference_time))
        else:
            synchronized_files.append(synchronize_for_single_file(
                file_name_list, reference_time))
    return synchronized_files


def synchronize_two_file_list(file_name_list, reference_file_list, is_find_nearest_file = False):
    reference_time_vec = obtain_file_time_vec(reference_file_list)
    return synchronize_file_list_and_time_vec(file_name_list, reference_time_vec, is_find_nearest_file)

def extract_files_within_time_interval(file_name_list, time_interval):
    '''
        file_name_list: a list of files that need to be processed.
        time_interval: a 2-element list, time_inverval[0] = start, time_inverval[1] = end
    '''
    file_time_vec = obtain_file_time_vec(file_name_list)
    valid_indices = np.bitwise_and(file_time_vec <= time_interval[1],
                                   file_time_vec >= time_interval[0])
    return file_name_list[valid_indices]

def extract_files_within_time_interval_of_reference_files(file_name_list, reference_file_list):
    reference_file_time_vec = obtain_file_time_vec(reference_file_list)
    time_interval = [np.min(reference_file_time_vec), np.max(reference_file_time_vec)]
    return extract_files_within_time_interval(file_name_list, time_interval)

def remove_files_far_from_reference_time_interval(file_name_list, reference_file_list):
    reference_file_time_vec = obtain_file_time_vec(reference_file_list)
    time_interval = [np.min(reference_file_time_vec), np.max(reference_file_time_vec)]
    file_time_vec = obtain_file_time_vec(file_name_list)
    invalid_indices = np.bitwise_or(file_time_vec <= time_interval[0] - 3,
                                    file_time_vec >= time_interval[1] + 3)
    for file_name in file_name_list[invalid_indices]:
        os.remove(file_name)


def tracker_to_rgbd_xyz(gaze_2d, img_depth, depth_scale = 1000):
    '''
    gaze 2d: u in [0, 1], v in [0, 1]
    image depth: unit mm
    xyz: unit m
    '''
    print(gaze_2d)
    uv_in_tracker = gaze_2d * np.asarray([1920, 1080])
    rgbd_tracker_paras = np.load('paras/rgbd_tracker_paras.npy', allow_pickle=True).item()
    u, v = tracker_to_rgbd_without_depth(uv_in_tracker, img_depth, rgbd_tracker_paras)
    d = img_depth[v, u] / depth_scale
    return uvz2xyz(u, v, d), (u,v)



def view_pcd_with_axis(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    world_axes.rotate(np.asarray([[1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, -1]]), center=np.asarray([0, 0, 0]))
    world_axes.translate(np.mean(np.asarray(pcd.points) + np.asarray([0, 1, -0.05]), axis=0))
    vis.add_geometry(pcd)
    vis.add_geometry(world_axes)
    vis.run()


def view_vis(vis, pcd_list, viewer_setting_file = None, img_name = None):
    # best view status
    vis.clear_geometries()
    for init_pcd in pcd_list:
        vis.add_geometry(init_pcd)
    if viewer_setting_file is not None:
        vis = load_view_point(vis, viewer_setting_file)
    vis.poll_events()
    vis.update_renderer()
    if img_name is not None:
        vis.capture_screen_image(img_name, do_render=False)
    return vis


def save_view_point(vis, viewer_setting_file):
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(viewer_setting_file, param)

def load_view_point(vis, viewer_setting_file):
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(viewer_setting_file)
    ctr.convert_from_pinhole_camera_parameters(param)
    return vis

def create_vis_window(init_pcd_list, fov_step=-45, r_x = 0, r_y = 650,
                      t_x = 0, t_y = 200, zoom = 0.2, img_name = None):
    # best view status
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for init_pcd in init_pcd_list:
        vis.add_geometry(init_pcd)
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=fov_step)
    ctr.rotate(r_x, r_y)  # the distance of mouse along x and y axes.
    ctr.translate(t_x, t_y)
    ctr.set_zoom(zoom)  # the distance of mouse along x and y axes.
    vis.poll_events()
    vis.update_renderer()
    if img_name is not None:
        vis.capture_screen_image(img_name)
    return vis


def custom_draw_geometry_with_custom_fov(vis, init_pcd_list, pcd_list,
                                         img_name = None):
    for i in range(len(init_pcd_list)):
        init_pcd = init_pcd_list[i]
        pcd = pcd_list[i]
        init_pcd.points = pcd.points
        if 0 == i:
            init_pcd.colors = pcd.colors
        vis.update_geometry(init_pcd)
    vis.poll_events()
    vis.update_renderer()
    if img_name is not None:
        vis.capture_screen_image(img_name)


def plot_significance_bar(data1, data2, y_label = 'error', x_label_tuple = None):
    means = np.asarray([np.mean(data1), np.mean(data2)])
    stds = np.asarray([np.std(data1), np.std(data2)])
    ind  = np.arange(2)    # the x locations for the groups
    width= 0.7
    if x_label_tuple is None:
        x_label_tuple = ('Subject 1', 'Subject 2')

    # Pull the formatting out here
    colors = plt.get_cmap("tab10")(ind / 10)
    bar_kwargs = {'width':width,'color':colors,'linewidth':2,'zorder':0}
    err_kwargs = {'zorder':1,'fmt': 'none','linewidth':2,'ecolor':'k', 'capsize': 5}  #for matplotlib >= v1.4 use 'fmt':'none' instead

    ax = plt.gca()
    ax.p1 = plt.bar(ind, means, **bar_kwargs)
    ax.errs = plt.errorbar(ind, means, yerr=stds, **err_kwargs)

    _, p = ttest_ind(data1, data2)
    if p >= 1e-3:
        p_text = 'p={:.3f}'.format(p)
    else:
        p_text = 'p<0.001'.format(p)
    label_diff(0, 1, p_text, ind, means + stds)
    plt.xticks(ind, x_label_tuple, color='k')
    plt.ylabel(y_label)


# Custom function to draw the diff bars
def label_diff(i,j,text,X,Y):
    height = 1.1*max(Y[i], Y[j])
    bar_centers = np.array([X[i], X[j]])
    significance_bar(bar_centers[0], bar_centers[1], height, text)
    plt.ylim([0, 1.1 * height])

def significance_bar(start,end,height,displaystring,linewidth = 1.2,markersize = 8,boxpad  =0.3,fontsize = 10,color = 'k'):
    # draw a line with downticks at the ends
    plt.plot([start,end],[height]*2,'-',color = color,lw=linewidth,marker = TICKDOWN,markeredgewidth=linewidth,markersize = markersize)
    # draw the text with a bounding box covering up the line
    plt.text(0.5*(start+end),height,displaystring,ha = 'center',va='center',bbox=dict(facecolor='1.', edgecolor='none',boxstyle='Square,pad='+str(boxpad)),size = fontsize)

def plot_bar(data1, data2, y_label = 'error'):
    means = [np.mean(data1), np.mean(data2)]
    ind = np.arange(2)  # the x locations for the groups
    width = 0.7
    labels = ('Subject 1', 'Subject 2')

    # Pull the formatting out here
    colors = plt.get_cmap("tab20")(ind / 20)
    bar_kwargs = {'width': width, 'color': colors, 'linewidth': 2, 'zorder': 5}

    ax = plt.gca()
    ax.p1 = plt.bar(ind, means, **bar_kwargs)
    plt.xticks(ind, labels, color='k')
    plt.ylabel(y_label)




def plot_scatter_with_confidence_ellipse(x = None, y=None, ax = None, xlim=None, ylim=None,
                                         color = np.array([[0, 0, 1]]), s = 1):
    if x is None or y is None:
        data = np.random.random((1000, 2))
        x = data[:, 0]
        y = data[:, 1]
    if ax is None:
        plt.figure(1)
        ax = plt.subplot(111)
    scatter_color = copy.deepcopy(color)
    scatter_color[0, -1] = 0.5
    plt.scatter(x, y, s=s, c=tuple(scatter_color), label=None)
    plt.axvline(c='gray', label=None)
    plt.axhline(c='gray', label=None)
    plt.scatter(np.mean(x), np.mean(y), c=color, s=20, label=None)
    confidence_ellipse(x, y, ax, n_std=1.0, edgecolor=color[0], linewidth = 2, label=None)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    # plt.show()



def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def orb_rgbd_odometry(ref_depth, cur_color, cur_depth,
                      orb, matcher, kp_ref, d_ref, init_current_in_previous = None):
    kp_cur, d_cur = orb.detectAndCompute(cur_color, None)
    # make match
    matches = matcher.knnMatch(d_ref, d_cur, 2)
    ref_pts = []
    cur_pts = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            ref_pts.append(kp_ref[m.queryIdx].pt)
            cur_pts.append(kp_cur[m.trainIdx].pt)

    # create 2 points clouds
    pc_ref = uv_vec2cloud(np.array(ref_pts), ref_depth)
    pc_cur = uv_vec2cloud(np.array(cur_pts), cur_depth)

    # # ICP
    current_in_previous, distances, iterations = icp.icp(pc_cur, pc_ref, tolerance=1e-6, max_iterations=50,
                                                         init_pose=init_current_in_previous)
    # current_in_previous = np.matmul(current_in_previous, init_current_in_previous)
    kp_ref = kp_cur
    d_ref = d_cur
    return current_in_previous, kp_ref, d_ref

def read_rgbd_pcd(rgb_img_name, depth_img_name, pinhole_camera_intrinsic = None):
    if pinhole_camera_intrinsic is None:
        pinhole_camera_intrinsic = read_cam_intrinsic()
    current_color = o3d.io.read_image(rgb_img_name)
    current_depth = o3d.io.read_image(depth_img_name)
    current_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        current_color, current_depth, depth_trunc=5.0, convert_rgb_to_intensity=False)
    current_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        current_rgbd_image, pinhole_camera_intrinsic)  # unit: m
    return current_pcd

def calc_cloud_in_ground(current_pcd):
    current_pcd_small = current_pcd.uniform_down_sample(30)
    cloud_pcd, plane_model, _ = remove_cloud_background(current_pcd_small)
    current_in_ground = calc_transformation_from_plane_model(plane_model, offset_y=True)
    return current_in_ground

def remove_leg(cloud_pcd):
    plane_model, ground_indices = cloud_pcd.segment_plane(
        distance_threshold=0.02, ransac_n=3, num_iterations=100)
    leg_indices = calc_leg_indices(cloud_pcd, ground_indices, plane_model)
    terrain_indices = set(np.arange(len(cloud_pcd.points)).tolist()) \
                      - set(leg_indices.tolist())
    terrain_indices = np.asarray(list(terrain_indices))
    if 0 == len(terrain_indices):
        terrain_indices = 0
    cloud_pcd = copy.deepcopy(cloud_pcd)
    cloud_no_ground = np.asarray(cloud_pcd.points)[terrain_indices]
    valid_indices = cloud_no_ground[..., 2] > 1
    cloud_pcd.points = o3d.utility.Vector3dVector(
        cloud_no_ground[valid_indices])
    cloud_pcd.colors = o3d.utility.Vector3dVector(
        np.asarray(cloud_pcd.colors)[terrain_indices][valid_indices])
    return cloud_pcd



def remove_cloud_background(cloud_pcd, remove_ground = True):
    plane_model, ground_indices = cloud_pcd.segment_plane(
        distance_threshold=0.02, ransac_n=3, num_iterations=100)
    outlier_indices = calc_outliers_of_terrain(cloud_pcd, ground_indices, plane_model)
    if remove_ground:
        terrain_indices = set(np.arange(len(cloud_pcd.points)).tolist()) \
                          - set(outlier_indices.tolist())\
                          - set(ground_indices)

    else:
        terrain_indices = set(np.arange(len(cloud_pcd.points)).tolist())\
                          - set(outlier_indices.tolist())
    terrain_indices = np.asarray(list(terrain_indices))
    if 0 == len(terrain_indices):
        terrain_indices = 0
    cloud_pcd = copy.deepcopy(cloud_pcd)
    cloud_no_ground = np.asarray(cloud_pcd.points)[terrain_indices]
    valid_indices = cloud_no_ground[..., 2] > 1
    cloud_pcd.points = o3d.utility.Vector3dVector(
        cloud_no_ground[valid_indices])
    cloud_pcd.colors = o3d.utility.Vector3dVector(
        np.asarray(cloud_pcd.colors)[terrain_indices][valid_indices])
    return cloud_pcd, plane_model, ground_indices


def calc_leg_indices(cloud_pcd, ground_indices, plane_model):
    transform_mat = calc_transformation_from_plane_model(plane_model)
    cloud_pcd_temp = copy.deepcopy(cloud_pcd)
    cloud_pcd_temp.transform(transform_mat)
    cloud = np.asarray(cloud_pcd_temp.points)
    cloud_ground = cloud[ground_indices]
    z_ground = np.mean(cloud_ground[..., 2])
    leg_indices = cloud[..., 2] < z_ground - 0.25
    leg_indices = np.nonzero(leg_indices)[0]
    return leg_indices


def calc_outliers_of_terrain(cloud_pcd, ground_indices, plane_model):
    transform_mat = calc_transformation_from_plane_model(plane_model)
    cloud_pcd_temp = copy.deepcopy(cloud_pcd)
    cloud_pcd_temp.transform(transform_mat)
    cloud  = np.asarray(cloud_pcd_temp.points)
    cloud_ground = cloud[ground_indices]
    z_ground = np.mean(cloud_ground[..., 2])
    # outliers = np.bitwise_and(np.abs(cloud[..., 1]) < 0.1, np.abs(cloud[..., 0]) < 0.2)
    # outliers = np.bitwise_or(cloud[..., 2] < z_ground - 0.25, outliers)
    outliers = np.bitwise_or(cloud[..., 2] < z_ground - 0.25, cloud[..., 2] > z_ground - 0.03)
    outlier_indices = np.nonzero(outliers)[0]
    return outlier_indices


def calc_transformation_from_plane_model(plane_model, offset_y = True):
    [a, b, c, d] = plane_model
    z_now = np.asarray([a, b, c])
    z_now = z_now/np.linalg.norm(z_now)
    y_now = np.asarray([0, 1, 0])
    x_now = np.cross(y_now, z_now)
    if offset_y:
        y_now = np.cross(z_now, x_now)
    coordinate_system_now = np.c_[x_now.reshape((3, 1)),
                                  y_now.reshape((3, 1)),
                                  z_now.reshape((3, 1))]

    rot_mat = np.linalg.inv(coordinate_system_now)
    transform_mat = np.identity(4)
    transform_mat[:3, :3] = rot_mat
    return transform_mat


def draw_registration_result(source, target, transformation = np.identity(4)):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def remove_invalid_points_from_pcd(pcd):
    cloud = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    indices_valid = cloud[..., 2] > 0
    colors_valid = colors[indices_valid]
    cloud_valid = cloud[indices_valid]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_valid)
    pcd.colors = o3d.utility.Vector3dVector(colors_valid)
    return pcd

def read_cam_intrinsic():
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    pinhole_camera_intrinsic.width = 640
    pinhole_camera_intrinsic.height = 480
    rgbd_tracker_paras = np.load('paras/rgbd_tracker_paras.npy', allow_pickle=True).item()
    pinhole_camera_intrinsic.intrinsic_matrix = rgbd_tracker_paras['cam_matrix_depth']
    return pinhole_camera_intrinsic


# def create_cylinder(start_point, end_point, cylinder_color = [1, 0, 0], radius=0.02):
#     z_to_gaze_mat = np.identity(4)
#     z_vec = end_point - start_point
#     z_to_gaze_mat[:3, 2] = z_vec / np.linalg.norm(z_vec)
#     z_to_gaze_mat[:3, 0] = np.cross(z_to_gaze_mat[:3, 1], z_vec)
#     z_to_gaze_mat[:3, 3] = (start_point + end_point)/2
#     height = np.linalg.norm(start_point - end_point)
#     mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius,
#                                                               height=height)
#     mesh_cylinder.paint_uniform_color(cylinder_color)
#     mesh_cylinder_origin = copy.deepcopy(mesh_cylinder)
#     mesh_cylinder.transform(z_to_gaze_mat)
#     return mesh_cylinder, mesh_cylinder_origin


def create_cylinder(start_point, end_point, cylinder_color = [1, 0, 0], radius=0.02):
    z_to_gaze_mat = np.identity(4)
    z_vec = end_point - start_point
    z_to_gaze_mat[:3, 2] = z_vec / np.linalg.norm(z_vec)
    z_to_gaze_mat[:3, 0] = np.cross(z_to_gaze_mat[:3, 1], z_to_gaze_mat[:3, 2])
    z_to_gaze_mat[:3, 0] /= np.linalg.norm(z_to_gaze_mat[:3, 0])
    z_to_gaze_mat[:3, 3] = (start_point + end_point)/2

    height = np.linalg.norm(start_point - end_point)
    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius,
                                                              height=height)
    mesh_cylinder.paint_uniform_color(cylinder_color)
    mesh_cylinder_origin = copy.deepcopy(mesh_cylinder)
    mesh_cylinder.transform(z_to_gaze_mat)
    return mesh_cylinder, mesh_cylinder_origin

def calc_init_transform(previous_rgbd_img, current_rgbd_image):
    pinhole_camera_intrinsic = read_cam_intrinsic()
    rgbd_img_list = [previous_rgbd_img, current_rgbd_image]
    cloud_in_ground_list = []
    for rgbd_img in rgbd_img_list:
        cloud_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_img, pinhole_camera_intrinsic)  # unit: m
        cloud_pcd, plane_model, _ = remove_cloud_background(cloud_pcd)
        cloud_in_ground = calc_transformation_from_plane_model(plane_model, offset_y=True)
        cloud_in_ground_list.append(cloud_in_ground)
    current_in_previous = np.matmul(np.linalg.inv(cloud_in_ground_list[0]), cloud_in_ground_list[1])
    return current_in_previous


def read_and_crop_color_img(rgb_name):
    current_color = o3d.io.read_image(rgb_name)
    color_array = np.asarray(current_color)
    color_img = np.zeros(color_array.shape, np.uint8)
    color_img[:-100, 200:, :] =  color_array[:-100, 200:, :]
    current_color = o3d.geometry.Image(color_img)
    return current_color


'''
0. Organize names of image files
'''
def organize_folders(val_dir, exp_num = 5):
    folders = np.asarray(sorted(glob.glob("{}/*/test*/".format(val_dir))))
    folders = folders.reshape((-1, exp_num))  # sensor number * experimental number
    folders_num = obtain_folder_number(folders[0])
    folders = folders[:, np.argsort(folders_num)]
    for c in range(folders.shape[-1]):
        '''Mover the sensor data in the same trial to the same test folder'''
        test_c_folder = '{}/test{}'.format(val_dir, c)
        if not os.path.exists(test_c_folder):
            os.mkdir(test_c_folder)
            for folder in folders[:, c]:
                base_folder_name = os.path.basename(os.path.dirname(os.path.dirname(folder)))
                shutil.copytree(folder, '{}/{}'.format(test_c_folder, base_folder_name))
        '''copy heel strike folder to a separate folder'''
        orbbec_frame_folder = '{}/orbbec_frame'.format(test_c_folder)
        heel_strike_original_folder = '{}/heel_strike'.format(orbbec_frame_folder)
        heel_strike_folder = '{}/heel_strike'.format(test_c_folder)
        if os.path.exists(heel_strike_original_folder) and not os.path.exists(heel_strike_folder):
            shutil.copytree(heel_strike_original_folder, heel_strike_folder)
        '''copy orbbec rgb and depth images to a separate folder'''
        orbbec_img_folder_list = ['orbbec_rgb', 'orbbec_depth']
        orbbec_img_type_list = ['jpg', 'png']
        for i in range(2):
            img_names = glob.glob('{}/*.{}'.format(orbbec_frame_folder, orbbec_img_type_list[i]))
            img_folder = '{}/{}'.format(test_c_folder, orbbec_img_folder_list[i])
            if not os.path.exists(img_folder):
                os.mkdir(img_folder)
            for file in img_names:
                file_name = os.path.basename(file)
                shutil.copyfile(file, '{}/{}'.format(img_folder, file_name))
        '''remove orbbec frame folder'''
        shutil.rmtree(orbbec_frame_folder)

    ''' Remove redundant folder'''
    folders = set(glob.glob("{}/*/".format(val_dir))) - \
              set(glob.glob("{}/test*/".format(val_dir)))
    print(folders)
    for folder in folders:
        shutil.rmtree(folder)


def organize_files(val_dir):
    folders = np.asarray(sorted(glob.glob("{}/test*/".format(val_dir))))
    print(folders)
    for r in range(folders.shape[0]):
        os.rename(folders[r], '{}/test{}/'.format(val_dir, r))


def rename_file_time(file_name):
    file_time = obtain_file_time(file_name)
    dir_name = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    root, extension = os.path.splitext(base_name)
    os.rename(file_name, '{}/{:.3f}{}'.format(dir_name, file_time, extension))

def calc_camera_pose(val_dir, test_idx = 1, save_data = True):
    ''' Calculate the pose of each frame and save it'''
    ''' Start from the first heel strike, end at the last heel strike '''
    rgb_img_names, depth_img_names, test_dir = obtain_image_names(val_dir, test_idx = test_idx)
    frame_num = len(depth_img_names)-1
    current_in_origin_mat = np.identity(4)
    transform_matrices = np.zeros((frame_num, 4, 4))

    ''' Initialize ORB SLAM '''
    orb = cv2.ORB_create(nfeatures=10000, nlevels=8, scoreType=cv2.ORB_FAST_SCORE)
    # Init feature matcher
    matcher = cv2.DescriptorMatcher_create("BruteForce-L1")
    '''Detect the features of reference image'''
    previous_color = cv2.imread(rgb_img_names[0], -1)
    ''' avoid seeing the foot '''
    half_rows = int(previous_color.shape[0] / 2)
    half_cols = int(previous_color.shape[1] / 2)
    half_block_width = 150

    previous_color[half_rows:, half_cols - half_block_width:half_cols + half_block_width] = 0
    previous_color = cv2.cvtColor(previous_color, cv2.COLOR_BGR2GRAY)

    previous_depth = cv2.imread(depth_img_names[0], -1)
    kp_previous, d_previous = orb.detectAndCompute(previous_color, None)
    previous_pcd = read_rgbd_pcd(rgb_img_names[0], depth_img_names[0])

    previous_in_ground = calc_cloud_in_ground(previous_pcd)

    for i in tqdm(range(frame_num)):
        current_pcd = read_rgbd_pcd(rgb_img_names[i + 1], depth_img_names[i + 1])
        current_in_ground = calc_cloud_in_ground(current_pcd)

        current_color = cv2.imread(rgb_img_names[i + 1], -1)
        ''' avoid seeing the foot '''
        current_color[half_rows:, half_cols - half_block_width:half_cols + half_block_width] = 0
        cv2.imshow('current color', current_color)
        cv2.waitKey(30)
        current_color = cv2.cvtColor(current_color, cv2.COLOR_BGR2GRAY)


        current_depth = cv2.imread(depth_img_names[i + 1], -1)

        start = time.time()
        init_current_in_previous = np.matmul(np.linalg.inv(previous_in_ground), current_in_ground)

        current_in_previous, kp_previous, d_previous = orb_rgbd_odometry(previous_depth, current_color, current_depth,
                                                                     orb, matcher, kp_previous, d_previous,
                                                                     init_current_in_previous)
        print('Computing time: {:0f} ms'.format(1000 * (time.time() - start)))

        previous_in_ground = current_in_ground
        previous_depth = current_depth

        current_in_origin_mat = np.matmul(current_in_origin_mat, current_in_previous)
        print(rgb_img_names[i + 1], current_in_origin_mat)
        transform_matrices[i] = current_in_origin_mat

    result_dir = '{}/results'.format(test_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if save_data:
        np.save('{}/transform_matrices.npy'.format(result_dir),
                transform_matrices)


def obtain_image_names(val_dir, test_idx = 1):
    print(val_dir)
    test_dir = '{}/test{}'.format(val_dir, test_idx)
    reference_file_list = np.array(sorted(glob.glob("{}/heel_strike/*.jpg".format(test_dir))))
    file_name_list = np.array(sorted(glob.glob('{}/orbbec_rgb/*.jpg'.format(test_dir))))
    rgb_img_names = extract_files_within_time_interval_of_reference_files(
        file_name_list, reference_file_list)
    '''Extract the depth image that is save at the same time as the rgb image'''
    rgb_img_times = obtain_file_time_vec(rgb_img_names)
    depth_img_names = []
    for rgb_img_time in rgb_img_times:
        depth_img_names.append('{}/orbbec_depth/{:.3f}.png'.format(test_dir, rgb_img_time))
    return rgb_img_names, depth_img_names, test_dir


def calc_original_offset_matrix(val_dir, test_idx = 3):
    rgb_img_names, depth_img_names, test_dir = obtain_image_names(val_dir, test_idx=test_idx)
    current_pcd = read_rgbd_pcd(rgb_img_names[0], depth_img_names[0])
    offset_mat = calc_cloud_in_ground(current_pcd)
    return offset_mat

'''
2. Fuse multiple point cloud and save it.
'''
def fuse_multi_clouds(val_dir, test_idx=1, remove_ground = True,
                      save_all_cloud = False):
    '''Only fuse the cloud while at the time of heel strike'''
    rgb_img_names, depth_img_names, test_dir = obtain_image_names(val_dir, test_idx=test_idx)
    '''Avoid seeing feet'''
    reference_file_list = np.array(sorted(glob.glob("{}/swing/*.jpg".format(test_dir))))
    rgb_img_sub_time_vec = obtain_file_time_vec(reference_file_list)
    origin_in_world = calc_original_offset_matrix(val_dir, test_idx=test_idx)
    frame_num = len(rgb_img_names)
    if remove_ground:
        cloud_name = 'terrain_cloud'
    else:
        cloud_name = 'terrain_and_background_cloud'
    result_dir = '{}/results'.format(test_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if save_all_cloud:
        cloud_dir = '{}/{}'.format(result_dir, cloud_name)
        if not os.path.exists(cloud_dir):
            os.mkdir(cloud_dir)
        segmented_result_dir = '{}/segmented_terrain_pcd'.format(result_dir)
        if not os.path.exists(segmented_result_dir):
            os.mkdir(segmented_result_dir)
    depth_in_origin_matrices = np.load('{}/transform_matrices.npy'.format(result_dir))
    original_pcd = None
    for i in tqdm(range(1, frame_num)):
        current_time = obtain_file_time(rgb_img_names[i])
        if current_time not in rgb_img_sub_time_vec:
            continue
        start = time.time()
        current_pcd = read_rgbd_pcd(rgb_img_names[i], depth_img_names[i]).uniform_down_sample(30)
        current_pcd = remove_leg(current_pcd)
        if remove_ground:
            current_pcd, plane_model, _ = remove_cloud_background(current_pcd, remove_ground=remove_ground)
        current_in_origin_mat = depth_in_origin_matrices[i-1]
        print(i, rgb_img_names[i])
        current_pcd = current_pcd.transform(current_in_origin_mat)
        if original_pcd is None:
            original_pcd = current_pcd
        else:
            original_pcd += current_pcd
        original_pcd = original_pcd.voxel_down_sample(1e-2)
        print('Computing time of fusing point clouds: {:0f} ms'.format(1000 * (time.time() - start)))
        if remove_ground:
            terrain_pcd = copy.deepcopy(original_pcd)
            start = time.time()
            terrain_pcd.transform(origin_in_world)  # offset the rotation of the point cloud
            # view_pcd_with_axis(terrain_pcd)
            segmented_terrain_pcd, segmented_labels = label_connected_area(terrain_pcd)
            print('Computing time of segmenting point clouds: {:0f} ms'.format(1000 * (time.time() - start)))
            # utils.view_pcd_with_axis(segmented_terrain_pcd)
        if save_all_cloud:
            terrain_pcd = copy.deepcopy(original_pcd)
            terrain_pcd.transform(origin_in_world)
            o3d.io.write_point_cloud(
                '{}/{:.3f}.pcd'.format(cloud_dir, current_time),
                terrain_pcd)
            if remove_ground and (i > 30):
                o3d.io.write_point_cloud('{}/{:.3f}.pcd'.format(segmented_result_dir, current_time),
                                         segmented_terrain_pcd)
                np.save('{}/{:.3f}.npy'.format(segmented_result_dir, current_time), segmented_labels)

    # original_pcd.transform(origin_in_world)  # offset the rotation of the point cloud
    view_pcd_with_axis(original_pcd)
    o3d.io.write_point_cloud('{}/{}.pcd'.format(result_dir, cloud_name),
                             original_pcd)
    if remove_ground:
        view_pcd_with_axis(segmented_terrain_pcd)
        o3d.io.write_point_cloud('{}/segmented_terrain_pcd.pcd'.format(result_dir), segmented_terrain_pcd)
        np.save('{}/segmented_labels.npy'.format(result_dir), segmented_labels)


def label_connected_area(cloud_pcd):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        eps = 0.1
        min_points = 30
        labels = np.array(cloud_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    '''
    DBSCAN [Ester1996] that is a density based clustering algorithm
    '''
    cloud_pcd.points = o3d.utility.Vector3dVector(np.asarray(cloud_pcd.points)[labels >= 0])
    labels = labels[labels>=0]
    labels = sort_clusters(switch_pcd_xy_axes(cloud_pcd), labels)
    print(f"point cloud has {labels.max() + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / 20)
    # colors[labels < 0] = 1
    cloud_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return cloud_pcd, labels


def switch_pcd_xy_axes(cloud_temp_pcd):
    cloud_temp_pcd = copy.deepcopy(cloud_temp_pcd)
    cloud = np.asarray(cloud_temp_pcd.points)
    cloud[:, :2] = -cloud[:, [1, 0]]
    cloud_temp_pcd.points = o3d.utility.Vector3dVector(cloud)
    return cloud_temp_pcd

def switch_array_xy_axes(array_temp):
    array_temp = copy.deepcopy(array_temp)
    array_temp[:, :2] = -array_temp[:, [1, 0]]
    return array_temp


def switch_xy_axes(cloud_temp_pcd, gaze_points_in_world, foot_placements_in_world):
    cloud_temp_pcd = copy.deepcopy(cloud_temp_pcd)
    cloud = np.asarray(cloud_temp_pcd.points)
    cloud[:, :2] = -cloud[:, [1, 0]]
    cloud_temp_pcd.points = o3d.utility.Vector3dVector(cloud)
    gaze_points_in_world[:, :2] = -gaze_points_in_world[:, [1, 0]]
    foot_placements_in_world[:, :2] = -foot_placements_in_world[:, [1, 0]]
    return cloud_temp_pcd, gaze_points_in_world, foot_placements_in_world

def sort_clusters(cloud_pcd, labels):
    points = np.asarray(cloud_pcd.points)
    center_mat = np.zeros((np.max(labels) + 1, 3))
    for i in range(np.max(labels) + 1):
        center_mat[i] = np.mean(points[labels == i], axis=0)
    indices = np.argsort(center_mat[:, 0])
    sorted_labels = np.zeros(labels.shape, np.int)
    for i in range(len(indices)):
        sorted_labels[labels == indices[i]] = i
    return sorted_labels

'''
3. Find corresponding gaze and calculate the median of 3D gaze.
'''
def synchronize_gaze_2d(val_dir, test_idx = 1, save_video = True, video_name = None):
    rgb_img_names, depth_img_names, test_dir = obtain_image_names(val_dir, test_idx = test_idx)
    gaze_2d_dir  = '{}/test{}/orbbec_gaze_2D/*.npy'.format(val_dir, test_idx)
    gaze_2d_names = sorted(glob.glob(gaze_2d_dir))
    synchronized_gaze_2d_names = synchronize_two_file_list(
        gaze_2d_names, depth_img_names)
    gaze_in_depth_dir = '{}/test{}/gaze_in_depth'.format(val_dir, test_idx)
    if os.path.exists(gaze_in_depth_dir):
        shutil.rmtree(gaze_in_depth_dir)
    os.mkdir(gaze_in_depth_dir)
    if save_video:
        result_dir = '{}/results'.format(test_dir)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        video_img_dir = '{}/gaze_image'.format(result_dir)
        if os.path.exists(video_img_dir):
           shutil.rmtree(video_img_dir)
        os.mkdir(video_img_dir)
    for i in range(len(synchronized_gaze_2d_names)-1):
        gaze_2d_list = []
        for gaze_2d_name in synchronized_gaze_2d_names[i]:
            gaze_2d_list.append(np.load(gaze_2d_name))
        gaze_2d_vec = np.asarray(gaze_2d_list)
        median_gaze_2d = np.median(gaze_2d_vec, axis=0)
        depth_name = depth_img_names[i]
        img_depth = cv2.imread(depth_name, cv2.IMREAD_UNCHANGED)
        gaze_3d_in_depth, gaze_2d_in_depth = tracker_to_rgbd_xyz(median_gaze_2d, img_depth)

        img_rgb = cv2.imread(rgb_img_names[i])
        cv2.circle(img_rgb, gaze_2d_in_depth,
                   radius = 20, color=(0, 255, 0), thickness = 5)
        cv2.imshow('img_rgb', img_rgb)
        cv2.waitKey(30)
        if save_video:
            img_time = float(os.path.splitext(os.path.basename(depth_name))[0])
            cv2.imwrite('{}/{:.2f}.jpg'.format(video_img_dir, img_time), img_rgb)
        current_time = obtain_file_time(depth_name)
        np.save('{}/{}.npy'.format(gaze_in_depth_dir, current_time),
                {'gaze_3d_in_depth': gaze_3d_in_depth,
                 'gaze_2d_in_depth': gaze_2d_in_depth})

    if save_video:
        if video_name is None:
            video_name = '{}/gaze_video.mp4'.format(result_dir)
        read_image_to_video(glob.glob('{}/*.jpg'.format(video_img_dir)), video_name, fps=20)

'''
4. Fuse sequential gaze vectors and save videos.
'''
def fuse_multi_gazes(val_dir, test_idx = 1):
    test_dir = '{}/test{}'.format(val_dir, test_idx)
    gaze_dir = '{}/gaze_in_depth/*.npy'.format(test_dir)
    result_dir ='{}/results'.format(test_dir)
    gaze_names = sorted(glob.glob(gaze_dir))
    gaze_points = np.zeros((len(gaze_names), 3))
    depth_in_origin_matrices = np.load(
        '{}/transform_matrices.npy'.format(result_dir))
    print(depth_in_origin_matrices.shape, len(gaze_names))
    for i in range(len(gaze_names)):
        gaze_dict = np.load(gaze_names[i], allow_pickle=True).item()
        gaze_3d = gaze_dict['gaze_3d_in_depth']
        if 0 == i:
            gaze_points[i] = gaze_3d
        else:
            depth_in_origin_mat = depth_in_origin_matrices[i - 1]
            gaze_3d_in_depth = np.ones((4, 1))
            gaze_3d_in_depth[:3, 0] = gaze_3d

            gaze_3d_in_origin = np.matmul(depth_in_origin_mat, gaze_3d_in_depth)
            gaze_points[i] = gaze_3d_in_origin[:3, 0]

    return gaze_points


def calc_foot_placements_in_origin(val_dir, test_idx):
    rgb_img_names, depth_img_names, test_dir = obtain_image_names(val_dir, test_idx)
    result_dir = '{}/results'.format(test_dir)
    '''We don't use the first two foot placements because they are on the ground rather than on the uneven terrain'''
    foot_uv_names = sorted(glob.glob('{}/heel_strike/*.npy'.format(test_dir)))[2:]
    heel_strike_times = obtain_file_time_vec(foot_uv_names)
    img_times = obtain_file_time_vec(rgb_img_names)
    depth_in_origin_matrices = np.load('{}/transform_matrices.npy'.format(result_dir))
    foot_placements_in_origin = np.zeros((len(foot_uv_names), 3))
    for i in range(len(heel_strike_times)):
        current_time = heel_strike_times[i]
        current_idx = np.where(np.abs(img_times - current_time) < 1e-3)[0][0]
        '''Sometime the depth image may not have depth value, we need to find valid depth value in the neighboring 
        area. '''
        foot_uv = np.load(foot_uv_names[i])
        depth_img = cv2.imread(depth_img_names[i], -1)
        for w in [3, 5, 10, 20, 50]:
            depth_mat = depth_img[foot_uv[1]-w:min(480, foot_uv[1] + w),
                        max(0, foot_uv[0]-w):min(640, foot_uv[0]+w)].reshape(-1).astype(np.float)
            depth_mat *= 1e-3 # mm to m
            '''The depth of foot placement should between 1m and 3m'''
            depth_mat = depth_mat[np.bitwise_and(depth_mat>1, depth_mat< 3)]
            depth = np.median(depth_mat)
            if depth > 0:
                break
        foot_in_depth_homo = np.ones((4, 1))
        foot_in_depth_homo[:3, 0] = uvz2xyz(foot_uv[0], foot_uv[1], depth)[:]
        if 0 == depth:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Foot in depth: ', foot_in_depth_homo)
        '''The idx of depth in origin = the idx of image - 1, 
        the reason is that we did not save the transformation matrix from the first image to iteself.'''
        depth_in_origin = depth_in_origin_matrices[current_idx-1]
        foot_placements_in_origin[i] = np.matmul(depth_in_origin, foot_in_depth_homo)[:3, 0]
    return foot_placements_in_origin


'''
6. Analyze gaze and foot placements
'''
def analyze_gaze_and_footplacement(val_dir, test_idx=0, render_every_gait=False,
                                   save_image=True, change_phase=False, end_time=None,
                                   lead_time=None, window_length=None, remove_outliers=False):
    '''
        split the whole code to two parts: before or after labeling the foot placements
    '''
    test_dir = '{}/test{}'.format(val_dir, test_idx)
    result_dir = '{}/results'.format(test_dir)
    origin_in_world_mat = calc_original_offset_matrix(val_dir, test_idx=test_idx)
    cloud_pcd = o3d.io.read_point_cloud('{}/segmented_terrain_pcd.pcd'.format(result_dir))
    gaze_points_in_origin = fuse_multi_gazes(val_dir, test_idx=test_idx)
    gaze_points_in_world = origin_2_world_system(gaze_points_in_origin, origin_in_world_mat)
    foot_placements_in_origin = calc_foot_placements_in_origin(val_dir, test_idx)
    foot_placements_in_world = origin_2_world_system(foot_placements_in_origin, origin_in_world_mat)
    '''
        Switch x and y axes, because the current walking direction is -y, which is not suitable to visualize
    '''
    cloud_pcd, gaze_points_in_world, foot_placements_in_world= switch_xy_axes(
        cloud_pcd, gaze_points_in_world, foot_placements_in_world)
    # read gait time:
    gaze_names = sorted(glob.glob('{}/gaze_in_depth/*.npy'.format(test_dir)))
    gaze_time_vec = obtain_file_time_vec(gaze_names)
    gait_names = sorted(glob.glob('{}/heel_strike/*.jpg'.format(test_dir)))
    step_time_vec = obtain_file_time_vec(gait_names)

    # plot labels of terrains
    labels = np.load('{}/segmented_labels.npy'.format(result_dir))
    center_mat = calc_cluster_center(cloud_pcd, labels)

    actual_foot_placement_indices = np.genfromtxt('{}/results/foot_placements.txt'.format(test_dir), delimiter=',', dtype=np.int)
    # predicted_foot_placements, actual_foot_placement_indices = calc_foot_placements(val_dir, center_mat, test_idx=test_idx)

    if (lead_time is None) or (window_length is None):
        if change_phase:
            window_length, lead_time = calc_best_gaze_window(
                foot_placements_in_world, actual_foot_placement_indices,
                gaze_points_in_world, gaze_time_vec, step_time_vec,
                cloud_pcd, center_mat, result_dir)
            print('Window length: {}, lead time: {}'.format(window_length, lead_time))
        else:
            window_length = 0.5
            lead_time = -0.7
    predicted_foot_placements = 0
    np.save('{}/sequential_gaze_and_foot_placements.npy'.format(result_dir),
            {'step_time_vec': step_time_vec,
             'predicted_foot_placements': predicted_foot_placements,
             'gaze_time_vec': gaze_time_vec,
             'gaze_points_in_world': gaze_points_in_world,
             'foot_placements_in_world': foot_placements_in_world,
             })

    if end_time is not None:
        gaze_points_in_world = gaze_points_in_world[gaze_time_vec < end_time]
        gaze_time_vec = gaze_time_vec[gaze_time_vec < end_time]
    gaze_points_list = calc_gaze_points(gaze_points_in_world,
                                        gaze_time_vec,
                                        step_time_vec,
                                        end_phase=lead_time,
                                        phase_width=window_length)

    if remove_outliers:
        for j in range(len(gaze_points_list)):
            gaze_points = gaze_points_list[j][:, :2]
            inlier_indices = np.abs(gaze_points[:, 0] - np.mean(gaze_points[:, 0])) < np.std(gaze_points[:, 0])
            gaze_points_list[j] = gaze_points[inlier_indices]
    if save_image:
        if render_every_gait:
            '''Plot 2D gaze, foot placements, and blocks'''
            foot_placements_dir = '{}/2d_gaze_and_foot_placements'.format(result_dir)
            if not os.path.exists(foot_placements_dir):
                os.mkdir(foot_placements_dir)
            for idx in range(len(gaze_points_list) + 1):
                fig = scatter_cloud_pcd(cloud_pcd, gaze_points_list[:idx + 1],
                                        center_mat,
                                        foot_placements=foot_placements_in_world[:idx])
                plt.savefig('{}/{}.jpg'.format(
                    foot_placements_dir, idx), bbox_inches='tight',
                    pad_inches=0.0)
                plt.close(fig)

            read_image_to_video(
                glob.glob('{}/*.jpg'.format(foot_placements_dir)),
                video_name='{}/2d_gaze_and_foot_placements.mp4'.format(result_dir),
                fps=1)
        else:
            time_delay = 0  # unit s
            folder_name = 'all_2d_gaze_and_foot_placements'
            foot_placements_dir = '{}/{}'.format(result_dir, folder_name)
            if not os.path.exists(foot_placements_dir):
                os.mkdir(foot_placements_dir)

            gaze_points_list = [None]
            idx = 0
            for gaze_time in gaze_time_vec[2:]:  # The first step is two steps before standing on a brick
                gait_time = step_time_vec[min(idx, len(step_time_vec) - 1)]
                if gaze_time > gait_time and idx < len(step_time_vec):
                    idx += 1
                    gaze_points_list.append(None)
                if 0 == idx:
                    pre_gait_time = 0
                else:
                    pre_gait_time = step_time_vec[idx - 1]
                gaze_indices = np.bitwise_and(
                    gaze_time_vec <= gaze_time,
                    gaze_time_vec > pre_gait_time + time_delay)
                gaze_points_list[idx] = gaze_points_in_world[gaze_indices]

                print(predicted_foot_placements)
                fig = scatter_cloud_pcd(
                    cloud_pcd, gaze_points_list, center_mat,
                    foot_placements=predicted_foot_placements[:idx])
                start = time.time()
                fig.canvas.print_png('{}/{:.2f}.png'.format(
                    foot_placements_dir, gaze_time))
                # plt.savefig('{}/{:.2f}.png'.format(
                #     foot_placements_dir, gaze_time), bbox_inches='tight',
                #     pad_inches=0.0)
                print('Time: {:.3f}'.format(time.time() - start))

            read_image_to_video(
                glob.glob('{}/*.png'.format(foot_placements_dir)),
                video_name='{}/{}.mp4'.format(result_dir, folder_name),
                fps=20)

    # switch back to the original axes
    for i in range(len(gaze_points_list)):
        gaze_points_list[i] = switch_array_xy_axes(gaze_points_list[i])
    return window_length, lead_time


def origin_2_world_system(points_in_origin, offset_mat):
    points = np.ones((len(points_in_origin), 4))
    points[:, :3] = points_in_origin
    points_in_world = np.matmul(
        offset_mat, np.transpose(points)).transpose()[:, :3]
    return points_in_world


def analyze_gait_parameters(val_dir_list, exp_num = 5):
    subject_num = len(val_dir_list)
    step_length_list_list = []
    block_gap_list_list = []
    step_duration_list_list = []
    velocity_list_list = []
    for s in range(subject_num):
        val_dir = val_dir_list[s]
        step_length_list = []
        block_gap_list = []
        step_duration_list = []
        velocity_list = []
        for e in range(exp_num):
            test_dir = '{}/test{}'.format(val_dir, e)
            result_dir = '{}/results'.format(test_dir)
            gaze_and_foot_placement_data = np.load('{}/gaze_and_foot_placements.npy'.format(result_dir),
                                                   allow_pickle=True).item()
            foot_placements_in_world = gaze_and_foot_placement_data['foot_placements_in_world']
            actual_support_block_center_in_world = gaze_and_foot_placement_data['actual_support_block_center_in_world']
            step_time_vec = gaze_and_foot_placement_data['step_time_vec'][2:]
            step_duration_vec = step_time_vec[1:] - step_time_vec[:-1]# unit: s
            block_gap_vec = np.linalg.norm(actual_support_block_center_in_world[:-1, :2]
                                           - actual_support_block_center_in_world[1:, :2], axis=-1)# unit: m
            step_length_vec = np.linalg.norm(foot_placements_in_world[:-1, :2]
                                           - foot_placements_in_world[1:, :2], axis=-1) # unit: m
            print(step_duration_vec.shape, block_gap_vec.shape, step_length_vec.shape)
            velocity_vec = step_length_vec/step_duration_vec
            step_length_list.append(step_length_vec)
            block_gap_list.append(block_gap_vec)
            step_duration_list.append(step_duration_vec)
            velocity_list.append(velocity_vec)

        step_length_list_list.append(step_length_list)
        block_gap_list_list.append(block_gap_list)
        step_duration_list_list.append(step_duration_list)
        velocity_list_list.append(velocity_list)

    '''
        Columns: step_length, block_gap, step duration, velocity
        Rows: 0:subject_num, mean; subject_num:2*subject_num, std;
    '''
    gait_parameters_mat = np.zeros((2 * subject_num, 4))
    gait_data_list = [step_length_list_list, block_gap_list_list, step_duration_list_list, velocity_list_list]
    for g in range(4):
        gait_data = gait_data_list[g]
        for s in range(subject_num):
            gait_data_s = gait_data[s]
            gait_data_vec = np.concatenate(gait_data_s, axis=0)
            gait_parameters_mat[s, g] = np.mean(gait_data_vec)
            gait_parameters_mat[s+subject_num, g] = np.std(gait_data_vec)

    np.savetxt("results/result_data/gait_parameters_mat.csv", gait_parameters_mat, delimiter=",")



def plot_gaze_foot_error_bar(val_dir_list, exp_num = 5):
    ''' 1. Analyze the prediction error of foot placements using four methods:
            1. Baseline: using only gaze
            2. Gaze + environmental context
            3. Gaze + user-dependent window
            4. Gaze + environmetal context + user-dependent window
        2. Plot the best time window for each step
        3. Plot the distributions of gazes, block centers, and foot placements.
    '''
    subject_num = len(val_dir_list)
    foot_placements_in_world_list_list = []
    actual_support_block_center_in_world_list_list = []
    gaze_location_list_list = []
    predicted_foot_placements_in_world_list_list = []
    cls_acc_every_step_mat_list_list = []
    for s in range(subject_num):
        val_dir = val_dir_list[s]
        foot_placements_in_world_list = []
        gaze_location_list = []
        predicted_foot_placements_in_world_list = []
        actual_support_block_center_in_world_list = []
        cls_acc_every_step_mat_list = []
        for e in range(exp_num):
            test_dir = '{}/test{}'.format(val_dir, e)
            result_dir = '{}/results'.format(test_dir)
            gaze_and_foot_placement_data = np.load('{}/gaze_and_foot_placements.npy'.format(result_dir),
                                                   allow_pickle=True).item()
            foot_placements_in_world_list.append(gaze_and_foot_placement_data['foot_placements_in_world'])
            gaze_location_list.append(gaze_and_foot_placement_data['gaze_location_mat'])
            predicted_foot_placements_in_world_list.append(gaze_and_foot_placement_data['predicted_foot_placements_in_world'])
            actual_support_block_center_in_world_list.append(gaze_and_foot_placement_data['actual_support_block_center_in_world'])
            cls_acc_every_step_mat_list.append(gaze_and_foot_placement_data['cls_acc_every_step_mat'])
        foot_placements_in_world_list_list.append(foot_placements_in_world_list)
        gaze_location_list_list.append(gaze_location_list)
        predicted_foot_placements_in_world_list_list.append(predicted_foot_placements_in_world_list)
        actual_support_block_center_in_world_list_list.append(actual_support_block_center_in_world_list)
        cls_acc_every_step_mat_list_list.append(cls_acc_every_step_mat_list)

    error_mean_mat = np.zeros((subject_num, 4))
    error_std_mat = np.zeros((subject_num, 4))
    best_end_phase_mat = np.zeros(subject_num, dtype=object)
    best_phase_width_mat = np.zeros(subject_num, dtype=object)

    '''First calculate the global best window'''
    cls_acc_mat_all = []
    for val_dir in val_dir_list:
        cls_acc_mat, gaze_foot_error_mat, predicted_foot_error_mat, end_phase_vec, phase_width_vec = \
            read_and_calc_cls_and_error_mat(val_dir)
        cls_acc_mat_all.append(cls_acc_mat)
    cls_acc_mean_mat = np.mean(np.asarray(cls_acc_mat_all), axis=0)
    global_r_best, global_c_best = np.unravel_index(cls_acc_mean_mat.argmax(), cls_acc_mat.shape)
    print(cls_acc_mean_mat)
    print('global_r_best: {}, global_c_best: {}'.format(global_r_best, global_c_best))
    all_prediction_error_mat = np.zeros((subject_num, 4), dtype=np.object)

    for s in range(subject_num):
        cls_acc_every_step_mat = np.concatenate(cls_acc_every_step_mat_list_list[s], axis=-1)

        actual_support_block_center_in_world_mat = np.concatenate(actual_support_block_center_in_world_list_list[s], axis=-2)
        foot_placements_in_world_mat = np.concatenate(foot_placements_in_world_list_list[s], axis=-2)
        gaze_location_mat = np.concatenate(gaze_location_list_list[s], axis=-2)
        ''' predicted_foot_placements_in_world_mat: number of end phases * number of phase width * step number * 2'''
        predicted_foot_placements_in_world_mat = np.concatenate(predicted_foot_placements_in_world_list_list[s], axis=-2)
        ''' gaze_foot_error_mat: number of end phases * number of phase width * step number'''
        gaze_foot_error_mat = np.linalg.norm(
            gaze_location_mat - np.reshape(foot_placements_in_world_mat[:, :2], (1, 1, -1, 2)), axis=-1)
        ''' predicted_foot_error_mat: number of end phases * number of phase width * step number'''
        predicted_foot_error_mat = np.linalg.norm(
            predicted_foot_placements_in_world_mat - np.reshape(foot_placements_in_world_mat[:, :2], (1, 1, -1, 2)), axis=-1)
        ''' gaze_foot_error_mat: number of end phases * number of phase width * step number'''
        gaze_block_center_error_mat = np.linalg.norm(
            gaze_location_mat - np.reshape(actual_support_block_center_in_world_mat[:, :2], (1, 1, -1, 2)), axis=-1)
        gaze_block_center_error_mean_mat = np.mean(gaze_block_center_error_mat, axis=-1)
        '''First ensure the classification accuracy is the maximum, 
        then select the window in which the gaze_block_center distance is minimum'''
        gaze_block_center_error_mean_mat[cls_acc_mat_all[s] < cls_acc_mat_all[s].max()] = 1000
        r_best, c_best = np.unravel_index(gaze_block_center_error_mean_mat.argmin(), gaze_block_center_error_mean_mat.shape)
        # r_best, c_best = np.unravel_index(cls_acc_mat_all[s].argmax(), cls_acc_mat.shape)
        # print('r_best: {}, c_best: {}'.format(r_best, c_best))
        all_prediction_error_mat[s, 0] = gaze_foot_error_mat[global_r_best, global_c_best]
        all_prediction_error_mat[s, 1] = predicted_foot_error_mat[global_r_best, global_c_best]
        all_prediction_error_mat[s, 2] = gaze_foot_error_mat[r_best, c_best]
        all_prediction_error_mat[s, 3] = predicted_foot_error_mat[r_best, c_best]

        step_num = gaze_block_center_error_mat.shape[-1]
        best_end_phase_vec = np.zeros(step_num)
        best_phase_width_vec = np.zeros(step_num)

        for i in range(step_num):
            '''First ensure the classification accuracy is the maximum, 
            then select the window in which the gaze_block_center distance is minimum'''
            gaze_block_center_error_mat[:, :, i][cls_acc_every_step_mat[:, :, i] < cls_acc_every_step_mat[:, :, i].max()] = 1000

            r_best_i, c_best_i = np.unravel_index(gaze_block_center_error_mat[:, :, i].argmin(),
                                                  gaze_block_center_error_mat[:, :, i].shape)
            best_end_phase_vec[i] = np.abs(end_phase_vec[r_best_i])
            best_phase_width_vec[i] = phase_width_vec[c_best_i]
        best_end_phase_mat[s] = best_end_phase_vec
        best_phase_width_mat[s] = best_phase_width_vec

    for s in range(subject_num):
        for m in range(all_prediction_error_mat.shape[1]):
            error_mean_mat[s, m] = np.mean(all_prediction_error_mat[s, m])
            error_std_mat[s, m] = np.std(all_prediction_error_mat[s, m])

    np.savetxt("results/result_data/prediction_error_mat.csv", np.c_[error_mean_mat, error_std_mat], delimiter=",")

    ''' 1. Analyze the prediction error of foot placements using four methods:'''
    plot_error_bar(error_mean_mat, error_std_mat, result_dir='results')
    # ''' 2. Plot the best time window for each step'''
    print(best_end_phase_mat, best_phase_width_mat)
    plot_best_time_windows(best_end_phase_mat, best_phase_width_mat)
    ''' 3. Plot the distributions of gazes, block centers, and foot placements.'''
    plot_all_gazes_and_predicted_foot_placements_points(foot_placements_in_world_list_list, gaze_location_list_list,
                                                        predicted_foot_placements_in_world_list_list,
                                                        actual_support_block_center_in_world_list_list,
                                                        cls_acc_mat_all)
    ''' 4. Calculate the P value of prediction errors'''
    analyze_p_vals_of_prediction_errors(all_prediction_error_mat)

    ''' 5. Calculate the P value of best time window'''
    analyze_p_vals_of_best_time_window(best_end_phase_mat, best_phase_width_mat)

def analyze_p_vals_of_best_time_window(best_end_phase_mat, best_phase_width_mat):
    subject_num = len(best_end_phase_mat)
    p_val_lead_time, is_normal_lead_time = calc_half_p_matrix(best_end_phase_mat)
    p_val_window_length, is_normal_window_length = calc_half_p_matrix(best_phase_width_mat)
    '''Mean of lead time, std of lead time, mean of window length, std of window length'''
    time_window_mean_std_mat = np.zeros((4, subject_num))
    for s in range(subject_num):
        time_window_mean_std_mat[0, s] = np.mean(best_end_phase_mat[s])
        time_window_mean_std_mat[1, s] = np.std(best_end_phase_mat[s])
        time_window_mean_std_mat[2, s] = np.mean(best_phase_width_mat[s])
        time_window_mean_std_mat[3, s] = np.std(best_phase_width_mat[s])
    np.savetxt("results/result_data/time_window_mean_std_mat.csv", time_window_mean_std_mat, delimiter=",")
    np.savetxt("results/result_data/p_vals_time_window.csv", np.c_[p_val_lead_time, p_val_window_length], delimiter=",")
    print('is_normal_lead_time : {}, is_normal_window_length: {}'.format(is_normal_lead_time, is_normal_window_length))


def analyze_p_vals_of_prediction_errors(all_prediction_error_mat):
    subject_num, method_num = all_prediction_error_mat.shape
    p_val_methods = np.zeros((subject_num, method_num, method_num))
    is_normal_methods = np.zeros(subject_num)
    p_val_subjects = np.zeros((method_num, subject_num, subject_num))
    is_normal_subjects = np.zeros(method_num)
    for s in range(subject_num):
        p_val_methods[s], is_normal_methods[s] = calc_half_p_matrix(all_prediction_error_mat[s])
    for m in range(method_num):
        p_val_subjects[m], is_normal_subjects[m]= calc_half_p_matrix(all_prediction_error_mat[:, m])
    print('is_normal_methods: ', is_normal_methods)
    print('is_normal_subjects: ', is_normal_subjects)
    np.savetxt("results/result_data/p_val_methods.csv", p_val_methods.reshape((-1, method_num)), delimiter=",")
    np.savetxt("results/result_data/p_val_subjects.csv", p_val_subjects.reshape((-1, subject_num)), delimiter=",")


def plot_all_gazes_and_predicted_foot_placements_points(foot_placements_in_world_list_list, gaze_location_list_list,
                                                        predicted_foot_placements_in_world_list_list,
                                                        actual_support_block_center_in_world_list_list,
                                                        cls_acc_mat_all):
    subject_num = len(foot_placements_in_world_list_list)
    fig = plt.figure(figsize=(16, 3))
    plt.rcParams["font.size"] = 10
    cls_acc_mean_mat = np.mean(np.asarray(cls_acc_mat_all), axis=0)
    global_r_best, global_c_best = np.unravel_index(cls_acc_mean_mat.argmax(), cls_acc_mean_mat.shape)
    for s in range(subject_num):
        foot_placements_in_world_mat = np.concatenate(foot_placements_in_world_list_list[s], axis=-2)
        gaze_location_mat = np.concatenate(gaze_location_list_list[s], axis=-2)
        ''' predicted_foot_placements_in_world_mat: number of end phases * number of phase width * step number * 2'''
        predicted_foot_placements_in_world_mat = np.concatenate(predicted_foot_placements_in_world_list_list[s], axis=-2)
        actual_support_block_center_in_world_mat = np.concatenate(actual_support_block_center_in_world_list_list[s],
                                                                  axis=-2)
        ''' gaze_foot_error_mat: number of end phases * number of phase width * step number'''
        gaze_block_center_error_mat = np.linalg.norm(
            gaze_location_mat - np.reshape(actual_support_block_center_in_world_mat[:, :2], (1, 1, -1, 2)), axis=-1)
        gaze_block_center_error_mean_mat = np.mean(gaze_block_center_error_mat, axis=-1)
        '''First ensure the classification accuracy is the maximum, 
                then select the window in which the gaze_block_center distance is minimum'''
        gaze_block_center_error_mean_mat[cls_acc_mat_all[s] < cls_acc_mat_all[s].max()] = 1000
        r_best, c_best = np.unravel_index(gaze_block_center_error_mean_mat.argmin(),
                                          gaze_block_center_error_mean_mat.shape)


        error_data_list = [gaze_location_mat[global_r_best, global_c_best] - foot_placements_in_world_mat[:, :2],
                           predicted_foot_placements_in_world_mat[global_r_best, global_c_best] - foot_placements_in_world_mat[:, :2],
                           gaze_location_mat[r_best, c_best] - foot_placements_in_world_mat[:, :2],
                           predicted_foot_placements_in_world_mat[r_best, c_best] - foot_placements_in_world_mat[:, :2]]

        plot_gazes_and_predicted_foot_placements_points(subject_num, s, error_data_list)
        plt.xlabel(r'$\hat{x} - x$ (m)' + '\nSubject {}'.format(s+1))
        if s == 0:
            plt.ylabel(r'$\hat{y} - y$ (m)')
            plt.yticks([-0.4, -0.2,0, 0.2, 0.4])
        else:
            plt.yticks([])


    fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.49, 0.90), frameon=False)
    figName = 'results/{}'.format('exp3_gaze_and_foot_scatters')
    fig.tight_layout()
    plt.savefig(figName + '.pdf', bbox_inches='tight')
    plt.savefig(figName + '.png', bbox_inches='tight')
    plt.show()


def plot_gazes_and_predicted_foot_placements_points(subject_num, subject_idx, error_data_list):
    ax = plt.subplot(1, subject_num, subject_idx + 1)
    legend_vec = ['Gaze', 'Gaze + environmental context', 'Gaze + user-dependent window', 'Gaze + environmetal context + user-dependent window']
    for k in range(len(error_data_list)):
        color = np.array(list(plt.get_cmap("tab10")(k / 10))).reshape((1, -1))
        if 0 == subject_idx:
            label = legend_vec[k]
        else:
            label = None
        plt.scatter(np.mean(error_data_list[k][:, 0]), np.mean(error_data_list[k][:, 1]), c=color, s=20, label=label)
        plot_scatter_with_confidence_ellipse(error_data_list[k][:, 0], error_data_list[k][:, 1], ax,
                                                   xlim=[-0.4, 0.4], ylim=[-0.4, 0.4],
                                                   color=color, s=5)
        print('{} bias: {} m'.format(legend_vec[k], np.linalg.norm(np.mean(error_data_list[k], axis=0))))


def plot_best_time_windows(best_end_phase_mat, best_phase_width_mat):
    plot_box(best_end_phase_mat, y_label='Best lead time (% of step)', img_path='results/exp3_lead_time')
    plot_box(best_phase_width_mat, y_label='Best length of time window (% of step)', img_path='results/exp3_window_length')

def plot_box(best_phase_mat, y_label, img_path):
    fig = plt.figure(figsize=(4, 3))
    plt.rcParams["font.size"] = 8
    plt.boxplot(best_phase_mat)
    plt.ylabel(y_label)
    x_tick_list = []
    for r in range(len(best_phase_mat)):
        x_tick_list.append('Subject {}'.format(r + 1))
    print(x_tick_list)
    plt.xticks(np.arange(1, len(best_phase_mat)+1), x_tick_list)
    fig.tight_layout()
    plt.savefig(img_path + '.pdf', bbox_inches='tight')
    plt.savefig(img_path + '.png', bbox_inches='tight')
    plt.show()

def plot_error_bar(error_mean_mat, error_std_mat, result_dir):
    '''
        rows: subject
        cols: methods
    '''
    fig = plt.figure(figsize=(16, 3))
    plt.rcParams.update({'font.size': 10})
    plt.tight_layout()
    subject_num = error_mean_mat.shape[0]
    method_num = error_mean_mat.shape[1]
    # color_vec = cm.get_cmap('Set3', 12)
    color_vec = cm.get_cmap("tab10", 10)
    hatch_vec = ['', '-', '/', '\\', 'x', '-/', '-\\', '-x']
    legend_vec = ['Gaze', 'Gaze + environmental context', 'Gaze + user-dependent window', 'Gaze + environmetal context + user-dependent window']
    x_label_vec = []
    for s in range(subject_num):
        x_vec = np.arange(start=s * (method_num + 1) + 0.5, stop=(s+1) * (method_num + 1), step=1)
        for m in range(method_num):
            plt.bar(x_vec[m], error_mean_mat[s, m], color=color_vec(m), edgecolor='black')
                    # hatch=hatch_vec[m % len(hatch_vec)])
        plt.errorbar(x_vec[:-1], error_mean_mat[s], yerr = np.clip(error_std_mat[s], a_min=0, a_max=error_mean_mat[s]), fmt='.',
                     solid_capstyle='projecting', capsize=5, color='black')
        x_label_vec.append('Subject {}'.format(s+1))
    plt.xticks(np.arange(start=2, stop = subject_num * (method_num+1), step=method_num+1), x_label_vec)
    plt.ylabel('Error of predicted foot placements')
    plt.yticks([0, 0.2, 0.4])
    fig.legend(legend_vec, loc='lower center',
               ncol=4, bbox_to_anchor=(0.49, 0.92), frameon=False)
    fig.tight_layout()
    figName = '{}/{}'.format(result_dir, 'exp3_error_mean_std')
    plt.savefig(figName + '.pdf', bbox_inches='tight')
    plt.savefig(figName + '.png', bbox_inches='tight')
    plt.show()




def calc_best_gaze_window(foot_placements_in_world, actual_foot_placement_indices, gaze_points_offset, gaze_time_vec, step_time_vec,
                          cloud_pcd, center_mat, result_dir):
    end_phase_vec = np.arange(-1.1, 0, 0.2)
    phase_width_vec = np.arange(0.3, 1.5, 0.2)
    gaze_location_mat = np.zeros((len(end_phase_vec), len(phase_width_vec),
                                  len(foot_placements_in_world), 2))
    predicted_foot_placements_in_world = np.zeros((len(end_phase_vec), len(phase_width_vec),
                                                   len(foot_placements_in_world), 2))
    actual_support_block_center_in_world = center_mat[:, :].reshape((-1, 1, 3))[actual_foot_placement_indices, 0, :]
    predicted_foot_placement_indices = np.zeros((len(end_phase_vec),
                                                 len(phase_width_vec),
                                                 len(actual_foot_placement_indices)))
    for r in range(len(end_phase_vec)):
        for c in range(len(phase_width_vec)):
            gaze_points_list = calc_gaze_points(gaze_points_offset,
                                                gaze_time_vec,
                                                step_time_vec,
                                                phase_width=phase_width_vec[c],
                                                end_phase=end_phase_vec[r])
            gaze_location_mat[r, c] = calc_mean_of_gaze(gaze_points_list)
            predicted_foot_placement_indices[r, c], predicted_foot_placements_in_world[r, c] \
                = predict_foot_placements(gaze_location_mat[r, c], center_mat)
    cls_acc_every_step_mat = predicted_foot_placement_indices == np.reshape(actual_foot_placement_indices, (1, 1, -1))
    cls_acc_mat = np.mean(cls_acc_every_step_mat, axis=-1)
    print('Classification acc: ', cls_acc_mat)
    gaze_foot_error_mat = np.linalg.norm(gaze_location_mat - np.reshape(foot_placements_in_world[:, :2], (1, 1, -1, 2)), axis=-1)
    # print('distance_error_mat: {}'.format(distance_error_mat))
    print('Mean of the shortest gaze-foot error: {}'.format(np.mean(np.min(gaze_foot_error_mat, axis=(0, 1)))))
    gaze_foot_error_mat = np.mean(gaze_foot_error_mat, axis=-1)
    print('Gaze-foot error: ', gaze_foot_error_mat)

    gaze_block_center_error_mat = np.linalg.norm(
        gaze_location_mat - np.reshape(actual_support_block_center_in_world[:, :2], (1, 1, -1, 2)), axis=-1)
    gaze_block_center_error_mat = np.mean(gaze_block_center_error_mat, axis=-1)

    predicted_foot_error_mat =  np.linalg.norm(predicted_foot_placements_in_world - np.reshape(foot_placements_in_world[:, :2], (1, 1, -1, 2)), axis=-1)
    print('Mean of the shortest predicted foot placement error: {}'.format(np.mean(np.min(predicted_foot_error_mat, axis=(0, 1)))))
    predicted_foot_error_mat = np.mean(predicted_foot_error_mat, axis=-1)
    print('Predicted foot placement error: ', predicted_foot_error_mat)

    np.save('{}/gaze_and_foot_placements.npy'.format(result_dir),
            {'foot_placements_in_world': foot_placements_in_world,
             'gaze_location_mat': gaze_location_mat,
             'predicted_foot_placements_in_world': predicted_foot_placements_in_world,
             'actual_support_block_center_in_world': actual_support_block_center_in_world,
             'gaze_foot_error_mat': gaze_foot_error_mat,
             'predicted_foot_error_mat': predicted_foot_error_mat,
             'cls_acc_every_step_mat':cls_acc_every_step_mat,
             'cls_acc_mat': cls_acc_mat,
             'end_phase_vec': end_phase_vec,
             'phase_width_vec': phase_width_vec,
             'actual_foot_placement_indices': actual_foot_placement_indices,
             'predicted_foot_placement_indices': predicted_foot_placement_indices,
             'step_time_vec': step_time_vec,
             'gaze_points_offset': gaze_points_offset,
             'gaze_time_vec': gaze_time_vec,
             'center_mat': center_mat})

    r_max, c_max = np.unravel_index(gaze_block_center_error_mat.argmin(), cls_acc_mat.shape)
    phase_width = phase_width_vec[c_max]
    end_phase = end_phase_vec[r_max]
    print('Best phase width: {}, end_phase: {}'.format(phase_width, end_phase))
    return phase_width, end_phase


def calc_mean_of_gaze(gaze_points_list):
    gaze_location_vec = np.zeros((len(gaze_points_list), 2))
    for i in range(len(gaze_points_list)):
        gaze_location_vec[i] = np.mean(gaze_points_list[i], axis=0)[:2]
    return gaze_location_vec


def calc_gaze_points(gaze_points_offset, gaze_time_vec,
                     step_time_vec, end_phase=-0.2, phase_width=0.3):
    gaze_points_list = []
    for idx in range(2, len(step_time_vec)):
        start_gaze_time = phase_to_time(step_time_vec, idx + end_phase - phase_width)
        end_gaze_time = phase_to_time(step_time_vec, idx + end_phase)
        gaze_indices = np.bitwise_and(
            gaze_time_vec < end_gaze_time,
            gaze_time_vec > start_gaze_time)
        gaze_points_list.append(gaze_points_offset[gaze_indices])
    return gaze_points_list


def predict_foot_placements(gaze_location_vec, center_mat):
    gaze_location_vec = np.reshape(gaze_location_vec, (1, -1, 2))
    center_mat = np.reshape(center_mat[:, :2], (-1, 1, 2))
    distance_error_mat = np.linalg.norm(center_mat - gaze_location_vec, axis=-1)
    predicted_foot_placement_indices = np.argmin(distance_error_mat, axis=0)
    predicted_foot_placements = center_mat[predicted_foot_placement_indices, 0, :]
    return predicted_foot_placement_indices, predicted_foot_placements


def phase_to_time(heel_strike_time_vec, phase):
    if phase <= 0:
        return heel_strike_time_vec[0]
    start_idx = int(np.floor(phase))
    end_idx = int(np.ceil(phase))
    phase_in_step = phase - start_idx
    return phase_in_step * (heel_strike_time_vec[end_idx] - heel_strike_time_vec[start_idx]) \
           + heel_strike_time_vec[start_idx]


# def calc_sum_of_gaussian(xy_points):
def calc_cluster_center(cloud_pcd, labels):
    points = np.asarray(cloud_pcd.points)
    center_mat = np.zeros((np.max(labels) + 1, 3))
    for i in range(np.max(labels) + 1):
        center_mat[i] = np.mean(points[labels == i], axis=0)
    indices = np.argsort(center_mat[:, 0])
    center_mat = center_mat[indices]
    return center_mat



def calc_foot_placements(val_dir, center_mat, test_idx = 3):
    test_idx = test_idx
    test_dir = '{}/test{}'.format(val_dir, test_idx)
    foot_placement_indices = np.genfromtxt(
        '{}/results/foot_placements.txt'.format(test_dir),
        delimiter = ',', dtype=np.int)
    foot_placements = center_mat[foot_placement_indices]
    return foot_placements, foot_placement_indices



def create_rgba_heat_map(gaze_points_list, map_range, heat_scale = 2e3, bins=1000):
    heat_map_list = []
    for i in range(len(gaze_points_list)):
        gaze_points = gaze_points_list[i]
        if 0 == len(gaze_points):
            continue
        heat_map, extent, xy_max = create_Gaussian_map(
            gaze_points[:, 0], gaze_points[:, 1], map_range=map_range, bins=bins)
        # heat_map, extent, xy_max = create_heat_map(
        #     gaze_points[:, 0], gaze_points[:, 1], map_range=map_range, bins=bins)
        heat_map_list.append(heat_map)
    heat_map = np.sum(np.asarray(heat_map_list), axis=0)

    cmap = matplotlib.cm.get_cmap('seismic') # cmap alternatives: seismic,
    heat_RGBA = np.zeros(heat_map.shape + (4,))
    heat_RGBA[:] = cmap(heat_map[:])
    heat_RGBA[..., 2] = 0
    heat_RGBA[..., 3] = heat_map
    return heat_RGBA, extent, xy_max



def create_Gaussian_map(x, y, map_range = None, bins=1000):
    if map_range is None:
        map_range = np.asarray([[0, 5],
                                [-1, 1.5]])
    extent = [map_range[0, 0], map_range[0, -1], map_range[1, 0], map_range[1, -1]]
    # Our 2-dimensional distribution will be over variables X and Y
    X = np.linspace(map_range[0, 0], map_range[0, -1], bins)
    Y = np.linspace(map_range[1, 0], map_range[1, -1], bins)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([np.mean(x), np.mean(y)])
    var = np.array([[np.var(x), 0],
                    [0, np.var(y)]])
    if np.linalg.norm(var) > 0:
        var = 0.2**2 * var / np.linalg.norm(var)
    var[[0, 1], [0, 1]] = np.clip(var[[0, 1], [0, 1]], 1e-2, 5e-2)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    # The distribution on the variables X, Y packed into pos.
    heatmap = multivariate_gaussian(pos, mu, var)
    heatmap = heatmap / np.max(heatmap)
    return heatmap, extent, mu


def multivariate_gaussian(pos, mu, var):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    var_det = np.linalg.det(var)
    var_inv = np.linalg.inv(var)
    N = np.sqrt((2 * np.pi) ** n * var_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, var_inv, pos - mu)

    return np.exp(-fac / 2) / N


def scatter_cloud_pcd(cloud_pcd, gaze_points_list, center_mat,
                      foot_placements):
    points = np.asarray(cloud_pcd.points)
    colors = np.asarray(cloud_pcd.colors)
    margin = 0.2
    map_range = np.asarray([[np.min(points[:, 0])-margin, np.max(points[:, 0]+margin)],
                            [np.min(points[:, 1])-margin, np.max(points[:, 1]+margin)]])

    heat_RGBA, extent, xy_max = create_rgba_heat_map(gaze_points_list, map_range)

    fig = plt.figure(figsize=(6, 3))
    plt.tight_layout()
    # plt.imshow(heat_RGBA, extent=extent, origin='lower', zorder=1)

    plt.scatter(points[:, 0], points[:, 1], c=colors, zorder=0)
    plt.plot(xy_max[0], xy_max[1], c='red',
             markersize=10, marker='o', zorder=2)
    plt.plot(foot_placements[::2, 0], foot_placements[::2, 1], c='blue',
             markersize = 20, marker='*', zorder=2)
    plt.plot(foot_placements[1::2, 0], foot_placements[1::2, 1], c='green',
             markersize=20, marker='*', zorder=2)
    plt.plot(center_mat[:, 0], center_mat[:, 1], '+--', c='purple',
             markersize=20, zorder=2)
    for i in range(len(center_mat)):
        plt.annotate("{}".format(i),
                     xy=(center_mat[i, 0], center_mat[i, 1]),
                     color="white", fontsize=20,
                     ha="center", va="center",
                     zorder=i+3)

    plt.xlim(map_range[0])
    plt.ylim(map_range[1])
    plt.axis('off')
    fig.tight_layout()
    return fig

def plot_all_prediction_accuracy(val_dir_list):
    cls_acc_mat_all = []
    gaze_foot_error_mat_all = []
    predicted_foot_error_mat_all = []
    for val_dir in val_dir_list:
        cls_acc_mat, gaze_foot_error_mat, predicted_foot_error_mat, end_phase_vec, phase_width_vec = \
            read_and_calc_cls_and_error_mat(val_dir)
        cls_acc_mat_all.append(cls_acc_mat)
        gaze_foot_error_mat_all.append(gaze_foot_error_mat)
        predicted_foot_error_mat_all.append(predicted_foot_error_mat)

    cls_acc_mean_mat = np.mean(np.asarray(cls_acc_mat_all), axis=0)
    gaze_foot_error_mean_mat = np.mean(np.asarray(gaze_foot_error_mat_all), axis=0)
    predicted_foot_error_mean_mat = np.mean(np.asarray(predicted_foot_error_mat_all), axis=0)

    cls_map = pd.DataFrame(np.round(cls_acc_mean_mat * 100).astype(int), index=np.abs(end_phase_vec * 100).astype(int),
                           columns=(phase_width_vec * 100).astype(int))
    plot_foot_placement_prediction('results', cls_map, file_name='exp3_cls_acc', is_classification=True)

    gaze_foot_error_map = pd.DataFrame(gaze_foot_error_mean_mat, index=np.abs(end_phase_vec * 100).astype(int),
                                columns=(phase_width_vec * 100).astype(int))
    plot_foot_placement_prediction('results', gaze_foot_error_map, file_name='exp3_gaze_foot_error', is_classification=False)

    predicted_foot_error_map = pd.DataFrame(predicted_foot_error_mean_mat, index=np.abs(end_phase_vec * 100).astype(int),
                                       columns=(phase_width_vec * 100).astype(int))
    plot_foot_placement_prediction('results', predicted_foot_error_map, file_name='exp3_predicted_foot_error', is_classification=False)


def read_and_calc_cls_and_error_mat(val_dir):
    cls_acc_mat_all = []
    gaze_foot_error_mat_all = []
    predicted_foot_error_mat_all = []
    if 'large' in val_dir:
        test_num = 4
    else:
        test_num = 5
    for test_idx in range(test_num):
        result_dir = '{}/test{}/results'.format(val_dir, test_idx)
        gaze_and_foot_placement_data = np.load('{}/gaze_and_foot_placements.npy'.format(result_dir),
                                               allow_pickle=True).item()
        cls_acc_mat = gaze_and_foot_placement_data['cls_acc_mat']
        cls_acc_mat_all.append(cls_acc_mat)

        gaze_foot_error_mat_all.append(gaze_and_foot_placement_data['gaze_foot_error_mat'])
        predicted_foot_error_mat_all.append(gaze_and_foot_placement_data['predicted_foot_error_mat'])
        # print('idx: ', test_idx)
        # print(cls_acc_mat)

    end_phase_vec = gaze_and_foot_placement_data['end_phase_vec']
    phase_width_vec = gaze_and_foot_placement_data['phase_width_vec']

    cls_acc_mat = np.mean(np.asarray(cls_acc_mat_all), axis=0)
    gaze_foot_error_mat = np.mean(np.asarray(gaze_foot_error_mat_all), axis=0)
    predicted_foot_error_mat = np.mean(np.asarray(predicted_foot_error_mat_all), axis=0)
    return cls_acc_mat, gaze_foot_error_mat, predicted_foot_error_mat, end_phase_vec, phase_width_vec


def plot_prediction_accuracy(val_dir):
    cls_acc_mean_mat, gaze_foot_error_mean_mat, predicted_foot_error_mean_mat, end_phase_vec, phase_width_vec = \
        read_and_calc_cls_and_error_mat(val_dir)

    cls_map = pd.DataFrame(np.round(cls_acc_mean_mat * 100).astype(int), index=np.abs(end_phase_vec * 100).astype(int),
                           columns=(phase_width_vec * 100).astype(int))
    plot_foot_placement_prediction(val_dir, cls_map, file_name='exp3_cls_acc', is_classification=True)

    gaze_foot_error_map = pd.DataFrame(gaze_foot_error_mean_mat, index=np.abs(end_phase_vec * 100).astype(int),
                                       columns=(phase_width_vec * 100).astype(int))
    plot_foot_placement_prediction(val_dir, gaze_foot_error_map, file_name='exp3_gaze_foot_error',
                                   is_classification=False)

    predicted_foot_error_map = pd.DataFrame(predicted_foot_error_mean_mat,
                                            index=np.abs(end_phase_vec * 100).astype(int),
                                            columns=(phase_width_vec * 100).astype(int))
    plot_foot_placement_prediction(val_dir, predicted_foot_error_map, file_name='exp3_predicted_foot_error',
                                   is_classification=False)



def plot_foot_placement_prediction(img_dir, result_map, file_name, is_classification = True, fig = None):
    if fig is None:
        fig = plt.figure(figsize=(4, 3))
    plt.rcParams["font.size"] = 8
    sn.set(font_scale=0.8)  # for label size
    if is_classification:
        cmap = 'Blues'
    else:
        cmap = 'coolwarm'
    if is_classification:
        fmt = ".3g"
    else:
        fmt = ".2g"
    ax = sn.heatmap(result_map, annot=True, annot_kws={"size": 8}, cmap=cmap, cbar=False, fmt=fmt)
    # bottom, top = ax.get_ylim()
    # ax.set_ylim(bottom + 0.5, top - 0.5)
    if is_classification:
        for t in ax.texts: t.set_text(t.get_text() + "%")
    plt.xlabel('Length of time window (% of step)')
    plt.ylabel('Lead time (% of step)')
    fig.tight_layout()
    plt.savefig('{}/{}.pdf'.format(img_dir, file_name))
    plt.show()


def mark_target(img_rgb):
    def draw_circle(event,x,y, flags,param):
        global target_u, target_v
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img_rgb, (x, y), 20, color = (0, 255, 0), thickness= 5)
            target_u, target_v = x,y

    global target_u, target_v
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    while (1):
        cv2.imshow('image', img_rgb)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # Esc key to stop
            print('u: {}, v: {}'.format(target_u, target_v))
            break
    return np.asarray([target_u, target_v]), img_rgb

def mark_foot_position(val_dir):
    heel_strike_img_names = np.array(sorted(glob.glob("{}/*/heel_strike/16086913*.jpg".format(val_dir))))
    for heel_strike_img_name in heel_strike_img_names:
        img_rgb = cv2.imread(heel_strike_img_name, -1)
        target_uv, img_rgb = mark_target(img_rgb)
        cv2.imwrite(heel_strike_img_name, img_rgb)
        root, extension = os.path.splitext(heel_strike_img_name)
        np.save('{}{}'.format(root, '.npy'), target_uv)


def calc_half_p_matrix(data_list):
    rows = len(data_list)
    cols = rows
    p_matrix = np.ones((rows, cols))
    is_normal = True
    for r in range(rows):
        '''We can use t test only if all data belong to the normal distribution, 
        if one data vector doesn't belong to the normal distribution, we need to use Wilcoxon rank-sum statistic method'''
        if is_normal:
            statistic, critical_values, significance_level = stats.anderson(data_list[r])
            # print(statistic, critical_values[2]) # 5% significance level
            '''If value > 5%, the distribution is not normal'''
            if statistic > critical_values[2]:
                is_normal = False
    for r in range(rows):
        for c in range(cols):
            p_matrix[r, c] = calc_p_value(data_list[r], data_list[c], is_normal)
    return p_matrix, is_normal

def calc_p_value(a_vec, b_vec, is_normal = True):
    if is_normal:
        _, p_val = stats.ttest_ind(a_vec, b_vec)
    else:
        _, p_val = stats.ranksums(a_vec, b_vec)
    return p_val


def save_global_cam_video(val_dir, test_idx=0):
    rgb_img_names, depth_img_names, test_dir = obtain_image_names(val_dir, test_idx=test_idx)
    global_img_dir = '{}/web_camera_2'.format(test_dir)
    result_dir = '{}/results'.format(test_dir)
    synchronized_global_img_dir = '{}/synchronized_global_img'.format(result_dir)
    if os.path.exists(synchronized_global_img_dir):
        shutil.rmtree(synchronized_global_img_dir)

    global_img_list = sorted(glob.glob('{}/*.jpg'.format(global_img_dir)))
    synchronized_global_img_list = synchronize_two_file_list(global_img_list, rgb_img_names, is_find_nearest_file=True)
    if os.path.exists(synchronized_global_img_dir):
        shutil.rmtree(synchronized_global_img_dir)
    os.mkdir(synchronized_global_img_dir)
    for synchronized_global_img_name in synchronized_global_img_list[:-1]:
        img = cv2.imread(synchronized_global_img_name, -1)
        # img = cv2.flip(img, flipCode=1)
        cv2.imshow('img', img)
        cv2.waitKey(5)
        cv2.imwrite('{}/{}'.format(synchronized_global_img_dir, os.path.basename(synchronized_global_img_name)), img)

    read_image_to_video(glob.glob('{}/*.jpg'.format(synchronized_global_img_dir)),
                        '{}/synchronized_global_cam_video.mp4'.format(result_dir),
                        fps=20)


def render_gaze_and_foot_placements_video(val_dir, test_idx = 0, lead_time = -0.3, window_length = 0.7):
    # fuse_multi_clouds(val_dir, test_idx=test_idx, remove_ground=False, save_all_cloud=True)
    # fuse_multi_clouds(val_dir, test_idx=test_idx, remove_ground=True, save_all_cloud=True)
    # window_length, lead_time = analyze_gaze_and_footplacement(val_dir, test_idx=test_idx, render_every_gait=True,
    #                                                           change_phase=True)
    view_3D_gaze_and_foot_placement(val_dir, test_idx=test_idx, lead_time=lead_time, window_length=window_length,
                                    is_set_view_direction = True)
    view_3D_gaze_and_foot_placement(val_dir, test_idx=test_idx, lead_time=lead_time, window_length=window_length)


def view_3D_gaze_and_foot_placement(val_dir, test_idx = 0, lead_time = -0.3, window_length = 0.7,
                                    is_set_view_direction = False):
    ''' Render 3D point cloud, gazes, and human model'''
    '''1. First visualize point cloud'''
    rgb_img_names, depth_img_names, test_dir = obtain_image_names(val_dir, test_idx=test_idx)
    result_dir = '{}/results'.format(test_dir)
    gaze_dir = '{}/3D_gaze_and_foot_placements'.format(result_dir)
    if not os.path.exists(gaze_dir):
        os.mkdir(gaze_dir)
    '''Avoid seeing feet'''
    reference_file_list = np.array(sorted(glob.glob("{}/terrain_and_background_cloud/*.pcd".format(result_dir))))
    rgb_img_sub_time_vec = obtain_file_time_vec(reference_file_list)
    '''Default terrain cloud'''
    terrain_with_ground_pcd = o3d.io.read_point_cloud('{}/terrain_and_background_cloud/{:.3f}.pcd'.format(
        result_dir, rgb_img_sub_time_vec[0]))
    sequential_gaze_and_foot_placements_data = np.load('{}/sequential_gaze_and_foot_placements.npy'
                                                       .format(result_dir), allow_pickle=True).item()
    gaze_points_in_world = switch_array_xy_axes(sequential_gaze_and_foot_placements_data['gaze_points_in_world'])
    ''' Visualization setting'''
    viewer_setting_file = '{}/slam_viewer_setting.json'.format(result_dir)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, visible=True)
    origin_in_world = calc_original_offset_matrix(val_dir, test_idx=test_idx)
    current_in_origin_matrices = np.load('{}/transform_matrices.npy'.format(result_dir))
    world_axes = create_world_axes(origin_in_world, current_in_origin_matrices)
    human_body_mesh = []
    gaze_line_mesh = None

    if is_set_view_direction:
        img_indices = range(len(rgb_img_names)-2, len(rgb_img_names)-1)
    else:
        img_indices = range(len(rgb_img_names) - 1)

    for i in tqdm(img_indices):
        current_time = obtain_file_time(rgb_img_names[i])
        step_idx, heel_strike_times = calc_step_idx(test_dir, current_time)
        '''1. Plot cloud'''
        idx_nearest = np.argmin(np.abs(rgb_img_sub_time_vec-(current_time+0.5)))
        cloud_time = rgb_img_sub_time_vec[idx_nearest]
        terrain_with_ground_pcd = o3d.io.read_point_cloud('{}/terrain_and_background_cloud/{:.3f}.pcd'.format(
            result_dir, cloud_time))
        segmented_terrain_pcd = upsample_segmented_cloud(
            o3d.io.read_point_cloud('{}/segmented_terrain_pcd/{:.3f}.pcd'.format(result_dir, cloud_time)))

        if step_idx > 1:
            '''2. Plot gaze distribution'''
            rendered_terrain_with_ground_pcd = render_cloud_with_gaze_distribution(
                copy.copy(terrain_with_ground_pcd), val_dir, test_idx, current_time, lead_time, window_length, step_idx)
            '''3. Plot human model'''
            current_in_origin_mat = current_in_origin_matrices[i - 1]
            current_in_world_mat = np.matmul(origin_in_world, current_in_origin_mat)
            human_body_mesh, head_point = render_human_model(result_dir, current_time, step_idx, heel_strike_times, current_in_world_mat)

            '''4. Plot real-time gaze line'''
            gaze_line_mesh, _ = create_cylinder(start_point=head_point, end_point=gaze_points_in_world[i])
        else:
            rendered_terrain_with_ground_pcd = copy.copy(terrain_with_ground_pcd)
        img_name = '{}/{:.3f}.jpg'.format(gaze_dir, current_time)
        if gaze_line_mesh is None:
            vis = view_vis(vis, [rendered_terrain_with_ground_pcd, segmented_terrain_pcd, world_axes]
                           + human_body_mesh,
                           viewer_setting_file=viewer_setting_file,
                           img_name = img_name)
        else:
            vis = view_vis(vis, [rendered_terrain_with_ground_pcd, segmented_terrain_pcd, world_axes, gaze_line_mesh]
                           + human_body_mesh,
                           viewer_setting_file=viewer_setting_file,
                           img_name = img_name)

    if is_set_view_direction:
        vis.run()
        save_view_point(vis, viewer_setting_file=viewer_setting_file)

    read_image_to_video(glob.glob('{}/*.jpg'.format(gaze_dir)),
                        video_name='{}/3D_gaze_and_foot_placement.mp4'.format(result_dir), fps=20)



def view_gaze_and_foot_placement_3D(val_dir, test_idx = 0, end_idx = 73, lead_time = -0.5, window_length = 0.7,
                                    gaze_dir = None):
    rgb_img_names, depth_img_names, test_dir = obtain_image_names(test_idx=test_idx)
    current_time = obtain_file_time(rgb_img_names[end_idx])

    test_dir = '{}/test{}'.format(val_dir, test_idx)
    result_dir = '{}/results'.format(test_dir)

    # 1. Read and offset the original point cloud
    origin_in_world = calc_original_offset_matrix(val_dir, test_idx=test_idx)

    terrain_with_ground_pcd = o3d.io.read_point_cloud('{}/terrain_and_background_cloud/{:.2f}.pcd'.format(
        result_dir, current_time))
    terrain_with_ground_pcd.transform(origin_in_world)

    # 2. Read and offset the original terrain cloud
    terrain_pcd = o3d.io.read_point_cloud('{}/terrain_cloud/{:.2f}.pcd'.format(result_dir, current_time))
    terrain_pcd.transform(origin_in_world)
    terrain_pcd.translate([0, 0, -0.02])

    # 3. Read the segmented terrain cloud, which has been offset
    segmented_result_dir = '{}/segmented_terrain_pcd'.format(result_dir)
    segmented_terrain_pcd = o3d.io.read_point_cloud('{}/{:.2f}.pcd'.format(segmented_result_dir, current_time))

    # 4. Read and fuse gaze points
    gaze_points_in_origin = fuse_multi_gazes(test_idx=test_idx)
    gaze_points = np.ones((len(gaze_points_in_origin), 4))
    gaze_points[:, :3] = gaze_points_in_origin

    gaze_points_in_world = np.matmul(origin_in_world, np.transpose(gaze_points)).transpose()[:, :3]
    current_in_origin_matrices = np.load('{}/transform_matrices.npy'.format(result_dir))

    gaze_pcd = None
    gaze_end_point = gaze_points_in_world[end_idx]
    gaze_pcd = points_to_pcd(gaze_end_point, points_pcd=gaze_pcd)
    gaze_pcd.translate([0, 0, -0.03])

    current_in_origin_mat = current_in_origin_matrices[end_idx - 1]
    current_in_world_mat = \
        np.matmul(origin_in_world, current_in_origin_mat)
    head_point = current_in_world_mat[:3, 3]
    gaze_line_mesh, _ = \
        create_cylinder(start_point=head_point, end_point=gaze_end_point)

    # 5. Plot human head, waist, and foot placements
    # calculate the last two steps
    heel_strike_names = sorted(glob.glob('{}/heel_strike/*.jpg'.format(test_dir)))
    heel_strike_times = obtain_file_time_vec(heel_strike_names)
    gait_time = heel_strike_times[1:] - heel_strike_times[:-1]

    time_error_vec = heel_strike_times - current_time
    time_error_vec[time_error_vec > 0] = -100
    step_idx = np.argmax(time_error_vec)  # find the larget negative value, which indicates the last step
    # print('Time error: ', time_error_vec)
    # calculate foot_placements
    # sort along the x axis, and thus the axes should be changed before sorting
    segmented_all_terrain_pcd = o3d.io.read_point_cloud('{}/segmented_terrain_pcd.pcd'.format(result_dir, current_time))
    segmented_all_labels = np.load('{}/segmented_labels.npy'.format(result_dir))
    # print('segmented_all_labels: ', segmented_all_labels)
    center_mat = switch_array_xy_axes(calc_cluster_center(switch_pcd_xy_axes(segmented_all_terrain_pcd), segmented_all_labels))
    foot_placements, foot_placement_indices = calc_foot_placements(center_mat, test_idx=test_idx)

    # current_foot_placements = foot_placements[[step_idx-3, step_idx-2]]
    waist_point = head_point + np.asarray([0, 0, 0.7])
    start_points = np.zeros((3, 3))
    start_points[0] = head_point + np.asarray([0, 0, 0.2])
    start_points[1] = waist_point
    start_points[2] = waist_point

    end_points = np.zeros((3, 3))
    end_points[0] = waist_point
    last_idx = step_idx-3
    current_idx = step_idx-2
    if last_idx < 0:
        end_points[1] = waist_point + np.asarray([0, 0, 0.7])
    else:
        # print('Foot placements number: ', foot_placements.shape, ' step idx: ', current_idx + 1)
        next_idx = min(current_idx + 1, len(foot_placements)-1)
        end_points[1] = foot_placements[last_idx] + \
                        (foot_placements[next_idx] - foot_placements[last_idx]) * \
                        abs(time_error_vec[step_idx]) / gait_time[min(step_idx, len(gait_time)-1)]
        # end_points[1] += np.asarray([0, 0, -0.1])
    if current_idx < 0:
        end_points[2] = waist_point + np.asarray([0, 0, 0.7])
    else:
        end_points[2] = foot_placements[current_idx]

    human_body_mesh = []
    for i in range(len(start_points)):
        cylinder_color = plt.get_cmap("tab20b")((i+1) / 20)[:3]
        if 0 == i:
            cylinder_radius = 0.2
        else:
            cylinder_radius = 0.05
        segment_cylinder, _ = \
            create_cylinder(start_point=start_points[i], end_point=end_points[i], cylinder_color=cylinder_color,
                            radius=cylinder_radius)
        human_body_mesh.append(segment_cylinder)

    human_body_mesh.append(create_sphere_cloud(head_point + np.asarray([0, 0, 0.1]), color=plt.get_cmap("tab20b")(0)[:3], radius=0.1))

    # 6. Plot head map with the point cloud, like Fig.1.B in Matthins' paper.
    terrain_points = np.asarray(terrain_with_ground_pcd.points)
    map_range = np.asarray([[np.min(terrain_points[:, 0]) - 0.2, np.max(terrain_points[:, 0] + 0.2)],
                            [np.min(terrain_points[:, 1]) - 0.2, np.max(terrain_points[:, 1] + 0.2)]])

    gaze_points_list = analyze_gaze_and_footplacement(test_idx=test_idx, render_every_gait=False,
                                                      save_image=False, change_phase=False, end_time=current_time,
                                                      lead_time = lead_time, window_length = window_length)
    gaze_points_list = gaze_points_list[:step_idx+1]
    heat_map, _, _ = create_rgba_heat_map(gaze_points_list, map_range=map_range, heat_scale=8e2, bins=1000)

    terrain_colors = read_colors_from_heat_map(terrain_with_ground_pcd, heat_map, map_range, original_color_scale = 0.8)
    terrain_with_ground_pcd.colors = o3d.utility.Vector3dVector(terrain_colors)

    segmented_terrain_colors = read_colors_from_heat_map(segmented_terrain_pcd, heat_map, map_range, original_color_scale = 0.8)
    segmented_terrain_pcd.colors = o3d.utility.Vector3dVector(segmented_terrain_colors)
    segmented_points = np.array(segmented_terrain_pcd.points) + np.array([0, 0, -0.01])
    segmented_terrain_pcd.points = o3d.utility.Vector3dVector(segmented_points)

    # upsample point cloud
    for r in range(2):
        for c in range(2):
            new_segmented_terrain_pcd = copy.deepcopy(segmented_terrain_pcd)
            transform_mat = np.identity(4)
            transform_mat[0, 3] = r * 3e-3
            transform_mat[1, 3] = c * 3e-3
            new_segmented_terrain_pcd.transform(transform_mat)
            segmented_terrain_pcd += new_segmented_terrain_pcd

    # 7. Visualization.
    viewer_setting_file = 'paras/slam_viewer_setting.json'
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, visible = False)

    world_axes = create_world_axes(origin_in_world, current_in_origin_matrices)
    if gaze_dir is None:
        gaze_dir = '{}/gaze_and_foot_placements'.format(result_dir)
        img_name = '{}/gaze_{:.2f}.jpg'.format(gaze_dir, current_time)
    else:
        img_name =  '{}/lead_{:.2f}_length_{:.2f}.jpg'.format(gaze_dir, lead_time, window_length)

    if not os.path.exists(gaze_dir):
        os.mkdir(gaze_dir)
    vis = view_vis(vis,
                   [terrain_with_ground_pcd, gaze_line_mesh,
                    segmented_terrain_pcd,
                    world_axes
                    ] + human_body_mesh,
                   viewer_setting_file=viewer_setting_file,
                   img_name=img_name
                   )
    # vis.run()
    # utils.save_view_point(vis, viewer_setting_file=viewer_setting_file)


def render_human_model(result_dir, current_time, step_idx, heel_strike_times, current_in_world_mat):
    '''Plot human head, waist, and foot placements'''
    '''calculate foot_placements'''
    gaze_and_foot_placements_data = np.load('{}/gaze_and_foot_placements.npy'.format(result_dir), allow_pickle=True).item()
    # print(gaze_and_foot_placements_data['predicted_foot_placement_indices'])
    foot_placements = switch_array_xy_axes(gaze_and_foot_placements_data['actual_support_block_center_in_world'])

    head_point = current_in_world_mat[:3, 3]
    waist_point = head_point + np.asarray([0, 0, 0.7])

    start_points = np.zeros((3, 3))
    neck_point = head_point + np.asarray([0, 0, 0.2])
    start_points[0] = neck_point
    start_points[1] = waist_point
    start_points[2] = waist_point

    end_points = np.zeros((3, 3))
    end_points[0] = waist_point
    current_idx = step_idx - 2
    last_idx = current_idx - 1
    next_idx = min(current_idx + 1, len(foot_placements) - 1)
    gait_cycle_duration = heel_strike_times[step_idx + 1] - heel_strike_times[step_idx] # current heel strike to the next heel strike
    phase = (current_time - heel_strike_times[step_idx]) / gait_cycle_duration
    if last_idx < 0:
        end_points[1] = foot_placements[current_idx] + \
                        (foot_placements[next_idx] - foot_placements[current_idx]) * phase
    else:
        end_points[1] = foot_placements[last_idx] + \
                        (foot_placements[next_idx] - foot_placements[last_idx]) * phase
    print('last: {}, current : {}, next: {}, phase: {}'.format(
        last_idx, current_idx, next_idx, phase))
    if current_idx < 0:
        end_points[2] = waist_point + np.asarray([0, 0, 0.7])
    else:
        end_points[2] = foot_placements[current_idx]


    human_body_mesh = []
    for i in range(len(start_points)):
        cylinder_color = plt.get_cmap("tab20b")((i + 1) / 20)[:3]
        if 0 == i:
            cylinder_radius = 0.2
        else:
            cylinder_radius = 0.05
        segment_cylinder, _ = \
            create_cylinder(start_point=start_points[i], end_point=end_points[i], cylinder_color=cylinder_color,
                            radius=cylinder_radius)
        human_body_mesh.append(segment_cylinder)

    human_body_mesh.append(
        create_sphere_cloud(head_point + np.asarray([0, 0, 0.1]), color=plt.get_cmap("tab20b")(0)[:3], radius=0.1))
    return human_body_mesh, head_point

def render_cloud_with_gaze_distribution(terrain_with_ground_pcd, val_dir, test_idx, current_time, lead_time,
                                        window_length, step_idx):
    terrain_points = np.asarray(terrain_with_ground_pcd.points)
    map_range = np.asarray([[np.min(terrain_points[:, 0]) - 0.2, np.max(terrain_points[:, 0] + 0.2)],
                            [np.min(terrain_points[:, 1]) - 0.2, np.max(terrain_points[:, 1] + 0.2)]])
    gaze_points_list = extract_gaze_point_list_based_on_current_time(val_dir, test_idx, current_time, lead_time, window_length)
    gaze_points_list = gaze_points_list[:step_idx + 1]
    heat_map, _, _ = create_rgba_heat_map(gaze_points_list, map_range=map_range, heat_scale=8e2, bins=1000)
    terrain_colors = read_colors_from_heat_map(terrain_with_ground_pcd, heat_map, map_range, original_color_scale=0.8)
    terrain_with_ground_pcd.colors = o3d.utility.Vector3dVector(terrain_colors)
    return terrain_with_ground_pcd


def extract_gaze_point_list_based_on_current_time(val_dir, test_idx, current_time, lead_time, window_length):
    test_dir = '{}/test{}'.format(val_dir, test_idx)
    origin_in_world_mat = calc_original_offset_matrix(val_dir, test_idx=test_idx)
    gaze_points_in_origin = fuse_multi_gazes(val_dir, test_idx=test_idx)
    gaze_points_in_world = origin_2_world_system(gaze_points_in_origin, origin_in_world_mat)

    # read gait time:
    gaze_names = sorted(glob.glob('{}/gaze_in_depth/*.npy'.format(test_dir)))
    gaze_time_vec = obtain_file_time_vec(gaze_names)
    gait_names = sorted(glob.glob('{}/heel_strike/*.jpg'.format(test_dir)))
    step_time_vec = obtain_file_time_vec(gait_names)

    gaze_points_in_world = gaze_points_in_world[gaze_time_vec < current_time]
    gaze_time_vec = gaze_time_vec[gaze_time_vec < current_time]
    gaze_points_list = calc_gaze_points(gaze_points_in_world,
                                        gaze_time_vec,
                                        step_time_vec,
                                        end_phase=lead_time,
                                        phase_width=window_length)

    return gaze_points_list


def calc_step_idx(test_dir, current_time):
    # calculate the last two steps
    heel_strike_names = sorted(glob.glob('{}/heel_strike/*.jpg'.format(test_dir)))
    heel_strike_times = obtain_file_time_vec(heel_strike_names)
    time_error_vec = heel_strike_times - current_time
    time_error_vec[time_error_vec > 0] = -100
    step_idx = np.argmax(time_error_vec)  # find the larget negative value, which indicates the last step
    return step_idx, heel_strike_times

def create_world_axes(origin_in_world, current_in_origin_matrices):
    initial_in_origin_mat = current_in_origin_matrices[0]
    initial_in_world_mat = \
        np.matmul(origin_in_world, initial_in_origin_mat)
    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    world_axes.rotate(np.asarray([[1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, -1]]), center=np.asarray([0, 0, 0]))
    world_axes.translate(initial_in_world_mat[:3, 3] + np.asarray([0, 0, 1.5]))
    return world_axes


def read_colors_from_heat_map(terrain_with_ground_pcd, heat_map, map_range, bins=1000, original_color_scale = 0.5):
    terrain_points = np.asarray(terrain_with_ground_pcd.points)
    terrain_colors = np.asarray(terrain_with_ground_pcd.colors)

    x_vec = terrain_points[:, 0]
    # c = (x - x_min) / ((x_max - x_min) / bins)
    c_indices = (x_vec - map_range[0, 0])/((map_range[0, 1] - map_range[0, 0]) / bins)
    c_indices = np.clip(c_indices.astype(np.int), 0, 999)

    y_vec = terrain_points[:, 1]
    r_indices = (y_vec - map_range[1, 0]) / ((map_range[1, 1] - map_range[1, 0]) / bins)
    r_indices = np.clip(r_indices.astype(np.int), 0, 999)

    heat_colors = heat_map[tuple([tuple(r_indices.tolist()), tuple(c_indices.tolist())])][..., :3]

    terrain_colors = 0.5 * heat_colors + original_color_scale * terrain_colors

    return terrain_colors



def points_to_pcd(gaze_3d, points_pcd = None, gaze_color = [1, 0, 0]):
    gaze_pcd_temp = create_sphere_cloud(gaze_3d, color=gaze_color)
    if points_pcd is None:
        points_pcd = gaze_pcd_temp
    else:
        points_pcd += gaze_pcd_temp
    return points_pcd


def create_sphere_cloud(gaze_3d, color = [1.0, 0, 0], radius=0.025):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius, resolution=20)
    mesh_sphere.translate(gaze_3d)
    mesh_sphere.paint_uniform_color(color)
    gaze_pcd_temp = mesh_sphere.sample_points_uniformly(number_of_points=5000)
    return gaze_pcd_temp

def upsample_segmented_cloud(segmented_terrain_pcd):
    for r in range(2):
        for c in range(2):
            new_segmented_terrain_pcd = copy.deepcopy(segmented_terrain_pcd)
            transform_mat = np.identity(4)
            transform_mat[0, 3] = r * 3e-3
            transform_mat[1, 3] = c * 3e-3
            new_segmented_terrain_pcd.transform(transform_mat)
            segmented_terrain_pcd += new_segmented_terrain_pcd
    return segmented_terrain_pcd


def save_temporal_gaze_video(val_dir, test_idx=0, save_pdf = False, lead_time = -0.3, window_length = 0.7,
                             last_only = False, temporal_result_dir = None, initial_right = False):
    test_dir = '{}/test{}'.format(val_dir, test_idx)
    result_dir = '{}/results'.format(test_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if temporal_result_dir is None:
        temporal_result_dir = '{}/temporal_result'.format(result_dir)
    if not os.path.exists(temporal_result_dir):
        os.mkdir(temporal_result_dir)

    rgb_img_names, depth_img_names, test_dir = obtain_image_names(val_dir, test_idx=test_idx)
    gaze_times = obtain_file_time_vec(rgb_img_names)[:-1]

    sequential_gaze_and_foot_placements_data = np.load('{}/sequential_gaze_and_foot_placements.npy'
                                                       .format(result_dir), allow_pickle=True).item()
    gaze_points_in_world = switch_array_xy_axes(sequential_gaze_and_foot_placements_data['gaze_points_in_world'])

    gaze_and_foot_placements_data = np.load('{}/gaze_and_foot_placements.npy'.format(result_dir),
                                            allow_pickle=True).item()
    foot_placements = switch_array_xy_axes(gaze_and_foot_placements_data['actual_support_block_center_in_world'])

    heel_strike_names = sorted(glob.glob('{}/heel_strike/*.jpg'.format(test_dir)))
    heel_strike_times = obtain_file_time_vec(heel_strike_names) - gaze_times[0]

    if last_only:
        fig = plt.figure(figsize=(20, 4))
        plt.rcParams["font.size"] = 16
        marker_scale = 5
    else:
        if save_pdf:
            fig = plt.figure(figsize=(4, 3))
            plt.rcParams["font.size"] = 8
            marker_scale = 1
        else:
            fig = plt.figure(figsize=(20, 8))
            plt.rcParams["font.size"] = 32
            marker_scale = 5

    y_axis_name_list = ['x', 'y']
    x_min, y_min = tuple((np.min(gaze_points_in_world[:, :2], axis=0) - 0.2).tolist())
    x_max, y_max = tuple((np.max(gaze_points_in_world[:, :2], axis=0) + 0.2).tolist())
    y_limb_vec = np.asarray([[x_max, x_min],
                             [-y_max, -y_min]])

    if save_pdf or last_only:
        k_vec = [len(gaze_times) - 1]
    else:
        k_vec = tqdm(range(len(gaze_times)))

    for k in k_vec:
        current_time = gaze_times[k]
        for i in range(2):
            plt.subplot(2, 1, i+1)
            if 1 == i:
                direction = -1
            else:
                direction = 1
            gaze_indices = (gaze_times <= current_time)
            step_indices = (heel_strike_times <= (current_time - gaze_times[0]))
            foot_indices = step_indices[2:]
            plt.plot(gaze_times[gaze_indices] - gaze_times[0], direction * gaze_points_in_world[gaze_indices, i], '.',
                     markersize=marker_scale)
            if initial_right:
                plt.plot(heel_strike_times[step_indices][2::2], direction * foot_placements[foot_indices][0::2, i], '*',
                         markersize=4 * marker_scale)
                plt.plot(heel_strike_times[step_indices][3::2], direction * foot_placements[foot_indices][1::2, i], '*',
                         markersize=4 * marker_scale)
            else:
                plt.plot(heel_strike_times[step_indices][3::2], direction * foot_placements[foot_indices][1::2, i], '*',
                         markersize=4 * marker_scale)
                plt.plot(heel_strike_times[step_indices][2::2], direction * foot_placements[foot_indices][0::2, i], '*',
                         markersize=4 * marker_scale)
            plt.ylabel('{} (m)'.format(y_axis_name_list[i]))
            ax = plt.gca()
            if last_only:
                ax.get_yaxis().set_label_coords(-0.03, 0.5)
            else:
                ax.get_yaxis().set_label_coords(-0.1, 0.5)


            step_num = min(len(heel_strike_times[step_indices])+1, len(heel_strike_times))

            if initial_right:
                init_idx_list = [3, 2]
            else:
                init_idx_list = [2, 3]
            for l in range(len(init_idx_list)):
                color = plt.get_cmap("Set2")(l / 8)
                color = tuple(list(color[:3]) + [0.5,])
                for j in range(init_idx_list[l], step_num, 2):
                    t_end = phase_to_time(heel_strike_times, j + lead_time)
                    t_start = phase_to_time(heel_strike_times, j + lead_time - window_length)
                    rect = patches.Rectangle((t_start, -10), t_end - t_start, 20, color= color)
                    currentAxis = plt.gca()
                    currentAxis.add_patch(rect)

            plt.ylim(y_limb_vec[i])
            plt.xlim([0, np.max(gaze_times - gaze_times[0]) + 1])
        plt.xlabel('Time (s)')
        if save_pdf:
            fig.legend(['Gaze', 'Right foot step', 'Left foot step'], loc='lower center',
                       ncol=3, bbox_to_anchor=(0.49, 0.93), frameon=False)
        else:
            bbox_to_anchor = (0.49, 0.89)
            fig.legend(['Gaze', 'Right foot step', 'Left foot step'], loc='lower center',
                       ncol=3, bbox_to_anchor=bbox_to_anchor, frameon=False)
        fig.tight_layout()
        if last_only:
            plt.savefig('{}/lead_{:.2f}_length_{:.2f}.jpg'.format(temporal_result_dir, abs(lead_time), abs(window_length)))
        else:
            if save_pdf:
                plt.savefig('{}/{:.2f}.pdf'.format(temporal_result_dir, current_time))
            else:
                plt.savefig('{}/{:.2f}.jpg'.format(temporal_result_dir, current_time))
        plt.clf()

    if not last_only:
        read_image_to_video(sorted(glob.glob('{}/*.jpg'.format(temporal_result_dir))),
                            video_name = '{}/temporal_result.mp4'.format(result_dir), fps=20)


def combine_videos(val_dir_list, test_idx = 4):
    folder_name_list = ['gaze_image', 'synchronized_global_img', '3D_gaze_and_foot_placements', 'temporal_result']
    for folder_name in folder_name_list:
        img_names = []
        for val_dir in val_dir_list:
            img_dir = '{}/test{}/results/{}'.format(val_dir, test_idx, folder_name)
            print(img_dir)
            img_names += sorted(glob.glob('{}/*.jpg'.format(img_dir)))

        read_image_to_video(img_names, video_name='results/videos/{}.mp4'.format(folder_name), fps=20)





if __name__ == '__main__':
    # split_rgbd_data_to_folder()
    plot_scatter_with_confidence_ellipse()
