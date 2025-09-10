#!/usr/bin/env python3

import time

import numpy as np

depth_info_width = 1280
depth_info_height = 720
depth_info_focal_x = 691.029052734375
depth_info_focal_y = 691.053466796875
depth_info_center_x = 638.14892578125
depth_info_center_y = 351.4556884765625

y_coords, x_coords = np.meshgrid(np.arange(depth_info_height), np.arange(depth_info_width))
full_pixel_mask = np.vstack((x_coords.ravel(), y_coords.ravel())).T
depth_npy = np.array(np.random.rand(depth_info_height, depth_info_width), dtype=np.float32)
depth_npy[107, 100] = np.nan
depth_npy[108, 100] = np.inf
depth_npy[100, 10] = 0

filter_invalid = True
# filter_invalid = False

# pixels = full_pixel_mask
# num = 500

pixels = np.array([[11, 204], [100, 100], [150, 100], [100, 107], [100, 108], [120, 100], [10, 100]])
num = 1000000

print("filter_invalid", filter_invalid)
print("pixels.shape[0]", pixels.shape[0])
print("num", num)
print()

# a

tic = time.perf_counter()
for i in range(num):
    coords = pixels[:, 1], pixels[:, 0]
    Z = depth_npy[coords]
    Z_mask = np.logical_and(np.isfinite(Z), Z != 0)
    if filter_invalid:
        Z = Z[Z_mask]
        X = (coords[1][Z_mask] - depth_info_center_x) * Z / depth_info_focal_x
        Y = (coords[0][Z_mask] - depth_info_center_y) * Z / depth_info_focal_y
    else:
        Z[~Z_mask] = np.nan
        X = (coords[1] - depth_info_center_x) * Z / depth_info_focal_x
        Y = (coords[0] - depth_info_center_y) * Z / depth_info_focal_y
    cloud_a = np.stack([X, Y, Z], axis=-1)
toc = time.perf_counter()
print(f"a {toc - tic:.3}s")

# b

depth_normal = np.full(shape=(depth_info_height, depth_info_width), fill_value=1.0, dtype=np.float32)
Z = depth_normal[full_pixel_mask[:, 1], full_pixel_mask[:, 0]]
X = (full_pixel_mask[:, 0] - depth_info_center_x) * Z / depth_info_focal_x
Y = (full_pixel_mask[:, 1] - depth_info_center_y) * Z / depth_info_focal_y
vectors_depth_1 = np.stack([X, Y, Z], axis=-1)
vectors_depth_1 = np.reshape(vectors_depth_1, (depth_normal.shape[0], depth_normal.shape[1], 3), order='F')

tic = time.perf_counter()
for i in range(num):
    coords = pixels[:, 1], pixels[:, 0]
    Z = depth_npy[coords]
    Z_mask = np.logical_and(np.isfinite(Z), Z != 0)
    if filter_invalid:
        Z = Z[Z_mask]
        Z = np.repeat(Z[:, np.newaxis], 3, axis=1)
        cloud_b = vectors_depth_1[coords[0][Z_mask], coords[1][Z_mask]] * Z
    else:
        Z[~Z_mask] = np.nan
        Z_stack = np.repeat(Z[:, np.newaxis], 3, axis=1)
        cloud_b = vectors_depth_1[coords] * Z_stack
toc = time.perf_counter()
print(f"b {toc - tic:.3}s")

# c

K = np.array([[depth_info_focal_x, 0.0, depth_info_center_x],
              [0.0, depth_info_focal_y, depth_info_center_y],
              [0.0, 0.0, 1.0]])
K_inv = np.linalg.inv(K)

tic = time.perf_counter()
for i in range(num):
    coords = pixels[:, 1], pixels[:, 0]
    Z = depth_npy[coords]
    Z_mask = np.logical_and(np.isfinite(Z), Z != 0)
    if filter_invalid:
        Z = Z[Z_mask]
        pixel_coords = np.stack([coords[1][Z_mask], coords[0][Z_mask], np.ones(Z.shape[0])])
    else:
        Z[~Z_mask] = np.nan
        pixel_coords = np.stack([coords[1], coords[0], np.ones(Z.shape[0])])
    cloud_c = Z * np.dot(K_inv, pixel_coords)
    cloud_c = cloud_c.T
toc = time.perf_counter()
print(f"c {toc - tic:.3}s")

# d

K = np.array([[depth_info_focal_x, 0.0, depth_info_center_x],
              [0.0, depth_info_focal_y, depth_info_center_y],
              [0.0, 0.0, 1.0]])
K_inv = np.linalg.inv(K)

x_coords, y_coords = np.meshgrid(np.arange(depth_info_width), np.arange(depth_info_height))
homogeneous_pixel_coords = np.stack((x_coords, y_coords, np.ones_like(x_coords)), axis=2)
vectors_depth_1 = homogeneous_pixel_coords @ K_inv.T

tic = time.perf_counter()
for i in range(num):
    coords = pixels[:, 1], pixels[:, 0]
    Z = depth_npy[coords]
    Z_mask = np.logical_and(np.isfinite(Z), Z != 0)
    if filter_invalid:
        Z = Z[Z_mask]
        selected_vectors = vectors_depth_1[coords[0][Z_mask], coords[1][Z_mask], :]
    else:
        Z[~Z_mask] = np.nan
        selected_vectors = vectors_depth_1[coords[0], coords[1], :]
    cloud_d = selected_vectors * Z[:, np.newaxis]
toc = time.perf_counter()
print(f"d {toc - tic:.3}s")

#

print()
print(np.allclose(cloud_a, cloud_b), np.allclose(cloud_b, cloud_c), np.allclose(cloud_c, cloud_d))
print(cloud_a[3])
print(cloud_b[3])
print(cloud_c[3])
print(cloud_d[3])


# filter_invalid True
# pixels.shape[0] 921600
# num 500

# a 18.9s
# b 25.8s
# c 22.3s
# d 26.9s


# filter_invalid False
# pixels.shape[0] 921600
# num 500

# a 15.9s
# b 25.2s
# c 20.5s
# d 25.4s


# filter_invalid True
# pixels.shape[0] 7
# num 1000000

# a 27.5s
# b 18.2s
# c 30.6s
# d 13.6s


# filter_invalid False
# pixels.shape[0] 7
# num 1000000

# a 26.8s
# b 17.8s
# c 29.9s
# d 13.1s
