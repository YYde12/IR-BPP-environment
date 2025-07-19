import numpy as np
from pydelatin import Delatin
from pydelatin.util import rescale_positions
import trimesh
import coacd
from scipy.spatial import ConvexHull, Delaunay

def getConvexHullActions_new(posZValid, mask, coors):
    allCandidates = []
    for rotIdx in range(len(posZValid)):
        print(f"原始 posZValid 范围: min={np.min(posZValid)}, max={np.max(posZValid)}")

        heightmap = posZValid[rotIdx].copy()
        heightmap[heightmap == 10e3] = 5
        print("heightmap", heightmap)
        print(f"[rotIdx {rotIdx}] 替换后范围: min={np.min(heightmap)}, max={np.max(heightmap)}")

        # 重建 free space mesh
        free_space_mesh = preview_heightmap_tin(heightmap)

        # CoACD 凸分解
        mesh_coacd = coacd.Mesh(free_space_mesh.vertices, free_space_mesh.faces)
        parts = coacd.run_coacd(mesh_coacd, max_convex_hull=10, threshold=0.1)
        print(f"[rotIdx {rotIdx}] CoACD 分解得到 {len(parts)} 个凸包")

        # 提前构造 Delaunay hulls
        filtered_hulls = []
        view_hulls = []
        for i in range(len(parts)):
            vertices = np.around(parts[i][0], decimals=2)
            hull = ConvexHull(vertices)
            volume = hull.volume
            print("Convex hull num " , i, " volume ", volume)
            if volume > 1:
                filtered_hulls.append(Delaunay(vertices))
                view_hulls.append(parts[i])

        if not filtered_hulls:
            print(f"[rotIdx {rotIdx}] 无符合体积阈值的凸包")
            continue

        # # 找到所有在凸包内的有效点
        # valid_idx = np.where(mask[rotIdx] == 1)
        # pos_points = coors[valid_idx]  # shape: (N, 2), 格子索引
        # heights = posZValid[rotIdx][valid_idx]
        # 找到所有点
        idx = np.where((mask[rotIdx] == 1) | (mask[rotIdx] == 0))
        pos_points = coors[idx]  # shape: (N, 2), 格子索引
        heights = posZValid[rotIdx][idx]
        #  新建该旋转下的候选点
        candidates = []
        for (x, y), z in zip(pos_points, heights):  # 注意：x 是列索引，y 是行索引
            point = np.array([x, y, z])  # 注意这里点的顺序
            for hull in filtered_hulls:
                if hull.find_simplex(point) >= 0:  # 点在 hull 内
                    v = mask[rotIdx][x, y]
                    # 按原版顺序 (rotIdx, x, y, height, valid)
                    point = [rotIdx, x, y, z, v]
                    candidates.append(point)
                    break
        if len(candidates)!= 0:
            candidates = np.array(candidates)
            ROT = candidates[:, 0].reshape(-1, 1)
            X   = candidates[:, 1].reshape(-1, 1)
            Y   = candidates[:, 2].reshape(-1, 1)
            H   = candidates[:, 3].reshape(-1, 1)
            V   = candidates[:, 4].reshape(-1, 1)
            candidates = np.concatenate([ROT, X, Y, H, V], axis=1)
            allCandidates.append(candidates)
        else:
            continue
    if len(allCandidates)!= 0:
        allCandidates = np.concatenate(allCandidates, axis=0)
        return allCandidates , view_hulls
    else:
        return None


def preview_heightmap_tin(heightmap):
    h, w = heightmap.shape
    adjusted_height = 5  # bin 高度
    container_height = np.full((h, w), adjusted_height, dtype=np.float32)
    free_space = container_height - heightmap
    # 在 y 轴方向镜像
    free_space = np.flip(free_space.T, axis=0)

    tin = Delatin(free_space, height=h, width=w, base_height=0.01)
    verts = tin.vertices
    faces = tin.triangles

    free_space_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    free_space_mesh.apply_scale([1, 1, -1])
    free_space_mesh.apply_translation([0, 0, adjusted_height])
    return free_space_mesh

from cvTools_copy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import coacd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# 函数：用 matplotlib 可视化所有凸包
def visualize_convex_parts_matplotlib(parts):
    """
    用 matplotlib 可视化凸分解结果，每个凸包随机上色
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # 保持xyz等比例

    for i, (vertices, faces) in enumerate(parts):
        # 为每个凸包随机颜色
        color = np.random.rand(3,)
        mesh = Poly3DCollection(vertices[faces], alpha=0.6, facecolor=color, edgecolor='k', linewidths=0.2)
        ax.add_collection3d(mesh)

    # 自动缩放到数据范围
    all_vertices = np.vstack([v for v, _ in parts])
    ax.auto_scale_xyz(all_vertices[:, 0], all_vertices[:, 1], all_vertices[:, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("CoACD 凸分解可视化 (matplotlib)")
    plt.show()

if __name__ == "__main__":
    heightResolution = 0.01
    posZValid = [[[10e3, 10e3, 10e3, 10e3, 10e3, 10e3, 10e3, 10e3, 10e3], 
                [4, 4, 4, 10e3, 0, 0, 0, 0, 10e3], 
                [4, 4, 4, 10e3, 0, 0, 0, 0, 10e3],
                [4, 4, 4, 10e3, 0, 0, 0, 0, 10e3],
                [10e3, 10e3, 10e3, 10e3, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 10e3, 10e3, 10e3, 10e3],
                [0, 0, 0, 0, 0, 10e3, 2, 2, 10e3]],
                [[10e3, 10e3, 10e3, 10e3, 10e3, 10e3, 10e3, 10e3, 10e3], 
                [4, 4, 4, 10e3, 0, 0, 0, 0, 10e3], 
                [4, 4, 4, 10e3, 0, 0, 0, 0, 10e3],
                [4, 4, 4, 10e3, 0, 0, 0, 0, 10e3],
                [10e3, 10e3, 10e3, 10e3, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 10e3, 10e3, 10e3, 10e3],
                [0, 0, 0, 0, 0, 10e3, 2, 2, 10e3]],
                [[10e3, 10e3, 10e3, 10e3, 10e3, 10e3, 10e3, 10e3, 10e3], 
                [4, 4, 4, 10e3, 0, 0, 0, 0, 10e3], 
                [4, 4, 4, 10e3, 0, 0, 0, 0, 10e3],
                [4, 4, 4, 10e3, 0, 0, 0, 0, 10e3],
                [10e3, 10e3, 10e3, 10e3, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 10e3, 10e3, 10e3, 10e3],
                [0, 0, 0, 0, 0, 10e3, 2, 2, 10e3]],
                [[10e3, 10e3, 10e3, 10e3, 10e3, 10e3, 10e3, 10e3, 10e3], 
                [4, 4, 4, 10e3, 0, 0, 0, 0, 10e3], 
                [4, 4, 4, 10e3, 0, 0, 0, 0, 10e3],
                [4, 4, 4, 10e3, 0, 0, 0, 0, 10e3],
                [10e3, 10e3, 10e3, 10e3, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 0, 0, 0, 10e3],
                [0, 0, 0, 0, 0, 10e3, 10e3, 10e3, 10e3],
                [0, 0, 0, 0, 0, 10e3, 2, 2, 10e3]]]
    
    mask = [[[0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [1, 1, 1, 0, 1, 1, 1, 1, 0], 
            [1, 1, 1, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 1, 1, 0]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [1, 1, 1, 0, 1, 1, 1, 1, 0], 
            [1, 1, 1, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 1, 1, 0]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [1, 1, 1, 0, 1, 1, 1, 1, 0], 
            [1, 1, 1, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 1, 1, 0]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [1, 1, 1, 0, 1, 1, 1, 1, 0], 
            [1, 1, 1, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 1, 1, 0]]]
    # posZValid = [[[4, 4, 4, 4, 0, 0, 0, 0, 0], 
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0], 
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0]],
    #             [[4, 4, 4, 4, 0, 0, 0, 0, 0], 
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0], 
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0]],
    #             [[4, 4, 4, 4, 0, 0, 0, 0, 0], 
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0], 
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0]],
    #             [[4, 4, 4, 4, 0, 0, 0, 0, 0], 
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0], 
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
    #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0]],]

    # mask = [[[1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1]],
    #         [[1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1]],
    #         [[1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1]],
    #         [[1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1]]]

    coors = [[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8]],
            [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8]],
            [[2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8]],
            [[3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8]],
            [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8]],
            [[5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8]],
            [[6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8]],
            [[7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8]],
            [[8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8]],
            [[9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8]]]

    
coors = np.array(coors)
mask = np.array(mask)
posZValid = np.array(posZValid)
output_old = getConvexHullActions(posZValid, mask,  heightResolution)
# output_new, view_hulls = getConvexHullActions_new(posZValid, mask, coors)
print("output_old:", output_old)
# print("output_new:", output_new[:93])
# visualize_convex_parts_matplotlib(view_hulls)
