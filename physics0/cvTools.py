import numpy as np
import matplotlib.pyplot as plt
from pydelatin import Delatin
from pydelatin.util import rescale_positions
import trimesh
import coacd
from scipy.spatial import ConvexHull, Delaunay
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def getConvexHullActions(posZValid, mask, coors):
    allCandidates = []
    for rotIdx in range(len(posZValid)):
        print(f"init posZValid : min={np.min(posZValid)}, max={np.max(posZValid)}")

        heightmap = posZValid[rotIdx].copy()
        # Adjust invalid values to box height
        heightmap[heightmap == 1000] = 5
        print(f"[rotIdx {rotIdx}] heightmap: min={np.min(heightmap)}, max={np.max(heightmap)}")

        # create free space mesh
        free_space_mesh = preview_heightmap_tin(heightmap)

        # CoACD convex decomposition
        mesh_coacd = coacd.Mesh(free_space_mesh.vertices, free_space_mesh.faces)
        parts = coacd.run_coacd(mesh_coacd, max_convex_hull=50, threshold=0.1)
        print(f"[rotIdx {rotIdx}] CoACD decomposition {len(parts)} convex hulls")

        # create Delaunay hulls
        filtered_hulls = []
        for i in range(len(parts)):
            vertices = np.around(parts[i][0], decimals=2)
            # 检查维度退化
            if np.linalg.matrix_rank(vertices - vertices[0]) < 3:
                print(f"[rotIdx {rotIdx}] Skipping degenerate convex hull {i} (less than 3D)")
                continue
            hull = ConvexHull(vertices)
            volume = hull.volume
            print("Convex hull num " , i, " volume ", volume)
            if volume > 1:
                filtered_hulls.append(Delaunay(vertices))

        if not filtered_hulls:
            print(f"[rotIdx {rotIdx}] No convex hull over the volume threshold")
            continue

        # # Find all valid points in the convex hull
        # valid_idx = np.where(mask[rotIdx] == 1)
        # pos_points = coors[valid_idx]  # shape: (N, 2)
        # heights = posZValid[rotIdx][valid_idx]

        # # Find all points in the convex hull
        # idx = np.where((mask[rotIdx] == 1) | (mask[rotIdx] == 0))
        # pos_points = coors[idx]  # shape: (N, 2)
        # heights = posZValid[rotIdx][idx]
        # valid_mask = mask[rotIdx][idx]
        # # Create a new candidate point under this rotation
        # candidates = []
        # for (x, y), z, v in zip(pos_points, heights, valid_mask):  
        #     point = np.array([x, y, z])  
        #     for hull in filtered_hulls:
        #         if hull.find_simplex(point) >= 0:  # point in the convex hull
        #             candidate = [rotIdx, x, y, z, v]
        #             candidates.append(candidate)
        #             break

        # Find all valid points in the convex hull
        idx = np.where((mask[rotIdx] == 1))
        pos_points = coors[idx]  # shape: (N, 2)
        heights = posZValid[rotIdx][idx]
        valid_mask = mask[rotIdx][idx]
        # Create a new candidate point under this rotation, Each convex hull has at most five candidates
        candidates = []
        hull_counters = [0 for _ in filtered_hulls]
        max_per_hull = 5
        for (x, y), z, v in zip(pos_points, heights, valid_mask):  
            point = np.array([x, y, z])  
            for i, hull in enumerate(filtered_hulls):
                if hull_counters[i] >= max_per_hull:
                    continue
                if hull.find_simplex(point) >= 0:
                    candidate = [rotIdx, x, y, z, v]
                    candidates.append(candidate)
                    hull_counters[i] += 1
                    break

        if len(candidates)!= 0:
            candidates = np.array(candidates)
            candidates = np.unique(candidates, axis=0)
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
        return allCandidates
    else:
        return None
    


def preview_heightmap_tin(heightmap):
    h, w = heightmap.shape
    adjusted_height = 5  # bin heights
    container_height = np.full((h, w), adjusted_height, dtype=np.float32)
    free_space = container_height - heightmap
    # Mirror on the y axis
    free_space = np.flip(free_space.T, axis=0)

    tin = Delatin(free_space, height=h, width=w, base_height=0.01)
    verts = tin.vertices
    faces = tin.triangles

    free_space_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    free_space_mesh.apply_scale([1, 1, -1])
    free_space_mesh.apply_translation([0, 0, adjusted_height])
    return free_space_mesh


# Visualizing all convex hulls with matplotlib
def visualize_convex_parts_matplotlib(parts):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Keep xyz proportional

    for i, (vertices, faces) in enumerate(parts):
        # Random colors for each convex hull
        color = np.random.rand(3,)
        mesh = Poly3DCollection(vertices[faces], alpha=0.6, facecolor=color, edgecolor='k', linewidths=0.2)
        ax.add_collection3d(mesh)

    # Automatically scale to data range
    all_vertices = np.vstack([v for v, _ in parts])
    ax.auto_scale_xyz(all_vertices[:, 0], all_vertices[:, 1], all_vertices[:, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("CoACD convex decomposition visualization (matplotlib)")
    plt.show()

# if __name__ == "__main__":
#     heightResolution = 0.01
#     posZValid = [[[1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3], 
#                 [4, 4, 4, 1e3, 0, 0, 0, 0, 1e3], 
#                 [4, 4, 4, 1e3, 0, 0, 0, 0, 1e3],
#                 [4, 4, 4, 1e3, 0, 0, 0, 0, 1e3],
#                 [1e3, 1e3, 1e3, 1e3, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e3],
#                 [0, 0, 0, 0, 0, 1e3, 2, 2, 1e3]],
#                 [[1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3], 
#                 [4, 4, 4, 1e3, 0, 0, 0, 0, 1e3], 
#                 [4, 4, 4, 1e3, 0, 0, 0, 0, 1e3],
#                 [4, 4, 4, 1e3, 0, 0, 0, 0, 1e3],
#                 [1e3, 1e3, 1e3, 1e3, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e3],
#                 [0, 0, 0, 0, 0, 1e3, 2, 2, 1e3]],
#                 [[1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3], 
#                 [4, 4, 4, 1e3, 0, 0, 0, 0, 1e3], 
#                 [4, 4, 4, 1e3, 0, 0, 0, 0, 1e3],
#                 [4, 4, 4, 1e3, 0, 0, 0, 0, 1e3],
#                 [1e3, 1e3, 1e3, 1e3, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e3],
#                 [0, 0, 0, 0, 0, 1e3, 2, 2, 1e3]],
#                 [[1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3], 
#                 [4, 4, 4, 1e3, 0, 0, 0, 0, 1e3], 
#                 [4, 4, 4, 1e3, 0, 0, 0, 0, 1e3],
#                 [4, 4, 4, 1e3, 0, 0, 0, 0, 1e3],
#                 [1e3, 1e3, 1e3, 1e3, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 1e3],
#                 [0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e3],
#                 [0, 0, 0, 0, 0, 1e3, 2, 2, 1e3]]]
    
#     mask = [[[0, 0, 0, 0, 0, 0, 0, 0, 0], 
#             [1, 1, 1, 0, 1, 1, 1, 1, 0], 
#             [1, 1, 1, 0, 1, 1, 1, 1, 0],
#             [1, 1, 1, 0, 1, 1, 1, 1, 0],
#             [0, 0, 0, 0, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 0, 0, 0, 0],
#             [1, 1, 1, 1, 1, 0, 1, 1, 0]],
#             [[0, 0, 0, 0, 0, 0, 0, 0, 0], 
#             [1, 1, 1, 0, 1, 1, 1, 1, 0], 
#             [1, 1, 1, 0, 1, 1, 1, 1, 0],
#             [1, 1, 1, 0, 1, 1, 1, 1, 0],
#             [0, 0, 0, 0, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 0, 0, 0, 0],
#             [1, 1, 1, 1, 1, 0, 1, 1, 0]],
#             [[0, 0, 0, 0, 0, 0, 0, 0, 0], 
#             [1, 1, 1, 0, 1, 1, 1, 1, 0], 
#             [1, 1, 1, 0, 1, 1, 1, 1, 0],
#             [1, 1, 1, 0, 1, 1, 1, 1, 0],
#             [0, 0, 0, 0, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 0, 0, 0, 0],
#             [1, 1, 1, 1, 1, 0, 1, 1, 0]],
#             [[0, 0, 0, 0, 0, 0, 0, 0, 0], 
#             [1, 1, 1, 0, 1, 1, 1, 1, 0], 
#             [1, 1, 1, 0, 1, 1, 1, 1, 0],
#             [1, 1, 1, 0, 1, 1, 1, 1, 0],
#             [0, 0, 0, 0, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 1, 1, 1, 0],
#             [1, 1, 1, 1, 1, 0, 0, 0, 0],
#             [1, 1, 1, 1, 1, 0, 1, 1, 0]]]
#     # posZValid = [[[4, 4, 4, 4, 0, 0, 0, 0, 0], 
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0], 
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0]],
#     #             [[4, 4, 4, 4, 0, 0, 0, 0, 0], 
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0], 
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0]],
#     #             [[4, 4, 4, 4, 0, 0, 0, 0, 0], 
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0], 
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0]],
#     #             [[4, 4, 4, 4, 0, 0, 0, 0, 0], 
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0], 
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
#     #             [4, 4, 4, 4, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     #             [0, 0, 0, 0, 0, 0, 0, 0, 0]],]

#     # mask = [[[1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1], 
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1]],
#     #         [[1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1], 
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1]],
#     #         [[1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1], 
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1]],
#     #         [[1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1], 
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#     #         [1, 1, 1, 1, 1, 1, 1, 1, 1]]]

#     coors = [[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8]],
#             [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8]],
#             [[2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8]],
#             [[3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8]],
#             [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8]],
#             [[5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8]],
#             [[6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8]],
#             [[7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8]],
#             [[8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8]],
#             [[9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8]]]

    
# coors = np.array(coors)
# mask = np.array(mask)
# posZValid = np.array(posZValid)
# output_new = getConvexHullActions(posZValid, mask, coors)
# print("output_new:", output_new[:80])
