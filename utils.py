import numpy as np
import struct
import open3d as o3d


def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])


def cvt_pose(src):
    R = qvec2rotmat(src.qvec)  
    t = src.tvec              
    pose = np.zeros((3, 4))
    pose[:3, :3] = R            
    pose[:3, 3] = t            

    return pose


def cvt_intrinsic(intrinsic):
    f_x, f_y, c_x, c_y = intrinsic.flatten()
    intrinsic = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

    return intrinsic


def save_pcd2ply_w_normals(path, points, normals):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for point, normal in zip(points, normals):
            f.write(f"{point[0]} {point[1]} {point[2]} {normal[0]} {normal[1]} {normal[2]} " + 
                    f"{int(point[3])} {int(point[4])} {int(point[5])}\n")


def save_pcd2ply_wo_normarls(path, points):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]} " + 
                    f"{int(point[3])} {int(point[4])} {int(point[5])}\n")
            

def save_pcd2ply_binary(path, pcds, normals):
    with open(path, "wb") as f:
        header = f"""ply
        format binary_little_endian 1.0
        element vertex {pcds.shape[0]}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        """
        f.write(header.encode('utf-8'))
        
        for point in pcds:
            f.write(struct.pack('fffBBB', 
                                float(point[0]), float(point[1]), float(point[2]), 
                                int(point[3]), int(point[4]), int(point[5])))   
            


def project_points_to_image(points, intrinsic, pose, h, w):
    image = np.zeros((h, w, 3), dtype=np.uint8)

    points_3d = points[:, :3]
    points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_camera = (pose @ points_homogeneous.T).T
    
    valid_points = points_camera[:, 2] > 0
    points_camera = points_camera[valid_points]
    colors = points[:, 3:][valid_points]
    
    points_projected = (intrinsic @ points_camera[:, :3].T).T
    points_2d = points_projected[:, :2] / points_projected[:, 2, None]
    points_2d = np.round(points_2d).astype(int)
    
    mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    points_2d = points_2d[mask]
    colors = colors[mask]
    
    for (u, v), color in zip(points_2d, colors):
        image[v, u] = color  

    return image


def compute_normal(pcds, k=5):
    normals = []
    for points in pcds:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))

        cur_normal = np.asarray(pcd.normals)
        normals.append(cur_normal)

    return normals
