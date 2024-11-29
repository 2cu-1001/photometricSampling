import open3d as o3d
import numpy as np


class PointAligner:
    def numpy_to_pcd(self, points):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])  
            if points.shape[1] == 6: 
                pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)  

            return pcd


    def align_point_clouds(self, source_pcd, target_pcd):
        source_pcd = self.numpy_to_pcd(source_pcd)
        target_pcd = self.numpy_to_pcd(target_pcd)

        threshold = 25
        transformation_init = np.eye(4) 

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, transformation_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        source_pcd.transform(reg_p2p.transformation)

        aligned_points = np.hstack((
            np.asarray(source_pcd.points),  
            np.asarray(source_pcd.colors) * 255.0  
        ))
        return aligned_points
    

    def merge_point_clouds(self, pcd_list, voxel_size=0.05):
        pcd1 = self.numpy_to_pcd(pcd_list[0])
        pcd2 = self.numpy_to_pcd(pcd_list[1])

        merged_pcd = pcd1 + pcd2

        downsampled_pcd = merged_pcd.voxel_down_sample(voxel_size)

        merged_pcd = np.hstack((
            np.asarray(downsampled_pcd.points),  
            np.asarray(downsampled_pcd.colors) * 255
        ))

        return merged_pcd