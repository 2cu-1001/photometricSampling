import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import struct
import open3d as o3d
import utils

class PcdGenerator:
    def __init__(self, imgs, depths, intrinsics, poses):
        self.imgs = imgs
        self.depths = depths
        self.intrinsics = intrinsics
        self.poses = poses

        
    def generate_for_single_img(self, image, depth, intrinsic, pose):
        h, w = depth.shape

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = torch.from_numpy(u.flatten()).to("cuda:0")
        v = torch.from_numpy(v.flatten()).to("cuda:0")
        intrinsic = torch.from_numpy(intrinsic).to("cuda:0")
        pose = (torch.from_numpy(pose).to("cuda:0")).float()
        
        z = depth.flatten()

        valid = z > 0
        valid = valid.detach().cpu().numpy()
        u = u[valid]
        v = v[valid]
        z = z[valid]

        r = pose[:3, :3]
        t = pose[:3, 3]
        r_inv = r.T
        t_inv = -torch.matmul(r_inv, t)

        x = (u.float() - intrinsic[0, 2]) * z / intrinsic[0, 0]
        y = (v.float() - intrinsic[1, 2]) * z / intrinsic[0, 0]
        points_cam = torch.stack([x, y, z], dim=1)
        points_world = torch.matmul(points_cam, r_inv.T) + t_inv

        colors = image[v.detach().cpu().numpy(), u.detach().cpu().numpy(), :]  

        if len(colors.shape) == 1:  
            colors = colors.reshape(1, -1)

        colors = colors*255
        pcd =  torch.cat([points_world, torch.from_numpy(colors).to("cuda:0")], dim=1)

        return pcd


    def generate(self):
        pcds = []
        point_sizes = []

        for img, depth, intrinsic, pose in zip(self.imgs, self.depths, self.intrinsics, self.poses):
            pcd = self.generate_for_single_img(img, depth, intrinsic, pose)
            pcds.append(pcd)

        pcds = [pcd.detach().cpu().numpy() for pcd in pcds]
        merged_pcd = np.vstack(pcds)
        point_sizes = [len(pcd) for pcd in pcds]
        normals_list = utils.compute_normal(pcds)

        return merged_pcd, pcds, point_sizes, normals_list     