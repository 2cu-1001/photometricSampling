from PIL import Image
from imgProcessor import ImgProcessor
from pcdGenerator import PcdGenerator
from pointAligner import PointAligner
from pcdEvaluator import PcdEvaluator
import read_write_dense
import read_write_model
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import utils
import torch
import torchvision
import PIL
import struct
import open3d as o3d


def generate_PCD(maske_imgs, masked_depths, intrinsics, poses):
    pcdGenerator = PcdGenerator(maske_imgs, masked_depths, intrinsics, poses)
    merged_pcd, pcds, point_numbers, normals_list = pcdGenerator.generate()

    for i in range(len(pcds)):
        utils.save_pcd2ply_w_normals(f"./generatedPCD/pcd{i + 1}.ply", pcds[i], normals_list[i])
        print(f"generate pcd : {i + 1}/{len(pcds)}")
    return merged_pcd, pcds, point_numbers, normals_list


def mask_img_depth(imgs, depths, masks):
    masked_imgs = []
    masked_depths = []
    backgrounds = []

    for img in imgs:
        img_H, img_W, _ = img.shape
        background = np.zeros((img_H, img_W), dtype=int)
        block_size = 21
        for i in range(0, img_H - block_size, block_size):
            for j in range(0, img_W -block_size, block_size):
                center_i = i + block_size // 2
                center_j = j + block_size // 2
                background[center_i, center_j] = 1
        backgrounds.append(background)

    for i in range(len(imgs)):
        cur_mask = masks[i]
        cur_mask = cur_mask > (cur_mask.mean())
        cur_mask = np.logical_or(cur_mask, backgrounds[i])

        cur_depth = torch.Tensor(depths[i]).to("cuda:0")
        cur_mask = torch.Tensor(cur_mask).to("cuda:0")
        cur_masked_depth = cur_depth * cur_mask

        cv2.imwrite("./maps/depth_map"+str(i)+".png", (cur_depth*255).detach().cpu().numpy())
        cv2.imwrite("./maps/map"+str(i)+".png", (cur_mask*255).detach().cpu().numpy())

        cur_mask = cur_mask.unsqueeze(0).repeat(3, 1, 1)
        cur_img = imgs[i]
        cur_masked_img = cur_img * cur_mask.detach().cpu().numpy().transpose(1, 2, 0)

        masked_imgs.append(cur_masked_img)
        masked_depths.append(cur_masked_depth)

        cv2.imwrite("./maps/masked_image"+str(i)+".png", (cur_masked_img*255)[...,[2, 1, 0]])
        cv2.imwrite("./maps/masked_depth"+str(i)+".png", (cur_masked_depth*255).detach().cpu().numpy())

        print(f"masking : {i + 1}/{len(imgs)}")

    return masked_imgs, masked_depths


def process_img(imgs):
    mask_list = []
    cnt = 0
    for img in imgs:
        imgProcessor = ImgProcessor(img)

        entropy_map = imgProcessor.calc_entropy_map()
        edge_map = imgProcessor.calc_edge_map()
        merged_map = imgProcessor.merge_entropy_with_edge(entropy_map=entropy_map, edge_map=edge_map, edge_factor=0.6)

        cv2.imwrite("./maps/edge_map"+str(cnt)+".png", (edge_map*255).detach().cpu().numpy())
        cv2.imwrite("./maps/entropy_map"+str(cnt)+".png", (entropy_map*255).detach().cpu().numpy())

        mask_list.append(merged_map)
        cnt += 1
        print(f"img processing : {cnt}/{len(imgs)}")

    return mask_list


def load_data():
    img_path = "./colmapInOutput/images/*"
    img_path_list = glob.glob(img_path)
    img_path_list.sort()
    img_list = [cv2.imread(cur_img_path, cv2.IMREAD_COLOR) for cur_img_path in img_path_list]

    depth_path = "./colmapInOutput/stereo/depth_maps/*"
    depth_path_list = glob.glob(depth_path)
    depth_path_list = [depth_path for depth_path in depth_path_list if "geometric" in depth_path]
    # depth_path_list = [depth_path for depth_path in depth_path_list if "photometric" in depth_path]
    depth_path_list.sort()
    depth_list = [read_write_dense.read_array(depth_path) for depth_path in depth_path_list]

    intrinsic_path = "./colmapInOutput/sparse/cameras.bin"
    intrinsic_dict = read_write_model.read_cameras_binary(intrinsic_path)
    intrinsic_list = []
    if len(intrinsic_dict) == 1:
        for idx in range(1, len(img_list) + 1):
            intrinsic_list.append(utils.cvt_intrinsic(intrinsic_dict[1].params))
    else:
        for idx in range(len(img_list)):
            intrinsic_list.append(utils.cvt_intrinsic(intrinsic_dict[idx + 1].params))

    pose_path = "./colmapInOutput/sparse/images.bin"
    pose_dict = read_write_model.read_images_binary(pose_path)
    pose_list = []
    for idx in range(1, len(img_list) + 1):
        pose_list.append(utils.cvt_pose(pose_dict[idx]))

    # colmap_pcd_path = "./colmapInOutput/pcd/points3D.bin"
    # colmap_pcd_dict = read_write_model.read_points3D_binary(colmap_pcd_path)
    # colmap_pcd_list = []
    # for point in colmap_pcd_dict.items():
    #     x, y, z = point[1].xyz
    #     r, g, b = point[1].rgb
    #     colmap_pcd_list.append([x, y, z, r, g, b])
    # colmap_pcd = np.stack(colmap_pcd_list)

    return img_list, depth_list, intrinsic_list, pose_list, img_path_list


def main():
    imgs, depths, intrinsics, poses, img_path_list =  load_data()

    imgs = [(img/255)[...,[2, 1, 0]] for img in imgs]
    
    masks = process_img(imgs)
    masked_imgs, masked_depths = mask_img_depth(imgs, depths, masks)

    merged_pcd, pcds, point_numbers, normals_list = generate_PCD(masked_imgs, masked_depths, intrinsics, poses)

    pointAligner = PointAligner()
    fin_pcd = pcds[0]
    for i in range(1, len(pcds)):
        fin_pcd = pointAligner.merge_point_clouds([fin_pcd, pcds[i]])
    print(f"point count : {len(fin_pcd)}")

    normals_list = utils.compute_normal(pcds)
    normals = np.vstack(normals_list)

    utils.save_pcd2ply_w_normals("./generatedPCD/points3D.ply", fin_pcd, normals)

    print("done")


if __name__ == '__main__':
    main()
