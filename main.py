import PIL.Image
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


# def align_PCDs(pcds, fin_idx):
#     pointAlinger = PointAlinger()
#     aligned_pcd = pcds[fin_idx]
#     cnt = 0
#     for i in range(0, len(pcds)):
#         if i == fin_idx: continue
#         cur_pcd = pointAlinger.align_point_clouds(aligned_pcd, pcds[i])
#         aligned_pcd = pointAlinger.merge_point_clouds([cur_pcd, aligned_pcd], 0.1)

#         print(f"align pcd : {cnt + 1}/{len(pcds) - 1}")
#         cnt += 1

#     return aligned_pcd


def align_with_org_PCD(colmap_pcd, our_pcd, voxel_size = 0.1):
    pointAligner = PointAligner()
    our_pcd = pointAligner.align_point_clouds(our_pcd, colmap_pcd)
    # our_pcd = pointAlinger.numpy_to_pcd(our_pcd)
    # our_pcd = our_pcd.voxel_down_sample(voxel_size)
    # our_pcd = np.hstack((
    #     np.asarray(our_pcd.points),
    #     np.asarray(our_pcd.colors) * 255
    # ))

    return our_pcd


def generate_PCD(maske_imgs, masked_depths, intrinsics, poses):
    pcdGenerator = PcdGenerator(maske_imgs, masked_depths, intrinsics, poses)
    merged_pcd, pcds, point_numbers, normals_list = pcdGenerator.generate()

    # slice_size = point_numbers[0] + point_numbers[1] + point_numbers[2]
    # pcdGenerator.save_pcd2ply("./generatedPCD/pcd.ply", merged_pcd[:slice_size])
    for i in range(len(pcds)):
        utils.save_pcd2ply_w_normals(f"./generatedPCD/pcd{i + 1}.ply", pcds[i], normals_list[i])
        print(f"generate pcd : {i + 1}/{len(pcds)}")
    return merged_pcd, pcds, point_numbers, normals_list


def mask_img_depth(imgs, depths, masks):
    masked_imgs = []
    masked_depths = []
    img_H, img_W, _ = imgs[0].shape

    background = np.zeros((img_H, img_W), dtype=int)
    block_size = 21
    for i in range(0, img_H - block_size, block_size):
        for j in range(0, img_W -block_size, block_size):
            center_i = i + block_size // 2
            center_j = j + block_size // 2
            background[center_i, center_j] = 1

    for i in range(len(imgs)):
        cur_mask = masks[i]
        cur_mask = cur_mask > (cur_mask.mean() * 1.5)
        cur_mask = np.logical_or(cur_mask, background)

        cur_depth = torch.Tensor(depths[i]).to("cuda:0")
        cur_mask = torch.Tensor(cur_mask).to("cuda:0")
        cur_masked_depth = cur_depth * cur_mask

        cv2.imwrite("./maps/depth_map"+str(i)+".png", cur_depth.detach().cpu().numpy())
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
    for idx in range(1, len(img_list) + 1):
        intrinsic_list.append(utils.cvt_intrinsic(intrinsic_dict[1].params))

    pose_path = "./colmapInOutput/sparse/images.bin"
    pose_dict = read_write_model.read_images_binary(pose_path)
    pose_list = []
    for idx in range(1, len(img_list) + 1):
        pose_list.append(utils.cvt_pose(pose_dict[idx]))

    colmap_pcd_path = "./colmapInOutput/pcd/points3D.bin"
    colmap_pcd_dict = read_write_model.read_points3D_binary(colmap_pcd_path)
    colmap_pcd_list = []
    for point in colmap_pcd_dict.items():
        x, y, z = point[1].xyz
        r, g, b = point[1].rgb
        colmap_pcd_list.append([x, y, z, r, g, b])
    colmap_pcd = np.stack(colmap_pcd_list)

    return img_list, depth_list, intrinsic_list, pose_list, img_path_list, colmap_pcd


def main():
    imgs, depths, intrinsics, poses, img_path_list, colmap_pcd =  load_data()

    # depth_H ,depth_W = depths[0].shape
    # resize = torchvision.transforms.Resize(min(depth_H, depth_W))
    # imgs = [np.array(resize(PIL.Image.fromarray(img))) for img in imgs]
    imgs = [(img/255)[...,[2, 1, 0]] for img in imgs]

    # depths = [((depth - depth.min()) / (depth.max() - depth.min())) for depth in depths]
    
    masks = process_img(imgs)
    masked_imgs, masked_depths = mask_img_depth(imgs, depths, masks)

    merged_pcd, pcds, point_numbers, normals_list = generate_PCD(masked_imgs, masked_depths, intrinsics, poses)
    # pcd = align_PCD(pcds, point_numbers)

    # pcdEvaluator = PcdEvaluator(pcds)
    # pcdEvaluator.eval_pcd_score()
    # fin_pcd, fin_idx = pcdEvaluator.get_best_pcd()
    # fin_pcd = align_PCDs(pcds, fin_idx)
    # fin_pcd = postprocess_PCD(colmap_pcd, fin_pcd, voxel_size=0.1)

    pcds = [align_with_org_PCD(colmap_pcd, pcd) for pcd in pcds]

    pointAligner = PointAligner()
    fin_pcd = pcds[0]
    for i in range(1, len(pcds)):
        fin_pcd = pointAligner.merge_point_clouds([fin_pcd, pcds[i]])
    
    normals_list = utils.compute_normal(pcds)
    normals = np.vstack(normals_list)

    utils.save_pcd2ply_w_normals("./generatedPCD/points3D.ply", fin_pcd, normals)
    # save_pcd2ply_binary("./generatedPCD/pcd.ply", fin_pcd, normals)
    
    # print(f"final idx : {fin_idx}")
    # print(f"selected view : " + str(img_path_list[fin_idx]))
    print("done")


if __name__ == '__main__':
    main()