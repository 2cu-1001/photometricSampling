from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import os

import copy
from dust3r.imgProcessor import ImgProcessor
from dust3r.pcdGenerator import PcdGenerator

import matplotlib.pyplot as plt
import numpy as np


img_list = os.listdir("/home/lsw/dust3r/imgs")

# im = [im.split('/')[-1] for im in img_list]

if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    images = load_images(img_list, size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    confidences = scene.get_conf()
    depths = scene.get_depthmaps()
    intrinsics = scene.get_intrinsics()

    imgs_c = copy.deepcopy(imgs)
    mask_lst = []
    for img_idx in range(len(imgs_c)):
        img = imgs_c[img_idx]
        print(img.shape)
        imgProcessor = ImgProcessor(img)

        entropy_map = imgProcessor.calc_entropy_map()
        edge_map = imgProcessor.calc_edge_map()
        merged_map = imgProcessor.merge_entropy_with_edge(entropy_map=entropy_map, edge_map=edge_map)

        plt.imshow(merged_map.detach().cpu().numpy())
        plt.savefig("./fig/infomap"+str(img_idx)+".png")
        mask_lst.append(merged_map)
        # img_H = img.shape[0]
        # img_W = img.shape[1]

        # print(merged_map.max())
        # print(merged_map.min())
        # pl.imshow(merged_map.detach().cpu().numpy(), cmap=pl.cm.jet)
        # pl.savefig("fig"+str(img_idx//3)+".png")

        # for h in range(img_H):
        #     for w in range(img_W):
        #         if merged_map[h][w] < 0.2:
        #             pts3d[img_idx//3][h][w] = [-1, -1, -1]

    # visualize reconstruction
    # scene.show()

    # find 2D-2D matches between the two images
    from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

    masked_img = []
    masked_depth = []
    img_H, img_W, _ = imgs[0].shape
    background = np.zeros((img_H, img_W), dtype=int)
    block_size = 3
    for i in range(0, img_H - block_size, block_size):
        for j in range(0, img_W -block_size, block_size):
            center_i = i + block_size // 2
            center_j = j + block_size // 2
            background[center_i, center_j] = 1

    for i in range(len(imgs)):
        cur_mask = mask_lst[i]
        cur_mask = cur_mask > (cur_mask.mean() / 2)

        cur_conf = confidences[i]
        cur_conf = cur_conf > (cur_conf.mean())

        plt.imshow(cur_conf.detach().cpu().numpy())
        plt.savefig("./fig/conf"+str(i)+".png")

        cur_mask = np.logical_and(cur_mask.detach().cpu(), cur_conf.detach().cpu())
        cur_mask = np.logical_or(cur_mask.detach().cpu(), background)

        plt.imshow(cur_mask.detach().cpu().numpy())
        plt.savefig("./fig/mask"+str(i)+".png")

        cur_depth = depths[i]
        cur_mask = cur_mask.to("cuda:0")
        cur_masked_depth = cur_depth * cur_mask

        cur_mask = cur_mask.unsqueeze(0).repeat(3, 1, 1)
        cur_img = imgs[i]
        cur_masked_img = cur_img * cur_mask.detach().cpu().numpy().transpose(1, 2, 0)

        masked_img.append(cur_masked_img)
        masked_depth.append(cur_masked_depth)

    pcdGenerator = PcdGenerator(masked_img, masked_depth, intrinsics, poses)
    pcds = pcdGenerator.generate()
    pcdGenerator.save_pcd2ply("tmp.ply", pcds)

    pts2d_list, pts3d_list = [], []
    for i in range(2):
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    print(f'found {num_matches} matches')
    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # visualize a few matches
    import numpy as np
    from matplotlib import pyplot as pl
    n_viz = 10
    match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)
