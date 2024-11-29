import torch
import torchvision.transforms as T
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import copy
import torch
import torch.nn.functional as F


def compute_entropy_map(image, kernel_size=5, block_size=512):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert image to PyTorch tensor and move to GPU
    image_tensor = torch.tensor(image, dtype=torch.float32, device='cuda') / 255.0

    # Pad the image to handle borders
    pad_size = kernel_size // 2
    padded_image = F.pad(image_tensor.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
                         (pad_size, pad_size, pad_size, pad_size),
                         mode='reflect')

    # Initialize the output entropy map
    h, w = image.shape
    entropy_map = torch.zeros((h, w), device='cuda')

    # Process the image block by block
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Define block boundaries
            i_end = min(i + block_size, h)
            j_end = min(j + block_size, w)

            # Extract block from padded image
            block = padded_image[:, :, i:i_end + kernel_size - 1, j:j_end + kernel_size - 1]

            # Extract sliding windows within the block
            patches = F.unfold(block, kernel_size=kernel_size, stride=1)
            patches = patches.permute(0, 2, 1)  # Rearrange to (batch, patches, pixels)

            # Compute histogram for each patch
            hist_bins = 256
            histograms = torch.zeros((patches.size(1), hist_bins), device='cuda')
            for k in range(hist_bins):
                histograms[:, k] = (patches == k / 255.0).sum(dim=2).float()

            # Normalize histograms to probabilities
            # histograms = histograms / histograms.sum(dim=1, keepdim=True)

            # Compute entropy
            epsilon = 1e-9  # Small value to prevent division or log errors
            histograms = histograms / (histograms.sum(dim=1, keepdim=True) + epsilon)
            entropy = -torch.sum(histograms * torch.log2(histograms + epsilon), dim=1)

            # Reshape entropy values and assign to output
            block_entropy = entropy.view(i_end - i, j_end - j)
            entropy_map[i:i_end, j:j_end] = block_entropy[: i_end - i, : j_end - j]
            # entropy_map = torch.nan_to_num(entropy_map, nan=0.0, posinf=0.0, neginf=0.0)
    
    entropy_map = torch.nan_to_num(entropy_map, nan=0.0, posinf=0.0, neginf=0.0)
    entropy_map = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-9)

    # Move the result back to CPU and convert to NumPy array
    return entropy_map


class ImgProcessor:
    def __init__(self, org_img):
        self.org_img = org_img
        self.dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()


    def calc_entropy_map(self):
        img = (copy.deepcopy(self.org_img) * 255).astype("uint8")
        entropy_map = compute_entropy_map(img)    
        min_val = entropy_map[~torch.isnan(entropy_map)].min().item()
        max_val = entropy_map[~torch.isnan(entropy_map)].max().item()
        entropy_map = (entropy_map - min_val) / (max_val - min_val + 1e-9)

        return entropy_map
    

    def calc_edge_map(self):
        img = (copy.deepcopy(self.org_img) * 255).astype("uint8")
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edge_map = cv2.Canny(gray, 50, 200, 10)

        kernel = np.ones((2, 2), np.uint8)
        edge_map = cv2.dilate(edge_map, kernel, iterations=1)
        edge_map = torch.Tensor(edge_map).to('cuda')

        min_val = edge_map[~torch.isnan(edge_map)].min().item()
        max_val = edge_map[~torch.isnan(edge_map)].max().item()
        edge_map = (edge_map - min_val) / (max_val - min_val + 1e-9)

        return edge_map
    

    def merge_entropy_with_edge(self, entropy_map, edge_map, edge_factor=0.5):
        edge_map = edge_map * edge_factor
        merged_map = entropy_map
        img_H = edge_map.shape[0]
        img_W = edge_map.shape[1]
        edge_val = merged_map.min() + edge_factor * (merged_map.max() - merged_map.min())

        # for h in range(img_H):
        #     for w in range(img_W):
        #         if edge_map[h][w] != 0:
        #             merged_map[h][w] = max(edge_val, merged_map[h][w])
        merged_map = np.maximum(entropy_map.detach().cpu().numpy(), edge_map.detach().cpu().numpy())

        return merged_map


# path = "./cat.png" # 이미지 경로

# print(torch.cuda.is_available())
# # 전체 클래스에 대한 entropy 계산
# # 특정 클래스에 대한 entropy 계산
# # entropy = segment_and_entropy(dlab, path, classes=[0, 3, 5])

# print(entropy_map)

# # import torch
# # print(torch.cuda.is_available())

# print("done")

# imgc = cv2.imread(path)
# print(type(imgc))
# gray = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
# edge_map = cv2.Canny(gray, 50, 200, 10)

# # 커널 정의 (에지를 확장하는 요소)
# kernel = np.ones((2, 2), np.uint8)  # 커널 크기를 키우면 더 두꺼워짐
# edge_map = cv2.dilate(edge_map, kernel, iterations=1)

# plt.imshow(edge_map)
# plt.show()

# # print(edge_map.shape)
# # print("edge\n", edge_map)

# # print(entropy_map.shape)
# # print("entropy\n", entropy_map)

# edge_map = torch.Tensor(edge_map)
# print(edge_map.shape)
# print(edge_map)

# entropy_map = entropy_map.squeeze()
# print(entropy_map)

# fin_map = entropy_map

# img_H = edge_map.shape[0]
# img_W = edge_map.shape[1]
# edge_val = fin_map.min() + 0.85 * (fin_map.max() - fin_map.min())

# for h in range(img_H):
#     for w in range(img_W):
#         if edge_map[h][w] != 0:
#             fin_map[h][w] = max(edge_val, fin_map[h][w])

# print(fin_map)
# plt.imshow(fin_map.detach().cpu().numpy(), cmap=plt.cm.jet); plt.axis('off'); plt.show()
# plt.show()


# # Define the helper function
# def decode_segmap(self, image, nc=21):

#     label_colors = np.array([(0, 0, 0),  # 0=background
#             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
#             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
#             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
#             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
#             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
#             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
#             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
#             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

#     r = np.zeros_like(image).astype(np.uint8)
#     g = np.zeros_like(image).astype(np.uint8)
#     b = np.zeros_like(image).astype(np.uint8)

#     for l in range(0, nc):
#         idx = image == l
#         r[idx] = label_colors[l, 0]
#         g[idx] = label_colors[l, 1]
#         b[idx] = label_colors[l, 2]

#     rgb = np.stack([r, g, b], axis=2)
#     return rgb


# def segment_and_entropy(self, net, img, show_orig=False, classes=-1, dev='cuda'):
#     # img = Image.open(path).convert("RGB")
#     # if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
#     # Comment the Resize and CenterCrop for better inference results
#     trf = T.Compose([#T.Resize(640), 
#                 #T.CenterCrop(224), 
#                 T.ToTensor(), 
#                 T.Normalize(mean = [0.485, 0.456, 0.406], 
#                             std = [0.229, 0.224, 0.225])])
#     inp = trf(img).unsqueeze(0).to(dev)
#     out = net.to(dev)(inp)['out']
#     om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
#     # rgb = self.decode_segmap(om)
#     # plt.imshow(rgb); plt.axis('off'); plt.show()
    
#     out_soft = torch.softmax(out, dim=1)    
#     entropy_map = -out_soft * torch.log(out_soft + 1e-6)
        
#     if classes == -1:
#         entropy_map = torch.sum(entropy_map, dim=1)
#     else:
#         indices = torch.tensor(classes)
#         entropy_map = torch.index_select(entropy_map, 1, indices)
#         entropy_map = torch.sum(entropy_map, dim=1)
    
#     ## opencv version of jetmap entropy
#     # entropy_map = entropy_map.squeeze(0)
#     # entropy_map = entropy_map.detach().numpy()
#     # entropy_map = entropy_map / np.max(entropy_map) * 255
#     # entropy_map = entropy_map.astype(np.uint8)
#     # entropy_map = cv2.applyColorMap(entropy_map, cv2.COLORMAP_JET)
    
#     # plt.imshow(np.transpose(entropy_map.detach().cpu().numpy(), (1, 2, 0)), cmap=plt.cm.jet); plt.axis('off'); plt.show()
#     #entropy = np.sum(entropy_map)
    
#     return entropy_map