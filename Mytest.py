# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import argparse
import traceback
import sys
import torch.nn as nn
import shutil
from torchvision.models.mobilenetv2 import InvertedResidual
import os

torch.manual_seed(1)
np.random.seed(1)
# %% run inference
# set up the parser
parser = argparse.ArgumentParser(description='run inference on testing set')
parser.add_argument('-i', '--data_path', type=str, default='data/vit_b/COD_test', help='path to the data folder')
parser.add_argument('-o', '--seg_path_root', type=str, default='data/inference_npz/DSAM',
                    help='path to the segmentation folder')
parser.add_argument('--seg_png_path', type=str, default='data/inference_img/DSAM',
                    help='path to the segmentation folder')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--device', type=str, default='cuda:1', help='device')
parser.add_argument('-chk', '--checkpoint', type=str, default='work_dir_cod/DSAM/DSAM.pth',
                    help='path to the trained model')
args = parser.parse_args()


mDSC = []
num_of_mDSC = 0

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0, 0, 0, 0), lw=2))


def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def finetune_model_predict(img_np, box_np,  depth_np, boundary, sam_trans, sam_model_tune, device=args.device):
    H, W = img_np.shape[:2]
    # Original image processing
    resize_img = sam_trans.apply_image(img_np)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)  # (3, 1024, 1024)
    input_image = sam_model_tune.preprocess(resize_img_tensor[None, :, :, :])  # (1, 3, 1024, 1024)
    # Depth map processing
    resize_depth_img = sam_trans.apply_image(depth_np)
    resize_dep_tensor = torch.as_tensor(resize_depth_img.transpose(2, 0, 1)).to(device)
    depth_image = sam_model_tune.preprocess(resize_dep_tensor[None, :, :, :])  # (1, 3, 1024, 1024)

    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(input_image.to(device))  # (1, 256, 64, 64)
        depth_embedding = sam_model_tune.image_encoder(depth_image.to(device))  # (1, 256, 64, 64)

        # convert box to 1024x1024 grid
        box = sam_trans.apply_boxes(box_np, (H, W))
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)

        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        sparse_embeddings_box, dense_embeddings_box = sam_model_tune.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None
        )

        pvt_embedding = sam_model_tune.pvt(input_image)[3]
        bc_embedding, pvt_64 = sam_model_tune.BC(pvt_embedding)
        hybrid_embedding = torch.cat([pvt_64, bc_embedding], dim=1)
        high_frequency = sam_model_tune.DWT(hybrid_embedding)
        dense_embeddings, sparse_embeddings = sam_model_tune.ME(dense_embeddings_box,
                                                                high_frequency, sparse_embeddings_box)
        # predicted masks
        seg_prob, _ = sam_model_tune.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)  hybrid_embedding:(1, 512, 64, 64)
            image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),  #
            # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        final_mask = sam_model_tune.loop_finer(seg_prob, depth_embedding, depth_embedding)
        seg_prob = 0.1 * final_mask + 0.9 * seg_prob
        seg_prob = torch.sigmoid(seg_prob)
        # convert soft mask to hard mask
        seg_prob = seg_prob.cpu().numpy().squeeze()
        seg = (seg_prob > 0.5).astype(np.uint8)
    return seg

def delete_folder(folder_path):
    try:
        # 删除文件夹及其内容
        shutil.rmtree(folder_path)
        print(f"delete folder: {folder_path}")
    except Exception as e:
        print(f"fail to delete folder: {folder_path}: {e}")

def divide(x, y):
    try:
        result = x / y
        return result
    except ZeroDivisionError as e:
        # 处理除以零的情况
        delete_folder(args.seg_path_root)
        delete_folder(args.seg_png_path)
        print("division by zero:", e)


device = args.device
sam_model_tune = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)

sam_trans = ResizeLongestSide(sam_model_tune.image_encoder.img_size)

npz_folders = sorted(os.listdir(args.data_path))
os.makedirs(args.seg_png_path, exist_ok=True)
sam_dice_scores = []
for npz_folder in npz_folders:
    npz_data_path = join(args.data_path, npz_folder)
    save_path = join(args.seg_path_root, npz_folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        npz_files = sorted(os.listdir(npz_data_path))
        for npz_file in tqdm(npz_files):
            try:
                npz = np.load(join(npz_data_path, npz_file))
                print(npz_file)
                ori_imgs = npz['imgs']
                ori_gts = npz['gts']
                ori_number = npz['number']
                boundary = npz['boundary']
                dep_imgs = npz['depth_imgs']

                sam_segs = []
                sam_bboxes = []
                sam_dice_scores = []
                for img_id, ori_img in enumerate(ori_imgs):
                    # get bounding box from mask
                    gt2D = ori_gts[img_id]
                    bboundary = boundary[img_id]
                    depth_img = dep_imgs[img_id]

                    y_indices, x_indices = np.where(gt2D > 0)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    # add perturbation to bounding box coordinates
                    H, W = gt2D.shape
                    x_min = max(0, x_min - np.random.randint(0, 20))
                    x_max = min(W, x_max + np.random.randint(0, 20))
                    y_min = max(0, y_min - np.random.randint(0, 20))
                    y_max = min(H, y_max + np.random.randint(0, 20))
                    bbox = np.array([x_min, y_min, x_max, y_max])
                    seg_mask = finetune_model_predict(ori_img, bbox, depth_img, bboundary, sam_trans, sam_model_tune, device=device)
                    sam_segs.append(seg_mask)
                    sam_bboxes.append(bbox)
                    # these 2D dice scores are for debugging purpose.
                    # 3D dice scores should be computed for 3D images
                    sam_dice_scores.append(compute_dice(seg_mask > 0, gt2D > 0))

                # save npz, including sam_segs, sam_bboxes, sam_dice_scores
                np.savez_compressed(join(save_path, npz_file), medsam_segs=sam_segs, gts=ori_gts, number=ori_number, sam_bboxes=sam_bboxes)

                # visualize segmentation results
                img_id = np.random.randint(0, len(ori_imgs))
                # show ground truth and segmentation results in two subplots
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(ori_imgs[img_id])
                show_box(sam_bboxes[img_id], axes[0])
                show_mask(ori_gts[img_id], axes[0])
                axes[0].set_title('Ground Truth')
                axes[0].axis('off')

                axes[1].imshow(ori_imgs[img_id])
                show_box(sam_bboxes[img_id], axes[1])
                show_mask(sam_segs[img_id], axes[1])
                axes[1].set_title('DSAM: DSC={:.3f}'.format(sam_dice_scores[img_id]))
                axes[1].axis('off')
                # save figure
                fig.savefig(join(args.seg_png_path, npz_file.split('.npz')[0] + '.png'))
                # close figure
                plt.close(fig)
            except Exception:
                traceback.print_exc()
                print('error in {}'.format(npz_file))

    tmp_mDSC = divide(sum(sam_dice_scores), len(sam_dice_scores))
    mDSC.append(tmp_mDSC)

    print(str(npz_folder)+": " + str(tmp_mDSC))
# average number of
print("finial average mDSC: " + str((sum(mDSC)/len(mDSC))))