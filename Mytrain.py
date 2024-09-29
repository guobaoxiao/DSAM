import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling.CWDLoss import CriterionCWD
from torch.nn import functional as F
from torchvision.models.mobilenetv2 import InvertedResidual
# set seeds
torch.manual_seed(2024)
np.random.seed(2024)

def _upsample_like_1024(src):
    src = F.interpolate(src, size=(1024, 1024), mode='bilinear')
    return src


class NpzDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root))
        self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files]
        # this implementation is ugly but it works (and is also fast for feeding data to GPU)
        # if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        self.ori_imgs = np.vstack([d['imgs'] for d in self.npz_data])
        self.img_embeddings = np.vstack([d['img_embeddings'] for d in self.npz_data])
        self.boundary = np.vstack([d['boundary'] for d in self.npz_data])
        self.depth_embeddings = np.vstack([d['depth_embeddings'] for d in self.npz_data])
        print(f"img_embeddings.shape={self.img_embeddings.shape}, ori_gts.shape={self.ori_gts.shape}, "
              f"boundary.shape={self.boundary.shape}", f"depth_embeddings.shape={self.depth_embeddings.shape}")

    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        img = self.ori_imgs[index]
        boundary = self.boundary[index]
        depth_embed = self.depth_embeddings[index]
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(img).float(), torch.tensor(gt2D[None, :, :]).long(), torch.tensor(bboxes).float(),\
               torch.tensor(boundary[None, :, :]).long(), torch.tensor(depth_embed).float()

    # %% test dataset class and dataloader
npz_tr_path = 'data/vit_b/COD_train'
work_dir = './work_dir_cod'
task_name = 'DSAM'
# prepare SAM model
model_type = 'vit_b'
checkpoint = 'work_dir_cod/SAM/sam_vit_b_01ec64.pth'
device = 'cuda:0'
model_save_path = join(work_dir, task_name)
os.makedirs(model_save_path, exist_ok=True)
sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)

sam_model.train()
# Set up the optimizer, hyperparameter tuning will improve performance here
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
CWD_loss = CriterionCWD(norm_type='channel', divergence='kl', temperature=4.0)

num_epochs = 100
losses = []
best_loss = 1e10
train_dataset = NpzDataset(npz_tr_path)
mask_threshold = 0.0
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
for epoch in range(num_epochs+1):
    epoch_loss = 0
    # train
    for step, (image_embedding, img, gt2D, boxes, boundary, depth_embedding) in enumerate(tqdm(train_dataloader)):

        with torch.no_grad():
            box_np = boxes.numpy()
            sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
            box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            boundary = torch.as_tensor(boundary, dtype=torch.float, device=device)
            # boundary = boun_conv(boundary)
            image_embedding = torch.as_tensor(image_embedding, dtype=torch.float, device=device)
            depth_embedding = torch.as_tensor(depth_embedding, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)
            # get prompt embeddings
            sparse_embeddings_box, dense_embeddings_box = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None
            )

            resize_img_tensor = np.transpose(img, (0, 3, 1, 2)).to(device)
            input_image = _upsample_like_1024(resize_img_tensor)
            pvt_embedding = sam_model.pvt(input_image)[3]

        bc_embedding, pvt_64 = sam_model.BC(pvt_embedding)
        # bc_embedding shape:" 1, 256, 64, 64
        distill_loss = CWD_loss(bc_embedding, depth_embedding)
        hybrid_embedding = torch.cat([pvt_64, bc_embedding], dim=1)
        high_frequency = sam_model.DWT(hybrid_embedding)

        dense_embeddings, sparse_embeddings = sam_model.ME(dense_embeddings_box,
                                                           high_frequency, sparse_embeddings_box)

        # predicted masks
        mask_predictions, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        final_mask = sam_model.loop_finer(mask_predictions, depth_embedding, depth_embedding)

        mask_predictions = 0.1*final_mask + 0.9*mask_predictions

        loss = 0.9*seg_loss(mask_predictions, gt2D.to(device)) + 0.1*distill_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= step
    losses.append(epoch_loss)
    print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
    # save the latest model checkpoint
    if epoch >= 80 and epoch % 10 == 0:
        torch.save(sam_model.state_dict(), join(model_save_path, str(epoch) + 'sam_model.pth'))
    # save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_best.pth'))
# plot loss
plt.plot(losses)
plt.title('Dice + Cross Entropy Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.show() # comment this line if you are running on a server
plt.savefig(join(model_save_path, 'train_loss.png'))
plt.close()