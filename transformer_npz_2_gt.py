import numpy as np
from skimage import io
import os
join = os.path.join


path = "data/inference_npz/DSAM/"
save_path = 'data/inference_img/DSAM/'
# data = np.load("data/demo2D_vit_b/CVC-ColonDB/CVC-ColonDB.npz")
npz_folders = sorted(os.listdir(path))
for npz_folder in npz_folders:
    # 加载npz文件
    data = np.load(join(path, npz_folder,npz_folder) + '.npz')
    # 获取图像数据
    imgs = data["medsam_segs"]  # 假设图像数据保存在名为"imgs"的数组中
    name = data['number']

    # 将图像数据转换为正确的数据类型和范围
    imgs = imgs.astype(np.uint8)
    imgs = (imgs * 255.0).astype(np.uint8)  # 根据图像数据的范围进行调整

    # 保存图像数据为图片文件
    for i, img in enumerate(imgs):
        num = 149 + i
        img_path = join(save_path, npz_folder) + "/" + name[i]  # 设置保存图片的路径和文件名
        if not os.path.exists(join(save_path, npz_folder)):
            os.makedirs(join(save_path, npz_folder), exist_ok=True)
        io.imsave(img_path, img)

    print("Images saved successfully.")