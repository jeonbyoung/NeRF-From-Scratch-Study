import torch
import numpy as np
from get_rays import get_ray

def load_full_data(target='tiny_nerf_data.npz'):
    data = np.load(target)
    images = data['images']
    Rot = data['poses']
    focal = data['focal']

    images = torch.from_numpy(images).float()
    Rot = torch.from_numpy(Rot).float()
    focal = float(focal)

    img_Width, img_Height = images.shape[1:3]
    num_imgs = 100

    every_ray_o = []
    every_ray_d = []
    every_rgb = []

    for i in range(num_imgs):
        target_img = images[i]
        pose = Rot[i]

        ray_o, ray_d = get_ray(img_Height, img_Width, focal, pose)


        every_ray_o.append(ray_o.reshape(-1,3))
        every_ray_d.append(ray_d.reshape(-1,3))
        every_rgb.append(target_img.reshape(-1,3))

    merged_ray_o = torch.cat(every_ray_o, 0)
    merged_ray_d = torch.cat(every_ray_d, 0)
    merged_rgb = torch.cat(every_rgb, 0)

    return merged_ray_o, merged_ray_d, merged_rgb, focal, img_Height, img_Width