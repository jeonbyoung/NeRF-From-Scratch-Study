import torch
import numpy as np
from get_rays import get_ray
from get_rgb import get_rgb

def load_full_data(num_imgs=100, target='lego'):
    # 학습 데이터로 쓸 것들을 일단 다 뽑아낼 거다.
    # 그리고 각 지점에 대해 페어만 맞춰준 상태에서 얘네를 다 섞을 거다.
    # 생각해보면, 굳이 이미지 단위로 학습시킬 이유가 없다. 
    # 어짜피 한 시작점에서 어느 방향으로 가는 지, 그 지점의 색이 뭔지도 다 알고 있는 상황이니.

    every_ray_o = []
    every_ray_d = []
    every_rgb = []

    for i in range(num_imgs):
        ray_o, ray_d, img_path = get_ray('train', target,i)
            

        # 중앙부 뽑기 시작.
        rgb = get_rgb(img_path)
        
        if len(ray_o.shape) == 2:
            H = int(np.sqrt(ray_o.shape[0]))
            W = H
            ray_o = ray_o.reshape(H,W,3)
            ray_d = ray_d.reshape(H,W,3)

        if len(rgb.shape)== 2:
            H = int(np.sqrt(rgb.shape[0]))
            rgb = rgb.reshape(H,H,-1)

        H, W = ray_o.shape[:2]
        crop_ratio = 0.6
        crop_h = int(H*crop_ratio)
        crop_w = int(W*crop_ratio)

        start_h = (H-crop_h)//2
        end_h = start_h + crop_h
        start_w = (W-crop_w)//2
        end_w = start_w + crop_w

        ray_o_cropped = ray_o[start_h:end_h, start_w:end_w]
        ray_d_cropped = ray_d[start_h:end_h, start_w:end_w]
        rgb_cropped = rgb[start_h:end_h, start_w:end_w, :]

        every_ray_o.append(ray_o_cropped.reshape(-1,3))
        every_ray_d.append(ray_d_cropped.reshape(-1,3))
        every_rgb.append(rgb_cropped.reshape(-1,3))

    merged_ray_o = torch.from_numpy(np.concatenate(every_ray_o, axis=0)).float()
    merged_ray_d = torch.from_numpy(np.concatenate(every_ray_d, axis=0)).float()
    merged_rgb = torch.from_numpy(np.concatenate(every_rgb, axis=0)).float()

    return merged_ray_o, merged_ray_d, merged_rgb