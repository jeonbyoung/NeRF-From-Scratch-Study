import torch
import os
import numpy as np

def get_ray(img_Height,img_Width,focal,c2w):
    device = c2w.device
    pixel_x, pixel_y = torch.meshgrid(torch.arange(img_Width, dtype=torch.float32, device = device),
                                   torch.arange(img_Height, dtype=torch.float32, device= device),
                                   indexing='xy')

    pre_ray_direction = torch.stack([(pixel_x-img_Width/2)/focal,
                              -(pixel_y-img_Height/2)/focal,
                              -torch.ones_like(pixel_x)] ,-1)


    ray_direction = torch.sum(pre_ray_direction[...,None,:]*c2w[:3,:3],-1)

    ray_origin = c2w[:3,-1].expand(ray_direction.shape)

    return ray_origin, ray_direction