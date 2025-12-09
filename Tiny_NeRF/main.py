import torch
import gc

from set_device import set_device
from Model import Customed_NeRF
from train import train
from render_video import render_video

if __name__=="__main__":
    device = set_device()

    my_nerf = Customed_NeRF(pos_encd_dim = 6).to(device)
    my_optimizer = torch.optim.Adam(my_nerf.parameters(), lr=5e-3)

    real_path = "tiny_nerf_data.npz"

    train(my_nerf, my_optimizer, target=real_path)


    with torch.no_grad():
        print("renderign start")
        render_video(model=my_nerf, save_path='last.mp4')