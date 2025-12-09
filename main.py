import torch
import gc

from set_device import set_device
from Model import Customed_NeRF
from train import train
from render_video import render_video

if __name__=="__main__":
    device = set_device()

    my_nerf = Customed_NeRF().to(device)
    my_optimizer = torch.optim.Adam(my_nerf.parameters(), lr=5e-3)

    train(my_nerf, my_optimizer, target='lego')

    gc.collect()
    torch.mps.empty_cache()
    resume_path = 'NeRF_weights/NeRF_weights_72000.pth'
    checkpoint = torch.load(resume_path, map_location=device)
    my_nerf.load_state_dict(checkpoint)
    with torch.no_grad():
        print("renderign start")
        render_video(model=my_nerf, save_path='big_result.mp4')