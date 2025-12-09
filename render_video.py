import numpy as np
import torch
import imageio
from tqdm import tqdm

from set_device import set_device
from get_samples import get_samples
from volume_rendering import volume_rendering
from Model import Customed_NeRF

# 카메라가 구(Sphere) 표면을 따라 돌도록 행렬을 만드는 함수
def get_spherical_pose(theta, phi, radius):
    trans_t = lambda t : np.array([
        [1,0,0,0], [0,1,0,-0.09], [0,0,1,t], [0,0,0,1],
    ])
    rot_phi = lambda phi : np.array([
        [1,0,0,0], [0,np.cos(phi),-np.sin(phi),0], [0,np.sin(phi), np.cos(phi),0], [0,0,0,1],
    ])
    rot_theta = lambda th : np.array([
        [np.cos(th),0,-np.sin(th),0], [0,1,0,0], [np.sin(th),0, np.cos(th),0], [0,0,0,1],
    ])
    
    # 카메라 좌표계 변환 (NeRF는 -Z 방향을 바라봄)
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w[:3,:4]




def render_video(model= None, save_path='result.mp4'):

    print("Video Rendering Started.")

    device = set_device()
    model.eval()

    height = 800
    width = 800

    # np.pi/4.5는 Blender Dataset(lego, chair, drum) 등에 대해서 공통적으로 사용하는 값이지만,
    # 나중에 직접 찍은 게 있다면, 그에 맞는 렌즈각?을 줘야한다.
    focal_length = 800/2/ np.tan(np.pi/4.5)

    render_poses = [get_spherical_pose(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]] 

    # 진짜 최종으로 이미지 딴 걸 넣을 곳이다.
    frames = []


    for c2w in tqdm(render_poses):
        # 1 ray 생성
        # 위에서 임의로 구한 저 원형으로 도는 궤적의 위치를 따는 get_spherical_pose로,
        # 이전에 Rot에 해당하는 것을 이번엔 c2w(camera to world)로 구해보자.
        pixel_x, pixel_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                       np.arange(height, dtype=np.float32),
                                       indexing='xy')
    
        dirs = np.stack([(pixel_x - width/2)/focal_length, -(pixel_y - width/2)/focal_length, -np.ones_like(pixel_x)],axis= -1)

        rays_d = dirs @ c2w[:3, :3].T
        rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)

        # 2 slicing and rendering
        # 800 by 800 짜리를 한 번에 모델에 넣었다가는 부담이 크다고 한다.
        # slicing해서 넣어주고, frames_pixels에서 모아놓고 관리하자.
        # 그거 말고는 그냥 그동안에 했던 거랑 유사하다.
        rays_o_flat = torch.from_numpy(rays_o).float().reshape(-1,3).to(device)
        rays_d_flat = torch.from_numpy(rays_d).float().reshape(-1,3).to(device)

        frame_pixels = []
        chunk_size = 1024

        for i in range(0,rays_o_flat.shape[0], chunk_size):
            batch_o = rays_o_flat[i:i+chunk_size]
            batch_d = rays_d_flat[i:i+chunk_size]

            pts, t_values = get_samples(batch_d, batch_o, num_of_samples=64)

            pts_flat = pts.reshape(-1,3)

            dirs_expanded = batch_d[:,None,:].expand_as(pts)
            dirs_flat = dirs_expanded.reshape(-1,3)

            raw_rgb, raw_sigma = model.forward(pts_flat, dirs_flat)


            rgb_for_vr = raw_rgb.reshape(batch_o.shape[0],64,3)
            sigma_for_vr = raw_sigma.reshape(batch_o.shape[0],64)

            rgb_chunk = volume_rendering(rgb_for_vr, sigma_for_vr, t_values)

            # 중간에 갑자기 왜 cpu로 돌리느냐
            # 메모리 절약 위함이라고 한다. gpu에 모든 정보를 다 올렸다가 터질 수 있으니, 이런 처리를 한다고 한다.
            frame_pixels.append(rgb_chunk.cpu())

        # 3 Synthesize
        # 이제 frame_pixels에 모아뒀던 것을 합칠 시간이다.
        final_img_flat = torch.cat(frame_pixels, dim=0)
        final_img = final_img_flat.reshape(height, width, 3)

        # 실제 rgb는 0~1의 실수가 아닌, 0~255의 정수로 구성된다.
        final_img = (np.clip(final_img.numpy(), 0, 1)* 255).astype(np.uint8)

        frames.append(final_img)

    print(f"Saving Video to {save_path}, just wait for the set up ended...")
    imageio.mimwrite(save_path, frames, fps=30, quality=8)
    print("🎈🎉🥳 Done! 🎈🎉🥳")

    model.train()