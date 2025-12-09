# get_samples.py

import torch

def get_samples(rays_d, rays_o, near=2.0, far=6.0, num_of_samples=128, mode='train'):
    # 1. 0에서 1 사이의 등간격 구간 생성
    # shape: [num_of_pts_per_ray]
    t_vals = torch.linspace(0., 1., steps=num_of_samples, device=rays_d.device)
    
    # 2. near ~ far 사이의 거리로 변환
    # z_vals: [num_of_pts_per_ray]
    z_vals = near * (1.-t_vals) + far * (t_vals)

    # 3. 레이 개수만큼 확장 (Broadcasting)
    # z_vals: [N_rays, num_of_pts_per_ray]
    z_vals = z_vals.expand([rays_d.shape[0], num_of_samples])

    # 4. [핵심] 학습(Train)일 때만 랜덤 노이즈 추가 (Stratified Sampling) ⭐
    # 이게 없으면 샘플링 위치가 고정되어 학습이 뭉개집니다.
    if mode == 'train':
        # 각 구간의 길이
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        # 각 구간 내에서 랜덤한 위치 뽑기
        t_rand = torch.rand(z_vals.shape, device=rays_d.device)
        z_vals = lower + (upper - lower) * t_rand

    # 5. 실제 3D 좌표(Points) 계산
    # P = O + t * D
    # rays_o: [N_rays, 1, 3] / rays_d: [N_rays, 1, 3]
    # pts: [N_rays, N_samples, 3]
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    
    return pts, z_vals