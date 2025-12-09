# volume_rendering.py

import torch
import torch.nn.functional as F

def volume_rendering(rgb, sigma, z_vals):
    # rgb: [N_rays, N_samples, 3]
    # sigma: [N_rays, N_samples]
    # z_vals: [N_rays, N_samples]

    # 1. 각 샘플 간의 거리(delta) 계산
    # 마지막 샘플은 거리를 알 수 없으므로 무한대(1e10)로 설정
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[...,:1].shape)], -1)

    # [중요] rays_d가 정규화(normalized) 안 되어 있다면, 실제 거리로 스케일링 필요할 수 있음.
    # 하지만 보통 NeRF에서는 z_vals 차이만으로도 충분히 학습됨.

    # 2. Alpha(불투명도) 계산
    # Density(sigma)는 음수가 나올 수 없으므로 ReLU 통과
    # 공식: alpha = 1 - exp(-sigma * delta)
    alpha = 1.0 - torch.exp(-F.relu(sigma) * dists)

    # 3. Transmittance(투과율) 계산 - Exclusive Cumprod
    # T_i = (1-a_1) * (1-a_2) * ... * (1-a_{i-1})
    # 내 앞까지의 투명도를 다 곱한 것. (나는 포함 안 함!)
    
    # 3-1. (1-alpha) + epsilon(0 방지)
    ones = torch.ones((alpha.shape[0], 1), device=alpha.device)
    vis = torch.cat([ones, 1.0 - alpha + 1e-10], -1) 
    
    # 3-2. 누적 곱 (마지막 하나는 필요 없으니 버림)
    T = torch.cumprod(vis, -1)[..., :-1]

    # 4. 최종 가중치(Weights) 계산
    # w_i = T_i * alpha_i
    weights = T * alpha

    # 5. RGB 렌더링 (Weighted Sum)
    rgb_map = torch.sum(weights[...,None] * rgb, -2)

    # 6. Depth Map (디버깅용, 필요 시 사용)
    # depth_map = torch.sum(weights * z_vals, -1)



    return rgb_map