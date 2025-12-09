import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from get_rays import get_ray
from get_rgb import get_rgb
from get_samples import get_samples
from data_loader import load_full_data
from set_device import set_device
from volume_rendering import volume_rendering

def mse2pnsr(mse):
      return -10 * torch.log10(mse)

# 검증용 img 1장(test의 0번째 이미지로) 렌더링 함수
@torch.no_grad() # 학습 아니니, grad 하지 말라는 표시
def rendering_one_img_for_test(model, test_pose, true_img, focal, img_Height, img_Width, device,num_of_pts_per_ray):
      test_pose = test_pose.to(device)
      rays_o, rays_d = get_ray(img_Height, img_Width, focal, test_pose)

      flat_rays_o = rays_o.reshape(-1,3).float()
      flat_rays_d = rays_d.reshape(-1,3).float()

      
      chunk_size = 4096
      all_rgb = []

      for i in range(0, flat_rays_o.shape[0], chunk_size):
            batch_o = flat_rays_o[i : i+chunk_size]
            batch_d = flat_rays_d[i : i+chunk_size]
            
            
            pts, t_vals = get_samples(batch_d, batch_o, num_of_samples = num_of_pts_per_ray, mode='test')

            pts_flat = pts.reshape(-1,3)
            dirs_expanded = batch_d[:,None,:].expand_as(pts)
            dirs_flat = dirs_expanded.reshape(-1,3)

            raw_rgb, raw_sigma = model(pts_flat, dirs_flat)

            rgb_for_vr = raw_rgb.reshape(batch_o.shape[0],num_of_pts_per_ray,3)
            sigma_for_vr = raw_sigma.reshape(batch_o.shape[0],num_of_pts_per_ray)

            rgb_chunk = volume_rendering(rgb_for_vr, sigma_for_vr, t_vals)

            all_rgb.append(rgb_chunk.cpu())

      pred_img= torch.cat(all_rgb, dim=0).reshape(img_Height, img_Width, 3).numpy()

      pred_img = np.clip(pred_img, 0 ,1)

      return pred_img, true_img.cpu().numpy()



def train(model=None, optimizer=None, target = 'tiny_nerf_data.npz'):
      device = set_device()
      model = model.to(device)
      
      epoch = 50000
      scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1**(1/epoch))


      merged_ray_o, merged_ray_d, merged_rgb, focal, img_Height, img_Width = load_full_data(target)

      merged_ray_o = merged_ray_o.to(device)
      merged_ray_d = merged_ray_d.to(device)
      merged_rgb = merged_rgb.to(device)

      data = np.load(target)
      true_img = torch.from_numpy(data['images'][101]).float()
      Rot = torch.from_numpy(data['poses'][101]).float()

      num_of_rays = 1024
      num_of_pts_per_ray = 64

      x_epoch_history = []
      y_psnr_history = []


      pbar = tqdm(range(epoch),ncols=100)

      for i in pbar:
            idx = torch.randint(0, merged_ray_o.shape[0], (num_of_rays,))

            batch_o = merged_ray_o[idx]
            batch_d = merged_ray_d[idx]
            batch_rgb = merged_rgb[idx]

            pts, pts_dist_info = get_samples(batch_d, batch_o, mode='train', num_of_samples=num_of_pts_per_ray)

            direction_expanded = batch_d[:,None,:].expand_as(pts)

            # 그럼 방향 행렬과 pts는 현재, 1024 * 64 * 3(x,y,z) 형태이다. 저 앞에 1024 * 64의 점들을 합쳐서 dim = 2로 만들어줘야한다.
            pts_for_model = pts.reshape(-1,3)

            dir_for_model = direction_expanded.reshape(-1,3)

            # 이제 (pts_for_model, dir_for_model)이라는 페어가 만들어졌고, 이 값이 model의 input으로서 들어가진다.
            from_model_rgb, from_model_sigma = model.forward(pts_for_model,dir_for_model)

            rgb_for_vr = from_model_rgb.reshape(num_of_rays, num_of_pts_per_ray, 3) # 이때의 3은 rgb 값
            sigma_for_vr = from_model_sigma.reshape(num_of_rays,num_of_pts_per_ray) # density에 해당하는 sigma는 스칼라다.

            # 이제 volume rendering을 할 차례다.
            pred_rgb = volume_rendering(rgb_for_vr,sigma_for_vr,pts_dist_info)

            # loss도 구하자. using MSE
            loss = torch.mean((pred_rgb - batch_rgb)**2)
            
            if(i%10==0):
                  with torch.no_grad():
                        current_psnr = mse2pnsr(loss).item()
                  pbar.set_postfix({'Epoch': i,'PSNR': f'{current_psnr:.2f}'})

            # 이전 gradient값 초기화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i%500 == 0 and i>0:
                  psnr_val = mse2pnsr(loss).item()
                  x_epoch_history.append(i)
                  y_psnr_history.append(psnr_val)

            # 진짜와 비교 and 가중치 저장
            if i%2500 == 0 and i >0:
                  # PSNR과 진짜 이미지와의 비교를 통해, 직접 얼마나 성장했나 보기
                  psnr_val = mse2pnsr(loss).item()

                  model.eval()
                  with torch.no_grad():
                        pred_img, true_img_np = rendering_one_img_for_test(model, Rot, true_img, focal, img_Height, img_Width, device, num_of_pts_per_ray)
                  model.train()

                  combined_img = np.hstack((pred_img, true_img_np))

                  height, width, _ = combined_img.shape


                  save_dir = 'test_img'
                  if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                  plt.figure(figsize=(10,5))
                  ax = plt.axes([0,0,1,1])
                  plt.axis('off')

                  font_size = 12 if height > 200 else 8
                  plt.imshow(combined_img)
                  plt.text(width*0.02, height*0.8,
                           f"Epoch: {i}\nPSNR: {psnr_val:.2f} dB",
                           color = 'yellow', fontsize=font_size, fontweight='bold',
                           bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
                  
                  plt.text(width*0.02, height*0.02, 'Pred', color='white', fontweight='bold', fontsize=font_size)
                  plt.text(width*0.52, height*0.02, 'Truth', color='white', fontweight='bold', fontsize=font_size)

                  save_path = os.path.join(save_dir, f"test_{i}epoch.png")
                  plt.savefig(save_path, bbox_inches='tight', pad_inches= 0)
                  plt.close()


      plt.figure(figsize=(10,6))
      plt.plot(x_epoch_history,y_psnr_history, linestyle='-',color = 'blue', label='PSNR')

      plt.title('NeRF Training PSNR Curve')
      plt.xlabel('Epoch')
      plt.ylabel('PSNR (dB')
      plt.grid(True, linestyle='--',alpha=0.7)
      plt.legend()

      graph_save_path = 'psnr_graph.png'
      plt.savefig(graph_save_path)
      plt.close()

      print(f'graph saved')


        

