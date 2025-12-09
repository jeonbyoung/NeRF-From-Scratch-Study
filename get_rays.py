import json
import os
import numpy as np

def get_ray(category='train', target='lego', i=0):
    base_dir_path = 'nerf_synthetic/'+ target +'/'
    json_path = os.path.join(base_dir_path, 'transforms_'+category+'.json')

    with open(json_path,'r') as fp:
        meta = json.load(fp)

    theta = meta['camera_angle_x']
    img_file_path = meta['frames'][i]['file_path']
    img_file_path = base_dir_path + img_file_path[2:] + '.png'
    Rt = np.array(meta['frames'][i]['transform_matrix'])


    Rot,t = Rt[:-1,:-1], Rt[:-1,-1]

    img_Width = 800
    img_Height = 800

    focal_length = img_Width/2/np.tan(theta/2)

    #translation
    pixel_x, pixel_y = np.meshgrid(np.arange(img_Width, dtype=np.float32),
                                   np.arange(img_Height, dtype=np.float32),
                                   indexing='xy')
    
    beforeIn = np.stack([(pixel_x-img_Width/2)/focal_length,
                              -(pixel_y-img_Height/2)/focal_length,
                              -np.ones_like(pixel_x)] ,-1)
    
    # 수학적으로는 Rot @ beforeIn이 되어야 하나, 지금 beforeIn은 800*800*3짜리임.
    # 또한, [x y z]형태로 돼있는 beforeIn에 Rot을 그대로 곱하게 되면, 얼마나 틀어져있는 지에 대한 정보가 뭉개져서 그냥 사라져버림.
    # 그래서 각 [X | Y | Z]로 돼있는 R을 transpose하고, [x y z]를 곱하게 되면, [x*X | y*Y | z*Z]가 되어, 원하는 정보를 산출해낼 수 있게 된다.
    ray_direction = beforeIn @ Rot.T
    ray_origin = np.broadcast_to(t, ray_direction.shape)

    return ray_direction, ray_origin, img_file_path