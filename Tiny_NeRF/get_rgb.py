import cv2
import numpy as np

def get_rgb(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    # [수정] 채널 수에 따라 안전하게 변환
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0

    # RGBA to RGB (White Background)
    if img.shape[2] == 4:
        alpha = img[..., 3:]
        rgb = img[..., :3]
        # 배경을 흰색으로 합성
        img = rgb * alpha + (1.0 - alpha)
    
    return img