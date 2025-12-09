import torch

def volume_rendering(rgb, sigma, info_for_dist):
    
    # volume rendering 식은 github paper review에 있다. 이를 참고하여 작성하면 된다.
    
    # 먼저 delta를 정의한다. ray 위에 있는 point들간의 거리를 나타내며,
    # [카메라] ---> (점1) ------- (점2) ------- (점3) ... (점64) ---> [우주 끝]
    #                 |__ d1 __|    |__ d2 __|            |__ ??? __|
    # 위와 같아지기에, 마지막 점은 거리가 꽤 커지게 된다.

    delta = info_for_dist[...,1:] - info_for_dist[...,:-1]
    # delta.device를 해주는 이유는 pytorch는 다른 공간에 있으면 연산을 못 한다. 이를 맞춰주려 한 것이다.
    infinity = torch.tensor([1e+10], device=delta.device).expand(delta[...,:1].shape)

    delta = torch.cat([delta,infinity], dim = -1)

    # 이제 빛을 받은 정도에 해당하는 alpha를 바로 구해준다.
    # density가 음수가 되는 것을 방지하기 위해 relu를 한 번 더 취한다.
    alpha = 1.0-torch.exp(-torch.nn.functional.relu(sigma)*delta)

    # 이번엔 Transmittance, 불투명도에 대한 값을 가져올 것이다.
    # 이 값은 앞에 얼마나 불투명한 얘들이 많은 지에 대한 값이다.
    # 그냥 보면 for문이 떠오른다. 근데 pytorch에서 좋은 게 있단다.
    # torch.cumprod라는 걸 써보자.

    # 위에서 alpha 구한 걸 갖다 쓰자. 어짜피 식은 같으니.
    vis = 1.0 - alpha + 1e-10 # 근데 혹시 0이 되면, 곱해도 뭐가 안 나오니, 아주 작은 값인 1e-10을 더한다.

    # 또한 약간의 트릭을 사용한다.
    # 지수의 덧셈은 원래 값의 곱셈으로 나타낼 수 있다. ex) e^(A+B) = e^A*e^B.
    # 이 점을 이용해서, cumprod를 쓴다. 누적 곱을 해주는 것이다.
    accum_prod = torch.cumprod(vis, dim=-1)

    # 근데 cumprod는 본인까지 곱하게 된다.
    # 우리가 정한 transmittance는 우리 직전까지 있는 값들에 대한 것이니, shifting시켜, 우리를 제외한 것을 만들어줘야한다.
    ones = torch.ones((alpha.shape[0],1), device=alpha.device)
    T = torch.cat([ones, accum_prod[...,:-1]], dim = -1)


    # 이제 그동안 구한 alpha, T, 그리고 이전에서 가져온 rgb를 통해 값을 통해 예측 값을 도출한다.
    # 근데 또 문제가 있다. alpha,T는 차원이 맞는데, rgb는 1차원이 더 붙는다. 채널 컬러값이다.
    # 그래서 alpha, T의 차원을 늘린다.
    weight = T*alpha
    pred_rgb = weight[...,None]*rgb
    
    # 이제 마지막이다.
    # 한 ray 위에 올라간 색들을 모두 더해서, 해당 ray에서 보일 색을 추출하자.
    pred_rgb = torch.sum(pred_rgb, dim=-2) # 1024*64(points on ray)*3이니, 64에 있는 값들을 다 더한다.

    # 현재 volume rendering 공식은, 배경을 검은색으로 인식하게 된다.
    # acc_map은 광선이 물체에 부딪힌 총량(0~1)을 나타낸다.
    # 1이면 물체에 꽉 막힌다는 것이다. 0이면 뻥 뚫려있다. 즉, 배경이나, 빈 공간이다.
    # 1.0 - acc_map으로 빈 공간을 1.0으로 채워주는 코드를 만들자.
    acc_map = torch.sum(weight, dim=-1)
    pred_rgb = pred_rgb + (1.0 - acc_map[...,None])


    return pred_rgb
