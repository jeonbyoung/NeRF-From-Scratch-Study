import torch

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 장치 설정: NVIDIA GPU ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 장치 설정: Apple Silicon (MacBook M1/M2/M3)")
    else:
        device = torch.device("cpu")
        print("🐢 장치 설정: CPU (속도가 느릴 수 있습니다)")
    
    return device