"""장치 선택과 기록만 담당하는 제공 코드입니다."""
import torch

def choose_device():
    # CUDA 사용 가능 여부에 따른 장치 선택 및 실제 선택 결과 기록
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    info = {"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "selected_device": str(device)}
    if device.type == "cuda":
        # CPU 실행 시 GPU 전용 정보 조회 제외
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["vram_gib"] = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return device, info
