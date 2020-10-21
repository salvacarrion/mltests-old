import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gpu_info():
    if torch.cuda.is_available():
        s = f"- Using GPU: {torch.cuda.is_available()}\n" \
            f"- No. devices: {torch.cuda.device_count()}\n" \
            f"- Device name (0): {torch.cuda.get_device_name(0)}"
    else:
        s = "- Using CPU"
    return s

