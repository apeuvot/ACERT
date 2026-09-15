import torch
import torch.distributed as dist


def gather_confusion_matrix(cm_np):
    """Sum the confusion matrices computed on each DDP rank."""
    if not dist.is_initialized():
        return cm_np
    cm_tensor = torch.tensor(cm_np, device='cuda')
    gathered = [torch.zeros_like(cm_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, cm_tensor)
    cm_sum = torch.stack(gathered).sum(dim=0)
    return cm_sum.cpu().numpy()
