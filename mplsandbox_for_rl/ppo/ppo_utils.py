import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from accelerate import Accelerator

@torch.no_grad()
def get_global_statistics(accelerator: Accelerator, xs: torch.Tensor, mask=None, device='cpu') -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes
    """
    xs = xs.to(accelerator.device)
    sum_and_count = torch.tensor([xs.sum(), (xs.numel() if mask is None else mask.sum())], device=xs.device)
    sum_and_count = accelerator.reduce(sum_and_count)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
    sum_var = accelerator.reduce(sum_var)
    global_var = sum_var / count
    
    return global_mean.to(device), global_var.to(device), count.to(device)

def logprobs_from_logits(logits, labels):
    """Compute log softmax values from logits."""
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)

@torch.no_grad()
def whiten(xs: torch.Tensor, mask: torch.BoolTensor, shift_mean=True, accelerator: Accelerator=None) -> torch.Tensor:
    """Whitens values"""
    if accelerator != None and accelerator.use_distributed:
        mean, var, _ = get_global_statistics(accelerator, xs, mask=mask, device=accelerator.device)
    else:
        mean = xs.sum() / mask.sum()
        var = torch.sum(((xs - mean) ** 2).mul(mask)) / mask.sum()

    whitened = (xs - mean) * torch.rsqrt(var + 1e-6)
    if not shift_mean:
        whitened += mean
    return whitened

class RunningMoments:
    def __init__(self, accelerator: Accelerator):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/CarperAI/trlx/blob/a66a7da90d3b9d4b74cf968139896d6797a17286/trlx/utils/modeling.py#L281
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24
        self.accelerator = accelerator

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """Updates running moments from batch's moments computed across ranks"""
        if self.accelerator.use_distributed:
            xs_mean, xs_var, xs_count = get_global_statistics(self.accelerator, xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()