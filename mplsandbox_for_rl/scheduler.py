import torch.optim as optim
import math

def invsqrt_scheduler(warmup_steps: int):
    def _invsqrt_lr(step):
        return math.sqrt(warmup_steps) / math.sqrt(max(warmup_steps, step))
    def _warmup_lr(step):
        return max(step / warmup_steps, 0.1)
    def _invsqrt_lr_with_warmup(step):
        return max(_warmup_lr(step) if step < warmup_steps else _invsqrt_lr(step), 1e-8)
    
    return _invsqrt_lr_with_warmup

def calculate_noam_lr(dmodel: int, step: int, factor=1.):
    return factor * dmodel ** -0.5 * step ** -0.5
    
# same as Transformer's scheduler
def noam_scheduler(dmodel: int, warmup_steps: int, factor=1.):
    maxlr = calculate_noam_lr(dmodel, warmup_steps, factor)
    
    def _warmup_lr(step):
        return max(step / warmup_steps, 0.1)
    def _decay_lr(step):
        actual_lr = calculate_noam_lr(dmodel, step, factor)
        return actual_lr / maxlr
    def _norm_scheduler(step):
        return _warmup_lr(step) if step < warmup_steps else _decay_lr(step)
    
    return _norm_scheduler

def warmup_scheduler(warmup_steps: int, min_factor=0.):
    def _warmup_lr(step):
        return min(max(step / warmup_steps, min_factor), 1.)
    return _warmup_lr