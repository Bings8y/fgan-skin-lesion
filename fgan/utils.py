import copy
import random

import numpy as np
import torch


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EMA:
    """Exponential moving average of generator weights, used for sampling."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.num_updates = 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # Warm-up so early samples are not dominated by the random initialisation.
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p, 1 - decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def to_uint8(x):
    """[-1, 1] NCHW float -> NHWC uint8 numpy."""
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()
