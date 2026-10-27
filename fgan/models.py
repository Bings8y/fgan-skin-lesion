"""Conditional generator (CBN) and spectrally-normalised projection critic for f-GAN."""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class ConditionalBatchNorm2d(nn.Module):
    """CBN(x | c) = gamma_c * (x - mu) / sqrt(var + eps) + beta_c, with gamma/beta from the class embedding."""

    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Linear(embed_dim, num_features)
        self.beta = nn.Linear(embed_dim, num_features)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        self.gamma._keep_init = self.beta._keep_init = True

    def forward(self, x, e):
        out = self.bn(x)
        return self.gamma(e)[:, :, None, None] * out + self.beta(e)[:, :, None, None]


class GenBlock(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)
        self.cbn = ConditionalBatchNorm2d(out_ch, embed_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, e):
        return self.act(self.cbn(self.up(x), e))


class Generator(nn.Module):
    """G(z, c): z ~ N(0, I_128), c in {0..6} -> 128x128x3 image in [-1, 1].

    Embedding(c) is concatenated with z, projected to 512x4x4, then upsampled by five
    ConvTranspose2d stages (4 -> 8 -> 16 -> 32 -> 64 -> 128), each followed by CBN + ReLU.
    """

    def __init__(self, num_classes=7, z_dim=128, embed_dim=128, base_ch=512):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Linear(z_dim + embed_dim, base_ch * 4 * 4)
        self.fc_cbn = ConditionalBatchNorm2d(base_ch, embed_dim)
        chs = [base_ch, 512, 256, 128, 64, 32]
        self.blocks = nn.ModuleList(GenBlock(chs[i], chs[i + 1], embed_dim) for i in range(5))
        self.to_rgb = nn.Conv2d(chs[-1], 3, 3, 1, 1)
        self.base_ch = base_ch
        self.apply(init_weights)

    def forward(self, z, c):
        e = self.embed(c)
        h = self.fc(torch.cat([z, e], dim=1)).view(-1, self.base_ch, 4, 4)
        h = torch.relu(self.fc_cbn(h, e))
        for blk in self.blocks:
            h = blk(h, e)
        return torch.tanh(self.to_rgb(h))


class MinibatchStdDev(nn.Module):
    """Append the mean feature std-dev over groups of consecutive samples (Karras et al., 2018).

    Lets the critic see sample diversity, which counters within-class mode collapse. Groups are
    formed from consecutive samples so a concatenated [real; fake] batch is never mixed.
    """

    def __init__(self, group=4):
        super().__init__()
        self.group = group

    def forward(self, x):
        n, c, h, w = x.shape
        g = self.group if n % self.group == 0 else 1
        if g == 1:
            return torch.cat([x, torch.zeros(n, 1, h, w, device=x.device, dtype=x.dtype)], 1)
        y = x.view(n // g, g, c, h, w)
        y = (y.var(dim=1, unbiased=False) + 1e-8).sqrt().mean(dim=(1, 2, 3))  # one value per group
        y = y.repeat_interleave(g).view(n, 1, 1, 1).expand(n, 1, h, w)
        return torch.cat([x, y], 1)


class Critic(nn.Module):
    """Variational critic T(x, c) = w^T psi(x) + <psi(x), v_c> + b, with no output activation.

    Five spectrally-normalised 4x4 stride-2 conv blocks with LeakyReLU(0.2) reduce 128x128 to 4x4;
    blocks from `extra_from` on add a 3x3 conv, and a minibatch std-dev channel is appended at 8x8.
    Global (sum or mean) pooling gives psi(x) in R^512.
    """

    def __init__(self, num_classes=7, pool="sum", extra_from=2, mbstd=4):
        super().__init__()
        self.pool = pool
        chs = [3, 64, 128, 256, 512, 512]
        layers = []
        for i in range(5):
            in_ch = chs[i]
            if mbstd and i == 4:
                layers.append(MinibatchStdDev(mbstd))  # at 8x8, before the last downsampling
                in_ch += 1
            layers += [spectral_norm(nn.Conv2d(in_ch, chs[i + 1], 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
            if i >= extra_from:
                # optional 3x3 refinement conv at low resolution: more critic depth for little compute
                layers += [spectral_norm(nn.Conv2d(chs[i + 1], chs[i + 1], 3, 1, 1)), nn.LeakyReLU(0.2, inplace=True)]
        self.features = nn.Sequential(*layers)
        self.out = spectral_norm(nn.Linear(chs[-1], 1))
        self.embed = spectral_norm(nn.Embedding(num_classes, chs[-1]))

    def forward(self, x, c):
        h = self.features(x)
        psi = h.mean(dim=(2, 3)) if self.pool == "mean" else h.sum(dim=(2, 3))
        return self.out(psi).squeeze(1) + (psi * self.embed(c)).sum(dim=1)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and not getattr(m, "_keep_init", False):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
