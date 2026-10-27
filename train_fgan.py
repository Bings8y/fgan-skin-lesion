"""Train the class-conditional f-GAN on the HAM10000 training split.

Example:
    python train_fgan.py --div pearson --iters 8000 --out runs/fgan_pearson
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from torchvision.utils import save_image

from fgan.data import NUM_CLASSES, class_counts, gpu_augment, load_cache, tempered_sampling_weights
from fgan.losses import get_divergence
from fgan.models import Critic, Generator
from fgan.utils import EMA, get_device, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/ham10000_128.npz")
    p.add_argument("--div", default="pearson", choices=["pearson", "js", "kl", "rkl"])
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=4e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--g_loss", default="nonsat", choices=["saddle", "nonsat"],
                   help="saddle: L_G = -E[f*(T)];  nonsat: L_G = -E[T] (Nowozin et al. Sec. 3.2)")
    p.add_argument("--drift", type=float, default=None,
                   help="weight of the penalty E[T(real)^2]; 0.25 with Pearson turns the critic into a chi^2 estimate against the mixture (P+Q)/2 (default: 0.25 for pearson, 0 otherwise)")
    p.add_argument("--r1", type=float, default=0.0, help="R1 gradient penalty weight on real images")
    p.add_argument("--r1_every", type=int, default=4, help="lazy R1: apply every k critic steps")
    p.add_argument("--pool", default="sum", choices=["mean", "sum"])
    p.add_argument("--mbstd", type=int, default=4, help="minibatch std-dev group size in the critic (0 = off)")
    p.add_argument("--extra_from", type=int, default=2,
                   help="add a 3x3 conv after critic block i for i >= extra_from (5 = none)")
    p.add_argument("--ms", type=float, default=1.0,
                   help="mode-seeking regulariser weight (Mao et al., 2019) against within-class collapse")
    p.add_argument("--n_critic", type=int, default=1)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--sample_power", type=float, default=0.5,
                   help="class c is sampled with probability ~ n_c ** power (1 = natural, 0 = uniform)")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--sample_every", type=int, default=1000)
    p.add_argument("--ckpt_every", type=int, default=2500)
    p.add_argument("--snapshots", type=int, nargs="*", default=[],
                   help="iterations at which to keep a frozen copy of the EMA generator")
    p.add_argument("--out", default="runs/fgan_pearson")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.drift is None:
        args.drift = 0.25 if args.div == "pearson" else 0.0
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    dev = get_device(args.device)
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)

    images, labels = load_cache(args.data, "train")
    print(f"train images: {len(images)}  counts: {class_counts(labels).tolist()}  device: {dev}")
    real_u8 = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(dev)
    real_y = torch.from_numpy(labels).to(dev)
    weights = torch.from_numpy(tempered_sampling_weights(labels, args.sample_power)).float().to(dev)

    div = get_divergence(args.div)
    G = Generator(NUM_CLASSES, args.z_dim).to(dev)
    D = Critic(NUM_CLASSES, args.pool, args.extra_from, args.mbstd).to(dev)
    ema = EMA(G, args.ema)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    ckpt_path = os.path.join(args.out, "checkpoint.pt")
    start = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]); ema.model.load_state_dict(ck["G_ema"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        start = ck["iter"]
        ema.num_updates = start
        print(f"resumed from iter {start}")

    # Fixed noise for the progress grid: one row per class, 8 samples each.
    fixed_z = torch.randn(NUM_CLASSES * 8, args.z_dim, device=dev)
    fixed_c = torch.arange(NUM_CLASSES, device=dev).repeat_interleave(8)

    log_path = os.path.join(args.out, "train_log.csv")
    log_f = open(log_path, "a" if start else "w", newline="")
    log = csv.writer(log_f)
    if not start:
        log.writerow(["iter", "loss_d", "loss_g", "bound", "t_real", "t_fake", "sec"])

    def real_batch():
        idx = torch.multinomial(weights, args.batch, replacement=True)
        x = real_u8[idx].float() / 127.5 - 1.0
        return gpu_augment(x), real_y[idx]

    t0 = time.time()
    acc = np.zeros(5)
    n_acc = 0
    for it in range(start, args.iters):
        G.train(); D.train()
        for _ in range(args.n_critic):
            x, c = real_batch()
            z = torch.randn(args.batch, args.z_dim, device=dev)
            with torch.no_grad():
                fake = G(z, c)
            # One critic pass over [real; fake] (the critic has no batch-dependent layers).
            v_real, v_fake = D(torch.cat([x, fake]), torch.cat([c, c])).chunk(2)
            loss_d = div.critic_loss(v_real, v_fake)
            if args.drift > 0:
                loss_d = loss_d + args.drift * (v_real ** 2).mean()
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            if args.r1 > 0 and it % args.r1_every == 0:
                # Lazy R1 (Mescheder et al., 2018): penalise ||grad_x T(x)||^2 on real data.
                xr = x.detach().requires_grad_(True)
                grad, = torch.autograd.grad(D(xr, c).sum(), xr, create_graph=True)
                r1 = 0.5 * args.r1 * args.r1_every * grad.pow(2).sum(dim=(1, 2, 3)).mean()
                r1.backward()
            opt_d.step()

        # Generator step: fake labels follow the same tempered class prior as the real batch.
        c = real_y[torch.multinomial(weights, args.batch, replacement=True)]
        if args.ms > 0:
            c = torch.cat([c[: args.batch // 2]] * 2)  # pairs (z1, c), (z2, c) with the same class
        z = torch.randn(args.batch, args.z_dim, device=dev)
        fake_g = G(z, c)
        v_fake_g = D(fake_g, c)
        loss_g = div.generator_loss(v_fake_g, args.g_loss)
        if args.ms > 0:
            # Mode seeking: penalise small image distance relative to latent distance for one class.
            f1, f2 = fake_g.chunk(2)
            z1, z2 = z.chunk(2)
            ratio = (f1 - f2).abs().mean(dim=(1, 2, 3)) / (z1 - z2).abs().mean(dim=1)
            loss_g = loss_g + args.ms / (ratio.mean() + 1e-5)
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        ema.update(G)

        acc += np.array([loss_d.item(), loss_g.item(), div.bound(v_real, v_fake).item(),
                         div.activation(v_real).mean().item(), div.activation(v_fake).mean().item()])
        n_acc += 1
        step = it + 1
        if step % args.log_every == 0:
            m = acc / n_acc
            log.writerow([step, *np.round(m, 5), round(time.time() - t0, 1)])
            log_f.flush()
            print(f"[{step:6d}] L_D {m[0]:+.4f}  L_G {m[1]:+.4f}  bound {m[2]:+.4f}  "
                  f"T_real {m[3]:+.3f}  T_fake {m[4]:+.3f}  {time.time() - t0:.0f}s", flush=True)
            acc[:] = 0
            n_acc = 0
        if step % args.sample_every == 0 or step == args.iters:
            with torch.no_grad():
                grid = ema.model(fixed_z, fixed_c)
            save_image(grid * 0.5 + 0.5, os.path.join(args.out, "samples", f"iter_{step:06d}.png"), nrow=8)
        if step % args.ckpt_every == 0 or step == args.iters:
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "G_ema": ema.model.state_dict(),
                        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "iter": step,
                        "args": vars(args)}, ckpt_path)
        if step in args.snapshots:
            torch.save({"G_ema": ema.model.state_dict(), "args": vars(args), "iter": step},
                       os.path.join(args.out, f"generator_ema_{step:06d}.pt"))
    log_f.close()
    torch.save({"G_ema": ema.model.state_dict(), "args": vars(args)}, os.path.join(args.out, "generator_ema.pt"))
    print("done")


if __name__ == "__main__":
    main()
