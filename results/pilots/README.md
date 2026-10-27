# Stabilisation pilots

Short runs used to choose the final f-GAN configuration. "subset" = 258 images decoded from the
first part of the archives while the download was in progress; "full" = the 8,011-image training split.
Unless noted: batch 64, Adam beta1 = 0, beta2 = 0.9, EMA 0.999.

| Run | Data | Divergence / G loss | Critic setup | LRs (D / G) | Outcome |
|---|---|---|---|---|---|
| p_A | subset | Pearson, saddle | mean pool | 1e-4 / 2e-4 | T(real) grew steadily to ~59 by iteration 350 |
| p_B | subset | Pearson, saddle | mean pool, R1 1.0 | 1e-4 / 2e-4 | L_D oscillated between -22 and +78 |
| p_C, q_1 | subset | Pearson, non-saturating | mean pool, R1 | 1e-4 / 2e-4 | critic output collapsed to ~0, generator ignored z |
| q_2 | subset | Pearson, non-saturating | mean pool, drift 0.01 | 1e-4 / 2e-4 | same collapse |
| r_D | subset | Pearson, non-saturating | sum pool, R1 0.1 | 4e-4 / 1e-4 | critic ~0 after spikes, per-class mode collapse |
| r_E | subset | Pearson, saddle | sum pool, R1 0.1 | 4e-4 / 1e-4 | T(real) 50-160, L_D up to +308 |
| f_P | full | Pearson, non-saturating | sum pool, R1 0.1 | 4e-4 / 1e-4 | stable but per-class mode collapse at 1,500 it |
| f_J | full | JS, non-saturating | sum pool, R1 0.1 | 4e-4 / 1e-4 | NaN at ~400 it (naive conjugate; fixed with softplus form) |
| f_X | full | Pearson, non-saturating | sum pool | 4e-4 / 1e-4 | per-class mode collapse |
| f_Y, f_Z | full | JS, non-saturating | + extra 3x3 convs (+ minibatch std) | 4e-4 / 1e-4 | mostly collapsed |
| f_W | full | JS, non-saturating, 2 critic steps | + extra convs + minibatch std | 2e-4 / 5e-5 | critic strong, z still ignored |
| attempt_main_v1 | full | Pearson, non-saturating, 2 critic steps | + R1 0.1 | 2e-4 / 5e-5 | T(real) 150-650, colour-saturated samples |
| **f_M** | full | **Pearson + 1/4 E_P[T^2] (chi^2 to mixture), non-saturating + mode seeking** | sum pool, extra convs, minibatch std | 4e-4 / 1e-4 | **bounded critic (T ~ +/-0.8), diverse lesion-like samples by 800 it; chosen** |
