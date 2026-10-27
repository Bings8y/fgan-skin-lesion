"""Variational f-divergence objectives (Nowozin et al., 2016).

For a divergence D_f, the critic's raw output v is mapped through an output activation
g_f so that T = g_f(v) lies in dom(f*). Then

    D_f(P || Q) >= sup_T  E_P[T(x)] - E_Q[f*(T(x))]

The critic maximises the bound; the generator (Q) minimises it.
"""
import math

import torch
import torch.nn.functional as F

LOG2 = math.log(2.0)


class FDivergence:
    name = ""

    def activation(self, v):
        raise NotImplementedError

    def conjugate(self, t):
        raise NotImplementedError

    def conjugate_of_output(self, v):
        """f*(g_f(v)); overridden where composing the two naively is numerically unstable."""
        return self.conjugate(self.activation(v))

    def critic_loss(self, v_real, v_fake):
        # L_D = -E_P[T(x)] + E_Q[f*(T(G(z)))]
        return -self.activation(v_real).mean() + self.conjugate_of_output(v_fake).mean()

    def generator_loss(self, v_fake, mode="saddle"):
        if mode == "saddle":
            # L_G = -E_Q[f*(T(G(z)))], the saddle-point objective. Its gradient is weighted by
            # f*'(T), which for Pearson chi^2 is T/2 + 1 = p/q at the optimal critic.
            return -self.conjugate_of_output(v_fake).mean()
        # Non-saturating alternative (Nowozin et al., 2016, Sec. 3.2): L_G = -E_Q[g_f(V(G(z)))].
        return -self.activation(v_fake).mean()

    def bound(self, v_real, v_fake):
        """Monte-Carlo estimate of the variational lower bound on D_f(P_data || P_G)."""
        return -self.critic_loss(v_real, v_fake)


class PearsonChi2(FDivergence):
    """f(u) = (u - 1)^2, f*(t) = t^2 / 4 + t, dom(f*) = R, so g_f is the identity (no sigmoid)."""

    name = "pearson"

    def activation(self, v):
        return v

    def conjugate(self, t):
        return 0.25 * t * t + t


class JensenShannon(FDivergence):
    """f(u) = u log u - (u + 1) log((u + 1) / 2), f*(t) = -log(2 - e^t), g_f(v) = log 2 - log(1 + e^-v)."""

    name = "js"

    def activation(self, v):
        return LOG2 - F.softplus(-v)

    def conjugate(self, t):
        return -torch.log(2.0 - torch.exp(t))

    def conjugate_of_output(self, v):
        # 2 - e^T = 2 sigmoid(-v), so f*(g_f(v)) = softplus(v) - log 2 (no log(0) for large v)
        return F.softplus(v) - LOG2


class ForwardKL(FDivergence):
    """f(u) = u log u, f*(t) = exp(t - 1), g_f(v) = v."""

    name = "kl"

    def activation(self, v):
        return v

    def conjugate(self, t):
        return torch.exp(t - 1.0)


class ReverseKL(FDivergence):
    """f(u) = -log u, f*(t) = -1 - log(-t), g_f(v) = -exp(-v)."""

    name = "rkl"

    def activation(self, v):
        return -torch.exp(-v)

    def conjugate(self, t):
        return -1.0 - torch.log(-t)


DIVERGENCES = {d.name: d for d in (PearsonChi2(), JensenShannon(), ForwardKL(), ReverseKL())}


def get_divergence(name):
    return DIVERGENCES[name]
