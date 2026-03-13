"""Mamba-3 SSD layer.

h_t = dA * h_{t-1} + gamma * B_t x x_t + beta * B_{t-1} x x_{t-1}
phi_t = -sum_{i<=t} dt_i * theta_i  (data-dependent RoPE on B/C)
BX[p,n] = sum_r B[n,r] * x[p,r]    (MIMO rank contraction, state is rank-free)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssd import chunked_ssd, SSDOutput


class RMSNorm(nn.Module):
    __slots__ = ('eps', 'dim')

    def __init__(self, d: int, eps: float = 1e-5, dim: int = -1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.dim
        rms = torch.rsqrt(
            x.float().pow(2).mean(d, keepdim=True) + self.eps
        ).to(x.dtype)
        shape = [1] * x.ndim
        shape[d] = -1
        return x * rms * self.weight.view(*shape)


def _apply_rope(tensor, cos, sin, N, R):
    leading = tensor.shape[:-2]
    rp = tensor.reshape(*leading, N // 2, 2, R)
    rotated = torch.cat([
        rp[..., 0:1, :] * cos - rp[..., 1:2, :] * sin,
        rp[..., 0:1, :] * sin + rp[..., 1:2, :] * cos,
    ], dim=-2).reshape(*leading, N, R)
    return rotated


class KulaMamba(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int | None = None,
        mimo_rank: int = 2,
        chunk_size: int = 64,
        bias: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        use_trapezoidal: bool = True,
        use_complex_ssm: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.mimo_rank = max(mimo_rank, 1)
        self.chunk_size = chunk_size
        self.use_trapezoidal = use_trapezoidal
        self.use_complex_ssm = use_complex_ssm

        self.d_inner = expand * d_model
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim
        assert d_state % 2 == 0, "d_state must be even (RoPE pairing)"

        H, P, N, R = self.nheads, headdim, d_state, self.mimo_rank

        self.ngroups = ngroups if ngroups is not None else 1
        G = self.ngroups
        assert H % G == 0
        self.heads_per_group = H // G

        self._d_bc = G * N * R
        self._d_x = self.d_inner
        d_in_proj = self.d_inner + self._d_x + 2 * self._d_bc + H
        if use_trapezoidal:
            d_in_proj += H
        if use_complex_ssm:
            d_in_proj += N // 2
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias)

        self.A_log = nn.Parameter(torch.zeros(H))
        self.D = nn.Parameter(torch.ones(H))
        self.dt_bias = nn.Parameter(torch.zeros(H))

        # QK-Norm: joint N*R normalization replaces post-SSD norm
        self.B_norm = RMSNorm(N * R)
        self.C_norm = RMSNorm(N * R)
        self.B_bias = nn.Parameter(torch.ones(G, N, R))
        self.C_bias = nn.Parameter(torch.ones(G, N, R))

        self.mimo_x = nn.Parameter(torch.ones(H, P, R))
        self.mimo_z = nn.Parameter(torch.ones(H, P, R))
        self.mimo_down = nn.Parameter(torch.full((H, P, R), 1.0 / R))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self._init_weights(dt_min, dt_max)

    def _init_weights(self, dt_min: float, dt_max: float):
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(self.nheads)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            nn.init.uniform_(self.A_log, -4.0, -1.0)

    def _split_proj(self, proj):
        H, N = self.nheads, self.d_state
        sizes = [self.d_inner, self._d_x, self._d_bc, self._d_bc, H]
        if self.use_trapezoidal:
            sizes.append(H)
        if self.use_complex_ssm:
            sizes.append(N // 2)

        parts = proj.split(sizes, dim=-1)
        z, x, B, C, dt_raw = parts[0], parts[1], parts[2], parts[3], parts[4]
        idx = 5
        lam_raw = parts[idx] if self.use_trapezoidal else None
        if self.use_trapezoidal:
            idx += 1
        theta_raw = parts[idx] if self.use_complex_ssm else None
        return z, x, B, C, dt_raw, lam_raw, theta_raw

    def _expand_groups(self, t):
        if self.heads_per_group == 1:
            return t
        return t.repeat_interleave(self.heads_per_group, dim=-3)

    def _process_BC_train(self, B_flat, C_flat, dt, theta_raw):
        Bat, T = dt.shape[:2]
        H, G, N, R = self.nheads, self.ngroups, self.d_state, self.mimo_rank

        B_r = self.B_norm(B_flat.reshape(Bat, T, G, N * R)).reshape(Bat, T, G, N, R)
        C_r = self.C_norm(C_flat.reshape(Bat, T, G, N * R)).reshape(Bat, T, G, N, R)
        B_r = B_r + self.B_bias[None, None]
        C_r = C_r + self.C_bias[None, None]

        B_r = self._expand_groups(B_r)
        C_r = self._expand_groups(C_r)

        cum_angles = None
        if self.use_complex_ssm and theta_raw is not None:
            angle_inc = dt.unsqueeze(-1) * theta_raw.unsqueeze(-2)
            cum_angles = -angle_inc.cumsum(dim=1)
            c = torch.cos(cum_angles).unsqueeze(-1).unsqueeze(-1)
            s = torch.sin(cum_angles).unsqueeze(-1).unsqueeze(-1)
            B_r = _apply_rope(B_r, c, s, N, R)
            C_r = _apply_rope(C_r, c, s, N, R)

        return B_r, C_r, cum_angles

    def _process_BC_step(self, B_flat, C_flat, dt_t, theta_raw, angle_state):
        Bat = B_flat.shape[0]
        H, G, N, R = self.nheads, self.ngroups, self.d_state, self.mimo_rank

        B_r = self.B_norm(B_flat.reshape(Bat, G, N * R)).reshape(Bat, G, N, R)
        C_r = self.C_norm(C_flat.reshape(Bat, G, N * R)).reshape(Bat, G, N, R)
        B_r = B_r + self.B_bias[None]
        C_r = C_r + self.C_bias[None]

        B_r = self._expand_groups(B_r)
        C_r = self._expand_groups(C_r)

        if self.use_complex_ssm and theta_raw is not None:
            angle_inc = dt_t.unsqueeze(-1) * theta_raw.unsqueeze(-2)
            angle_state = angle_state - angle_inc
            c = torch.cos(angle_state).unsqueeze(-1).unsqueeze(-1)
            s = torch.sin(angle_state).unsqueeze(-1).unsqueeze(-1)
            B_r = _apply_rope(B_r, c, s, N, R)
            C_r = _apply_rope(C_r, c, s, N, R)

        return B_r, C_r, angle_state

    def _forward_impl(self, x, dt_scale=None, return_state=False):
        Bat, T, _ = x.shape
        assert T > 0, f"Sequence length must be positive, got {T}"
        H, P, N, R = self.nheads, self.headdim, self.d_state, self.mimo_rank

        proj = self.in_proj(x)
        z, x_flat, B_flat, C_flat, dt_raw, lam_raw, theta_raw = self._split_proj(proj)

        dt = F.softplus(dt_raw + self.dt_bias)
        if dt_scale is not None:
            dt = dt * dt_scale
        A = -torch.exp(self.A_log)
        dA = torch.exp(A[None, None, :] * dt)

        B_proc, C_proc, cum_angles = self._process_BC_train(
            B_flat, C_flat, dt, theta_raw,
        )

        x_r = x_flat.reshape(Bat, T, H, P, 1) * self.mimo_x[None, None]

        if self.use_trapezoidal and lam_raw is not None:
            lam = torch.sigmoid(lam_raw)
            scale_g = lam.unsqueeze(-1).unsqueeze(-1)
            scale_b = ((1.0 - lam) * dA).unsqueeze(-1).unsqueeze(-1)

            x_gamma = x_r * scale_g
            x_prev = F.pad(x_r[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
            B_prev = F.pad(B_proc[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
            x_beta = x_prev * scale_b
        else:
            x_gamma = x_r
            x_beta = None
            B_prev = None

        ssd_out: SSDOutput = chunked_ssd(
            x_gamma, x_beta,
            B_proc, B_prev,
            C_proc, dt, A, self.D,
            x_r, self.chunk_size,
            return_final_state=return_state,
        )

        z_heads = z.reshape(Bat, T, H, P)
        z_r = z_heads.unsqueeze(-1) * self.mimo_z[None, None]
        y_gated = ssd_out.y * F.silu(z_r)
        y = (y_gated * self.mimo_down[None, None]).sum(dim=-1)
        y = y.reshape(Bat, T, self.d_inner)
        output = self.out_proj(y)

        state = None
        if return_state:
            h = ssd_out.h_final

            angle = cum_angles[:, T - 1] if cum_angles is not None else \
                torch.zeros(Bat, H, N // 2, dtype=x.dtype, device=x.device)

            x_last = x_r[:, T - 1]
            B_last = B_proc[:, T - 1]
            prev_BX = torch.einsum("bhnr,bhpr->bhpn", B_last, x_last)

            state = {
                'h': h.float(),
                'angle': angle,
                'prev_BX': prev_BX.float(),
            }

        return output, state

    def forward(self, x, dt_scale=None):
        output, _ = self._forward_impl(x, dt_scale, return_state=False)
        return output

    def prefill(self, x, dt_scale=None):
        return self._forward_impl(x, dt_scale, return_state=True)

    def step(self, hidden_states, ssm_state, dt_scale=None):
        squeeze = hidden_states.dim() == 2
        if squeeze:
            hidden_states = hidden_states.unsqueeze(1)
        x_input = hidden_states.squeeze(1)

        H, P, N, R = self.nheads, self.headdim, self.d_state, self.mimo_rank

        proj = self.in_proj(x_input)
        z, x_flat, B_flat, C_flat, dt_raw, lam_raw, theta_raw = self._split_proj(proj)

        dt = F.softplus(dt_raw + self.dt_bias)
        if dt_scale is not None:
            dt = dt * dt_scale
        A = -torch.exp(self.A_log)
        dA = torch.exp(A * dt)

        B_proc, C_proc, angle_state = self._process_BC_step(
            B_flat, C_flat, dt, theta_raw, ssm_state['angle'],
        )

        x_r = x_flat.reshape(-1, H, P, 1) * self.mimo_x[None]

        if self.use_trapezoidal and lam_raw is not None:
            lam = torch.sigmoid(lam_raw)
            gamma = lam * dt
            beta = (1.0 - lam) * dt * dA
        else:
            gamma = dt
            beta = torch.zeros_like(dt)

        curr_BX = torch.einsum("bhnr,bhpr->bhpn", B_proc, x_r)

        h = ssm_state['h']
        prev_BX = ssm_state['prev_BX']

        # fp32 recurrence to match parallel path's factored cumsum precision
        h = (dA[:, :, None, None].float() * h.float()
             + gamma[:, :, None, None].float() * curr_BX.float()
             + beta[:, :, None, None].float() * prev_BX.float())

        wdtype = x_input.dtype
        y_per_rank = torch.einsum("bhnr,bhpn->bhpr", C_proc.float(), h).to(wdtype)
        y_per_rank = y_per_rank + self.D[None, :, None, None] * x_r

        z_heads = z.reshape(-1, H, P)
        z_r = z_heads.unsqueeze(-1) * self.mimo_z[None]
        y_gated = y_per_rank * F.silu(z_r)
        y = (y_gated * self.mimo_down[None]).sum(dim=-1)
        y = y.reshape(-1, self.d_inner)
        out = self.out_proj(y)

        if not squeeze:
            out = out.unsqueeze(1)

        return out, {'h': h, 'angle': angle_state, 'prev_BX': curr_BX.float()}

    def allocate_inference_state(self, batch_size, dtype=None, device=None):
        dtype = dtype or self.in_proj.weight.dtype
        device = device or self.in_proj.weight.device
        H, P, N = self.nheads, self.headdim, self.d_state
        return {
            'h': torch.zeros(batch_size, H, P, N, dtype=torch.float32, device=device),
            'angle': torch.zeros(batch_size, H, N // 2, dtype=dtype, device=device),
            'prev_BX': torch.zeros(batch_size, H, P, N, dtype=torch.float32, device=device),
        }
