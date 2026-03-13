"""Mamba-3 blocks and language model."""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .mamba3 import KulaMamba, RMSNorm


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int | None = None, bias: bool = False):
        super().__init__()
        d_ff = d_ff or int(d_model * 8 / 3 / 64 + 1) * 64
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class KulaMambaBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int | None = None,
        mimo_rank: int = 2,
        chunk_size: int = 64,
        use_mlp: bool = True,
        mlp_expand: float = 2.667,
        norm_eps: float = 1e-5,
        bias: bool = False,
        residual_scale: float = 1.0,
        gradient_checkpointing: bool = False,
        use_trapezoidal: bool = True,
        use_complex_ssm: bool = True,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.gradient_checkpointing = gradient_checkpointing

        self.ssm_norm = RMSNorm(d_model, eps=norm_eps)
        self.ssm = KulaMamba(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            bias=bias,
            use_trapezoidal=use_trapezoidal,
            use_complex_ssm=use_complex_ssm,
        )

        self.use_mlp = use_mlp
        if use_mlp:
            d_ff = int(d_model * mlp_expand / 64 + 1) * 64
            self.mlp_norm = RMSNorm(d_model, eps=norm_eps)
            self.mlp = SwiGLU(d_model, d_ff, bias=bias)

    def _ssm_forward(self, x_normed):
        return self.ssm(x_normed)

    def _mlp_forward(self, x_normed):
        return self.mlp(x_normed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            h = grad_checkpoint(self._ssm_forward, self.ssm_norm(x), use_reentrant=False)
        else:
            h = self.ssm(self.ssm_norm(x))
        x = x + h * self.residual_scale

        if self.use_mlp:
            if self.gradient_checkpointing and self.training:
                m = grad_checkpoint(self._mlp_forward, self.mlp_norm(x), use_reentrant=False)
            else:
                m = self.mlp(self.mlp_norm(x))
            x = x + m * self.residual_scale
        return x

    def step(self, x, ssm_state):
        residual = x
        h, new_state = self.ssm.step(self.ssm_norm(x), ssm_state)
        x = residual + h * self.residual_scale
        if self.use_mlp:
            x = x + self.mlp(self.mlp_norm(x)) * self.residual_scale
        return x, new_state

    def prefill(self, x):
        h, state = self.ssm.prefill(self.ssm_norm(x))
        x = x + h * self.residual_scale
        if self.use_mlp:
            x = x + self.mlp(self.mlp_norm(x)) * self.residual_scale
        return x, state


class KulaMambaLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int | None = None,
        mimo_rank: int = 2,
        chunk_size: int = 64,
        use_mlp: bool = True,
        tie_weights: bool = True,
        norm_eps: float = 1e-5,
        bias: bool = False,
        residual_scale: str = 'depth',
        gradient_checkpointing: bool = False,
        use_trapezoidal: bool = True,
        use_complex_ssm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        r_scale = 1.0 / math.sqrt(2 * n_layers) if residual_scale == 'depth' else 1.0

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            KulaMambaBlock(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                ngroups=ngroups,
                mimo_rank=mimo_rank,
                chunk_size=chunk_size,
                use_mlp=use_mlp,
                norm_eps=norm_eps,
                bias=bias,
                residual_scale=r_scale,
                gradient_checkpointing=gradient_checkpointing,
                use_trapezoidal=use_trapezoidal,
                use_complex_ssm=use_complex_ssm,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.lm_head:
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # N(0,0.02) above overwrites kaiming; re-run KulaMamba-specific init
        for m in self.modules():
            if isinstance(m, KulaMamba):
                m._init_weights(dt_min=0.001, dt_max=0.1)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        result = {'logits': logits}
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            result['loss'] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return result

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        B, T = input_ids.shape

        x = self.embedding(input_ids)
        states = [None] * self.n_layers
        for i, layer in enumerate(self.layers):
            x, states[i] = layer.prefill(x)
        x = self.norm(x)

        logits = self.lm_head(x[:, -1])
        generated = [input_ids]

        for _ in range(max_new_tokens):
            if temperature > 0:
                logits_scaled = logits / temperature
            else:
                logits_scaled = logits
            if top_k is not None:
                v, _ = torch.topk(logits_scaled, min(top_k, logits_scaled.size(-1)))
                logits_scaled = logits_scaled.masked_fill(
                    logits_scaled < v[:, -1:], float('-inf')
                )
            if temperature > 0:
                probs = F.softmax(logits_scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits_scaled.argmax(dim=-1, keepdim=True)
            generated.append(next_token)

            x = self.embedding(next_token.squeeze(1))
            for i, layer in enumerate(self.layers):
                x, states[i] = layer.step(x, states[i])
            x = self.norm(x)
            logits = self.lm_head(x)

        return torch.cat(generated, dim=1)
