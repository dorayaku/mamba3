"""Chunked SSD — Mamba-3 parallel training kernel.

h_t = alpha_t h_{t-1} + gamma_t (B_t x x_t) + beta_t (B_{t-1} x x_{t-1})
BX[p,n] = sum_r B[n,r] * (dt * x[p,r])    (MIMO rank contraction)
y[t,p,q] = sum_n C[t,n,q] * h[t,p,n] + D * x[t,p,q]

Intra-chunk: factored cumsum, O(CS) per chunk, no L-matrix.
Inter-chunk: sequential scan over K = T/CS boundaries.
Custom autograd recomputes fp32 intermediates in backward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .scan import ssd_scan


@dataclass
class SSDOutput:
    y: torch.Tensor
    h_final: torch.Tensor | None = None


class _IntraChunkFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        BX_total: torch.Tensor,
        cum_h: torch.Tensor,
        Cc_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Bat, K, H, CS, P, N = BX_total.shape
        R = Cc_w.shape[3]
        BKH = Bat * K * H
        BKHCS = BKH * CS
        wdtype = BX_total.dtype
        acc_dtype = torch.float32 if wdtype in (torch.float16, torch.bfloat16) else wdtype

        enc = (-cum_h).exp()[:, :, :, :, None, None]
        ec = cum_h.exp()[:, :, :, :, None, None]
        scaled_BX = BX_total.to(acc_dtype) * enc
        mid = (ec * scaled_BX.cumsum(dim=3)).to(wdtype)

        Cc_mm = Cc_w.permute(0, 1, 2, 4, 3, 5).reshape(BKHCS, R, N)
        mid_mm = mid.reshape(BKHCS, P, N).transpose(-2, -1)
        y_intra = (
            torch.bmm(Cc_mm, mid_mm)
            .reshape(BKH, CS, R, P)
            .permute(0, 2, 1, 3)
            .reshape(Bat, K, H, R, CS, P)
        )

        sBX_sum = scaled_BX.sum(dim=3)

        ctx.save_for_backward(BX_total, cum_h, Cc_w)
        ctx.shapes = (Bat, K, H, CS, P, N, R, BKH, BKHCS)
        ctx.wdtype = wdtype
        ctx.acc_dtype = acc_dtype

        return y_intra, sBX_sum

    @staticmethod
    def backward(
        ctx, d_y_intra: torch.Tensor, d_sBX_sum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        BX_total, cum_h, Cc_w = ctx.saved_tensors
        Bat, K, H, CS, P, N, R, BKH, BKHCS = ctx.shapes
        wdtype = ctx.wdtype
        acc_dtype = ctx.acc_dtype

        enc = (-cum_h).exp()[:, :, :, :, None, None]
        ec = cum_h.exp()[:, :, :, :, None, None]
        scaled_BX = BX_total.to(acc_dtype) * enc
        cumsBX = scaled_BX.cumsum(dim=3)
        mid = (ec * cumsBX).to(wdtype)

        Cc_mm = Cc_w.permute(0, 1, 2, 4, 3, 5).reshape(BKHCS, R, N).to(wdtype)
        mid_mm = mid.reshape(BKHCS, P, N).transpose(-2, -1)

        d_bmm = (
            d_y_intra
            .reshape(BKH, R, CS, P)
            .permute(0, 2, 1, 3)
            .reshape(BKHCS, R, P)
        ).to(wdtype)

        d_Cc_mm = torch.bmm(d_bmm, mid_mm.transpose(-2, -1))
        d_mid_mm = torch.bmm(Cc_mm.transpose(-2, -1), d_bmm)

        d_Cc_w = (
            d_Cc_mm
            .reshape(Bat, K, H, CS, R, N)
            .permute(0, 1, 2, 4, 3, 5)
        )

        d_mid = d_mid_mm.transpose(-2, -1).reshape(Bat, K, H, CS, P, N)

        d_mid_acc = d_mid.to(acc_dtype)
        d_cumsBX = ec * d_mid_acc
        d_ec_from_mid = (cumsBX * d_mid_acc).sum(dim=(-2, -1))

        d_scaled_BX = d_cumsBX.flip(3).cumsum(3).flip(3)
        d_scaled_BX = d_scaled_BX + d_sBX_sum.to(acc_dtype).unsqueeze(3)

        d_BX_total = (enc * d_scaled_BX).to(wdtype)
        d_enc_from_sBX = (BX_total.to(acc_dtype) * d_scaled_BX).sum(dim=(-2, -1))

        d_cum_h = ec[:, :, :, :, 0, 0] * d_ec_from_mid - enc[:, :, :, :, 0, 0] * d_enc_from_sBX

        return d_BX_total, d_cum_h, d_Cc_w


def chunked_ssd(
    x_g: torch.Tensor,
    x_b: torch.Tensor | None,
    B_g: torch.Tensor,
    B_b: torch.Tensor | None,
    C: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    x_raw: torch.Tensor,
    chunk_size: int = 64,
    h_init: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> SSDOutput:
    Bat, T, H, P, R = x_g.shape
    N = B_g.shape[3]
    CS = chunk_size

    has_beta = x_b is not None

    n_chunks = (T + CS - 1) // CS
    T_pad = n_chunks * CS
    pad = T_pad - T
    if pad > 0:
        x_g = F.pad(x_g, (0, 0, 0, 0, 0, 0, 0, pad))
        B_g = F.pad(B_g, (0, 0, 0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        x_raw = F.pad(x_raw, (0, 0, 0, 0, 0, 0, 0, pad))
        if has_beta:
            x_b = F.pad(x_b, (0, 0, 0, 0, 0, 0, 0, pad))
            B_b = F.pad(B_b, (0, 0, 0, 0, 0, 0, 0, pad))

    K = n_chunks

    log_decay = A[None, None, :] * dt
    cum_h = (
        log_decay.reshape(Bat, K, CS, H)
        .float()
        .cumsum(dim=2)
        .permute(0, 1, 3, 2)
    )

    exp_cum = cum_h.exp()
    chunk_decay = exp_cum[:, :, :, -1]

    def _chunk(t: torch.Tensor) -> torch.Tensor:
        X = t.shape[3]
        return (
            t.reshape(Bat, K, CS, H, X, R)
            .permute(0, 1, 3, 5, 2, 4)
            .contiguous()
        )

    x_gc = _chunk(x_g)
    Bgc = _chunk(B_g)
    Cc = _chunk(C)
    xrc = _chunk(x_raw)

    dt_w = dt.reshape(Bat, K, CS, H).permute(0, 1, 3, 2)
    wdtype = x_g.dtype

    dt_r = dt_w.to(wdtype)[:, :, :, None, :, None]

    BKH = Bat * K * H
    BKHCS = BKH * CS

    dtx_g = dt_r * x_gc

    if has_beta:
        x_bc = _chunk(x_b)
        Bbc = _chunk(B_b)
        dtx_b = dt_r * x_bc
        R2 = 2 * R

        dtx_fused = torch.cat([dtx_g, dtx_b], dim=3)
        B_fused = torch.cat([Bgc, Bbc], dim=3)

        dtx_r = dtx_fused.reshape(BKH, R2, CS, P).permute(0, 2, 3, 1).reshape(BKHCS, P, R2)
        B_r = B_fused.reshape(BKH, R2, CS, N).permute(0, 2, 1, 3).reshape(BKHCS, R2, N)
        BX_total = torch.bmm(dtx_r, B_r).reshape(Bat, K, H, CS, P, N)
    else:
        dtx_r = dtx_g.reshape(BKH, R, CS, P).permute(0, 2, 3, 1).reshape(BKHCS, P, R)
        B_r = Bgc.reshape(BKH, R, CS, N).permute(0, 2, 1, 3).reshape(BKHCS, R, N)
        BX_total = torch.bmm(dtx_r, B_r).reshape(Bat, K, H, CS, P, N)

    Cc_w = Cc.to(wdtype)
    y_intra, sBX_sum = _IntraChunkFn.apply(BX_total, cum_h, Cc_w)

    chunk_decay_pn = chunk_decay[:, :, :, None, None]
    h_chunk = (chunk_decay_pn * sBX_sum).to(wdtype)

    h_chunk_flat = h_chunk.reshape(Bat, K, H, P * N)
    h_init_flat = h_init.reshape(Bat, H, P * N) if h_init is not None else None
    h_states_flat = ssd_scan(chunk_decay, h_chunk_flat, h_init_flat)
    h_states = h_states_flat.reshape(Bat, K, H, P, N)

    Cc_inter = Cc_w.reshape(BKH, R * CS, N)
    h_mm = h_states.to(wdtype).reshape(BKH, P, N).transpose(-2, -1)
    y_inter = (
        torch.bmm(Cc_inter, h_mm)
        .reshape(Bat, K, H, R, CS, P)
        .float()
        * exp_cum[:, :, :, None, :, None]
    ).to(wdtype)

    y_d = D[None, None, :, None, None, None] * xrc.to(wdtype)

    y_total = y_intra + y_inter + y_d

    y = (
        y_total
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(Bat, T_pad, H, P, R)
        [:, :T]
    )

    h_final = None
    if return_final_state:
        # fp32: feeds recurrence in step(), quantization would accumulate
        h_final = (
            chunk_decay[:, K - 1, :, None, None]
            * h_states[:, K - 1].float()
            + h_chunk[:, K - 1].float()
        )

    return SSDOutput(y=y.to(x_g.dtype), h_final=h_final)
