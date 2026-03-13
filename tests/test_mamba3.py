"""KulaMamba test suite — 42 tests."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba3.mamba3 import KulaMamba
from mamba3.block import KulaMambaBlock, KulaMambaLM
from mamba3.ssd import _IntraChunkFn


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32


def make_layer(**kw):
    defaults = dict(d_model=64, d_state=16, headdim=16, mimo_rank=2, expand=2, chunk_size=8)
    defaults.update(kw)
    return KulaMamba(**defaults).to(DEVICE, DTYPE).eval()


def test_forward_shape():
    layer = make_layer()
    x = torch.randn(2, 32, 64, device=DEVICE, dtype=DTYPE)
    assert layer(x).shape == x.shape


def test_non_chunk_aligned():
    layer = make_layer()
    x = torch.randn(2, 37, 64, device=DEVICE, dtype=DTYPE)
    assert layer(x).shape == x.shape


def test_forward_step_consistency():
    layer = make_layer()
    B, T = 1, 24
    x = torch.randn(B, T, 64, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        y_par = layer(x)

    state = layer.allocate_inference_state(B, dtype=DTYPE, device=DEVICE)
    y_steps = []
    with torch.no_grad():
        for t in range(T):
            y_t, state = layer.step(x[:, t:t+1], state)
            y_steps.append(y_t.squeeze(1))
    y_seq = torch.stack(y_steps, dim=1)

    max_diff = (y_par - y_seq).abs().max().item()
    assert max_diff < 1e-4, f"Forward/step max diff = {max_diff:.6f}"


def test_prefill_step_consistency():
    layer = make_layer()
    B, T_prompt, T_gen = 1, 16, 8
    x_full = torch.randn(B, T_prompt + T_gen, 64, device=DEVICE, dtype=DTYPE)

    state_gt = layer.allocate_inference_state(B, dtype=DTYPE, device=DEVICE)
    outs_gt = []
    with torch.no_grad():
        for t in range(T_prompt + T_gen):
            y_t, state_gt = layer.step(x_full[:, t:t+1], state_gt)
            outs_gt.append(y_t.squeeze(1))

    with torch.no_grad():
        _, state_pf = layer.prefill(x_full[:, :T_prompt])
        outs_pf = []
        for t in range(T_prompt, T_prompt + T_gen):
            y_t, state_pf = layer.step(x_full[:, t:t+1], state_pf)
            outs_pf.append(y_t.squeeze(1))

    y_gt = torch.stack(outs_gt[T_prompt:], dim=1)
    y_pf = torch.stack(outs_pf, dim=1)

    max_diff = (y_gt - y_pf).abs().max().item()
    assert max_diff < 1e-4, f"Prefill/step max diff = {max_diff:.6f}"


def test_gradient_flow():
    layer = make_layer().train()
    x = torch.randn(2, 16, 64, device=DEVICE, dtype=DTYPE)
    y = layer(x)
    y.sum().backward()

    for name, p in layer.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
        assert p.grad.abs().max() > 0, f"Zero gradient in {name}"


def test_parity_task():
    torch.manual_seed(42)
    SEQ_LEN, D, TRAIN, TEST, BS = 16, 32, 2000, 500, 64

    def make_data(n):
        bits = torch.randint(0, 2, (n, SEQ_LEN), dtype=DTYPE, device=DEVICE)
        parity = bits.cumsum(dim=1) % 2
        return bits.unsqueeze(-1), parity

    train_x, train_y = make_data(TRAIN)
    test_x, test_y = make_data(TEST)

    model = nn.Sequential(
        nn.Linear(1, D),
        KulaMamba(d_model=D, d_state=8, expand=1, headdim=16, mimo_rank=1, chunk_size=8),
        nn.Linear(D, 2),
    ).to(DEVICE, DTYPE)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(30):
        perm = torch.randperm(TRAIN, device=DEVICE)
        for i in range(0, TRAIN, BS):
            logits = model(train_x[perm[i:i+BS]])
            loss = F.cross_entropy(logits.reshape(-1, 2), train_y[perm[i:i+BS]].long().reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        acc = (model(test_x).argmax(-1) == test_y.long()).float().mean().item()
    assert acc > 0.90, f"Parity accuracy = {acc:.3f}"


def test_block_forward():
    block = KulaMambaBlock(
        d_model=64, d_state=16, headdim=16, mimo_rank=2,
        chunk_size=8, use_mlp=True,
    ).to(DEVICE, DTYPE)
    x = torch.randn(2, 16, 64, device=DEVICE, dtype=DTYPE)
    assert block(x).shape == x.shape


def test_lm_forward():
    model = KulaMambaLM(
        vocab_size=256, d_model=64, n_layers=2,
        d_state=16, headdim=16, mimo_rank=2, chunk_size=8, use_mlp=False,
        residual_scale='none',
    ).to(DEVICE, DTYPE)
    ids = torch.randint(0, 256, (2, 16), device=DEVICE)
    result = model(ids, labels=ids)
    assert result['logits'].shape == (2, 16, 256)
    assert 'loss' in result


def test_lm_generate():
    model = KulaMambaLM(
        vocab_size=256, d_model=64, n_layers=2,
        d_state=16, headdim=16, mimo_rank=2, chunk_size=8, use_mlp=False,
        residual_scale='none',
    ).to(DEVICE, DTYPE).eval()
    prompt = torch.randint(0, 256, (1, 4), device=DEVICE)
    out = model.generate(prompt, max_new_tokens=8)
    assert out.shape == (1, 12)


def test_dt_scale():
    layer = make_layer()
    x = torch.randn(1, 16, 64, device=DEVICE, dtype=DTYPE)
    y1 = layer(x)
    dt_scale = torch.ones(1, 16, layer.nheads, device=DEVICE, dtype=DTYPE) * 2.0
    y2 = layer(x, dt_scale=dt_scale)
    assert not torch.allclose(y1, y2, atol=1e-5)


def test_ngroups():
    layer_g1 = make_layer(ngroups=1)
    layer_gh = make_layer(ngroups=layer_g1.nheads)
    x = torch.randn(1, 16, 64, device=DEVICE, dtype=DTYPE)
    y1 = layer_g1(x)
    yh = layer_gh(x)
    assert y1.shape == x.shape
    assert yh.shape == x.shape
    p1 = sum(p.numel() for p in layer_g1.parameters())
    ph = sum(p.numel() for p in layer_gh.parameters())
    assert p1 < ph, f"ngroups=1 ({p1}) should have fewer params than ngroups=H ({ph})"


def test_ngroups_step_consistency():
    layer = make_layer(ngroups=2)
    B, T = 1, 16
    x = torch.randn(B, T, 64, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        y_par = layer(x)

    state = layer.allocate_inference_state(B, dtype=DTYPE, device=DEVICE)
    y_steps = []
    with torch.no_grad():
        for t in range(T):
            y_t, state = layer.step(x[:, t:t+1], state)
            y_steps.append(y_t.squeeze(1))
    y_seq = torch.stack(y_steps, dim=1)

    max_diff = (y_par - y_seq).abs().max().item()
    assert max_diff < 1e-4, f"ngroups=2 forward/step diff = {max_diff:.6f}"


def test_euler_ablation():
    layer = make_layer(use_trapezoidal=False)
    x = torch.randn(1, 16, 64, device=DEVICE, dtype=DTYPE)
    y = layer(x)
    assert y.shape == x.shape

    layer.train()
    y.sum().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None


def test_euler_step_consistency():
    layer = make_layer(use_trapezoidal=False)
    B, T = 1, 16
    x = torch.randn(B, T, 64, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        y_par = layer(x)

    state = layer.allocate_inference_state(B, dtype=DTYPE, device=DEVICE)
    y_steps = []
    with torch.no_grad():
        for t in range(T):
            y_t, state = layer.step(x[:, t:t+1], state)
            y_steps.append(y_t.squeeze(1))
    y_seq = torch.stack(y_steps, dim=1)

    max_diff = (y_par - y_seq).abs().max().item()
    assert max_diff < 1e-4, f"Euler forward/step diff = {max_diff:.6f}"


def test_real_ssm_ablation():
    layer = make_layer(use_complex_ssm=False)
    x = torch.randn(1, 16, 64, device=DEVICE, dtype=DTYPE)
    y = layer(x)
    assert y.shape == x.shape


def test_real_ssm_step_consistency():
    layer = make_layer(use_complex_ssm=False)
    B, T = 1, 16
    x = torch.randn(B, T, 64, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        y_par = layer(x)

    state = layer.allocate_inference_state(B, dtype=DTYPE, device=DEVICE)
    y_steps = []
    with torch.no_grad():
        for t in range(T):
            y_t, state = layer.step(x[:, t:t+1], state)
            y_steps.append(y_t.squeeze(1))
    y_seq = torch.stack(y_steps, dim=1)

    max_diff = (y_par - y_seq).abs().max().item()
    assert max_diff < 1e-4, f"Real SSM forward/step diff = {max_diff:.6f}"


def test_mimo_rank1():
    layer = make_layer(mimo_rank=1)
    x = torch.randn(1, 16, 64, device=DEVICE, dtype=DTYPE)
    assert layer(x).shape == x.shape


def test_gradient_checkpointing():
    block = KulaMambaBlock(
        d_model=64, d_state=16, headdim=16, mimo_rank=2,
        chunk_size=8, use_mlp=True, gradient_checkpointing=True,
    ).to(DEVICE, DTYPE).train()
    x = torch.randn(2, 16, 64, device=DEVICE, dtype=DTYPE, requires_grad=True)
    y = block(x)
    y.sum().backward()
    assert x.grad is not None


def test_depth_scaled_residuals():
    model = KulaMambaLM(
        vocab_size=256, d_model=64, n_layers=4,
        d_state=16, headdim=16, mimo_rank=2, chunk_size=8,
        use_mlp=False, residual_scale='depth',
    ).to(DEVICE, DTYPE)
    import math
    expected = 1.0 / math.sqrt(2 * 4)
    actual = model.layers[0].residual_scale
    assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual}"

    ids = torch.randint(0, 256, (1, 16), device=DEVICE)
    result = model(ids)
    assert result['logits'].shape == (1, 16, 256)


def test_prefill_state_accuracy():
    layer = make_layer()
    B, T = 1, 24
    x = torch.randn(B, T, 64, device=DEVICE, dtype=DTYPE)

    state_gt = layer.allocate_inference_state(B, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        for t in range(T):
            _, state_gt = layer.step(x[:, t:t+1], state_gt)

    with torch.no_grad():
        _, state_pf = layer.prefill(x)

    h_diff = (state_gt['h'] - state_pf['h']).abs().max().item()
    assert h_diff < 1e-4, f"Prefill h state diff = {h_diff:.6f}"

    angle_diff = (state_gt['angle'] - state_pf['angle']).abs().max().item()
    assert angle_diff < 1e-4, f"Prefill angle diff = {angle_diff:.6f}"

    bx_diff = (state_gt['prev_BX'] - state_pf['prev_BX']).abs().max().item()
    assert bx_diff < 1e-4, f"Prefill prev_BX diff = {bx_diff:.6f}"


def test_scan_initial_state():
    from mamba3.scan import ssd_scan

    B, K, H, PN = 2, 4, 3, 8
    decay = torch.rand(B, K, H, device=DEVICE, dtype=DTYPE) * 0.9
    h_chunk = torch.randn(B, K, H, PN, device=DEVICE, dtype=DTYPE)
    h_init = torch.randn(B, H, PN, device=DEVICE, dtype=DTYPE)

    h_states = ssd_scan(decay, h_chunk, h_init)

    h = h_init
    expected = [h]
    for k in range(K - 1):
        h = h * decay[:, k, :, None] + h_chunk[:, k]
        expected.append(h)
    expected = torch.stack(expected, dim=1)

    max_diff = (h_states - expected).abs().max().item()
    assert max_diff < 1e-5, f"Scan h_init diff = {max_diff:.6f}"


def test_scan_h_init_gradient():
    from mamba3.scan import ssd_scan

    B, K, H, PN = 1, 3, 2, 4
    decay = torch.rand(B, K, H, device=DEVICE, dtype=DTYPE)
    h_chunk = torch.randn(B, K, H, PN, device=DEVICE, dtype=DTYPE)
    h_init = torch.randn(B, H, PN, device=DEVICE, dtype=DTYPE, requires_grad=True)

    h_states = ssd_scan(decay, h_chunk, h_init)
    loss = h_states.sum()
    loss.backward()

    assert h_init.grad is not None
    assert not torch.isnan(h_init.grad).any()
    assert h_init.grad.abs().max() > 0


def test_all_innovations_off():
    layer = make_layer(use_trapezoidal=False, use_complex_ssm=False, mimo_rank=1)
    B, T = 1, 16
    x = torch.randn(B, T, 64, device=DEVICE, dtype=DTYPE)
    y = layer(x)
    assert y.shape == x.shape

    with torch.no_grad():
        y_par = layer(x)

    state = layer.allocate_inference_state(B, dtype=DTYPE, device=DEVICE)
    y_steps = []
    with torch.no_grad():
        for t in range(T):
            y_t, state = layer.step(x[:, t:t+1], state)
            y_steps.append(y_t.squeeze(1))
    y_seq = torch.stack(y_steps, dim=1)

    max_diff = (y_par - y_seq).abs().max().item()
    assert max_diff < 1e-4, f"All-off forward/step diff = {max_diff:.6f}"


def test_mimo_cross_rank():
    layer = make_layer(mimo_rank=2)
    B, T = 1, 16
    x = torch.randn(B, T, 64, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        y_full = layer(x)

        mimo_x_backup = layer.mimo_x.data.clone()
        layer.mimo_x.data[:, :, 1] = 0.0
        y_r1_zeroed = layer(x)
        layer.mimo_x.data.copy_(mimo_x_backup)

    diff = (y_full - y_r1_zeroed).abs().max().item()
    assert diff > 1e-5, f"No cross-rank interaction detected, diff={diff:.8f}"


def test_state_shapes():
    layer = make_layer()
    B = 2
    H, P, N = layer.nheads, layer.headdim, layer.d_state

    state = layer.allocate_inference_state(B, dtype=DTYPE, device=DEVICE)
    assert state['h'].shape == (B, H, P, N)
    assert state['prev_BX'].shape == (B, H, P, N)
    assert state['angle'].shape == (B, H, N // 2)


def test_default_ngroups():
    layer = make_layer()
    assert layer.ngroups == 1


def test_numerical_stability_extreme_decay():
    layer = make_layer()
    with torch.no_grad():
        layer.A_log.fill_(0.0)
        layer.dt_bias.fill_(3.0)

    B, T = 1, 64
    x = torch.randn(B, T, 64, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        y = layer(x)

    assert torch.isfinite(y).all(), "NaN/Inf in output under extreme decay"
    assert y.abs().max() > 0


def test_stability_fwd_step_extreme_decay():
    layer = make_layer()
    with torch.no_grad():
        layer.A_log.fill_(-0.5)
        layer.dt_bias.fill_(2.0)

    B, T = 1, 24
    x = torch.randn(B, T, 64, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        y_par = layer(x)

    state = layer.allocate_inference_state(B, dtype=DTYPE, device=DEVICE)
    y_steps = []
    with torch.no_grad():
        for t in range(T):
            y_t, state = layer.step(x[:, t:t+1], state)
            y_steps.append(y_t.squeeze(1))
    y_seq = torch.stack(y_steps, dim=1)

    max_diff = (y_par - y_seq).abs().max().item()
    assert max_diff < 5e-3, f"Extreme decay forward/step diff = {max_diff:.6f}"


def test_gradient_flow_extreme_decay():
    layer = make_layer()
    with torch.no_grad():
        layer.A_log.fill_(0.0)
        layer.dt_bias.fill_(2.5)
    layer.train()

    x = torch.randn(1, 32, 64, device=DEVICE, dtype=DTYPE)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    for name, p in layer.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name}"


def test_intra_chunk_fn_gradcheck():
    torch.manual_seed(42)
    B, K, H, CS, P, N, R = 1, 2, 2, 4, 4, 8, 2

    BX = torch.randn(B, K, H, CS, P, N, dtype=torch.float64, requires_grad=True)
    cum_h = -torch.abs(torch.randn(B, K, H, CS, dtype=torch.float64)).requires_grad_(True)
    Cc = torch.randn(B, K, H, R, CS, N, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        _IntraChunkFn.apply,
        (BX, cum_h, Cc),
        eps=1e-6, atol=1e-5, rtol=1e-4,
    )


def test_e2e_gradient_consistency():
    torch.manual_seed(123)
    layer = make_layer()
    x = torch.randn(2, 16, 64, device=DEVICE, dtype=DTYPE)

    layer.zero_grad()
    y = layer(x)
    y.sum().backward()
    grads = {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}

    layer.zero_grad()
    y2 = layer(x)
    y2.sum().backward()
    grads2 = {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}

    for name in grads:
        diff = (grads[name] - grads2[name]).abs().max().item()
        assert diff == 0.0, f"Non-deterministic gradient in {name}: diff={diff}"

    for name, g in grads.items():
        assert torch.isfinite(g).all(), f"NaN/Inf in gradient {name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 requires CUDA")
def test_bf16_forward_step_consistency():
    layer = KulaMamba(
        d_model=64, d_state=16, headdim=16, mimo_rank=2, expand=2, chunk_size=8,
    ).to('cuda', torch.bfloat16).eval()
    B, T = 1, 24
    x = torch.randn(B, T, 64, device='cuda', dtype=torch.bfloat16)

    with torch.no_grad():
        y_par = layer(x)

    state = layer.allocate_inference_state(B, dtype=torch.bfloat16, device='cuda')
    y_steps = []
    with torch.no_grad():
        for t in range(T):
            y_t, state = layer.step(x[:, t:t+1], state)
            y_steps.append(y_t.squeeze(1))
    y_seq = torch.stack(y_steps, dim=1)

    max_diff = (y_par - y_seq).abs().max().item()
    assert max_diff < 0.05, f"bf16 forward/step max diff = {max_diff:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 requires CUDA")
def test_bf16_gradient_flow():
    layer = KulaMamba(
        d_model=64, d_state=16, headdim=16, mimo_rank=2, expand=2, chunk_size=8,
    ).to('cuda', torch.bfloat16).train()
    x = torch.randn(2, 16, 64, device='cuda', dtype=torch.bfloat16)
    y = layer(x)
    y.sum().backward()

    for name, p in layer.named_parameters():
        assert p.grad is not None, f"No gradient for {name} in bf16"
        assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name} (bf16)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 requires CUDA")
def test_bf16_prefill_state_accuracy():
    layer = KulaMamba(
        d_model=64, d_state=16, headdim=16, mimo_rank=2, expand=2, chunk_size=8,
    ).to('cuda', torch.bfloat16).eval()
    B, T = 1, 24
    x = torch.randn(B, T, 64, device='cuda', dtype=torch.bfloat16)

    state_gt = layer.allocate_inference_state(B, dtype=torch.bfloat16, device='cuda')
    with torch.no_grad():
        for t in range(T):
            _, state_gt = layer.step(x[:, t:t+1], state_gt)
        _, state_pf = layer.prefill(x)

    h_diff = (state_gt['h'] - state_pf['h']).abs().max().item()
    assert h_diff < 0.05, f"bf16 prefill h diff = {h_diff:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 requires CUDA")
def test_bf16_numerical_stability_extreme_decay():
    layer = KulaMamba(
        d_model=64, d_state=16, headdim=16, mimo_rank=2, expand=2, chunk_size=8,
    ).to('cuda', torch.bfloat16).eval()
    with torch.no_grad():
        layer.A_log.fill_(0.0)
        layer.dt_bias.fill_(3.0)

    x = torch.randn(1, 64, 64, device='cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        y = layer(x)

    assert torch.isfinite(y).all(), "NaN/Inf in bf16 output under extreme decay"


def _has_triton():
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not torch.cuda.is_available() or not _has_triton(),
                    reason="compile needs CUDA + Triton")
def test_torch_compile_forward():
    layer = KulaMamba(
        d_model=64, d_state=16, headdim=16, mimo_rank=2, expand=2, chunk_size=8,
    ).to('cuda', torch.float32).eval()

    compiled = torch.compile(layer, fullgraph=True)
    x = torch.randn(1, 16, 64, device='cuda', dtype=torch.float32)

    with torch.no_grad():
        y_eager = layer(x)
        y_compiled = compiled(x)

    max_diff = (y_eager - y_compiled).abs().max().item()
    assert max_diff < 1e-5, f"Compiled vs eager diff = {max_diff:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available() or not _has_triton(),
                    reason="compile needs CUDA + Triton")
def test_torch_compile_backward():
    layer = KulaMamba(
        d_model=64, d_state=16, headdim=16, mimo_rank=2, expand=2, chunk_size=8,
    ).to('cuda', torch.float32).train()

    compiled = torch.compile(layer, fullgraph=True)
    x = torch.randn(1, 16, 64, device='cuda', dtype=torch.float32)
    y = compiled(x)
    y.sum().backward()

    for name, p in layer.named_parameters():
        assert p.grad is not None, f"No gradient for {name} under compile"
        assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name} under compile"


def test_block_step_consistency():
    block = KulaMambaBlock(
        d_model=64, d_state=16, headdim=16, mimo_rank=2,
        chunk_size=8, use_mlp=True,
    ).to(DEVICE, DTYPE).eval()
    B, T = 1, 16
    x = torch.randn(B, T, 64, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        y_par = block(x)

    ssm_state = block.ssm.allocate_inference_state(B, dtype=DTYPE, device=DEVICE)
    y_steps = []
    with torch.no_grad():
        for t in range(T):
            y_t, ssm_state = block.step(x[:, t], ssm_state)
            y_steps.append(y_t)
    y_seq = torch.stack(y_steps, dim=1)

    max_diff = (y_par - y_seq).abs().max().item()
    assert max_diff < 1e-4, f"Block forward/step diff = {max_diff:.6f}"


def test_block_prefill_step_consistency():
    block = KulaMambaBlock(
        d_model=64, d_state=16, headdim=16, mimo_rank=2,
        chunk_size=8, use_mlp=True,
    ).to(DEVICE, DTYPE).eval()
    B, T_prompt, T_gen = 1, 12, 8
    x_full = torch.randn(B, T_prompt + T_gen, 64, device=DEVICE, dtype=DTYPE)

    state_gt = block.ssm.allocate_inference_state(B, dtype=DTYPE, device=DEVICE)
    outs_gt = []
    with torch.no_grad():
        for t in range(T_prompt + T_gen):
            y_t, state_gt = block.step(x_full[:, t], state_gt)
            outs_gt.append(y_t)

    with torch.no_grad():
        _, state_pf = block.prefill(x_full[:, :T_prompt])
        outs_pf = []
        for t in range(T_prompt, T_prompt + T_gen):
            y_t, state_pf = block.step(x_full[:, t], state_pf)
            outs_pf.append(y_t)

    y_gt = torch.stack(outs_gt[T_prompt:], dim=1)
    y_pf = torch.stack(outs_pf, dim=1)
    max_diff = (y_gt - y_pf).abs().max().item()
    assert max_diff < 1e-4, f"Block prefill/step diff = {max_diff:.6f}"


def test_ssd_h_init():
    from mamba3.ssd import chunked_ssd
    B, T, H, P, R, N = 1, 16, 2, 4, 2, 8
    CS = 8

    x_g = torch.randn(B, T, H, P, R, device=DEVICE, dtype=DTYPE)
    x_b = torch.randn(B, T, H, P, R, device=DEVICE, dtype=DTYPE)
    B_g = torch.randn(B, T, H, N, R, device=DEVICE, dtype=DTYPE)
    B_b = torch.randn(B, T, H, N, R, device=DEVICE, dtype=DTYPE)
    C_t = torch.randn(B, T, H, N, R, device=DEVICE, dtype=DTYPE)
    dt = torch.rand(B, T, H, device=DEVICE, dtype=DTYPE) * 0.1 + 0.01
    A = -torch.exp(torch.randn(H, device=DEVICE, dtype=DTYPE).clamp(-4, -1))
    D = torch.ones(H, device=DEVICE, dtype=DTYPE)
    x_raw = torch.randn(B, T, H, P, R, device=DEVICE, dtype=DTYPE)

    out0 = chunked_ssd(x_g, x_b, B_g, B_b, C_t, dt, A, D, x_raw, CS, return_final_state=True)
    h_init = torch.zeros(B, H, P, N, device=DEVICE, dtype=DTYPE)
    out1 = chunked_ssd(x_g, x_b, B_g, B_b, C_t, dt, A, D, x_raw, CS, h_init=h_init, return_final_state=True)

    y_diff = (out0.y - out1.y).abs().max().item()
    assert y_diff < 1e-5, f"h_init=zeros vs None diff = {y_diff:.6f}"

    h_init2 = torch.randn(B, H, P, N, device=DEVICE, dtype=DTYPE)
    out2 = chunked_ssd(x_g, x_b, B_g, B_b, C_t, dt, A, D, x_raw, CS, h_init=h_init2, return_final_state=True)
    y_diff2 = (out0.y - out2.y).abs().max().item()
    assert y_diff2 > 1e-3, f"h_init should change output, diff = {y_diff2:.6f}"


def test_long_sequence_stress():
    layer = make_layer(chunk_size=8)
    B, T = 1, 512
    x = torch.randn(B, T, 64, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        y = layer(x)
    assert torch.isfinite(y).all()
    assert y.shape == (B, T, 64)

    layer.train()
    y = layer(x)
    y.sum().backward()
    for name, p in layer.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name} (T=512)"

    layer.eval()
    with torch.no_grad():
        y_par = layer(x)

    state = layer.allocate_inference_state(B, dtype=DTYPE, device=DEVICE)
    y_steps = []
    with torch.no_grad():
        for t in range(T):
            y_t, state = layer.step(x[:, t:t+1], state)
            y_steps.append(y_t.squeeze(1))
    y_seq = torch.stack(y_steps, dim=1)

    max_diff = (y_par - y_seq).abs().max().item()
    assert max_diff < 1e-3, f"Long-sequence forward/step diff = {max_diff:.6f}"


def test_single_token():
    layer = make_layer(chunk_size=8)
    B = 2
    x = torch.randn(B, 1, 64, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        y = layer(x)
    assert y.shape == (B, 1, 64)
    assert torch.isfinite(y).all()

    layer.eval()
    with torch.no_grad():
        y_par = layer(x)

    state = layer.allocate_inference_state(B, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        y_step, _ = layer.step(x[:, 0:1], state)
        y_step = y_step.squeeze(1).unsqueeze(1)

    max_diff = (y_par - y_step).abs().max().item()
    assert max_diff < 1e-4, f"T=1 forward/step diff = {max_diff:.6f}"

    layer.train()
    y = layer(x)
    y.sum().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"No gradient for {name} at T=1"
        assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name} at T=1"

    layer.eval()
    with torch.no_grad():
        _, state_pf = layer.prefill(x)
    assert state_pf['h'].shape[0] == B
    assert torch.isfinite(state_pf['h']).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
