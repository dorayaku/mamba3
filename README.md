# Mamba3

Complete Mamba-3 implementation. The [official repo](https://github.com/state-spaces/mamba) does not include Mamba-3.

All three innovations from [Mamba-3 (ICLR 2026)](https://openreview.net/forum?id=HwCvaJOiCj):

- **Trapezoidal discretization** — two-term SSD decomposition handling the `B_{t-1} x_{t-1}` dependency, fused via rank-doubled bmm with shared decay.
- **Complex SSM** — data-dependent RoPE on B/C. Equivalent to complex-valued state without complex arithmetic.
- **MIMO** — rank-R contraction on input, expansion on output. State is rank-free (P, N) — cross-rank interaction, not R independent SISOs.

Chunked SSD with factored cumsum (no L-matrix), custom autograd recomputation backward, CUDA inter-chunk scan, O(1) step inference, parallel prefill, torch.compile zero graph breaks. 42 tests.

## Install

```bash
pip install -e .
```

## Usage

```python
import torch
from mamba3 import KulaMamba

layer = KulaMamba(d_model=768, d_state=128, expand=2, headdim=64, mimo_rank=2)

x = torch.randn(4, 1024, 768)
y = layer(x)
```

```python
# O(1) inference
state = layer.allocate_inference_state(batch_size=4)
for tok in stream:
    out, state = layer.step(tok, state)

# parallel prefill + generation
out, state = layer.prefill(prompt)
out, state = layer.step(next_tok, state)
```

```python
from mamba3 import KulaMambaLM

model = KulaMambaLM(
    vocab_size=32000, d_model=768, n_layers=24,
    d_state=128, headdim=64, mimo_rank=2,
    residual_scale='depth', gradient_checkpointing=True,
)
result = model(input_ids, labels=labels)
tokens = model.generate(prompt_ids, max_new_tokens=256, temperature=0.7, top_k=50)
```

## Structure

```
mamba3/
├── mamba3.py    KulaMamba layer
├── ssd.py       Chunked SSD (factored cumsum, custom autograd)
├── scan.py      Inter-chunk scan (CUDA JIT + PyTorch fallback)
├── block.py     KulaMambaBlock, KulaMambaLM
└── csrc/
    └── scan.cu  CUDA scan kernels
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
