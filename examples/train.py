"""Train KulaMambaLM from scratch on TinyShakespeare."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler

from mamba3 import KulaMambaLM

try:
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")

    def encode(text):
        return _enc.encode(text, allowed_special=set())

    def decode(ids):
        return _enc.decode(ids)

    VOCAB_SIZE = _enc.n_vocab

except ImportError:
    _chars = []
    _stoi = {}
    _itos = {}

    def _build_char_vocab(text):
        global _chars, _stoi, _itos, VOCAB_SIZE
        _chars = sorted(set(text))
        _stoi = {c: i for i, c in enumerate(_chars)}
        _itos = {i: c for i, c in enumerate(_chars)}
        VOCAB_SIZE = len(_chars)

    def encode(text):
        return [_stoi.get(c, 0) for c in text]

    def decode(ids):
        return "".join(_itos.get(i, "?") for i in ids)

    VOCAB_SIZE = 0


DATA_DIR = Path(__file__).parent / "data"
TINYSHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_tinyshakespeare():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "tinyshakespeare.txt"
    if not path.exists():
        print("Downloading TinyShakespeare...")
        urllib.request.urlretrieve(TINYSHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")


def load_data(dataset):
    global VOCAB_SIZE

    if dataset == "tinyshakespeare":
        text = download_tinyshakespeare()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if VOCAB_SIZE == 0:
        _build_char_vocab(text)

    ids = encode(text)
    print(f"Dataset: {dataset}, tokens: {len(ids):,}, vocab: {VOCAB_SIZE:,}")
    return torch.tensor(ids, dtype=torch.long)


def cosine_lr(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    if step >= total:
        return lr_min
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def get_batch(data, batch_size, seq_len, device):
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s : s + seq_len] for s in starts]).to(device)
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data, batch_size, seq_len, device, dtype, n_eval=10):
    model.eval()
    total = 0.0
    for _ in range(n_eval):
        x, y = get_batch(data, batch_size, seq_len, device)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=(dtype != torch.float32)):
            out = model(x, labels=y)
        total += out["loss"].item()
    model.train()
    return total / n_eval


PRESETS = {
    "tiny": dict(d_model=128, n_layers=4, d_state=32, headdim=32, expand=2, mimo_rank=2),
    "small": dict(d_model=256, n_layers=8, d_state=64, headdim=32, expand=2, mimo_rank=2),
    "medium": dict(d_model=512, n_layers=12, d_state=64, headdim=64, expand=2, mimo_rank=2),
    "base": dict(d_model=768, n_layers=24, d_state=128, headdim=64, expand=2, mimo_rank=2),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default=None, choices=list(PRESETS.keys()))
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--d_state", type=int, default=None)
    parser.add_argument("--headdim", type=int, default=None)
    parser.add_argument("--expand", type=int, default=None)
    parser.add_argument("--mimo_rank", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="tinyshakespeare")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--gen_tokens", type=int, default=200)
    parser.add_argument("--gen_prompt", type=str, default="First Citizen:\n")
    args = parser.parse_args()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = args.device
    use_amp = dtype != torch.float32

    cfg = PRESETS.get(args.preset, PRESETS["tiny"]).copy()
    for k in ["d_model", "n_layers", "d_state", "headdim", "expand", "mimo_rank"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    data = load_data(args.dataset)
    n_train = int(len(data) * 0.9)
    train_data = data[:n_train]
    val_data = data[n_train:]

    model = KulaMambaLM(
        vocab_size=VOCAB_SIZE,
        residual_scale="depth",
        gradient_checkpointing=args.gradient_checkpointing,
        **cfg,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {cfg}")
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"Device: {device}, Dtype: {args.dtype}, AMP: {use_amp}")
    print(f"Training: {args.steps} steps, batch={args.batch_size}, seq={args.seq_len}")
    print(f"LR: {args.lr} -> {args.lr_min} (cosine, warmup={args.warmup})\n")

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model...")
        model = torch.compile(model)

    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name or "bias" in name or "embedding" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), fused=(device == "cuda"))

    scaler = GradScaler("cuda", enabled=(use_amp and device == "cuda"))

    model.train()
    best_val_loss = float("inf")
    t_start = time.perf_counter()
    tokens_processed = 0

    for step in range(1, args.steps + 1):
        lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.lr_min)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.seq_len, device)

        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            out = model(x, labels=y)
            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        tokens_processed += args.batch_size * args.seq_len

        if step % args.log_interval == 0:
            dt = time.perf_counter() - t_start
            tok_per_sec = tokens_processed / dt
            print(
                f"step {step:>6d}/{args.steps} | "
                f"loss {loss.item():.4f} | "
                f"lr {lr:.2e} | "
                f"grad {grad_norm:.2f} | "
                f"{tok_per_sec:,.0f} tok/s"
            )

        if step % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_data, args.batch_size, args.seq_len, device, dtype)
            print(f"  >> val loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.save_dir:
                    save_path = Path(args.save_dir)
                    save_path.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path / "best.pt")
                    with open(save_path / "config.json", "w") as f:
                        json.dump({"vocab_size": VOCAB_SIZE, **cfg}, f, indent=2)
                    print(f"  >> saved best model (val_loss={val_loss:.4f})")
            model.train()

    total_time = time.perf_counter() - t_start
    print(f"\nDone: {args.steps} steps in {total_time:.1f}s")
    print(f"Final train loss: {loss.item():.4f}, Best val loss: {best_val_loss:.4f}")
    print(f"Throughput: {tokens_processed / total_time:,.0f} tok/s")

    print(f"\n{'='*60}")
    print(f"Generating {args.gen_tokens} tokens from: {args.gen_prompt!r}")
    print(f"{'='*60}\n")

    model.eval()
    prompt_ids = torch.tensor([encode(args.gen_prompt)], dtype=torch.long, device=device)

    with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
        generated = model.generate(prompt_ids, max_new_tokens=args.gen_tokens, temperature=0.8, top_k=40)

    print(decode(generated[0].tolist()))
    print()

    if args.save_dir:
        save_path = Path(args.save_dir)
        torch.save(model.state_dict(), save_path / "final.pt")
        print(f"Final model saved to {save_path / 'final.pt'}")


if __name__ == "__main__":
    main()
