"""Smoke run: build a tiny Mamba LM, run one forward + backward on random tokens."""

import time

import torch
import torch.nn.functional as F

from mamba_scratch import MambaConfig, MambaLM


def main() -> None:
    torch.manual_seed(0)
    vocab_size, batch, seq_len = 256, 4, 64
    config = MambaConfig(d_model=64, d_state=16)
    model = MambaLM(config, vocab_size=vocab_size, n_layers=2)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"config: {config}")
    print(f"d_inner={config.d_inner}, dt_rank={config.dt_rank}, params={n_params:,}")

    tokens = torch.randint(0, vocab_size, (batch, seq_len))
    start = time.perf_counter()
    logits = model(tokens[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), tokens[:, 1:].reshape(-1))
    loss.backward()
    elapsed = time.perf_counter() - start

    print(f"logits: {tuple(logits.shape)}, loss={loss.item():.4f} (ln vocab = {torch.log(torch.tensor(vocab_size)):.4f})")
    print(f"forward + backward: {elapsed * 1000:.1f} ms")


if __name__ == "__main__":
    main()
