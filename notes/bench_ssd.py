"""Benchmark & visualization suite for Mamba-2 SSD implementations.

Supports device-adaptive presets (``cpu`` vs ``cuda``) via ``--device``:
    python notes/bench_ssd.py --device auto
    python notes/bench_ssd.py --device cpu
    python notes/bench_ssd.py --device cuda
    python notes/bench_ssd.py --device all

Saves plots + JSON results under ``notes/images/`` and bundles them into
``notes/images/ssd_bench_outputs.zip``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
import zipfile

import matplotlib.pyplot as plt
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import FlopCounterMode

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamba_scratch.ssd import ssd_chunked, ssd_quadratic, ssd_sequential


OUT_DIR = REPO_ROOT / "notes" / "images"


@dataclass(frozen=True)
class BenchConfig:
    device: str
    dtype_name: str
    warmup: int
    iters: int
    # Exp 1: sweep (c, l) at fixed L
    exp1_b: int
    exp1_L: int
    exp1_h: int
    exp1_p: int
    exp1_n: int
    exp1_ls: tuple[int, ...]
    # Exp 2: sweep (P, h) at fixed d_inner = h * P
    exp2_b: int
    exp2_L: int
    exp2_d_inner: int
    exp2_l: int
    exp2_n: int
    exp2_ps: tuple[int, ...]

    @property
    def dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype_name)


CONFIGS: dict[str, BenchConfig] = {
    "cpu": BenchConfig(
        device="cpu",
        dtype_name="float32",
        warmup=1,
        iters=2,
        exp1_b=1,
        exp1_L=2048,
        exp1_h=2,
        exp1_p=64,
        exp1_n=64,
        exp1_ls=(4, 8, 16, 32, 64, 128, 256, 512, 1024),
        exp2_b=1,
        exp2_L=1024,
        exp2_d_inner=128,
        exp2_l=64,
        exp2_n=64,
        exp2_ps=(1, 4, 16, 32, 64, 128),
    ),
    "cuda": BenchConfig(
        device="cuda",
        dtype_name="float16",
        warmup=3,
        iters=10,
        exp1_b=4,
        exp1_L=8192,
        exp1_h=16,
        exp1_p=64,
        exp1_n=64,
        exp1_ls=(16, 32, 64, 128, 256, 512, 1024),
        exp2_b=2,
        exp2_L=4096,
        exp2_d_inner=1024,
        exp2_l=256,
        exp2_n=64,
        exp2_ps=(1, 4, 16, 32, 64, 128, 256),
    ),
}


class ExactMemoryTrafficMode(TorchDispatchMode):
    """Count exact bytes read (inputs) + written (outputs) across every ATen op."""

    def __init__(self) -> None:
        super().__init__()
        self.bytes_read = 0
        self.bytes_written = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        flat_args, _ = tree_flatten((args, kwargs))
        seen_in: set[int] = set()
        for a in flat_args:
            if isinstance(a, torch.Tensor) and id(a) not in seen_in:
                seen_in.add(id(a))
                self.bytes_read += a.numel() * a.element_size()

        out = func(*args, **kwargs)
        flat_out, _ = tree_flatten(out)
        seen_out: set[int] = set()
        for o in flat_out:
            if isinstance(o, torch.Tensor) and id(o) not in seen_out:
                seen_out.add(id(o))
                self.bytes_written += o.numel() * o.element_size()
        return out


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def measure_runtime_and_peak_mb(
    fn, X, A, B, C, block_len: int, device: str, warmup: int, iters: int
) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(warmup):
            fn(X, A, B, C, block_len)
        _sync(device)

        t0 = time.perf_counter()
        for _ in range(iters):
            fn(X, A, B, C, block_len)
        _sync(device)
        ms = (time.perf_counter() - t0) / iters * 1000.0

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            fn(X, A, B, C, block_len)
            _sync(device)
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
                fn(X, A, B, C, block_len)
            peak_bytes = 0
            cur_bytes = 0
            for evt in prof.events():
                cur_bytes += evt.cpu_memory_usage
                if cur_bytes > peak_bytes:
                    peak_bytes = cur_bytes
            peak_mb = peak_bytes / (1024 * 1024)
    return ms, peak_mb


def measure_unfused_exact(
    fn, X, A, B, C, block_len: int, device: str, warmup: int, iters: int
) -> tuple[int, int, float, float]:
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            fn(X, A, B, C, block_len)
        flops = fc.get_total_flops()

        with ExactMemoryTrafficMode() as mem:
            fn(X, A, B, C, block_len)
        io_bytes = mem.bytes_read + mem.bytes_written

        for _ in range(warmup):
            fn(X, A, B, C, block_len)
        _sync(device)

        t0 = time.perf_counter()
        for _ in range(iters):
            fn(X, A, B, C, block_len)
        _sync(device)
        ms = (time.perf_counter() - t0) / iters * 1000.0

    ai = flops / io_bytes if io_bytes > 0 else 0.0
    return flops, io_bytes, ai, ms


def run_exp1(cfg: BenchConfig) -> list[dict]:
    X = torch.randn(cfg.exp1_b, cfg.exp1_L, cfg.exp1_h, cfg.exp1_p, device=cfg.device, dtype=cfg.dtype)
    A = -torch.rand(cfg.exp1_b, cfg.exp1_L, cfg.exp1_h, device=cfg.device, dtype=cfg.dtype) * 0.2
    B = torch.randn(cfg.exp1_b, cfg.exp1_L, cfg.exp1_h, cfg.exp1_n, device=cfg.device, dtype=cfg.dtype)
    C = torch.randn(cfg.exp1_b, cfg.exp1_L, cfg.exp1_h, cfg.exp1_n, device=cfg.device, dtype=cfg.dtype)

    ls = list(cfg.exp1_ls)
    cs = [cfg.exp1_L // l for l in ls]
    mss, mbs, records = [], [], []

    print(
        f"=== Experiment 1 [{cfg.device.upper()}]: Sweep (c, l) at fixed "
        f"b={cfg.exp1_b}, L={cfg.exp1_L}, h={cfg.exp1_h}, p={cfg.exp1_p}, n={cfg.exp1_n} ==="
    )
    print(f"{'l':>6} {'c':>6} | {'ms':>8} {'peak(MB)':>10}")
    print("-" * 36)
    for l, c in zip(ls, cs):
        ms, mb = measure_runtime_and_peak_mb(
            ssd_chunked, X, A, B, C, l, cfg.device, cfg.warmup, cfg.iters
        )
        mss.append(ms)
        mbs.append(mb)
        records.append({"l": l, "c": c, "ms": round(ms, 4), "peak_mb": round(mb, 4)})
        print(f"{l:>6} {c:>6} | {ms:>8.2f} {mb:>10.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    x_idx = list(range(len(ls)))
    x_labels = [f"{l}\n(c={c})" for l, c in zip(ls, cs)]

    ax = axes[0]
    ax.plot(x_idx, mss, marker="o", linewidth=2, color="#2563eb")
    min_i = mss.index(min(mss))
    ax.scatter([min_i], [mss[min_i]], color="#dc2626", zorder=5, s=70, label=f"Min: l={ls[min_i]} ({mss[min_i]:.2f} ms)")
    ax.set_xticks(x_idx, x_labels)
    ax.set_xlabel("block_len l (and num chunks c = L / l)")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title(f"Exp 1 ({cfg.device.upper()}): Runtime vs (c, l) at L={cfg.exp1_L}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    ax = axes[1]
    ax.plot(x_idx, mbs, marker="s", linewidth=2, color="#059669")
    min_m = mbs.index(min(mbs))
    ax.scatter([min_m], [mbs[min_m]], color="#dc2626", zorder=5, s=70, label=f"Min: l={ls[min_m]} ({mbs[min_m]:.2f} MB)")
    ax.set_xticks(x_idx, x_labels)
    ax.set_xlabel("block_len l (and num chunks c = L / l)")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title(f"Exp 1 ({cfg.device.upper()}): Peak Memory vs (c, l) at L={cfg.exp1_L}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    out_path = OUT_DIR / f"ssd_exp1_cl_sweep_{cfg.device}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}\n")
    return records


def run_exp2(cfg: BenchConfig) -> list[dict]:
    ps = list(cfg.exp2_ps)
    impls = [("sequential", ssd_sequential), ("quadratic", ssd_quadratic), ("chunked", ssd_chunked)]
    results = {name: {"ai": [], "ms": []} for name, _ in impls}
    records: list[dict] = []

    print(
        f"=== Experiment 2 [{cfg.device.upper()}]: Sweep (P, h) at fixed "
        f"d_inner={cfg.exp2_d_inner}, L={cfg.exp2_L}, l={cfg.exp2_l}, N={cfg.exp2_n} ==="
    )
    print(f"{'Impl':>10} {'P':>4} {'h':>4} | {'FLOPs(M)':>10} {'IO(MB)':>10} {'AI(F/B)':>8} {'ms':>8}")
    print("-" * 68)

    for P in ps:
        h = cfg.exp2_d_inner // P
        X = torch.randn(cfg.exp2_b, cfg.exp2_L, h, P, device=cfg.device, dtype=cfg.dtype)
        A = -torch.rand(cfg.exp2_b, cfg.exp2_L, h, device=cfg.device, dtype=cfg.dtype) * 0.2
        B = torch.randn(cfg.exp2_b, cfg.exp2_L, h, cfg.exp2_n, device=cfg.device, dtype=cfg.dtype)
        C = torch.randn(cfg.exp2_b, cfg.exp2_L, h, cfg.exp2_n, device=cfg.device, dtype=cfg.dtype)

        for name, fn in impls:
            try:
                flops, io_bytes, ai, ms = measure_unfused_exact(
                    fn, X, A, B, C, cfg.exp2_l, cfg.device, cfg.warmup, cfg.iters
                )
                results[name]["ai"].append(ai)
                results[name]["ms"].append(ms)
                records.append({
                    "impl": name,
                    "P": P,
                    "h": h,
                    "flops_m": round(flops / 1e6, 4),
                    "io_mb": round(io_bytes / 1e6, 4),
                    "ai": round(ai, 4),
                    "ms": round(ms, 4),
                    "oom": False,
                })
                print(f"{name:>10} {P:>4} {h:>4} | {flops/1e6:>10.2f} {io_bytes/1e6:>10.2f} {ai:>8.2f} {ms:>8.2f}")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                results[name]["ai"].append(float("nan"))
                results[name]["ms"].append(float("nan"))
                records.append({"impl": name, "P": P, "h": h, "oom": True})
                print(f"{name:>10} {P:>4} {h:>4} | {'OOM':>10} {'OOM':>10} {'OOM':>8} {'OOM':>8}")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), dpi=150)
    x_idx = list(range(len(ps)))
    x_labels = [f"P={p}\n(h={cfg.exp2_d_inner // p})" for p in ps]
    colors = {"sequential": "#dc2626", "quadratic": "#d97706", "chunked": "#2563eb"}
    markers = {"sequential": "o", "quadratic": "^", "chunked": "s"}

    ax = axes[0]
    for name, _ in impls:
        ax.plot(x_idx, results[name]["ai"], marker=markers[name], linewidth=2, color=colors[name], label=name)
    ax.set_xticks(x_idx, x_labels)
    ax.set_xlabel(f"headdim P (and heads h = {cfg.exp2_d_inner} / P)")
    ax.set_ylabel("Arithmetic Intensity (FLOPs / Byte)")
    ax.set_title(f"Exp 2 ({cfg.device.upper()}): Arithmetic Intensity vs P (d_inner={cfg.exp2_d_inner})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    ax = axes[1]
    for name, _ in impls:
        ax.plot(x_idx, results[name]["ms"], marker=markers[name], linewidth=2, color=colors[name], label=name)
    ax.set_yscale("log")
    ax.set_xticks(x_idx, x_labels)
    ax.set_xlabel(f"headdim P (and heads h = {cfg.exp2_d_inner} / P)")
    ax.set_ylabel("Measured Runtime (ms, log)")
    ax.set_title(f"Exp 2 ({cfg.device.upper()}): Runtime vs P (d_inner={cfg.exp2_d_inner})")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    out_path = OUT_DIR / f"ssd_exp2_p_sweep_{cfg.device}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}\n")
    return records


def run_for_device(device: str) -> None:
    cfg = CONFIGS[device]
    exp1_records = run_exp1(cfg)
    exp2_records = run_exp2(cfg)

    hw_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    payload = {
        "device": device,
        "hardware": hw_name,
        "config": asdict(cfg),
        "exp1": exp1_records,
        "exp2": exp2_records,
    }
    json_path = OUT_DIR / f"ssd_bench_{device}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {json_path}\n")


def package_outputs() -> Path:
    zip_path = OUT_DIR / "ssd_bench_outputs.zip"
    files = sorted(list(OUT_DIR.glob("ssd_*.png")) + list(OUT_DIR.glob("ssd_*.json")))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    print(f"Packaged {len(files)} files into {zip_path}")
    return zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Mamba-2 SSD implementations.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "all"),
        default="auto",
        help="Device preset to run ('auto' picks cuda if available else cpu; 'all' runs both).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        devices = ["cuda" if torch.cuda.is_available() else "cpu"]
    elif args.device == "all":
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        if not torch.cuda.is_available():
            print("Warning: CUDA not available; running CPU preset only.")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available.")
        devices = [args.device]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    for dev in devices:
        run_for_device(dev)
    package_outputs()


if __name__ == "__main__":
    main()
