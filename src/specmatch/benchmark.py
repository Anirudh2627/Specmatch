"""Benchmark harness.

Runs baseline and/or speculative generation over a prompt set with warmup and
repetitions, and writes one JSON record per (mode, prompt, rep) run. Model
loading happens outside the timed region - records measure generation only.

Fairness rules this harness enforces:
- identical GenConfig (tokens, temperature, top_p/top_k) for both modes
- identical prompt set, order, and repetition count
- identical device/dtype, models kept resident between runs
- one warmup generation per mode before timing
"""

import json
import platform
import time
from pathlib import Path
from typing import Optional

import torch

try:
    import resource  # POSIX only - peak-RSS reporting is skipped on Windows
except ImportError:
    resource = None

from .config import SpecConfig, resolve_device, resolve_dtype
from .generate import GenStats, baseline_generate
from .models import load_pair
from .speculative import speculative_generate


def peak_memory_mb() -> Optional[float]:
    """Process peak RSS in MB where the platform exposes it (Linux/macOS).
    Returns None on Windows; accelerator peak memory is recorded separately
    (cuda_peak_memory_mb / xpu_peak_memory_mb in the run records)."""
    if resource is None:
        return None
    try:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mb = rss_kb / 1024.0
        if platform.system() == "Darwin":  # macOS reports bytes
            mb = rss_kb / (1024.0 * 1024.0)
        return round(mb, 1)
    except Exception:
        return None


def accelerator_peak_mb(device: torch.device) -> Optional[float]:
    """Peak allocated MB on the active accelerator (CUDA or Intel XPU), else None."""
    try:
        if device.type == "cuda":
            return round(torch.cuda.max_memory_allocated() / 1e6, 1)
        if device.type == "xpu" and hasattr(torch, "xpu"):
            return round(torch.xpu.max_memory_allocated() / 1e6, 1)
    except Exception:
        pass
    return None


def load_prompts(path) -> list:
    """Prompt file format: [{"id": ..., "category": ..., "prompt": ...}, ...]"""
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data["prompts"]


def _stats_to_record(stats: GenStats, mode: str, prompt_entry: dict, cfg: SpecConfig,
                     device, dtype, draft_name: str, target_name: str, rep: int,
                     accel_peak_mb: Optional[float] = None) -> dict:
    return {
        "mode": mode,
        "draft_model": draft_name if mode == "speculative" else None,
        "target_model": target_name,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "prompt_id": prompt_entry["id"],
        "category": prompt_entry.get("category", "unknown"),
        "prompt": prompt_entry["prompt"],
        "prompt_tokens": stats.prompt_tokens,
        "new_tokens": stats.new_tokens,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "seed": cfg.seed,
        "rep": rep,
        "wall_time": round(stats.wall_time, 4),
        "ttft": round(stats.ttft, 4) if stats.ttft is not None else None,
        "tokens_per_sec": round(stats.tokens_per_sec, 2),
        "stop_reason": stats.stop_reason,
        # speculative-decoding accounting (zeros for baseline)
        "K": cfg.num_speculative_tokens if mode == "speculative" else None,
        "proposed": stats.proposed,
        "accepted": stats.accepted,
        "acceptance_rate": round(stats.acceptance_rate, 4) if stats.acceptance_rate is not None else None,
        "accepted_per_iter": stats.accepted_per_iter,
        "target_forwards": stats.target_forwards,
        "draft_forwards": stats.draft_forwards,
        "target_forward_tokens": stats.target_forward_tokens,
        "mean_tokens_per_target_forward": round(stats.mean_tokens_per_target_forward, 3),
        "peak_memory_mb": peak_memory_mb(),
        "accelerator_peak_memory_mb": accel_peak_mb,
    }


def run_benchmark(prompts: list, cfg: SpecConfig, mode: str = "both",
                  device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
                  reps: int = 3, warmup: bool = True,
                  tokenizer=None, draft=None, target=None,
                  progress: bool = True) -> list:
    """Run the benchmark. Pass preloaded models to reuse them across calls;
    otherwise they are loaded once here. Returns the list of records."""
    device = device or resolve_device()
    dtype = dtype or resolve_dtype(cfg.dtype, device)
    if target is None:
        tokenizer, draft_m, target_m = load_pair(cfg.draft_model, cfg.target_model, dtype, device)
    else:
        tokenizer, draft_m, target_m = tokenizer, draft, target

    modes = ["baseline", "speculative"] if mode == "both" else [mode]
    if "speculative" in modes and draft_m is None:
        raise ValueError("speculative mode needs a draft model")

    records = []
    warm_prompt = {"id": "warmup", "category": "warmup", "prompt": "Warmup run."}

    # Warm up every requested mode before collecting any measurements.
    if warmup:
        wcfg = SpecConfig(**{
            **cfg.to_dict(),
            "max_new_tokens": 8,
            "seed": 0,
        })

        for m in modes:
            if m == "baseline":
                baseline_generate(
                    target_m, tokenizer, warm_prompt["prompt"], wcfg, device
                )
            else:
                speculative_generate(
                    draft_m, target_m, tokenizer, warm_prompt["prompt"],
                    wcfg, device
                )

    # Interleave modes for each prompt/repetition. Alternate which mode runs
    # first to reduce systematic ordering bias.
    for prompt_index, prompt_entry in enumerate(prompts):
        for rep in range(reps):
            run_modes = list(modes)

            if len(run_modes) == 2 and (prompt_index + rep) % 2 == 1:
                run_modes.reverse()

            for m in run_modes:
                rcfg = SpecConfig(**cfg.to_dict())

                # Match seeds across modes for each prompt/repetition.
                rcfg.seed = (
                    None if cfg.seed is None else cfg.seed + 1000 * rep
                )

                # Reset peak-memory counters before each measured run.
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                elif device.type == "xpu" and hasattr(torch, "xpu"):
                    torch.xpu.reset_peak_memory_stats()

                t0 = time.perf_counter()

                if m == "baseline":
                    _, stats = baseline_generate(
                        target_m, tokenizer, prompt_entry["prompt"],
                        rcfg, device
                    )
                else:
                    _, stats = speculative_generate(
                        draft_m, target_m, tokenizer, prompt_entry["prompt"],
                        rcfg, device
                    )

                if progress:
                    dt = time.perf_counter() - t0
                    print(
                        f"  [{m:13s}] {prompt_entry['id']:22s} rep{rep} "
                        f"{stats.new_tokens:3d} tok in {dt:5.2f}s "
                        f"({stats.tokens_per_sec:5.1f} tok/s"
                        + (
                            f", acc {stats.acceptance_rate:.2f}"
                            if stats.acceptance_rate is not None else ""
                        )
                        + ")"
                    )

                accel_peak = accelerator_peak_mb(device)

                record = _stats_to_record(
                    stats, m, prompt_entry, rcfg, device, dtype,
                    cfg.draft_model, cfg.target_model, rep, accel_peak
                )

                if m == "baseline" and len(modes) > 1:
                    record["comparison_K"] = cfg.num_speculative_tokens

                records.append(record)

    return records


def save_records(records: list, path) -> None:
    """Append-friendly save: one JSON file per benchmark invocation containing
    a header block plus the raw records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(resolve_device()),
            "cpu_count": torch.get_num_threads(),
        },
        "records": records,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"saved {len(records)} records -> {path}")
