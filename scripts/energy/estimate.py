"""Turn measured model calls (from ``measure.py``) into energy and CO2 per question.

Two numbers per question, because the gateway's GPUs are shared and nobody can
meter a single request:

* **Estimate (EcoLogits method).** GPU energy per generated token from the
  EcoLogits fit over active parameters at batch size 64, times the number of
  GPUs the model needs; plus the non-GPU share of the server for the measured
  call time; times the data centre's PUE. Source: EcoLogits LLM inference
  methodology, https://ecologits.ai/latest/methodology/llm_inference/ .
  Like EcoLogits, it counts generated tokens only (reasoning included), not
  the prompt.
* **Upper bound.** Every call is assumed to occupy its GPUs alone at full
  power for its whole measured duration (no batching with other users).

    python -m scripts.energy.estimate energy_runs.jsonl

Every assumption is a flag; the defaults are documented in ``docs/energy.md``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

# EcoLogits: f_E(P_active, B) = alpha * exp(beta * B) * P_active + gamma, in Wh
# per output token and GPU, P_active in billions.
_ALPHA, _BETA, _GAMMA = 1.17e-6, -1.12e-2, 4.05e-5
_BATCH = 64
# Non-GPU server power and GPUs per server (p5.48xlarge reference), EcoLogits.
_SERVER_W, _SERVER_GPUS = 1200.0, 8
# Model memory = 1.2 x parameters x bytes per parameter (EcoLogits).
_MEMORY_OVERHEAD = 1.2


def gpus_needed(total_params_b: float, bits: int, gpu_memory_gb: float) -> int:
    """GPUs to hold the model, rounded up to a power of two as EcoLogits does."""
    memory_gb = _MEMORY_OVERHEAD * total_params_b * bits / 8
    return 2 ** math.ceil(math.log2(math.ceil(memory_gb / gpu_memory_gb)))


def wh_per_token_per_gpu(active_params_b: float) -> float:
    return _ALPHA * math.exp(_BETA * _BATCH) * active_params_b + _GAMMA


def estimate_call(call: dict, args: argparse.Namespace, gpus: int) -> tuple[float, float]:
    """(EcoLogits estimate, upper bound) in Wh at the socket, PUE included."""
    tokens = call.get("completion_tokens") or 0
    hours = call["seconds"] / 3600
    gpu = gpus * tokens * wh_per_token_per_gpu(args.active_params)
    server = hours * _SERVER_W * (gpus / _SERVER_GPUS) / _BATCH
    estimate = args.pue * (gpu + server)
    bound = args.pue * hours * (gpus * args.gpu_watts + _SERVER_W * gpus / _SERVER_GPUS)
    return estimate, bound


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path)
    parser.add_argument("--active-params", type=float, default=10.0, help="billions (Qwen3.5-122B-A10B)")
    parser.add_argument("--total-params", type=float, default=122.0, help="billions")
    parser.add_argument("--bits", type=int, default=8, help="weight precision (FP8)")
    parser.add_argument("--gpu-memory", type=float, default=96.0, help="GB (RTX PRO 6000 Blackwell)")
    parser.add_argument("--gpu-watts", type=float, default=600.0, help="max board power per GPU")
    parser.add_argument("--pue", type=float, default=1.5, help="data centre power usage effectiveness")
    parser.add_argument("--grid", type=float, default=406.0, help="g CO2e per kWh (UBA 2026, 2025 incl. upstream)")
    args = parser.parse_args()

    runs = [json.loads(line) for line in args.runs.read_text(encoding="utf-8").splitlines() if line.strip()]
    gpus = gpus_needed(args.total_params, args.bits, args.gpu_memory)

    rows = []
    for run in runs:
        parts = [estimate_call(call, args, gpus) for call in run["calls"]]
        rows.append(
            {
                "question": run["question"],
                "calls": len(run["calls"]),
                "out_tokens": sum((c.get("completion_tokens") or 0) for c in run["calls"]),
                "in_tokens": sum((c.get("prompt_tokens") or 0) for c in run["calls"]),
                "llm_seconds": sum(c["seconds"] for c in run["calls"]),
                "wh": sum(p[0] for p in parts),
                "wh_bound": sum(p[1] for p in parts),
            }
        )

    print(f"GPUs per model instance: {gpus}; {wh_per_token_per_gpu(args.active_params) * 1000:.4f} mWh "
          f"per output token and GPU; PUE {args.pue}; grid {args.grid:.0f} g CO2e/kWh\n")
    print("| Question | Calls | Tokens in | Tokens out | Model time (s) | Wh (estimate) | Wh (upper bound) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        print(f"| {r['question']} | {r['calls']} | {r['in_tokens']} | {r['out_tokens']} | "
              f"{r['llm_seconds']:.1f} | {r['wh']:.3f} | {r['wh_bound']:.1f} |")

    n = len(rows)
    total, bound = sum(r["wh"] for r in rows), sum(r["wh_bound"] for r in rows)
    print(f"\nQuestions: {n}")
    print(f"Output tokens per question: mean {mean(r['out_tokens'] for r in rows):.0f}, "
          f"median {median(r['out_tokens'] for r in rows):.0f}")
    print(f"Model time per question: mean {mean(r['llm_seconds'] for r in rows):.1f} s")
    print(f"Estimate:    {total / n:.3f} Wh and {total / n * args.grid / 1000:.3f} g CO2e per question; "
          f"{total:.2f} Wh and {total * args.grid / 1000:.2f} g CO2e for all {n}")
    print(f"Upper bound: {bound / n:.1f} Wh and {bound / n * args.grid / 1000:.1f} g CO2e per question; "
          f"{bound:.0f} Wh and {bound * args.grid / 1000:.0f} g CO2e for all {n}")


if __name__ == "__main__":
    main()
