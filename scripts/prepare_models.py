"""Prepare local bf16 copies of the benchmark models.

Why: on a 1 GiB-RAM machine, mmap'ing gpt2's 548 MB fp32 safetensors file on
top of an already-resident draft model exceeds the cgroup commit limit and the
load dies with "unable to mmap ... Cannot allocate memory". Converting the
checkpoints to bf16 once (each in its own subprocess, so peak RSS stays low)
shrinks the mmap to ~250 MB and makes every later run lighter and faster.

    python scripts/prepare_models.py                 # gpt2 + distilgpt2
    python scripts/prepare_models.py gpt2            # just one
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

MODELS = ["gpt2", "distilgpt2"]


def convert_one(name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from specmatch.models import LOCAL_MODEL_DIR

    out_dir = LOCAL_MODEL_DIR / (name.replace("/", "--") + "-bf16")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] loading fp32 -> converting to bf16 ...")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(name).save_pretrained(out_dir)
    size_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
    print(f"[{name}] saved to {out_dir} ({size_mb:.0f} MB)")


def main():
    names = sys.argv[1:] or MODELS
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        convert_one(sys.argv[2])
        return
    # one subprocess per model: peak RSS stays ~ (torch overhead + one model)
    for name in names:
        subprocess.run([sys.executable, __file__, "--one", name], check=True)
    print("done. run scripts/download_models.py or any benchmark to use them.")


if __name__ == "__main__":
    main()
