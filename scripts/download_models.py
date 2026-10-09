"""Download (and cache) the models SpecMatch uses.

    python scripts/download_models.py                 # default set
    python scripts/download_models.py gpt2 distilgpt2 # explicit
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specmatch.models import load_model, load_tokenizer  # noqa: E402
from specmatch.config import resolve_device, resolve_dtype  # noqa: E402

DEFAULTS = ["gpt2", "distilgpt2", "sshleifer/tiny-gpt2"]


def main():
    names = sys.argv[1:] or DEFAULTS
    device = resolve_device()
    dtype = resolve_dtype("auto", device)
    for name in names:
        print(f"downloading {name} ...")
        load_tokenizer(name)
        model = load_model(name, dtype, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ok: {n_params / 1e6:.1f}M params, dtype {dtype}")


if __name__ == "__main__":
    main()
