"""Download wikitext-2-raw and pre-tokenize it for draft training.

Saves .models/wikitext2_tokens.bin (uint16 token ids, GPT-2 tokenizer).
Tokenizing once here keeps the training loop free of tokenizer overhead.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / ".models"


def download_text() -> str:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                           filename="wikitext-2-raw-v1/train-00000-of-00001.parquet")
    table = pq.read_table(path, columns=["text"])
    lines = table.column("text").to_pylist()
    # wikitext-raw: keep everything, normalize whitespace-only lines away
    text = "\n".join(l for l in lines if l.strip())
    return text


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("downloading wikitext-2-raw-v1 ...")
    text = download_text()
    print(f"corpus: {len(text)} chars")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    print("tokenizing in chunks (whole-corpus tokenization OOMs on 1 GiB) ...")
    out = OUT_DIR / "wikitext2_tokens.bin"
    total = 0
    CHUNK_CHARS = 200_000
    with open(out, "wb") as f:
        for i in range(0, len(text), CHUNK_CHARS):
            ids = tok(text[i: i + CHUNK_CHARS])["input_ids"]
            np.asarray(ids, dtype=np.uint16).tofile(f)
            total += len(ids)
            del ids
            if total % 500_000 < CHUNK_CHARS:
                print(f"  {total} tokens", flush=True)
    print(f"saved {total} tokens ({out.stat().st_size / 1e6:.1f} MB) -> {out}")


if __name__ == "__main__":
    main()
