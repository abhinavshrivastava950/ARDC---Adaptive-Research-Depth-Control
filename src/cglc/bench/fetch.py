"""Fetch HotpotQA (distractor, validation) from Hugging Face and convert it
to the original HotpotQA JSON format that ``bench.datasets`` reads.

    python -m cglc.bench.fetch hotpotqa --out data/

Explicit, user-invoked only; the benchmark itself never downloads anything.
Needs ``pyarrow`` (pip install pyarrow). The download is ~27 MB.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

HF_HOTPOT = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
             "distractor/validation-00000-of-00001.parquet")


def hf_hotpot_to_original(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """HF columnar rows -> original ``{_id, question, answer, supporting_facts, context}``."""
    out = []
    for r in rows:
        sf = r["supporting_facts"]
        ctx = r["context"]
        out.append({
            "_id": r["id"], "question": r["question"], "answer": r["answer"],
            "type": r.get("type", ""), "level": r.get("level", ""),
            "supporting_facts": [[t, int(i)] for t, i in zip(sf["title"], sf["sent_id"])],
            "context": [[t, list(s)] for t, s in zip(ctx["title"], ctx["sentences"])],
        })
    return out


def fetch_hotpot(out_dir: Path) -> Path:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pyarrow is required: pip install pyarrow") from e
    out_dir.mkdir(parents=True, exist_ok=True)
    pq_path = out_dir / "hotpot_validation.parquet"
    if not pq_path.exists():
        print(f"downloading {HF_HOTPOT} ...")
        urllib.request.urlretrieve(HF_HOTPOT, pq_path)
    rows = pq.read_table(pq_path).to_pylist()
    dest = out_dir / "hotpot_dev_distractor_v1.json"
    dest.write_text(json.dumps(hf_hotpot_to_original(rows)), encoding="utf-8")
    print(f"{len(rows)} items -> {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["hotpotqa"])
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    fetch_hotpot(Path(a.out))


if __name__ == "__main__":
    main()
