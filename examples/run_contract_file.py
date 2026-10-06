"""Run one contract from a contracts file on your own documents (real model, bring your own key).

    export GROQ_API_KEY=...            # PowerShell: $env:GROQ_API_KEY="..."
    python examples/run_contract_file.py contracts.json rbc4-02397 page1.txt page2.txt

The file may be a single contract or a list; the second argument picks one by contract_id (or by
position, e.g. 0). Both the native contract format and ``cglc-contract-v1`` are accepted. Documents
are plain-text files. Provider/model: GROQ_API_KEY (default model openai/gpt-oss-120b) or
ANTHROPIC_API_KEY with PROVIDER=anthropic. Nothing is sent anywhere except to that provider.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cglc import service  # noqa: E402


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    path, pick, doc_paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else [data]
    chosen = next((c for c in items if isinstance(c, dict) and c.get("contract_id") == pick), None)
    if chosen is None and pick.isdigit() and int(pick) < len(items):
        chosen = items[int(pick)]
    if chosen is None:
        ids = [c.get("contract_id") for c in items if isinstance(c, dict)]
        print(f"No contract {pick!r} in {path}. Available: {ids}")
        return 2

    provider = os.environ.get("PROVIDER", "groq")
    key = os.environ.get("ANTHROPIC_API_KEY" if provider == "anthropic" else "GROQ_API_KEY", "")
    if not key and provider != "offline":
        print(f"Set {'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'GROQ_API_KEY'} first.")
        return 2
    docs = [{"name": Path(p).stem, "text": Path(p).read_text(encoding="utf-8")} for p in doc_paths]
    status, r = service.handle_run(json.dumps({"provider": provider, "api_key": key,
                                               "model": os.environ.get("MODEL", ""),
                                               "contract": chosen, "documents": docs}).encode())
    if not r.get("ok"):
        print(f"HTTP {status}: {r.get('error')}")
        return 1
    print("DECISION:", r["decision"], f"({r['mode']} mode, {r['spend']['elapsed_seconds']}s)")
    if r["decision"] != "ALLOW_FINALIZE":
        for why in r["reasons"]:
            print("  not approved because:", why)
        if r["blocked_condition"]:
            print("  blocked condition:", r["blocked_condition"])
    print("CONTRACT SOURCE:", r["contract"]["provenance"])
    for e in r["evidence"]:
        print(f"  [{e['status']:14}] {e['proposition'][:100]}  ({len(e['receipts'])} quotes)")
    print("\nANSWER" + ("" if r["decision"] == "ALLOW_FINALIZE" else " (UNAPPROVED DRAFT)") + ":\n" + r["draft"])
    for c in r["checkpoints"]:
        print(f"  checkpoint {c['id']}: {c['decision']}  {c['gates'].get('reasons') or ''}")
    for n in r["contract"]["notes"]:
        print("  note:", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
