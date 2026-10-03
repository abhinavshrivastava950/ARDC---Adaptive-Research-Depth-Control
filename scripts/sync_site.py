"""Copy the demo site from demo/ (canonical) to docs/ (GitHub Pages) and public/ (Vercel).

    python scripts/sync_site.py          # copy
    python scripts/sync_site.py --check  # exit 1 if any copy is stale
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["index.html", "try.js", "try.css"]
TARGETS = ["docs", "public"]


def main() -> int:
    check = "--check" in sys.argv
    stale = []
    for t in TARGETS:
        (ROOT / t).mkdir(exist_ok=True)
        for f in FILES:
            src, dst = ROOT / "demo" / f, ROOT / t / f
            if not dst.exists() or dst.read_bytes() != src.read_bytes():
                stale.append(f"{t}/{f}")
                if not check:
                    shutil.copyfile(src, dst)
    if check and stale:
        print("stale copies (run scripts/sync_site.py):", ", ".join(stale))
        return 1
    print("in sync" if not stale or check else f"updated: {', '.join(stale)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
