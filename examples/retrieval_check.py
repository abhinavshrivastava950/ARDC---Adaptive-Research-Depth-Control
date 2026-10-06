"""Compare BM25 with the BM25 + embedding hybrid on a small synthetic corpus.

Not part of the test suite (it may download a model). Run it with an
embedding backend installed to see real numbers::

    pip install "fastembed>=0.3"        # or: pip install -e ".[embed]"
    python examples/retrieval_check.py          # BM25 vs hybrid, per query
    python examples/retrieval_check.py --sims   # also dense similarity of the
                                                # right chunk vs the best wrong one

The corpus is tiny and written by hand, the queries are paraphrases (little
word overlap with the answer, though not always none) plus two plain keyword
controls. Treat the output
as a smoke test of the plumbing, not as a retrieval benchmark: with so few
queries it says nothing about how either retriever scales.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cglc.embeddings import HashedNgramEmbedder, get_embedder  # noqa: E402
from cglc.retrieval import make_index  # noqa: E402

HANDBOOK = """
Northwind is a small company that builds scheduling software for clinics. This handbook describes how the company operates.

Creators who publish templates on the marketplace keep 88 percent of every sale. The remaining 12 percent is retained by Northwind to cover hosting and payment fees.

Employees may work from their own homes up to three days a week, provided their manager has signed off on the arrangement in writing.

If a client is unhappy within thirty days of purchase, the full price is returned to the original card. After thirty days only a prorated amount is paid back.

Personal records about patients are erased twenty-four months after the last appointment, unless a regulator requires the file to be kept longer.

Anyone who suspects an intruder has broken into a production system must page the on-call engineer immediately and write down what they saw. The engineer has one hour to open a formal incident.

New mothers and fathers receive sixteen weeks of fully paid time away from work, which can be taken in one block or split across the first year.

Login secrets for internal tools must be replaced every ninety days. Reusing any of the last ten secrets is blocked by the system, and sharing them in chat is a disciplinary matter.

The booking service promises to be reachable 99.9 percent of the time each calendar month. If Northwind misses that goal, affected clients receive a credit on their next invoice.

Travel and meals are reimbursed when receipts are uploaded within fourteen days. Alcohol is never reimbursed, and flights above 600 dollars need a manager's prior approval.

Every code change needs a second engineer to review it before merging. Reviewers should look at correctness first and style second, and they must not approve their own work.
""".strip()

POLICY = """
The support desk is staffed from 8 am to 8 pm on weekdays. Messages that arrive overnight are answered first thing the next morning.

Each engineer receives a yearly budget of 1,500 dollars for courses, books and conferences. Unused money does not roll over into the next year.

The Lisbon office is open to everyone at any time, while the Austin office requires a badge after 7 pm. Visitors must be signed in at the front desk.

Laptops are replaced every three years. Departing staff return their equipment within five working days of their last day.

Pricing is split into three plans: Starter, Team and Enterprise. The Enterprise plan includes single sign-on and a dedicated account manager.

API calls are limited to 600 requests per minute per key. Requests above the limit receive an error and should be retried with exponential backoff.

After every serious outage the team writes a blameless review within five days, listing the timeline, the root cause and the follow-up tasks.

New vendors need approval from finance and security before any contract is signed. The review covers pricing, data handling and exit terms.
""".strip()

DOCS = {"handbook.txt": HANDBOOK, "policy.txt": POLICY}

# (query, a phrase that must appear in the right chunk, kind)
QUERIES = [
    ("how is the money divided between content makers and the company", "88 percent", "paraphrase"),
    ("can staff work from home", "own homes", "paraphrase"),
    ("what if a customer wants their payment back", "full price is returned", "paraphrase"),
    ("how long do we keep health information about people we treat", "erased twenty-four months", "paraphrase"),
    ("what should I do if a hacker gets into our servers", "suspects an intruder", "paraphrase"),
    ("time off for new parents", "New mothers and fathers", "paraphrase"),
    ("how often must credentials be changed", "Login secrets", "paraphrase"),
    ("how fast must the service respond to requests", "99.9 percent", "paraphrase"),
    ("who keeps most of the earnings from the template store", "keep 88 percent", "paraphrase"),
    ("how many weeks of vacation after having a baby", "New mothers and fathers", "paraphrase"),
    ("rules for getting back the cost of business trips", "Travel and meals", "paraphrase"),
    ("limit on how often a program may call us per minute", "600 requests", "paraphrase"),
    ("refund thirty days", "full price is returned", "keyword control"),
    ("code change review before merging", "second engineer", "keyword control"),
]


def _rank(index, query: str, needle: str, k: int = 3):
    hits = index.search(query, k)
    for r, h in enumerate(hits, start=1):
        if needle in h.text:
            return r, hits
    return None, hits


def _show(label: str, rank, hits) -> str:
    top = hits[0].chunk_id if hits else "-"
    return f"{label}: {'rank %d' % rank if rank else 'miss':8s} (top1 {top})"


def main() -> None:
    show_sims = "--sims" in sys.argv
    emb, note = get_embedder()
    print(f"embedder: {note}")
    bm, _ = make_index(DOCS, "bm25")
    hy, info = make_index(DOCS, "hybrid", embedder=emb)
    print(f"hybrid info: {info}")
    hashed, hinfo = make_index(DOCS, "hybrid", embedder=HashedNgramEmbedder())
    print(f"chunks: {len(bm)}; hits counted within top 3\n")

    tally = {"bm25": 0, "hybrid": 0, "hashed hybrid": 0}
    for q, needle, kind in QUERIES:
        rb, hb = _rank(bm, q, needle)
        rh, hh = _rank(hy, q, needle)
        rs, hs = _rank(hashed, q, needle)
        tally["bm25"] += rb is not None
        tally["hybrid"] += rh is not None
        tally["hashed hybrid"] += rs is not None
        print(f"[{kind}] {q}")
        print("   " + _show("BM25         ", rb, hb))
        print("   " + _show("hybrid       ", rh, hh))
        print("   " + _show("hashed hybrid", rs, hs))
        if show_sims and info["used"] == "hybrid":
            qv = emb.embed_query(q)
            sims = [(emb.similarity(qv, v), c) for v, c in zip(hy._vecs, hy.chunks)]
            right = max(s for s, c in sims if needle in c.text)
            wrong = max(s for s, c in sims if needle not in c.text)
            print(f"   dense cosine: right chunk {right:.3f}, best wrong chunk {wrong:.3f}, "
                  f"cutoff {info['min_similarity']}")
    print("\nfound in top 3:", {k: f"{v}/{len(QUERIES)}" for k, v in tally.items()})
    if not emb.semantic:
        print("NOTE: no semantic backend is installed; 'hybrid' above used the lexical "
              "hashed embedder, so paraphrases are not expected to improve.")


if __name__ == "__main__":
    main()
