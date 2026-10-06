"""Test-suite environment.

Retrieval picks a real embedding model automatically when one is installed (``fastembed``).
Tests must stay hermetic and fast, so unless a test sets it itself the lexical hashing
embedder is forced and no model is ever loaded or downloaded.
"""
import os

os.environ.setdefault("CGLC_EMBEDDER", "hashed")
