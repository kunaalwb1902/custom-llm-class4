"""Cosine neighbours of a token before and after training, from checkpoint.json.

`embedding-viewer.html` shows the same thing interactively (load a run's
checkpoint.json into it). This script prints the numbers so they can be quoted in
the README and checked without opening a browser. Cosine similarity uses the full
64 dimensions; the viewer's 3D map is a PCA projection, so two tokens can look
close on screen while their cosine similarity is not.

Run:  python scripts/embedding_neighbours.py llm_runs/<run>/checkpoint.json customer salmon
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def neighbours(vectors: list[list[float]], vocabulary: list[str], word: str, count: int):
    index = vocabulary.index(word)
    normed = [unit(v) for v in vectors]
    target = normed[index]
    scored = []
    for other, vector in enumerate(normed):
        if other == index or vocabulary[other].startswith("<"):
            continue
        scored.append((sum(a * b for a, b in zip(target, vector)), vocabulary[other]))
    scored.sort(reverse=True)
    return scored[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("words", nargs="+")
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()

    data = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    vocabulary = data["vocabulary"]
    before, after = data["initial_embeddings"], data["weights"]["wte"]
    counts = data.get("token_counts") or [None] * len(vocabulary)

    for word in args.words:
        if word not in vocabulary:
            print(f"{word!r}: not in this run's vocabulary ({len(vocabulary)} types)")
            continue
        index = vocabulary.index(word)
        moved = math.sqrt(sum((a - b) ** 2 for a, b in zip(before[index], after[index])))
        print(f"\n{word!r}  id={index}  occurrences in corpus={counts[index]}  "
              f"vector moved {moved:.3f} in 64-D space")
        rows_before = neighbours(before, vocabulary, word, args.count)
        rows_after = neighbours(after, vocabulary, word, args.count)
        print(f"  {'before training':<34} {'after training':<34}")
        for (score_b, word_b), (score_a, word_a) in zip(rows_before, rows_after):
            print(f"  {word_b:<22}{score_b:>+7.3f}     {word_a:<22}{score_a:>+7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
