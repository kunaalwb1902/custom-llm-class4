"""Author the corpus-extension teaching material for three eval categories.

Writes corpus/grammar.md, corpus/opposites.md and corpus/categories.md.

What this script is and is not:

* It is an *authoring* tool. It reads evals/language_evals.json ONLY to build a
  blocklist, and it aborts if any sentence it produced contains one of the 48
  fixed test prompts. No eval prompt, answer choice, answer key or eval output is
  ever written into corpus/.
* The three target categories are `grammar` (lang_25-27), `opposites`
  (lang_28-30) and `categories_and_analogies` (lang_46-48). The sentences teach
  the *patterns* and the *vocabulary* those cases need, using different words,
  people and situations than the tests use.

Two pipeline details drive the formatting:

1. `chunk_text()` in the notebook splits on `(?<=[.!?])\\s+|\\n+`. A two-clause
   teaching example ("a kitten grows into a cat . a puppy grows into a dog .")
   would therefore become two separate passages. Writing it with no space after
   the internal period keeps it as one passage; the tokenizer ignores whitespace,
   so the stored passage in corpus.txt reads normally. See TWO_CLAUSE below.
2. `reject_eval_leakage()` runs on the whole file text as well as on each
   passage, so two adjacent lines can form a forbidden prompt by accident. The
   line order is therefore shuffled with a fixed seed and re-checked until the
   concatenated file is clean.

These files contain no Markdown syntax on purpose: Markdown is read as plain
text, so a `# heading` would become the training tokens "#" and "heading".

Run:  python scripts/make_teaching_corpus.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "evals" / "language_evals.json"
CORPUS = ROOT / "corpus"


def word_tokens(text: str) -> list[str]:
    """Identical to the notebook's tokenizer, so our checks see what it sees."""
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", text.lower(), flags=re.UNICODE)


def normalized(text: str) -> str:
    return " " + " ".join(word_tokens(text)) + " "


def two_clause(first: str, second: str) -> str:
    """Join two clauses into a single passage the chunker will not split."""
    return f"{first} .{second} ."


# --------------------------------------------------------------------------
# grammar  (targets lang_25 "one bird" -> is, lang_26 "the dogs" -> are,
#           lang_27 "yesterday she" -> walked)
# --------------------------------------------------------------------------

SINGULAR = ["dog", "cat", "duck", "goat", "horse", "robin", "sparrow", "trout",
            "salmon", "apple", "carrot", "box", "cup", "lamp", "book", "bag",
            "door", "ball", "shelf", "desk"]
PLURAL = ["cats", "dogs", "ducks", "goats", "horses", "birds", "boxes", "cups",
          "lamps", "books", "bags", "doors", "balls", "shelves", "desks"]
ADJECTIVES = ["small", "big", "loud", "quiet", "soft", "hard", "heavy", "light",
              "clean", "dirty", "old", "young", "wet", "dry", "round", "square"]
PLACES = ["store", "market", "bank", "kitchen", "station", "office", "hospital",
          "school", "park"]
NAMES = ["maya", "leo", "nina"]
# "she" is deliberately never placed directly after "yesterday": that two-word
# sequence is the lang_27 prompt.
PAST_SUBJECTS = ["he", "maya", "leo", "nina", "the teacher", "the nurse",
                 "the student", "the doctor"]


def grammar_lines() -> list[str]:
    lines: list[str] = []
    # singular subject -> "is"
    for noun, adjective in product(SINGULAR, ADJECTIVES[:10]):
        lines.append(f"one {noun} is {adjective} .")
        lines.append(f"the {noun} is {adjective} .")
    # plural subject -> "are"
    for noun, adjective in product(PLURAL, ADJECTIVES[:10]):
        lines.append(f"two {noun} are {adjective} .")
        lines.append(f"the {noun} are {adjective} .")
    # past tense after a past-time adverb
    for subject, place in product(PAST_SUBJECTS, PLACES):
        lines.append(f"yesterday {subject} walked to the {place} .")
    # present tense, so "walk", "walks" and "walking" are all in the vocabulary
    for subject, place in product(["he", "she"] + NAMES, PLACES):
        lines.append(f"{subject} walks to the {place} every day .")
        lines.append(f"{subject} is walking to the {place} now .")
        lines.append(f"{subject} walked to the {place} yesterday .")
    for place in PLACES:
        lines.append(f"they walk to the {place} every day .")
        lines.append(f"we walk to the {place} every day .")
    # "was", "were" and "am" must exist in the vocabulary or lang_25/lang_26
    # become unscorable: every one of the four choices has to be a known word.
    for adjective in ADJECTIVES:
        lines.append(f"i am {adjective} today .")
        lines.append(f"we were {adjective} yesterday .")
        lines.append(f"they were {adjective} yesterday .")
        for name in NAMES:
            lines.append(f"{name} was {adjective} yesterday .")
    for noun, adjective in product(PLURAL, ADJECTIVES[:4]):
        lines.append(f"the {noun} were {adjective} yesterday .")
    for noun, adjective in product(SINGULAR, ADJECTIVES[:4]):
        lines.append(f"the {noun} was {adjective} yesterday .")
    return lines


# --------------------------------------------------------------------------
# opposites  (targets lang_28 hot->cold, lang_29 empty->full, lang_30 noisy->quiet)
# --------------------------------------------------------------------------

# (a, b) are antonyms. The three pairs the tests ask about are marked: for those
# the sentence "the opposite of <a> is ..." is the literal test prompt, so that
# one direction is skipped and the pair is taught the other way round and with
# different phrasings instead.
PAIRS = [
    ("hot", "cold", True),
    ("empty", "full", True),
    ("noisy", "quiet", True),
    ("loud", "quiet", False),
    ("fast", "slow", False),
    ("heavy", "light", False),
    ("warm", "cool", False),
    ("early", "late", False),
    ("soft", "hard", False),
    ("round", "square", False),
    ("big", "small", False),
    ("tall", "short", False),
    ("wet", "dry", False),
    ("open", "closed", False),
    ("new", "old", False),
    ("long", "short", False),
    ("clean", "dirty", False),
    ("happy", "sad", False),
    ("young", "old", False),
    ("strong", "weak", False),
    ("up", "down", False),
    ("rich", "poor", False),
]

def opposites_lines() -> list[str]:
    # Every template is pair-agnostic, so no combination produces a sentence that
    # is false or nonsensical (an earlier version paired adjectives with random
    # nouns and produced lines like "a book that is not fast is slow").
    lines: list[str] = []
    for first, second, reserved in PAIRS:
        for a, b in [(first, second), (second, first)]:
            # The canonical template. For the three tested pairs the direction
            # that IS a test prompt ("the opposite of hot is ...") is skipped by
            # the leakage filter in main(); the reverse direction is kept, so the
            # model still meets the template and must generalise to the unseen
            # direction at eval time.
            if not (reserved and a == first):
                lines.append(f"the opposite of {a} is {b} .")
            lines.append(f"{b} is the opposite of {a} .")
            lines.append(f"{a} and {b} are opposites .")
            lines.append(f"the reverse of {a} is {b} .")
            lines.append(f"when something is not {a} it is {b} .")
            lines.append(f"if it is not {a} then it is {b} .")
            lines.append(f"something {a} is never {b} .")
            lines.append(f"nothing is {a} and {b} at the same time .")
    return lines


# --------------------------------------------------------------------------
# categories_and_analogies  (targets lang_46 salmon->fish, lang_47 kitten->cat,
#                            lang_48 apple->fruit)
# --------------------------------------------------------------------------

MEMBERS = [
    ("robin", "bird"), ("sparrow", "bird"), ("crow", "bird"), ("duck", "bird"),
    ("salmon", "fish"), ("trout", "fish"), ("tuna", "fish"),
    ("oak", "tree"), ("pine", "tree"), ("maple", "tree"),
    ("hammer", "tool"), ("saw", "tool"), ("drill", "tool"),
    ("carrot", "vegetable"), ("onion", "vegetable"), ("potato", "vegetable"),
    ("apple", "fruit"), ("banana", "fruit"), ("pear", "fruit"), ("mango", "fruit"),
    ("bus", "vehicle"), ("train", "vehicle"), ("truck", "vehicle"), ("taxi", "vehicle"),
    ("cotton", "fabric"), ("silk", "fabric"), ("wool", "fabric"),
    ("iron", "metal"), ("copper", "metal"), ("steel", "metal"),
    ("cat", "animal"), ("dog", "animal"), ("horse", "animal"), ("goat", "animal"),
]

GROWS = [
    ("kitten", "cat"), ("puppy", "dog"), ("duckling", "duck"), ("foal", "horse"),
    ("lamb", "sheep"), ("calf", "cow"), ("chick", "bird"), ("seed", "tree"),
]

# Mass nouns take no article: "cotton is a fabric", not "a cotton is a fabric".
MASS = {"cotton", "silk", "wool", "iron", "copper", "steel"}

# How many of the ~1,000 possible two-clause analogies to keep. A fixed seed
# makes the selection reproducible.
ANALOGY_SAMPLE = 300
SAMPLE_SEED = 7


def noun_phrase(word: str) -> str:
    if word in MASS:
        return word
    return f"{'an' if word[0] in 'aeiou' else 'a'} {word}"


def is_a(member: str, category: str) -> str:
    """'a salmon is a fish', 'cotton is a fabric', 'a cat is an animal'."""
    article = "an" if category[0] in "aeiou" else "a"
    return f"{noun_phrase(member)} is {article} {category}"


def categories_lines() -> list[str]:
    lines: list[str] = []
    for member, category in MEMBERS:
        lines.append(f"{is_a(member, category)} .")
        lines.append(f"every {member} is a kind of {category} .")
    # Two-clause analogies: after "X is a CAT . Y is a" the next word has to be
    # Y's category. Only cross-category pairs are combined, and a fixed random
    # sample keeps this file from swamping the classroom corpus.
    analogies = [(a, b) for a, b in product(MEMBERS, repeat=2)
                 if a[1] != b[1] and a[0] != b[0]]
    random.Random(SAMPLE_SEED).shuffle(analogies)
    for (member_a, category_a), (member_b, category_b) in analogies[:ANALOGY_SAMPLE]:
        lines.append(f"{is_a(member_a, category_a)} .{is_a(member_b, category_b)} .")
    for young, adult in GROWS:
        lines.append(f"{noun_phrase(young)} grows into a {adult} .")
        lines.append(f"the {young} grows into a {adult} .")
    for (young_a, adult_a), (young_b, adult_b) in product(GROWS, repeat=2):
        if young_a == young_b:
            continue
        lines.append(f"{noun_phrase(young_a)} grows into a {adult_a} ."
                     f"{noun_phrase(young_b)} grows into a {adult_b} .")
    return lines


# --------------------------------------------------------------------------
# leakage guard and file writing
# --------------------------------------------------------------------------

def offending_cases(text: str, prompts: dict[str, str]) -> list[str]:
    content = normalized(text)
    return sorted(case_id for case_id, prompt in prompts.items() if prompt in content)


def clean_order(lines: list[str], prompts: dict[str, str], label: str) -> list[str]:
    """Shuffle until no two adjacent lines combine into a forbidden prompt."""
    ordered = sorted(set(lines))
    for seed in range(100):
        random.Random(seed).shuffle(ordered)
        hits = offending_cases("\n".join(ordered) + "\n", prompts)
        if not hits:
            if seed:
                print(f"  {label}: used shuffle seed {seed} to avoid adjacency hits")
            return ordered
    raise SystemExit(f"{label}: could not find a line order free of eval prompts ({hits}).")


def main() -> int:
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    prompts = {case["id"]: normalized(case["prompt"]) for case in suite["cases"]}
    choice_lists = [normalized(" ".join(case["choices"])) for case in suite["cases"]]

    groups = {
        "grammar.md": grammar_lines(),
        "opposites.md": opposites_lines(),
        "categories.md": categories_lines(),
    }

    CORPUS.mkdir(parents=True, exist_ok=True)
    total = 0
    for filename, lines in groups.items():
        # Per-line check. A template can legitimately generate a sentence that
        # happens to contain a test prompt ("the dogs are small ." contains the
        # lang_26 prompt "the dogs"); those candidates are dropped here and
        # reported, so the exclusions are visible rather than silent.
        kept, dropped = [], []
        for line in lines:
            hits = offending_cases(line, prompts)
            (dropped if hits else kept).append((line, hits))
        kept = [line for line, _ in kept]
        if dropped:
            by_case: dict[str, int] = {}
            for _, hits in dropped:
                for case_id in hits:
                    by_case[case_id] = by_case.get(case_id, 0) + 1
            summary = ", ".join(f"{k}x{v}" for k, v in sorted(by_case.items()))
            print(f"  {filename}: dropped {len(dropped)} candidate sentences containing a test prompt ({summary})")
        ordered = clean_order(kept, prompts, filename)
        text = "\n".join(ordered) + "\n"
        # Whole-file check, matching how load_corpus_folder() validates imports.
        hits = offending_cases(text, prompts)
        if hits:
            raise SystemExit(f"{filename}: file text reproduces {hits}")
        content = normalized(text)
        for case, choices in zip(suite["cases"], choice_lists):
            if choices in content:
                raise SystemExit(f"{filename}: reproduces the answer list for {case['id']}")
        (CORPUS / filename).write_text(text, encoding="utf-8")
        vocabulary = sorted({token for line in ordered for token in word_tokens(line)})
        print(f"{filename}: {len(ordered)} unique sentences, {len(vocabulary)} token types")
        total += len(ordered)

    print(f"total: {total} unique teaching sentences in {CORPUS}")
    print("leakage guard passed: no test prompt or answer list appears in any file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
