"""Deep probe-support module: shared machinery for FibQuant fidelity probes.

This is the substrate for the notebook-style probe scripts
(``scripts/key_recall.py``, ``scripts/multi_needle.py``, ``scripts/logit_kl.py``,
``scripts/eval_longbench.py``) -- the pieces that are identical across those
scripts because they answer the same questions ("is this filler sentence
unique and marker-free?", "does this marker configuration make sense?", "how
many tokens must generation budget for?", "which (name, spec) rows does this
run compare?"), not a general utility grab bag. Each script still owns its
own scenario-specific trial layout, generation loop, and metrics.

Owns:
  - deterministic unique filler-sentence generation (``SENTENCE_TEMPLATES``,
    ``SENTENCE_POOLS``, ``unique_filler_sentence``): every filler sentence is
    one template + iid slot words, astronomically unlikely to collide by
    construction and made impossible by a per-build ``used`` set.
  - continuation normalization (``normalize_continuation``) so marker
    matching is whitespace/case-insensitive without over-matching.
  - marker validation (``validate_markers``): empty, duplicate, and
    substring-overlap rejection, plus filler-pool reachability.
  - the verbose answer-token budget calculation (``required_answer_tokens``).
  - repeated (name, spec) config-matrix assembly from
    baseline/BITS/SPEC_PATHS (``build_spec_matrix``).
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from pathlib import Path

from .spec import FibQuantSpec

# --- filler: template-generated unique sentences -------------------------
# Every filler sentence is one template + iid slot words (no config, no
# files, no repeating passages). Pool products make exact collisions
# astronomically unlikely; a per-build `used` set (see unique_filler_sentence)
# makes it impossible regardless.
SENTENCE_TEMPLATES = [
    "The {animal} {verb} through the {place} {when}.",
    "A {animal} {verb} {adv} near the {place}.",
    "The {adj} {animal} {verb} {adv} {when}.",
    "The {occupation} {verb} {adv} beside a {noun}.",
    "A {occupation} {verb} through the {place} {when}.",
    "The {adj} {occupation} examined a {noun} {when}.",
    "The {occupation} carried a {noun} across the {place}.",
    "A {noun} {verb} {adv} on the {place} {when}.",
    "The {adj} {noun} was {adv} visible {when}.",
    "The {animal} watched the {adj} {noun} {when}.",
    "A {noun} hung {adv} over the {place}.",
    "The {adj} {occupation} saw a {noun} at the {place}.",
    "Every {occupation} at the {place} owns a {noun}.",
    "The {animal} hid {adv} behind the {noun} {when}.",
]

SENTENCE_POOLS: dict[str, list[str]] = {
    "animal": ["dog", "cat", "fox", "hawk", "heron", "otter", "badger", "lynx",
               "eel", "newt", "crow", "seal", "wolf", "deer", "moose", "bison",
               "gecko", "crane", "swan", "mole"],
    "verb": ["wandered", "crept", "peered", "dashed", "drifted", "clambered",
             "lingered", "scurried", "ambled", "soared", "trudged", "bounded",
             "veered", "nestled", "vanished", "circled", "rested", "paced"],
    "adv": ["quietly", "slowly", "steadily", "briefly", "softly", "gradually",
            "warily", "eagerly", "haphazardly", "carefully", "awkwardly"],
    "place": ["meadow", "harbor", "orchard", "courtyard", "thicket", "valley",
              "station", "street", "rooftop", "corridor", "garden", "plateau",
              "market", "tunnel", "attic", "promenade", "forest", "bridge"],
    "noun": ["lantern", "ledger", "crate", "saddle", "hatbox", "map", "basket",
             "kettle", "telescope", "compass", "bundle", "barrel", "mirror",
             "whistle"],
    "occupation": ["librarian", "baker", "blacksmith", "cartographer",
                   "apothecary", "cartwright", "beekeeper", "clockmaker",
                   "ferryman", "mason", "weaver", "oarsman", "lamplighter"],
    "adj": ["weathered", "mottled", "sturdy", "curious", "weary", "faded",
            "coiled", "glazed", "hollow", "bronze", "mossy", "threadbare"],
    "when": ["at dusk", "before dawn", "in the rain", "under a thin moon",
             "amid fog", "past midnight", "at first light", "in a stiff wind"],
}

_SLOT_RE = re.compile(r"\{(\w+)\}")

# Default per-marker generation frame ("Special token: {word}."), shared by
# every probe that plants a recallable marker in the filler.
DEFAULT_MARKER_FRAME = "Special token: {word}."


def normalize_continuation(text: str) -> str:
    """Lowercase and collapse whitespace before marker matching.

    Multi-token markers are phrases, and the model may insert extra spaces or
    line breaks between the words ("blue   whale") without forgetting them --
    those must not count as recall misses. (A word glued mid-word --
    "bluewhale" -- still does; keep markers to words/short phrases.)
    """
    return " ".join(text.lower().split())


def validate_markers(markers: Sequence[str], pools: dict[str, list[str]] = SENTENCE_POOLS) -> None:
    """Startup invariants for one or more marker phrases (raises ValueError).

    Multi-token markers are allowed; the single-token rule was dropped. What
    must hold for every marker:

      - non-empty (not whitespace-only)
      - no exact duplicate (case-insensitive) -- a repeated marker can never
        be told apart from itself in the continuation, so it adds nothing
        and signals a config mistake
      - no marker is a substring of another marker (self-overlap makes hits
        unassignable: which needle did a matched substring satisfy?)
      - no marker text is reachable from the filler pools (a marker that can
        legitimately appear in filler breaks the test; unique_filler_sentence
        also re-verifies every generated sentence against `avoid`)
    """
    marker_ls = [m.lower() for m in markers]

    seen: set[str] = set()
    for m, m_l in zip(markers, marker_ls):
        if not m_l.strip():
            raise ValueError(f"MARKER {m!r} is empty/whitespace-only")
        if m_l in seen:
            raise ValueError(f"MARKER {m!r} is a duplicate; markers must be unique")
        seen.add(m_l)

    for m, m_l in zip(markers, marker_ls):
        for other in marker_ls:
            if other != m_l and other in m_l:
                raise ValueError(f"MARKER {m!r} overlaps another marker ({other!r}); use non-substring words")
        for slot, words in pools.items():
            for w in words:
                if m_l in w.lower():
                    raise ValueError(f"MARKER {m!r} appears in filler pool '{slot}' as {w!r}")


def required_answer_tokens(
    tokenizer,
    markers: Sequence[str],
    slack: int,
    frame: str = DEFAULT_MARKER_FRAME,
) -> int:
    """Minimum generation budget for one trial: verbose marker listing + slack.

    The model may answer with full framing ("Special token: blue whale.") per
    marker rather than a bare word/list, so the budget is sized against that
    worst-case verbose listing -- one frame per marker, concatenated. A
    smaller budget risks a truncated continuation, which counts as a recall
    miss (an underestimate, never an overestimate).
    """
    verbose = "".join(f" {frame.format(word=m)}" for m in markers)
    return len(tokenizer.encode(verbose, add_special_tokens=False)) + slack


def minimal_answer_tokens(tokenizer, markers: Sequence[str]) -> int:
    """Smallest continuation that still contains every marker: the bare comma list.

    Sizes the generation budget (see multi_needle.py's answer-allowance
    constant): any response with every marker needs at least this many tokens,
    however terse; the budget is a generous multiple of it so the cap never
    cuts off a model that is still answering.
    """
    return len(tokenizer.encode(", ".join(markers), add_special_tokens=False))


def marker_hits(text: str, markers: Sequence[str]) -> list[bool]:
    """Word-boundary marker hits in one continuation (format-agnostic).

    The old check (substring of the whitespace-normalized continuation)
    probed response *format* as much as recall: "wren" inside "wrenches"
    counted as a hit, while any phrasing the strict form didn't anticipate
    (answers framed as prose, or with commas/case/punctuation variations)
    could score a miss even when the marker was recalled. This is an
    existence check: a marker counts as retrieved when it appears as a word
    (or, for multi-token markers, the phrase with whitespace collapsed to
    single spaces) anywhere in the continuation, in any case/format.
    Multi-token markers glued into one token ("bluewhale") still miss -- keep
    markers to words/short phrases (see validate_markers).
    """
    text = normalize_continuation(text)
    return [bool(re.search(rf"(?<!\w){re.escape(m.lower())}(?!\w)", text)) for m in markers]


def unique_filler_sentence(
    tokenizer,
    rng: random.Random,
    used: set[str],
    avoid: Sequence[str] = (),
    templates: Sequence[str] = SENTENCE_TEMPLATES,
    pools: dict[str, list[str]] = SENTENCE_POOLS,
) -> tuple[str, list[int]]:
    """One fresh, unique filler sentence as (text, token ids).

    Encoded with a leading space so the sentence-initial word is the
    *mid-text* token variant (" A" != "A" in BPE tokenizers), keeping
    concatenated filler properly spaced at the token level. `avoid` (marker
    words/phrases, already lowercased by the caller) are rejected as
    substrings of the decoded sentence, not just the raw template text, so a
    marker split across slot words is still caught.
    """
    for _ in range(100):
        template = rng.choice(templates)
        text = template.format(**{s: rng.choice(pools[s]) for s in _SLOT_RE.findall(template)})
        if text in used:
            continue
        ids = tokenizer.encode(" " + text, add_special_tokens=False)
        if avoid:
            dec = tokenizer.decode(ids).lower()
            if any(a in dec for a in avoid):
                continue
        used.add(text)
        return text, ids
    raise ValueError("could not generate a unique, marker-free filler sentence")


def build_spec_matrix(
    include_baseline: bool,
    bits: Sequence[int],
    spec_paths: Sequence[str | Path],
    d: int = 256,
    k: int = 4,
) -> list[tuple[str, FibQuantSpec | None]]:
    """Assemble the (name, spec) config matrix from baseline/BITS/SPEC_PATHS.

    One row per requested configuration, in a stable order: an optional fp16
    baseline (`spec=None`) first, then one `FibQuantSpec` per BITS entry (via
    `FibQuantSpec.from_bits(d, k, bits)` -- resolves a repo-relative default
    checkpoint path, only present where `models/` is checked out), then one
    per SPEC_PATHS entry (via `FibQuantSpec.from_path`, the Databricks-volume
    checkpoint case). Row names are stable ("baseline", "fq-b{bits}",
    "fq-N{n_levels}") so scripts can key result tables by name across
    configs. Does not validate that the matrix is non-empty -- callers raise
    with their own actionable message (probes differ on whether a baseline
    row is required).
    """
    specs: list[tuple[str, FibQuantSpec | None]] = []
    if include_baseline:
        specs.append(("baseline", None))
    for b in bits:
        specs.append((f"fq-b{b}", FibQuantSpec.from_bits(d=d, k=k, bits=b)))
    for p in spec_paths:
        spec = FibQuantSpec.from_path(p)
        specs.append((f"fq-N{spec.n_levels}", spec))
    return specs
