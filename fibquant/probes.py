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
#
# The space is sized for long runs: a 65k-token multi-needle depth consumes
# ~300k-1M sentences (more if the tokenizer merges aggressively), so pools
# are wide and every template family has >= ~200k combinations (most in the
# millions; total ~10^9). With the old 14-template space the small families
# (2.7k-27k combos) exhausted mid-run and unique_filler_sentence started
# burning its 100-attempt budget on families with no fresh sentences left.
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
    # --- expanded: more structures, more subjects, more voices -----------
    "In the {place}, a {animal} {verb} {adv}.",
    "At the {place}, the {occupation} {verb} a {noun}.",
    "A {animal} and a {bird} {verb} {adv} {when}.",
    "The {occupation}'s {noun} was {adv} {adj} {when}.",
    "There was a {adj} {noun} beside the {place} {when}.",
    "The {adj} {noun} leaned against the {place}.",
    "A {plant} grew {adv} beside the {place} {when}.",
    "The {tool} lay {adv} on the {place} {when}.",
    "A {vehicle} {verb} through the {place} {when}.",
    "The {weather} {verb} {adv} over the {place}.",
    "A {instrument} {verb} {adv} in the {place}.",
    "The {adj} {water} wound through the {place} {when}.",
    "Near the {place}, a {bird} {verb} {adv} {when}.",
    "Beyond the {place}, the {animal} {verb} {adv}.",
    "A {occupation} {verb} {adv} through the {place}.",
    "The {adj} {occupation} {verb} a {noun} {when}.",
    "Each {occupation} {verb} {adv} at the {place}.",
    "The {animal} slept {adv} beneath the {plant} {when}.",
    "A {bird} built a nest in the {adj} {plant} {when}.",
    "The {adj} {plant} {verb} {adv} {when}.",
    "A {vehicle} stood {adv} by the {place} {when}.",
    "The {occupation} hung a {noun} in the {place} {when}.",
    "A {tool} lay {adv} beside the {noun} {when}.",
    "The {weather} cleared {adv} over the {place} {when}.",
    "Behind the {place}, the {bird} {verb} {adv}.",
    "The {adj} {animal} {verb} toward the {place}.",
    "A {adj} {occupation} carried a {noun} {when}.",
    "The {noun} fell {adv} beside the {place} {when}.",
    "Every {occupation} {verb} at the {place} {when}.",
    "The {vehicle} rolled {adv} along the {place}.",
    "A {fish} {verb} {adv} through the {water}.",
    "The {adj} {noun} {verb} {adv} {when}.",
    "The {water} {verb} {adv} under the {place}.",
    "A {plant} stood {adv} in the {place} {when}.",
    "The {occupation} set a {noun} on the {place} {when}.",
    "A {animal} {verb} by the {water} {when}.",
    "The {tool} {verb} {adv} near the {noun}.",
    "A {bird} rested on the {adj} {plant} {when}.",
    "The {weather} hung {adv} above the {place} {when}.",
    "A {fish} circled the {noun} {adv} {when}.",
    "The {insect} {verb} {adv} beside the {plant}.",
    "A {insect} landed on the {adj} {place} {when}.",
    "The {adj} {insect} {verb} across the {water}.",
    "A {occupation} watched the {bird} {adv} {when}.",
    "A {vehicle} {verb} {adv} across the {place}.",
    "A {plant} {verb} {adv} beside the {water}.",
    "The {bird} {verb} {adv} above the {place} {when}.",
]

SENTENCE_POOLS: dict[str, list[str]] = {
    "animal": ["dog", "cat", "fox", "hawk", "heron", "otter", "badger", "lynx",
               "eel", "newt", "crow", "seal", "wolf", "deer", "moose", "bison",
               "gecko", "crane", "swan", "mole",
               "beaver", "ferret", "weasel", "marten", "stoat", "skunk",
               "raccoon", "squirrel", "chipmunk", "marmot", "porcupine",
               "hedgehog", "armadillo", "sloth", "tapir", "capybara", "peccary",
               "ibex", "chamois", "gazelle", "antelope", "caribou", "yak",
               "zebra", "okapi", "camel", "donkey", "pony", "goat", "sheep",
               "pig", "calf", "foal", "stag", "doe", "buck", "hind", "boar",
               "ram", "ewe", "kid", "colt", "puma", "cougar", "jaguar",
               "ocelot", "caracal", "serval", "mongoose", "meerkat",
               "wolverine", "mink", "fisher", "walrus", "manatee", "dolphin",
               "porpoise", "orca", "coyote", "jackal", "hyena", "dhole",
               "dingo", "leopard", "cheetah", "lion", "tiger", "vole",
               "shrew", "hamster", "gerbil", "chinchilla", "dormouse",
               "pika", "lemming", "hare", "rat", "mouse", "muskrat",
               "gopher", "groundhog", "woodchuck", "lemur", "gibbon",
               "macaque", "mandrill", "baboon", "marmoset", "tamarin",
               "coati", "kinkajou"],
    "bird": ["gull", "tern", "plover", "finch", "robin", "sparrow", "stork",
             "ibis", "egret", "rook", "jay", "magpie", "raven", "vulture",
             "falcon", "eagle", "kite", "kestrel", "merlin", "goshawk", "owl",
             "puffin", "penguin", "albatross", "petrel", "gannet", "cormorant",
             "loon", "grebe", "coot", "bittern", "mallard", "pintail",
             "wigeon", "pochard", "gadwall", "shelduck", "scaup", "eider",
             "merganser", "smew", "greylag", "lapwing", "curlew", "whimbrel",
             "godwit", "sandpiper", "oystercatcher", "avocet", "phalarope",
             "dotterel", "turnstone", "dunlin", "sanderling", "ruff",
             "redshank", "greenshank", "woodcock", "snipe", "nightjar",
             "swift", "martin", "swallow", "pipit", "wagtail", "dipper",
             "kingfisher", "hoopoe", "roller", "nuthatch", "treecreeper",
             "dove", "pigeon", "cuckoo", "starling", "oriole", "bunting",
             "bullfinch", "chaffinch", "siskin", "serin", "crossbill",
             "waxwing", "wheatear", "stonechat", "redstart", "blackcap",
             "chiffchaff", "whitethroat", "goldfinch", "greenfinch", "dunnock"],
    "insect": ["moth", "beetle", "ant", "bee", "wasp", "hornet", "dragonfly",
               "damselfly", "firefly", "cricket", "grasshopper", "mantis",
               "cicada", "locust", "caterpillar", "butterfly", "spider",
               "scorpion", "snail", "slug", "worm", "leech", "midge", "gnat",
               "hoverfly", "lacewing", "mayfly", "caddisfly", "stonefly",
               "alderfly", "sawfly", "harvestman", "centipede", "millipede",
               "woodlouse", "earwig", "silverfish", "booklouse", "springtail",
               "tadpole", "grub", "larva", "pupa", "chrysalis", "cocoon"],
    "fish": ["salmon", "trout", "pike", "perch", "bass", "carp", "roach",
             "bream", "chub", "dace", "minnow", "gudgeon", "rudd", "tench",
             "grayling", "charr", "smelt", "shad", "herring", "mackerel",
             "cod", "haddock", "plaice", "sole", "turbot", "brill", "dab",
             "flounder", "angelfish", "limpet", "whelk", "crab", "lobster",
             "prawn", "shrimp", "mussel", "oyster", "clam", "scallop",
             "cockle", "urchin", "starfish", "jellyfish", "squid", "octopus",
             "cuttlefish", "conger"],
    "plant": ["oak", "pine", "fern", "ivy", "orchid", "tulip", "moss", "reed",
             "clover", "heather", "juniper", "aspen", "birch", "laurel",
             "thyme", "sage", "fennel", "marigold", "crocus", "daisy", "poppy",
             "lily", "iris", "holly", "yew", "cedar", "elm", "willow",
             "alder", "maple", "walnut", "chestnut", "hawthorn", "blackthorn",
             "elder", "hazel", "rowan", "bramble", "gorse", "broom",
             "lavender", "rosemary", "mint", "basil", "chive", "sorrel",
             "dock", "sedge", "rush", "cattail", "tansy", "yarrow",
             "coltsfoot", "buttercup", "dandelion", "foxglove", "cowslip",
             "primrose", "snowdrop", "bluebell", "violet", "pansy", "peony",
             "aster"],
    "tool": ["hammer", "chisel", "ladder", "anvil", "plane", "tongs", "auger",
             "trowel", "spade", "sickle", "shears", "mallet", "pincers",
             "level", "square", "clamp", "rasp", "file", "saw", "adze",
             "drawknife", "froe", "maul", "wedge", "pick", "hoe", "rake",
             "fork", "scythe", "billhook", "slasher", "mattock", "quern",
             "mortar", "pestle", "flint", "bellows", "crucible", "winch",
             "block", "bollard", "cleat", "oar", "mast", "tiller", "keel",
             "gaff", "spear", "net", "creel"],
    "weather": ["mist", "fog", "rain", "snow", "hail", "sleet", "frost",
                "drizzle", "thunder", "wind", "haze", "cloud", "gale",
                "breeze", "squall", "storm", "gloom", "vapor", "spray",
                "spindrift"],
    "vehicle": ["wagon", "cart", "sleigh", "barge", "skiff", "canoe", "raft",
                "carriage", "trolley", "ferry", "dinghy", "yawl", "tram",
                "zeppelin", "chariot", "caravan", "punt", "curricle",
                "phaeton", "landau", "barouche", "cab", "hansom", "brougham",
                "chaise", "dray", "tumbrel", "tender", "launch", "cutter",
                "sloop", "ketch", "schooner", "brig", "clipper", "galleon",
                "dhow", "felucca", "longboat", "cog", "shallop", "pinnace"],
    "water": ["stream", "creek", "river", "brook", "beck", "burn", "rill",
              "runnel", "canal", "channel", "race", "sluice", "flume",
              "millpond", "tarn", "loch", "mere", "lake", "pond", "pool",
              "lagoon", "bay", "bight", "sound", "strait", "reach",
              "shallows", "ford", "crossing", "spring", "falls"],
    "instrument": ["flute", "lyre", "drum", "lute", "horn", "harp", "viola",
                   "oboe", "bagpipe", "fiddle", "organ", "cello", "mandolin",
                   "clarinet", "dulcimer", "zither", "psaltery", "rebec",
                   "gittern", "shawm", "crumhorn", "sackbut", "cornetto",
                   "ocarina", "cittern", "bandore", "vihuela", "tabor",
                   "bell", "chime", "cymbal", "tambourine"],
    "verb": ["wandered", "crept", "peered", "dashed", "drifted", "clambered",
             "lingered", "scurried", "ambled", "soared", "trudged", "bounded",
             "veered", "nestled", "vanished", "circled", "rested", "paced",
             "returned", "paused", "waited", "lurked", "hovered", "glided",
             "flitted", "hopped", "leapt", "sprinted", "trotted", "galloped",
             "prowled", "stalked", "slunk", "sneaked", "scampered",
             "skittered", "bustled", "hastened", "hurried", "rushed",
             "bolted", "fetched", "hauled", "carried", "lifted", "lowered",
             "raised", "placed", "stored", "stacked", "piled", "sorted",
             "counted", "measured", "weighed", "folded", "wrapped", "tied",
             "bound", "fastened", "secured", "opened", "closed", "unlocked",
             "latched", "shuttered", "rustled", "creaked", "groaned",
             "hummed", "murmured", "whispered", "chattered", "clattered",
             "rattled", "thumped", "knocked", "drummed", "trickled",
             "gurgled", "rippled", "lapped", "swung", "swayed", "rocked",
             "tilted", "spun", "rolled", "trembled", "shivered", "quivered",
             "flickered", "glimmered", "gleamed", "shimmered", "sparkled",
             "flashed", "glowed", "shone", "stood", "sat", "lay", "leaned",
             "hung", "fell", "dropped", "tumbled", "tottered", "wobbled",
             "tipped", "slid", "slipped", "crawled", "wriggled", "squirmed",
             "writhed", "twisted", "coiled", "wound", "curved", "looped",
             "snaked", "arched", "bent", "stooped", "crouched", "ducked",
             "dodged", "swerved"],
    "adv": ["quietly", "slowly", "steadily", "briefly", "softly", "gradually",
            "warily", "eagerly", "haphazardly", "carefully", "awkwardly",
            "sternly", "dimly", "faintly", "sharply", "crisply", "stealthily",
            "silently", "noisily", "wearily", "sleepily", "gloomily",
            "brightly", "lazily", "idly", "aimlessly", "purposefully",
            "methodically", "hastily", "gingerly", "randomly", "often",
            "again", "onward"],
    "place": ["meadow", "harbor", "orchard", "courtyard", "thicket", "valley",
              "station", "street", "rooftop", "corridor", "garden", "plateau",
              "market", "tunnel", "attic", "promenade", "forest", "bridge",
              "grove", "glade", "copse", "hollow", "moor", "heath", "steppe",
              "marsh", "bog", "fen", "levee", "delta", "estuary", "inlet",
              "cove", "shoal", "reef", "dune", "bluff", "mesa", "butte",
              "canyon", "ravine", "gully", "ridge", "summit", "slope",
              "prairie", "savanna", "tundra", "taiga", "clearing", "pasture",
              "paddock", "barn", "stable", "forge", "kiln", "smithy",
              "stall", "plaza", "square", "arcade", "patio", "terrace",
              "balcony", "veranda", "portico", "cloister", "chapel", "nave",
              "belfry", "steeple", "gatehouse", "rampart", "parapet",
              "bastion", "turret", "hall", "parlor", "cellar", "pantry",
              "larder", "scullery", "study", "den", "loft", "mezzanine",
              "landing", "hallway", "foyer", "atrium"],
    "noun": ["lantern", "ledger", "crate", "saddle", "hatbox", "map", "basket",
             "kettle", "telescope", "compass", "bundle", "barrel", "mirror",
             "whistle", "basin", "bowl", "jug", "pitcher", "carafe",
             "decanter", "tumbler", "goblet", "chalice", "tankard", "flagon",
             "crock", "urn", "chest", "coffer", "strongbox", "casket",
             "satchel", "knapsack", "haversack", "pouch", "quiver",
             "scabbard", "banner", "pennant", "standard", "torch", "candle",
             "taper", "sconce", "bridle", "halter", "stirrup", "harness",
             "rein", "quill", "inkpot", "scroll", "tome", "folio", "codex",
             "parchment", "vellum", "manuscript", "loom", "spindle",
             "distaff", "shuttle", "bobbin", "cup", "mug", "plate",
             "platter", "tray", "ladle", "skillet", "cauldron"],
    "occupation": ["librarian", "baker", "blacksmith", "cartographer",
                   "apothecary", "cartwright", "beekeeper", "clockmaker",
                   "ferryman", "mason", "weaver", "oarsman", "lamplighter",
                   "tinker", "farrier", "cooper", "cutler", "hostler",
                   "drover", "shepherd", "warden", "reeve", "steward",
                   "minstrel", "scribe", "herald", "bowyer", "fletcher",
                   "glover", "hatter", "saddler", "wheelwright", "joiner",
                   "turner", "fowler", "beadle", "sexton", "innkeeper",
                   "miller", "thatcher", "woodman", "pilgrim", "friar",
                   "abbot", "tanner", "currier", "chandler", "brewer",
                   "vintner", "barber", "fishmonger", "cobbler", "tailor",
                   "fuller", "dyer", "distiller", "haberdasher", "mercer",
                   "draper"],
    "adj": ["weathered", "mottled", "sturdy", "curious", "weary", "faded",
            "coiled", "glazed", "hollow", "bronze", "mossy", "threadbare",
            "gilded", "silvered", "carved", "inlaid", "engraved", "polished",
            "burnished", "tarnished", "amber", "crimson", "azure", "ivory",
            "ochre", "russet", "sepia", "umber", "indigo", "mauve",
            "scarlet", "vermilion", "ebony", "sienna", "saffron", "ashen",
            "oaken", "brass", "copper", "pewter", "tin", "iron", "stalwart",
            "stolid", "dour", "odd", "strange", "peculiar", "quaint",
            "rustic", "pastoral", "idle", "ancient", "silent", "modest",
            "humble", "crooked", "crumpled", "twisted", "forked", "notched",
            "pierced", "yawning", "vaulted", "gabled", "tiled", "cobbled",
            "worn", "frayed"],
    "when": ["at dusk", "before dawn", "in the rain", "under a thin moon",
             "amid fog", "past midnight", "at first light", "in a stiff wind",
             "after sunset", "at noon", "at nightfall", "during the storm",
             "in the daylight", "under the stars", "in early spring",
             "by the first frost", "just after rain", "while the tide was low",
             "before the storm", "during the eclipse", "when the bells rang",
             "after the harvest", "at the equinox", "near the end of summer"],
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
