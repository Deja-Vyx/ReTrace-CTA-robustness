"""Lexicons for the perturbation operators: group aliases (E4, E8), tool aliases (E3),
abbreviations (T3), dates (T6), temporal connectives (E5), generic phrases replacing IOCs (E1).

CLI:
    python -m retrace.lexicons export-review    # -> outputs/retrace/alias_review.csv
    python -m retrace.lexicons stats            # print group / tool alias counts

PITFALLS
  - Many group aliases are ordinary English words (SEA, Silence, Patchwork, Tick, Nitro,
    Scarab, Snake, Krypton, ZINC, MERCURY...). These aliases only match with the EXACT casing
    AND with context (group/APT/actor/...) within +-4 words, or in the same sentence as an
    unambiguous alias of the same group. Review alias_review.csv once.
  - Tool aliases that overlap with group names (e.g. "Snake" is both Turla and Uroburos;
    "Backdoor.SofacyX" contains APT28's "Sofacy") are excluded from E3.
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from src.utils import ATTCK_DIR, PROCESSED, load_json, read_jsonl, sent_split
from .common import ADV_DIR

# ------------------------------------------------------------------ groups
# Same as explain._attck_group_map (not imported to avoid a heavy dependency chain)
MANUAL_ATTCK = {"hidden cobra": "Lazarus Group", "waterbug": "Turla",
                "energetic bear": "Dragonfly", "quedagh": "Sandworm Team",
                "tick": "BRONZE BUTLER", "magecart": "FIN6"}
# Manual extra aliases (missing from ATT&CK v14, or short forms common in reports)
EXTRA_ALIASES = {
    "SEA": ["Syrian Electronic Army"],
    "Hidden Cobra": ["Lazarus"],
    "Quedagh": ["Sandworm"],
    "Waterbug": ["Uroburos group", "Snake group"],
    "MageCart": ["Magecart", "Magecart Group"],
    "TwoForOne": ["Two For One"],
    "APT28": ["Fancy Bear", "STRONTIUM", "Tsar Team"],
    "APT29": ["Cozy Bear", "The Dukes"],
    "FIN7": ["Carbanak Group", "Navigator Group"],
}
# Aliases that are common words -> require context
AMBIGUOUS = {"sea", "silence", "patchwork", "tick", "nitro", "scarab", "snake", "krypton",
             "zinc", "mercury", "monsoon", "swallowtail", "hangover", "comment group",
             "dropping elephant", "chinastrats", "electrum", "iridium", "telebots", "rocke",
             "group 88", "whitebear", "static kitten", "earth vetala", "nickel academy",
             "guardians of peace", "group 74", "iron twilight", "iron hunter", "iron viking",
             "iron liberty", "iron ritual", "iron hemlock", "dark halo", "nobelium"}
CONTEXT_WORDS = {"group", "groups", "apt", "actor", "actors", "gang", "team", "hackers",
                 "hacker", "operators", "operator", "campaign", "campaigns", "crew", "threat",
                 "attackers", "attacker", "adversary", "adversaries", "intrusion", "set",
                 "cyberespionage", "espionage", "state-sponsored", "sponsored", "linked",
                 "attributed", "attribution", "malware", "backdoor", "implant", "toolset"}
MIN_ALIAS_LEN = 3


def _attck() -> dict:
    fp = ATTCK_DIR / "attck_v14.json"
    return load_json(fp) if fp.exists() else {"groups": {}, "techniques": {}, "tactic_order": []}


@lru_cache(maxsize=1)
def dataset_groups() -> tuple:
    fp = PROCESSED / "groups.json"
    return tuple(load_json(fp)) if fp.exists() else tuple()


@lru_cache(maxsize=1)
def attck_group_map() -> dict:
    a = _attck()
    by_alias = {}
    for name, info in a.get("groups", {}).items():
        for al in [name] + info.get("aliases", []):
            by_alias.setdefault(al.lower(), name)
    out = {}
    for g in dataset_groups():
        out[g] = by_alias.get(g.lower()) or by_alias.get(MANUAL_ATTCK.get(g.lower(), "").lower())
    return out


@lru_cache(maxsize=1)
def group_aliases() -> dict:
    """{dataset group: [alias,...]} with longer aliases first (so longer phrases match first)."""
    a = _attck()
    gm = attck_group_map()
    out = {}
    for g in dataset_groups():
        al = {g}
        name = gm.get(g)
        if name:
            al.add(name)
            al.update(a["groups"][name].get("aliases", []))
        al.update(EXTRA_ALIASES.get(g, []))
        al = {x.strip() for x in al if x and len(x.strip()) >= MIN_ALIAS_LEN}
        al = {x for x in al if "(" not in x}          # vd. "BlackEnergy (Group)"
        out[g] = sorted(al, key=lambda s: (-len(s), s))
    return out


def is_ambiguous(alias: str) -> bool:
    return alias.lower() in AMBIGUOUS


def _alias_regex(alias: str) -> re.Pattern:
    esc = re.escape(alias).replace(r"\ ", r"[\s\-]+")
    flags = 0 if is_ambiguous(alias) else re.I
    return re.compile(rf"(?<![\w\-]){esc}(?![\w\-])", flags)


@lru_cache(maxsize=4096)
def _cached_regex(alias: str) -> re.Pattern:
    return _alias_regex(alias)


def _has_context(text: str, start: int, end: int, window: int = 4) -> bool:
    left = re.findall(r"[\w\-]+", text[:start])[-window:]
    right = re.findall(r"[\w\-]+", text[end:])[:window]
    return any(w.lower() in CONTEXT_WORDS for w in left + right)


def find_group_mentions(text: str, group: str) -> list[tuple[int, int, str]]:
    """Return NON-overlapping (start, end, alias) mentions of one group in a piece of text."""
    ensure_overrides()
    aliases = group_aliases().get(group, [group])
    unamb_hit = any(_cached_regex(a).search(text) for a in aliases if not is_ambiguous(a))
    spans: list[tuple[int, int, str]] = []
    taken = [False] * (len(text) + 1)
    for a in aliases:
        for m in _cached_regex(a).finditer(text):
            s, e = m.span()
            if any(taken[s:e]):
                continue
            if is_ambiguous(a) and not (unamb_hit or _has_context(text, s, e)):
                continue
            spans.append((s, e, a))
            for i in range(s, e):
                taken[i] = True
    return sorted(spans)


def find_any_group_mentions(text: str, groups=None) -> dict:
    groups = groups or dataset_groups()
    return {g: find_group_mentions(text, g) for g in groups if find_group_mentions(text, g)}


def mask_spans(text: str, spans: list[tuple[int, int, str]], repl: str) -> str:
    out, last = [], 0
    for s, e, _a in sorted(spans):
        out.append(text[last:s]); out.append(repl); last = e
    out.append(text[last:])
    return "".join(out)


# ------------------------------------------------------------------ tools (E3)
TOOL_STOP = {"net", "ping", "empire", "cmd", "at", "tor", "page", "ras", "wevtutil", "reg",
             "certutil", "schtasks", "systeminfo", "ipconfig", "netstat", "tasklist", "arp",
             "route", "whoami", "ftp", "expand", "xcopy", "esentutl", "nltest", "dsquery",
             "cachedump", "pass-the-hash toolkit", "lslsass", "hikit", "carbon"}


@lru_cache(maxsize=1)
def tool_aliases() -> dict:
    """{software_name: [alias,...]} for software with >= 2 valid aliases, excluding aliases that
    overlap with group names or are shared by several software entries."""
    fp = ATTCK_DIR / "enterprise-attack-14.1.json"
    if not fp.exists():
        return {}
    bundle = json.load(open(fp, encoding="utf-8"))
    alive = lambda o: not o.get("revoked") and not o.get("x_mitre_deprecated")  # noqa: E731
    group_tokens = set()
    for als in group_aliases().values():
        for a in als:
            group_tokens.add(a.lower())
            for tok in re.findall(r"[A-Za-z]{4,}", a):
                group_tokens.add(tok.lower())
    raw = {}
    for o in bundle["objects"]:
        if o.get("type") in ("malware", "tool") and alive(o):
            als = [o["name"]] + list(o.get("x_mitre_aliases", []))
            raw[o["name"]] = list(dict.fromkeys(a.strip() for a in als if a and a.strip()))
    owners = defaultdict(set)
    for name, als in raw.items():
        for a in als:
            owners[a.lower()].add(name)
    out = {}
    for name, als in raw.items():
        good = []
        for a in als:
            low = a.lower()
            if len(a) < 4 or low in TOOL_STOP or len(owners[low]) > 1:
                continue
            if low in group_tokens or any(t in low for t in group_tokens if len(t) >= 5):
                continue
            good.append(a)
        if len(good) >= 2:
            out[name] = good
    return out


def find_tool_mentions(text: str) -> dict:
    """{software_name: [(start,end,alias),...]} (case-sensitive match on word boundaries)."""
    res = {}
    for name, als in tool_aliases().items():
        spans = []
        for a in sorted(als, key=len, reverse=True):
            for m in re.finditer(rf"(?<![\w\-.]){re.escape(a)}(?![\w\-])", text):
                s, e = m.span()
                if any(not (e <= s2 or s >= e2) for s2, e2, _ in spans):
                    continue
                spans.append((s, e, a))
        if spans:
            res[name] = sorted(spans)
    return res


# ------------------------------------------------------------------ abbreviations (T3)
# (full form, abbreviation, whether it touches the ioc_extract Protocol regex)
ABBREV = [
    ("command and control", "C2", False), ("command-and-control", "C2", False),
    ("Remote Desktop Protocol", "RDP", True), ("Server Message Block", "SMB", True),
    ("Domain Name System", "DNS", True), ("Transport Layer Security", "TLS", True),
    ("Secure Shell", "SSH", True), ("File Transfer Protocol", "FTP", True),
    ("Windows Management Instrumentation", "WMI", False), ("remote access trojan", "RAT", False),
    ("remote access tool", "RAT", False), ("advanced persistent threat", "APT", False),
    ("indicators of compromise", "IOCs", False), ("indicator of compromise", "IOC", False),
    ("Active Directory", "AD", False), ("dynamic-link library", "DLL", False),
    ("dynamic link library", "DLL", False), ("virtual private server", "VPS", False),
    ("virtual private network", "VPN", False), ("PowerShell", "PS", False),
    ("Visual Basic for Applications", "VBA", False), ("user account control", "UAC", False),
    ("Local Security Authority Subsystem Service", "LSASS", False),
    ("distributed denial of service", "DDoS", False), ("point of sale", "PoS", False),
    ("point-of-sale", "PoS", False), ("proof of concept", "PoC", False),
    ("operating system", "OS", False), ("Internet Relay Chat", "IRC", True),
]


# ------------------------------------------------------------------ dates (T6)
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
_MON = {m.lower(): i + 1 for i, m in enumerate(MONTHS)}
_MON.update({m[:3].lower(): i + 1 for i, m in enumerate(MONTHS)})
RE_DATE_MDY = re.compile(r"\b(" + "|".join(MONTHS) + r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{1,2}),\s+(\d{4})\b")
RE_DATE_DMY = re.compile(r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})\b")
RE_DATE_ISO = re.compile(r"\b(20\d{2}|19\d{2})-(\d{2})-(\d{2})\b")


def reformat_dates(text: str) -> tuple[str, int]:
    """Reformat dates: 'March 3, 2021' / '3 March 2021' -> '2021-03-03'; '2021-03-03' ->
    'March 3, 2021'. Single pass, so dates are never flipped back. Returns (text, n_changes)."""
    n = 0
    pat = re.compile("|".join(f"(?:{r.pattern})" for r in (RE_DATE_MDY, RE_DATE_DMY, RE_DATE_ISO)))

    def repl(m):
        nonlocal n
        g = m.group(0)
        mm = RE_DATE_ISO.fullmatch(g)
        if mm:
            mo, d = int(mm.group(2)), int(mm.group(3))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                n += 1
                return f"{MONTHS[mo - 1]} {d}, {mm.group(1)}"
            return g
        mm = RE_DATE_MDY.fullmatch(g)
        if mm:
            mon = _MON.get(mm.group(1).lower().rstrip("."), _MON.get(mm.group(1)[:3].lower()))
            if mon:
                n += 1
                return f"{mm.group(3)}-{mon:02d}-{int(mm.group(2)):02d}"
            return g
        mm = RE_DATE_DMY.fullmatch(g)
        if mm:
            n += 1
            return f"{mm.group(3)}-{_MON[mm.group(2).lower()]:02d}-{int(mm.group(1)):02d}"
        return g

    return pat.sub(repl, text), n


# ------------------------------------------------------------------ temporal connectives (E5)
TEMPORAL_MARKERS = ["After that", "Afterwards", "Afterward", "Subsequently", "Then", "Later",
                    "Next", "Finally", "First", "Firstly", "Initially", "Once inside",
                    "Following this", "Following that", "Shortly after", "Shortly afterwards",
                    "At a later stage", "In the next stage", "In the final stage", "Eventually",
                    "Thereafter", "Meanwhile", "In parallel", "Before that", "Prior to this"]
RE_TEMPORAL_START = re.compile(
    r"^(?:" + "|".join(re.escape(m) for m in sorted(TEMPORAL_MARKERS, key=len, reverse=True)) + r")\s*,?\s+",
    re.I)


def strip_temporal_marker(sent: str) -> str | None:
    m = RE_TEMPORAL_START.match(sent)
    if not m:
        return None
    rest = sent[m.end():]
    if len(rest) < 10:
        return None
    return rest[0].upper() + rest[1:]


# ------------------------------------------------------------------ generic phrases replacing IOCs (E1)
GENERIC_IOC = {
    "Domain": "a malicious domain", "URL": "a malicious URL", "IP": "a remote server",
    "Email": "an attacker-controlled mailbox", "Hash": "a related sample",
    "Filename": "a malicious file", "CVE": "a known vulnerability", "Account": "a local user",
    "Port": "a high port", "Protocol": "a network protocol", "MAC": "a hardware address",
    "Registry": "a registry location", "Header": "a custom header",
}


# ------------------------------------------------------------------ CLI
def cmd_export_review(args):
    fp0 = ADV_DIR / "alias_review.csv"
    if fp0.exists() and not getattr(args, "overwrite", False):
        with open(fp0, encoding="utf-8") as f:
            n_drop = sum(1 for r in csv.DictReader(f) if str(r.get("keep(1/0)", "1")).strip() == "0")
        if n_drop:
            print(f"-> keeping {fp0} (already reviewed: {n_drop} aliases with keep=0). Use --overwrite to export again.")
            return
    docs = [d for d in read_jsonl(PROCESSED / "docs.jsonl") if d["split"] == args.split]
    ga = group_aliases()
    counts = Counter(); examples = defaultdict(list)
    for d in docs:
        for g in ga:
            for s, e, a in find_group_mentions(d["text"], g):
                counts[(g, a)] += 1
                if len(examples[(g, a)]) < 2:
                    lo = max(0, s - 60); hi = min(len(d["text"]), e + 60)
                    examples[(g, a)].append(f"[{d['doc_id']}|{d['group']}] ...{d['text'][lo:hi]}...")
    ADV_DIR.mkdir(parents=True, exist_ok=True)
    out = ADV_DIR / "alias_review.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "alias", "ambiguous", "n_mentions_test", "example_1", "example_2",
                    "keep(1/0)"])
        for g, als in ga.items():
            for a in als:
                ex = examples.get((g, a), [])
                w.writerow([g, a, int(is_ambiguous(a)), counts.get((g, a), 0),
                            ex[0] if ex else "", ex[1] if len(ex) > 1 else "", 1])
    print(f"-> {out} ({sum(len(v) for v in ga.values())} aliases / {len(ga)} groups)")
    print("Review the keep column: set 0 for false-positive aliases; lexicons re-reads this file if present.")


def cmd_stats(args):
    ga = group_aliases(); ta = tool_aliases()
    print(f"{len(ga)} groups, {sum(len(v) for v in ga.values())} group aliases")
    for g, als in ga.items():
        print(f"  {g:<15} {len(als):>3} alias: {', '.join(als[:6])}{' ...' if len(als) > 6 else ''}")
    print(f"{len(ta)} software entries with >= 2 valid aliases for E3 (e.g. "
          f"{', '.join(f'{k}: {v}' for k, v in list(ta.items())[:3])})")


def _apply_review_overrides():
    """If alias_review.csv exists, drop the aliases marked keep=0."""
    fp = ADV_DIR / "alias_review.csv"
    if not fp.exists():
        return
    drop = defaultdict(set)
    with open(fp, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("keep(1/0)", "1")).strip() == "0":
                drop[r["group"]].add(r["alias"])
    if not drop:
        return
    ga = group_aliases()
    for g, als in drop.items():
        if g in ga:
            ga[g][:] = [a for a in ga[g] if a not in als]


_apply_review_overrides_done = False


def ensure_overrides():
    global _apply_review_overrides_done
    if not _apply_review_overrides_done:
        _apply_review_overrides()
        _apply_review_overrides_done = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export-review", "stats"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--overwrite", action="store_true", help="overwrite an already reviewed alias_review.csv")
    args = ap.parse_args()
    {"export-review": cmd_export_review, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main()
