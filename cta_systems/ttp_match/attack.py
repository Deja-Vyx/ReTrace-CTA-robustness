"""ATT&CK Enterprise v14 technique list for TTP-Match: ID, name, description to embed.

The original code scrapes https://attack.mitre.org/techniques/enterprise/ (columns ID, Name,
Description), then rid_technique.py drops parent techniques that have sub-techniques. Here the
list is read from the local STIX file data/attck/enterprise-attack-14.1.json (the same v14 that
ThreatMAMBA uses), so no network access is needed and the version is pinned.
"""
import json
import re

from src.utils import ATTCK_DIR

STIX = ATTCK_DIR / "enterprise-attack-14.1.json"
_CIT = re.compile(r"\(Citation:[^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_TAG = re.compile(r"</?code>|<[^>]+>")
_WS = re.compile(r"\s+")


def clean_desc(s: str, max_chars: int) -> str:
    s = _CIT.sub("", s or "")
    s = _LINK.sub(r"\1", s)
    s = _TAG.sub("", s)
    para = s.strip().split("\n\n")[0]           # first paragraph = the short description on the ATT&CK list page
    para = _WS.sub(" ", para).strip()
    if len(para) > max_chars:
        cut = para[:max_chars]
        dot = cut.rfind(". ")
        para = cut[: dot + 1] if dot > max_chars // 2 else cut
    return para


def load_catalogue(max_chars: int = 1200, prune_parents: bool = True) -> list[dict]:
    if not STIX.exists():
        raise SystemExit(f"{STIX} not found (run scripts/download_data.sh of the base repository)")
    data = json.load(open(STIX, encoding="utf-8"))
    tech = {}
    for o in data["objects"]:
        if o.get("type") != "attack-pattern" or o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        if "enterprise-attack" not in (o.get("x_mitre_domains") or ["enterprise-attack"]):
            continue
        tid = next((r["external_id"] for r in o.get("external_references", [])
                    if r.get("source_name") == "mitre-attack"), None)
        if not tid or not re.fullmatch(r"T\d{4}(\.\d{3})?", tid):
            continue
        tech[tid] = {"id": tid, "name": o["name"],
                     "text": f"{o['name']}: {clean_desc(o.get('description', ''), max_chars)}"}
    if prune_parents:
        parents = {t.split(".")[0] for t in tech if "." in t}
        tech = {t: v for t, v in tech.items() if t not in parents}
    return [tech[t] for t in sorted(tech)]
