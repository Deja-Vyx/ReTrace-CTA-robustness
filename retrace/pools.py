"""Sentence and IOC pools for the insertion operators (T5, E7, E8). Built ONCE, then frozen.

    python -m retrace.pools build --what carriers,distractors
    python -m retrace.pools build --what false-flag        # needs a GPU (scores TTPs per sentence)

Data sources:
  - distractors (T5): benign sentences (no IOC / group name / tool name) from a seed list +
    (optionally) LLM-generated ones. The TTP labels the extractor assigns to each sentence are
    recorded (the false-positive rate of the TTP extractor).
  - carriers (E7/E8): template sentences that carry exactly one IOC, checked with the IOC extractor
    itself; TTP labels are recorded.
  - false_flag (E8): REUSED from the EXISTING training documents of each group in the public corpus
    (distinctive IOCs + real sentences that trigger the group's distinctive techniques). No new
    infrastructure is created; this only rearranges existing data into an offline test set.

Every sentence is checked with the reproduction's OWN pipeline (ioc_extract + TTP extractor).
"""
import argparse
import json
import random
from collections import Counter, defaultdict

from src.ioc_extract import extract_iocs
from src.utils import PROCESSED, load_config, read_jsonl, sent_split
from .common import POOLS_DIR, log
from .lexicons import dataset_groups, find_any_group_mentions, find_tool_mentions

DEFAULT_DISTRACTORS = [
    "Our team will present new research at several industry conferences later this year.",
    "We would like to thank our readers for the valuable feedback on previous articles.",
    "Organizations should review their security policies on a regular basis.",
    "Security teams are encouraged to keep all software up to date.",
    "The research team continues to monitor the threat landscape closely.",
    "This article is part of an ongoing series about emerging security trends.",
    "We will publish further updates as our investigation progresses.",
    "Readers can subscribe to the newsletter to receive future posts.",
    "Employees remain the first line of defense for most companies.",
    "Many businesses have increased their security budgets over the past year.",
    "Training programs can help staff recognize suspicious messages at work.",
    "The findings were shared with relevant partners before publication.",
    "We appreciate the collaboration of the wider research community.",
    "The analysis in this post reflects the information available at the time of writing.",
    "Defenders often face limited time and resources during incident response.",
    "Regular backups remain an important part of any resilience strategy.",
    "Several customers contacted our support team with questions about this topic.",
    "The security community has discussed similar topics in recent months.",
    "Additional details will be provided in a follow-up publication.",
    "Our analysts reviewed a large number of reports while preparing this summary.",
    "Clear communication between teams makes incident handling more effective.",
    "The report was prepared by members of our research department.",
    "Some of the observations described here may change as new data appears.",
    "We encourage organizations to share information with trusted peers.",
    "Strong governance helps companies respond to incidents in a consistent way.",
    "Many small companies still lack a dedicated security team.",
    "Public awareness of online risks has grown steadily in recent years.",
    "The conference featured talks from researchers across many countries.",
    "This summary is intended for a general audience as well as practitioners.",
    "We thank the partners who helped review an early draft of this post.",
    "Planning ahead can reduce the impact of unexpected disruptions.",
    "The team hopes these notes will be useful for defenders everywhere.",
    "Feedback from the community helps us improve future publications.",
    "Budget planning for the next year is already under way at many firms.",
    "Good documentation makes it easier to onboard new team members.",
    "Our blog covers a wide range of topics related to digital safety.",
    "Some sections of this post have been shortened for readability.",
    "Industry groups have published several guides on this subject.",
    "Readers interested in the topic may find the earlier posts helpful.",
    "The findings were reviewed internally before this article was released.",
    "Our editorial team welcomes suggestions for topics to cover in future posts.",
    "The annual summit will take place in the spring and registration is now open.",
    "We are hiring analysts and engineers for several teams across the company.",
    "This post was updated to correct a minor typo in the introduction.",
    "Thanks to everyone who attended the webinar and asked thoughtful questions.",
    "A recording of the presentation will be made available on our website soon.",
    "Customers with questions can reach their account manager for more details.",
    "The views expressed in this article are those of the authors alone.",
    "Many readers asked us to explain the background of this topic in more depth.",
    "We plan to expand this series with more case studies next quarter.",
    "Leadership support is essential for any long term improvement program.",
    "The survey collected responses from professionals in many different industries.",
    "Our team spent several weeks reviewing the material before writing this post.",
    "The previous article in this series received a great deal of attention.",
    "Small improvements in daily routines can add up over the course of a year.",
    "We would like to acknowledge the help of our colleagues in preparing this note.",
    "Interested readers can find a glossary of common terms at the end of the page.",
    "The team will host an open discussion session for partners next month.",
    "Several universities have launched new programs focused on this field.",
    "It is always a good idea to discuss new policies with all stakeholders.",
    "The author thanks the reviewers for their careful reading of the draft.",
    "Our newsletter is published once a month and covers a variety of subjects.",
    "The meeting agenda included updates from each of the regional offices.",
    "We look forward to continuing this conversation with the community.",
    "The report is available in several languages for readers around the world.",
    "Participants shared their experiences during a panel held last week.",
    "Some companies have created dedicated roles to coordinate these efforts.",
    "Further reading on the subject is listed at the bottom of this article.",
    "This overview is based on public information and conversations with experts.",
    "The team celebrated the tenth anniversary of the research program this year.",
    "Clear goals make it easier to measure progress over time.",
    "We hope that this summary helps readers plan their next steps.",
    "The workshop was organized together with several local partners.",
    "Opinions on the best approach continue to differ among practitioners.",
    "Readers are welcome to share this article with their colleagues.",
    "The introduction gives a short overview of the topics discussed below.",
    "Our organization supports several community initiatives every year.",
    "The case study describes lessons learned from a long running project.",
    "More information about the program can be found on the official page.",
    "The next edition of this report is expected to be released later this year.",
]

# Template sentences that "carry" one IOC. {v} is the IOC slot. Covers prose and IOC-appendix lines.
CARRIER_TEMPLATES = {
    "IP": ["Additional traffic was also observed to {v}.",
           "The investigation also identified {v} in the logs.",
           "Another address, {v}, appeared in the collected data.",
           "Indicator: {v}.",
           "The list of indicators also includes {v}."],
    "Domain": ["The domain {v} was also identified during the investigation.",
               "Analysts additionally noted {v} in the collected data.",
               "Another domain, {v}, appeared in the logs.",
               "Indicator: {v}.",
               "The list of indicators also includes {v}."],
    "URL": ["The address {v} was also identified during the investigation.",
            "Analysts additionally noted {v} in the collected data.",
            "Indicator: {v} .",
            "The list of indicators also includes {v} ."],
    "Hash": ["A related sample with hash {v} was also identified.",
             "The investigation also found the sample {v}.",
             "Indicator: {v}.",
             "The list of indicators also includes {v}."],
    "Email": ["The address {v} was also identified during the investigation.",
              "Indicator: {v}."],
    "Filename": ["A file named {v} was also identified during the investigation.",
                 "Indicator: {v}."],
    "CVE": ["The report also mentions {v}.",
            "Indicator: {v}."],
}
CHECK_TYPES = ["IP", "Domain", "URL", "Hash", "Email", "Filename", "CVE"]


def _check_values(typ: str, n: int = 3) -> list[str]:
    """RANDOM values (with a real TLD so ioc-finder accepts them), ONLY to CHECK that a template extracts the right type."""
    from .operators.evidence_ops import random_ioc
    rng = random.Random(1234)
    if typ in ("IP", "Domain", "URL", "Hash"):
        return [random_ioc(typ, rng) for _ in range(n)]
    if typ == "Email":
        return [f"info@{random_ioc('Domain', rng)}" for _ in range(n)]
    if typ == "Filename":
        return ["update-helper.exe", "svc-report.dll", "invoice-scan.exe"][:n]
    if typ == "CVE":
        return ["CVE-2021-40444", "CVE-2017-11882", "CVE-2019-0708"][:n]
    return []


class Pools:
    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self._dis = None
        self._car = None
        self._ff = None

    def distractors(self, kind: str = "benign") -> list[str]:
        """kind=benign: benign to a HUMAN reader (no IOC / group name / tool name) - the TTP
        extractor may still label it (that is exactly what is measured). kind=ttp_free: additionally
        requires that the TTP extractor assigns no label."""
        if self._dis is None:
            fp = POOLS_DIR / "distractors.json"
            self._dis = json.load(open(fp, encoding="utf-8")) if fp.exists() \
                else build_distractors(DEFAULT_DISTRACTORS, self.pipeline)
            if isinstance(self._dis, list):                   # old format
                self._dis = {"benign": self._dis, "ttp_free": self._dis}
        return self._dis.get(kind, [])

    def carriers(self, typ: str) -> list[str]:
        if self._car is None:
            fp = POOLS_DIR / "carriers.json"
            self._car = json.load(open(fp, encoding="utf-8")) if fp.exists() \
                else filter_carriers(self.pipeline)
        v = self._car.get(typ, [])
        return v if isinstance(v, list) else []

    def false_flag(self, group: str) -> dict:
        if self._ff is None:
            fp = POOLS_DIR / "false_flag.json"
            self._ff = json.load(open(fp, encoding="utf-8")) if fp.exists() else {}
            if not self._ff:
                log("[pools] false_flag.json not found - E8 kind=ioc/ttp is skipped. "
                    "Run: python -m retrace.pools build --what false-flag")
        return self._ff.get(group, {})


def _ttp_free(pipeline, sents: list[str]) -> list[bool]:
    if pipeline is None:
        return [True] * len(sents)
    sets = pipeline.ttp_per_sentence(sents)
    out = [not s for s in sets]
    return out + [True] * (len(sents) - len(out))


def _ttp_labels(pipeline, sents: list[str]) -> list[list[str]]:
    if pipeline is None or not sents:
        return [[] for _ in sents]
    sets = pipeline.ttp_per_sentence(sents)
    out = [sorted(x) for x in sets]
    return out + [[] for _ in range(len(sents) - len(out))]


def filter_distractors(cands: list[str], pipeline) -> list[str]:
    """Backward compatible: returns the sentences WITHOUT TTPs (ttp_free)."""
    return build_distractors(cands, pipeline)["ttp_free"]


def build_distractors(cands: list[str], pipeline) -> dict:
    cands = [c.strip() for c in cands if isinstance(c, str) and 8 <= len(c.split()) <= 40]
    cands = [c for c in cands if len(sent_split(c)) == 1]
    cands = sorted(set(cands), key=cands.index)
    benign = []
    for c in cands:
        if extract_iocs(c, [c]) or find_any_group_mentions(c) or find_tool_mentions(c):
            continue
        benign.append(c)
    labels = _ttp_labels(pipeline, benign)
    ttp_free = [c for c, lab in zip(benign, labels) if not lab]
    lab_count = Counter(l for lab in labels for l in lab)
    return {"benign": benign, "ttp_free": ttp_free,
            "ttp_labels": {c: lab for c, lab in zip(benign, labels) if lab},
            "stats": {"n_candidates": len(cands), "n_benign": len(benign),
                      "n_ttp_free": len(ttp_free),
                      "ttp_fp_rate": round(1 - len(ttp_free) / max(1, len(benign)), 4),
                      "top_labels": lab_count.most_common(10)}}


def filter_carriers(pipeline) -> dict:
    """Keep templates that extract EXACTLY one IOC of the right type (checked with the real
    ioc_extract). Templates labelled by the TTP extractor are not dropped (almost every sentence
    is); the labels are stored in '_ttp' for analysis and templates with fewer labels come first."""
    out, ttp_info = {}, {}
    for typ, tmpls in CARRIER_TEMPLATES.items():
        good = []
        for t in tmpls:
            vals = _check_values(typ)
            passed = True
            for val in vals:
                s = t.format(v=val)
                found = extract_iocs(s, [s])
                n_typ = Counter(i["type"] for i in found).get(typ, 0)
                others = [i for i in found if i["type"] != typ]
                # a URL/Email may also yield its own Domain -> accepted
                if typ in ("URL", "Email"):
                    others = [i for i in others if i["type"] != "Domain"]
                if n_typ != 1 or others or len(sent_split(s)) != 1:
                    passed = False
                    break
            if passed:
                good.append(t)
        labs = _ttp_labels(pipeline, [t.format(v=_check_values(typ, 1)[0]) for t in good])
        order = sorted(range(len(good)), key=lambda i: (len(labs[i]), i))
        out[typ] = [good[i] for i in order]
        for i in order:
            ttp_info[good[i]] = labs[i]
    out["_ttp"] = ttp_info
    return out


def build_false_flag(pipeline, per_group: int = 40, max_iocs: int = 20, max_sents: int = 40,
                     seed: int = 17) -> dict:
    """For every group B, taken from B's TRAINING documents already in the corpus:
      - iocs: IOCs that occur only in B's documents (not shared with any other group in train)
      - ttp_sents: REAL sentences from B's documents that trigger B's distinctive techniques,
        contain no IOC and do not name any group (so they do not give themselves away when
        inserted into a document of group A).
    """
    rng = random.Random(seed)
    train = [d for d in read_jsonl(PROCESSED / "docs.jsonl") if d["split"] == "train"]
    by_g = defaultdict(list)
    for d in train:
        by_g[d["group"]].append(d)

    # IOCs owned by exactly one group
    ioc_owner = defaultdict(set)
    ioc_info = {}
    for d in train:
        for i in extract_iocs(d["text"]):
            if i["type"] in ("Protocol", "Port", "Account"):
                continue
            key = (i["type"], i["value"].lower())
            ioc_owner[key].add(d["group"])
            ioc_info[key] = {"type": i["type"], "value": i["value"]}
    exclusive = defaultdict(list)
    for key, owners in ioc_owner.items():
        if len(owners) == 1:
            exclusive[next(iter(owners))].append(ioc_info[key])

    # distinctive techniques of each group (frequent in B, rare in other groups)
    tech_by_g = defaultdict(Counter)
    for g, ds in by_g.items():
        for d in ds[:per_group]:
            techs = set()
            for labs in pipeline.ttp_per_sentence(sent_split(d["text"])):
                techs |= {t for t in labs if not t.startswith("TA")}
            for t in techs:
                tech_by_g[g][t] += 1
    global_df = Counter()
    for g, c in tech_by_g.items():
        for t in c:
            global_df[t] += 1

    out = {}
    for g, ds in by_g.items():
        # distinctive technique: common in B but used by few other groups (tf-idf style), top 8
        import math
        n_g = max(1, len(by_g))
        scored = sorted(((cnt * math.log((n_g + 1) / (1 + global_df[t])), t)
                         for t, cnt in tech_by_g[g].items() if cnt >= 2), reverse=True)
        signature = {t for sc, t in scored[:8] if sc > 0}
        ttp_sents = []
        for d in ds[:per_group]:
            sents = sent_split(d["text"])
            labsets = pipeline.ttp_per_sentence(sents)
            for i, labs in enumerate(labsets):
                sig_hit = signature & {t for t in labs if not t.startswith("TA")}
                s = sents[i]
                if not sig_hit or extract_iocs(s, [s]):
                    continue
                if find_any_group_mentions(s) or len(s.split()) > 40:
                    continue
                ttp_sents.append({"tech": sorted(sig_hit)[0], "sent": s})
                if len(ttp_sents) >= max_sents:
                    break
            if len(ttp_sents) >= max_sents:
                break
        iocs = exclusive.get(g, [])
        rng.shuffle(iocs)
        out[g] = {"iocs": iocs[:max_iocs], "ttp_sents": ttp_sents[:max_sents],
                  "signature": sorted(signature)}
    return out


def cmd_build(args):
    cfg = load_config(args.config)
    what = set(args.what.split(","))
    POOLS_DIR.mkdir(parents=True, exist_ok=True)
    pipe = None
    if what & {"carriers", "distractors", "false-flag"}:
        from .pipeline import AdvPipeline
        docs = [d for d in read_jsonl(PROCESSED / "docs.jsonl")]
        pipe = AdvPipeline(cfg, device=args.device, warm_docs=docs[:200] if not args.no_warm else None)

    if "distractors" in what:
        cands = list(DEFAULT_DISTRACTORS)
        if args.llm > 0:
            from .llm import P_DISTRACTORS, check_sentences, get_llm
            llm = get_llm(cfg)
            got = 0
            while got < args.llm:
                obj = llm.json(P_DISTRACTORS.format(n=30), check=check_sentences,
                               temperature=1.0, seed=1000 + got, tag="distract")
                if not obj:
                    break
                cands += obj; got += len(obj)
        dis = build_distractors(cands, pipe)
        json.dump(dis, open(POOLS_DIR / "distractors.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        st = dis["stats"]
        log(f"[pools] distractors: {st['n_benign']} benign sentences, {st['n_ttp_free']} without TTP labels "
            f"(the TTP extractor labels {st['ttp_fp_rate']*100:.0f}% of benign sentences) -> "
            f"{POOLS_DIR / 'distractors.json'}")
        if st["n_benign"] < 10:
            log("[pools] WARNING: very few distractor sentences - T5 will have few variants")

    if "carriers" in what:
        car = filter_carriers(pipe)
        json.dump(car, open(POOLS_DIR / "carriers.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        cnt = {k: len(v) for k, v in car.items() if k != "_ttp"}
        n_ttp = sum(1 for v in car["_ttp"].values() if v)
        log(f"[pools] carriers: {cnt} | {n_ttp}/{len(car['_ttp'])} templates labelled by the TTP extractor")
        for typ in ("IP", "Domain", "URL", "Hash"):
            if not car.get(typ):
                raise SystemExit(f"[pools] ERROR: no carrier template for {typ} -> E7 would be empty. Please report this.")

    if "false-flag" in what:
        ff = build_false_flag(pipe, per_group=args.per_group, seed=int(cfg.get("run", {}).get("seed", 17)))
        json.dump(ff, open(POOLS_DIR / "false_flag.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        summ = {g: {"iocs": len(v["iocs"]), "ttp_sents": len(v["ttp_sents"])} for g, v in ff.items()}
        log(f"[pools] false_flag per group: {summ}")
    if pipe is not None:
        pipe.save_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--what", default="carriers,distractors",
                    help="carriers,distractors,false-flag (comma-separated)")
    ap.add_argument("--llm", type=int, default=0, help="number of extra distractor sentences generated by the LLM")
    ap.add_argument("--per-group", type=int, default=40)
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    {"build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
