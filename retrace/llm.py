"""Ollama client for ReTrace (model set in configs/retrace.yaml -> llm.model), prompts and SQLite cache.

Host: llm.host in the config > OLLAMA_HOST environment variable > OLLAMA_HOST in .env > localhost:11434.

- Uses the same call as ioc_hunter/llm_agent.OllamaAgent: /api/generate, format=json,
  think=False (Qwen3 thinking mode OFF to save tokens).
- Every response is cached by sha1(model | prompt | options), so re-runs give identical
  results and an interrupted run does not repeat calls.
- A fixed seed in the options makes Ollama more stable (not guaranteed 100% deterministic).
- num_predict (default 2048, set with llm.num_predict) stops endless responses; it is not part
  of the cache key, so cached results remain reusable.
"""
import json
import os
import sqlite3
import threading
from pathlib import Path

from .common import CACHE_DIR, sha1


def _load_ollama_env():
    """Load OLLAMA_HOST / OLLAMA_MODEL from the repository .env (like ioc_hunter) if the shell
    has not set them. Only these two variables are read; no other key is touched."""
    try:
        from src.utils import ROOT
        fp = ROOT / ".env"
        if not fp.exists():
            return
        for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            if k in ("OLLAMA_HOST", "OLLAMA_MODEL") and v.strip():
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))
    except Exception:  # noqa: BLE001
        pass


def _norm_host(h: str) -> str:
    h = h.strip().rstrip("/")
    if not h.startswith("http"):
        h = "http://" + h
    if h.count(":") == 1:          # port missing
        h += ":11434"
    return h


class LLMClient:
    def __init__(self, model: str | None = None, host: str | None = None, cache: Path | None = None,
                 temperature: float = 0.7, seed: int = 17, timeout: int = 300, retries: int = 3,
                 think: bool = False, num_ctx: int = 4096, num_predict: int = 2048):
        _load_ollama_env()
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        self.host = _norm_host(host or os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.retries = retries
        self.think = think
        self.num_ctx = num_ctx
        # cap on generated tokens: prevents endless generation (one call once hung for ~15 min).
        # The longest valid response (30 distractor sentences) is ~1000 tokens; a truncated
        # response -> invalid JSON -> retried with another seed.
        self.num_predict = num_predict
        self.cache_path = Path(cache) if cache else CACHE_DIR / "llm.sqlite"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.cache_path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self._db.commit()
        self.calls = 0
        self.cache_hits = 0

    def check(self) -> tuple[bool, str]:
        """(ok, reason). Checks the connection to Ollama AND that the model has been pulled (exact tag)."""
        if os.getenv("RETRACE_FAKE_LLM"):
            return True, "fake"
        import requests
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=(5, 15))
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as e:  # noqa: BLE001
            return False, (f"cannot connect to Ollama at {self.host} ({type(e).__name__}). "
                           f"Run `ollama serve`, or if Ollama runs on another host (e.g. Windows "
                           f"from WSL) set OLLAMA_HOST=http://<host-ip>:11434 (see README)")
        want = {self.model, self.model + ":latest"} if ":" not in self.model else {self.model}
        if not any(n in want for n in names):
            return False, (f"Ollama at {self.host} does not have model '{self.model}'. Run: ollama pull {self.model} "
                           f"(available: {', '.join(names[:8]) or 'no models'})")
        return True, f"{self.model} @ {self.host}"

    def ping(self) -> bool:
        return self.check()[0]

    def _raw(self, prompt: str, temperature: float, seed: int) -> str:
        if os.getenv("RETRACE_FAKE_LLM"):
            return _fake_response(prompt, seed)
        import requests
        payload = {"model": self.model, "prompt": prompt, "stream": False, "format": "json",
                   "options": {"temperature": temperature, "seed": seed, "num_ctx": self.num_ctx,
                               "num_predict": self.num_predict},
                   "think": self.think}
        # (connect 10 s, read at most self.timeout) -> never hangs when Ollama does not answer
        r = requests.post(f"{self.host}/api/generate", json=payload, timeout=(10, self.timeout))
        if r.status_code == 400 and "think" in payload:
            payload.pop("think")
            r = requests.post(f"{self.host}/api/generate", json=payload, timeout=(10, self.timeout))
        r.raise_for_status()
        return r.json().get("response", "")

    def json(self, prompt: str, check=None, temperature: float | None = None, seed: int | None = None,
             tag: str = "") -> dict | None:
        """Call the LLM, parse JSON, run check(obj) (None = invalid -> retry). Returns None on failure."""
        temperature = self.temperature if temperature is None else temperature
        seed = self.seed if seed is None else seed
        key = sha1(f"{self.model}|{temperature}|{seed}|{tag}|{prompt}")
        with self._lock:
            row = self._db.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
        if row:
            self.cache_hits += 1
            obj = json.loads(row[0])
            return check(obj) if check else obj
        last = None
        for attempt in range(self.retries):
            try:
                self.calls += 1
                raw = self._raw(prompt, temperature, seed + attempt)
                obj = json.loads(raw)
                out = check(obj) if check else obj
                if out is not None:
                    with self._lock:
                        self._db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (key, json.dumps(obj)))
                        self._db.commit()
                    return out
            except Exception as e:  # noqa: BLE001
                last = e
        if last is not None:
            print(f"[llm] failed after {self.retries} attempts: {type(last).__name__}: {str(last)[:120]}")
        return None

    def stats(self) -> dict:
        return {"calls": self.calls, "cache_hits": self.cache_hits}


def _fake_response(prompt: str, seed: int) -> str:
    """ONLY for tests on machines without Ollama (RETRACE_FAKE_LLM=1): returns JSON in the right
    format, rewriting sentences with a few fixed transformations and keeping placeholders."""
    import re as _re
    if '"is_cti"' in prompt:
        return json.dumps({"is_cti": True, "reason": "fake"})
    if '"equivalent"' in prompt:
        return json.dumps({"equivalent": True, "same_actor": True, "reason": "fake"})
    if '"sentences"' in prompt:
        return json.dumps({"sentences": [f"Our team shared general news with readers in issue {i}." for i in range(5)]})
    m = _re.search(r"\nSentence: (.*)\n", prompt)
    sent = m.group(1).strip() if m else "Activity was observed."
    body = sent[0].lower() + sent[1:] if sent[:1].isupper() and not sent.startswith("[P") else sent
    body = body.rstrip(".")
    outs = [f"According to the report, {body}.", f"It was noted that {body}.", f"Notably, {body}."]
    return json.dumps({"rewrites": outs})


# ---------------------------------------------------------------------- prompt
P_PARAPHRASE = """You rewrite sentences from cyber threat intelligence (CTI) reports.
Rewrite the sentence below {n} different ways. Every rewrite MUST:
- keep exactly the same meaning and the same technical facts (no new facts, nothing removed);
- keep every placeholder such as [P1], [P2] exactly once and unchanged;
- be ONE single sentence;
- use clearly different wording and sentence structure from the original.

Sentence: {sentence}

Return ONLY JSON: {{"rewrites": ["...", "..."]}}"""

P_RESTRUCTURE = """You rewrite sentences from cyber threat intelligence (CTI) reports.
Restructure the sentence below {n} different ways: split it into two shorter sentences, or
switch between active and passive voice. Every rewrite MUST keep exactly the same meaning and
technical facts and keep every placeholder such as [P1] exactly once and unchanged.

Sentence: {sentence}

Return ONLY JSON: {{"rewrites": ["...", "..."]}}"""

P_NEUTRALIZE = """You edit sentences from cyber threat intelligence (CTI) reports.
Rewrite the sentence below so that it NO LONGER describes the attack behavior
"{tech_name}" (MITRE ATT&CK {tech_id}). Keep all other information and keep every placeholder
such as [P1] exactly once and unchanged. If the whole sentence is only about this behavior,
return a neutral sentence saying that additional activity was observed, keeping the placeholders.
Give {n} different versions, each ONE sentence.

Sentence: {sentence}

Return ONLY JSON: {{"rewrites": ["...", "..."]}}"""

P_BLUR = """You edit sentences from cyber threat intelligence (CTI) reports.
Rewrite the sentence below replacing specific tool names, malware names and exact command lines
with generic descriptions (for example "a credential dumping tool", "a scripting interpreter"),
while keeping the described attacker behavior and every placeholder such as [P1] exactly once.
Give {n} different versions, each ONE sentence.

Sentence: {sentence}

Return ONLY JSON: {{"rewrites": ["...", "..."]}}"""

P_DISTRACTORS = """Write {n} different sentences that could appear in a cybersecurity blog post
but carry NO technical evidence: no IP addresses, domains, URLs, file names, hashes, registry keys,
port numbers, protocol names, product names, malware names, threat actor names, CVE identifiers,
or descriptions of attack techniques. Good topics: company announcements, general advice,
conference news, hiring, reader feedback. Each item is ONE sentence of 12 to 30 words.

Return ONLY JSON: {{"sentences": ["...", "..."]}}"""

P_IS_CTI = """Decide whether the document below is a cyber threat intelligence report, i.e. it
describes malicious cyber activity, malware, an intrusion campaign or a threat actor. Marketing
posts, product announcements and general IT articles are NOT CTI.

Document (truncated):
\"\"\"{text}\"\"\"

Return ONLY JSON: {{"is_cti": true or false, "reason": "<one short sentence>"}}"""

P_JUDGE = """You check rewritten sentences from cyber threat intelligence reports.
Sentence A (original): {a}
Sentence B (rewritten): {b}

Question 1: Do A and B describe the same attacker behavior with the same technical evidence
(nothing added, nothing removed)?
Question 2: Would an analyst attribute B to the same threat actor as A?

Return ONLY JSON: {{"equivalent": true or false, "same_actor": true or false, "reason": "<short>"}}"""


def check_rewrites(obj, n_min: int = 1):
    rw = obj.get("rewrites") if isinstance(obj, dict) else None
    if not isinstance(rw, list):
        return None
    rw = [r for r in rw if isinstance(r, str) and r.strip()]
    return rw if len(rw) >= n_min else None


def check_sentences(obj):
    s = obj.get("sentences") if isinstance(obj, dict) else None
    if not isinstance(s, list):
        return None
    return [x for x in s if isinstance(x, str) and x.strip()] or None


def check_is_cti(obj):
    if isinstance(obj, dict) and isinstance(obj.get("is_cti"), bool):
        return obj
    return None


def check_judge(obj):
    if isinstance(obj, dict) and isinstance(obj.get("equivalent"), bool):
        return obj
    return None


def cmd_check():
    """python -m retrace.llm   -> check Ollama + model, try one rewrite with placeholders."""
    from .config import get_config
    llm = get_llm(get_config())
    ok, why = llm.check()
    print(f"[llm] host={llm.host} model={llm.model} -> {'OK' if ok else 'ERROR'}: {why}")
    if not ok:
        raise SystemExit(1)
    import time
    t = time.time()
    obj = llm.json(P_PARAPHRASE.format(n=2, sentence="The actor used [P1] to download a payload from [P2]."),
                   check=lambda o: check_rewrites(o, 1), temperature=0.7, seed=1, tag="selftest")
    print(f"[llm] test rewrite ({time.time()-t:.1f}s): {obj}")
    good = bool(obj) and all(("[P1]" in r and "[P2]" in r) for r in obj)
    print("[llm] => " + ("PASS" if good else "FAIL - the model did not keep placeholders / did not return JSON"))
    if not good:
        raise SystemExit(1)


def get_llm(cfg: dict) -> LLMClient:
    c = (cfg or {}).get("llm", {})
    return LLMClient(model=c.get("model"), host=c.get("host"), temperature=float(c.get("temperature", 0.7)),
                     seed=int(c.get("seed", 17)), think=bool(c.get("think", False)),
                     num_ctx=int(c.get("num_ctx", 4096)), num_predict=int(c.get("num_predict", 2048)),
                     timeout=int(c.get("timeout", 300)))


if __name__ == "__main__":
    cmd_check()
