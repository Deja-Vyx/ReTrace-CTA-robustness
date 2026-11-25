"""Check the environment and FREEZE the comparison baseline before running ReTrace.

    python -m retrace.preflight

Writes outputs/retrace/freeze.json: git commit, SHA-256 of every checkpoint and of the TTP
extractor, library versions, Ollama model. Catches missing files and version drift early.
"""
import json
import platform
import subprocess
import sys
from pathlib import Path

from src.utils import CSKG_DIR, OUTPUTS, PROCESSED, ROOT
from .common import ADV_DIR, sha256_file


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def _pkg(name):
    try:
        return __import__(name).__version__
    except Exception:  # noqa: BLE001
        return None


def main():
    ok = True
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "git_commit": _git("rev-parse", "HEAD"), "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "packages": {p: _pkg(p) for p in ("torch", "transformers", "sklearn", "numpy",
                                              "pandas", "ioc_finder")},
            "checkpoints": {}, "data": {}}

    print("== PREFLIGHT ==")
    print(f"git: {info['git_branch']} @ {str(info['git_commit'])[:10]} "
          f"{'(UNCOMMITTED changes)' if info['git_dirty'] else '(clean)'}")
    if info["git_branch"] == "master" or info["git_branch"] == "main":
        print("  ! Consider a dedicated branch: git checkout -b retrace")

    for tag in ("main", "no_cl", "no_mamba"):
        fp = OUTPUTS / f"model_{tag}.pt"
        if fp.exists():
            info["checkpoints"][tag] = {"sha256": sha256_file(fp), "bytes": fp.stat().st_size}
            print(f"  checkpoint {tag}: OK ({fp.stat().st_size} B)")
        else:
            ok = False
            print(f"  checkpoint {tag}: MISSING {fp}")
    for extra in ("ttp_extractor",):
        fp = OUTPUTS / f"{extra}.pt"
        if fp.exists():
            info["checkpoints"][extra] = {"sha256": sha256_file(fp)}
            print(f"  {extra}: OK")
        else:
            ok = False
            print(f"  {extra}: MISSING {fp}")

    n_cskg = len(list(CSKG_DIR.glob("d*.pt")))
    info["data"]["n_cskg"] = n_cskg
    info["data"]["docs_jsonl"] = (PROCESSED / "docs.jsonl").exists()
    info["data"]["attck_v14"] = (Path(PROCESSED).parent / "attck" / "attck_v14.json").exists()
    info["data"]["stix_14_1"] = (Path(PROCESSED).parent / "attck" / "enterprise-attack-14.1.json").exists()
    info["data"]["cleaned_csv"] = (Path(PROCESSED).parent / "raw" / "MuscleFish" / "cleaned_CTI2Attacker.csv").exists()
    print(f"  CSKG: {n_cskg} files (.pt) in data/cskg  {'OK' if n_cskg >= 3000 else '! fewer than 3144'}")
    for k in ("docs_jsonl", "attck_v14", "stix_14_1", "cleaned_csv"):
        print(f"  data/{k}: {'OK' if info['data'][k] else 'MISSING'}")
        ok = ok and info["data"][k]
    if n_cskg < 3000:
        ok = False

    # torch / cuda / mamba
    try:
        import torch
        info["torch_cuda"] = torch.cuda.is_available()
        print(f"  torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
        ck = torch.load(OUTPUTS / "model_main.pt", map_location="cpu", weights_only=False)
        impl = ck.get("ssm_impl")
        info["checkpoints"].setdefault("main", {})["ssm_impl"] = impl
        print(f"  model_main ssm_impl = {impl}")
        if impl == "mamba" and not torch.cuda.is_available():
            print("  ! model_main uses the mamba kernel but CUDA is not available -> only the unit "
                  "tests can run (THREATMAMBA_SSM=simple + FakeEncoder). Real runs need a GPU machine.")
    except Exception as e:  # noqa: BLE001
        print(f"  torch: could not be loaded ({e})")

    info["passed"] = bool(ok)
    info["time"] = __import__("time").strftime("%Y-%m-%d %H:%M:%S")
    ADV_DIR.mkdir(parents=True, exist_ok=True)
    (ADV_DIR / "freeze.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"-> {ADV_DIR/'freeze.json'}")
    print("== " + ("PASS" if ok else "FAIL - see the MISSING/! lines above") + " ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
