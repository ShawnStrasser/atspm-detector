"""Fresh-venv install smoke test of the SHIPPED wheel: take `dist/atspm_detector-<version>-py3-none-any.whl` (the
file that is handed out, not a fresh build), check that every file in it equals the source tree, install it into a NEW
virtual environment (dependencies resolved by pip from the minimums in pyproject.toml), then -- from a directory outside
the source tree -- import the package, run the CLI on the bundled sample (with --out-atspm, and once with the ignored
--profile: same CSV), confirm predict() and to_atspm_config() print no warning, and run every check
(`atspm-detector-check`).

    python tests/smoke_install.py [--wheel <file>] [--venv <dir>] [--keep] [--pip <extra pins, e.g. onnxruntime==1.21.1>]
Exit code 0 = the shipped wheel matches the source, installs and passes."""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if r.returncode:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit(f"FAILED: {cmd[0]} ... (exit {r.returncode})")
    return r.stdout + r.stderr


def shipped_wheel() -> Path:
    ver = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
    whl = ROOT / "dist" / f"atspm_detector-{ver}-py3-none-any.whl"
    if not whl.exists():
        raise SystemExit(f"FAILED: no shipped wheel {whl}")
    return whl


def matches_source(whl: Path) -> int:
    """every package file in the wheel is byte-identical to src/ and every src/ file is in the wheel."""
    src = ROOT / "src"
    with zipfile.ZipFile(whl) as z:
        names = [n for n in z.namelist() if n.startswith("atspm_detector/")]
        for name in names:
            f = src / name
            assert f.exists(), f"{name} is in the wheel but not in src/"
            assert hashlib.sha256(z.read(name)).digest() == hashlib.sha256(f.read_bytes()).digest(), \
                f"{name} in the wheel differs from src/ (rebuild the wheel)"
    srcs = {p.relative_to(src).as_posix() for p in (src / "atspm_detector").rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}
    missing = srcs - set(names)
    assert not missing, f"in src/ but not in the wheel: {sorted(missing)[:5]}"
    return len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", default=None)
    ap.add_argument("--venv", default=None)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--pip", nargs="*", default=[])
    a = ap.parse_args()
    whl = Path(a.wheel) if a.wheel else shipped_wheel()
    print(f"{whl.name}: {matches_source(whl)} package files, all identical to src/", flush=True)
    work = Path(a.venv) if a.venv else Path(tempfile.mkdtemp(prefix="dc_smoke_"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    venv = work / "venv"
    sh([sys.executable, "-m", "venv", venv])
    py = venv / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python")
    sh([py, "-m", "pip", "install", "--upgrade", "pip"])
    sh([py, "-m", "pip", "install", whl, *a.pip])
    print(sh([py, "-m", "pip", "list"]))
    run = work / "run"
    run.mkdir()
    out = sh([py, "-c", "import atspm_detector as d, pathlib; print(d.__version__, pathlib.Path(d.__file__).parent)"],
             cwd=run)
    print(out)
    assert "site-packages" in out, "imported the source tree instead of the installed wheel"
    sample = sh([py, "-c", "from atspm_detector.check import SAMPLE; print(SAMPLE)"], cwd=run).strip()
    exe = venv / ("Scripts" if sys.platform == "win32" else "bin")
    print(sh([exe / "atspm-detector", "--events", sample, "--out", run / "preds.csv",
              "--out-phases", run / "phases.csv", "--out-atspm", run / "atspm_config.csv", "--min-actuations", "1",
              "--quiet"], cwd=run))
    cfg = (run / "atspm_config.csv").read_text(encoding="utf-8").splitlines()
    assert cfg[0] == "DeviceId,Phase,Parameter,Function" and len(cfg) > 1, "--out-atspm wrote no config"
    print(f"--out-atspm: {len(cfg) - 1} atspm detector_config rows")
    dep = sh([exe / "atspm-detector", "--events", sample, "--out", run / "preds_le2h.csv", "--profile", "le2h",
              "--min-actuations", "1", "--quiet"], cwd=run)
    assert "FutureWarning" in dep, "the ignored --profile gave no FutureWarning"
    assert (run / "preds_le2h.csv").read_bytes() == (run / "preds.csv").read_bytes(), "--profile changed the answers"
    print("--profile: ignored with a FutureWarning, CSV identical")
    # predict() with every warning shown (resource / performance warnings included): nothing may be printed
    w = sh([py, "-X", "dev", "-W", "always", "-c",
            "from atspm_detector import predict; from atspm_detector.check import SAMPLE; "
            "from atspm_detector import to_atspm_config; "
            "o = predict(SAMPLE, min_actuations=1); print('rows', len(o), 'atspm rows', len(to_atspm_config(o)))"],
           cwd=run)
    noise = [ln for ln in w.splitlines() if "Warning" in ln]
    assert not noise, "predict() printed warnings:\n" + "\n".join(noise[:10])
    print(sh([exe / "atspm-detector-check"], cwd=run))
    print(f"SMOKE OK: {whl.name} ({whl.stat().st_size / 1e6:.1f} MB) installed into a fresh venv and passed")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
