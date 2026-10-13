#!/usr/bin/env python3
"""Stage (download) the DSPredict-Hard competition data from Kaggle.

The DSPredict loader downloads lazily at run time, but that mixes 403
rules-acceptance failures into the harness run. This script pre-downloads all 54
hard competitions up front so we know exactly which succeed before launching v4.

Data lands in --root (default /path/to/scratch/dspredict-hard, which is symlinked
from data/data/dspredict-hard so it does NOT consume the small home quota).

Competitions return 403 Forbidden until you accept their rules while logged in as
the KAGGLE_USERNAME account. See the Kaggle competition pages for the list
of rules pages to accept; this script re-run skips anything already downloaded.

Usage:
  PY=python
  export KAGGLE_USERNAME=$($PY -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")
  export KAGGLE_KEY=$($PY -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['key'])")
  PYTHONPATH=. PYTHONUNBUFFERED=1 $PY scripts/adapra/stage_dspredict_hard.py
"""
import argparse
import json
import os
from pathlib import Path

from dsgym.datasets.config import get_task_path
from dsgym.datasets.loaders.kaggle_downloader import KaggleChallengeDownloader


def _has_data(cdir: Path) -> bool:
    return cdir.exists() and any(
        f.is_file() and f.name != "description.txt" for f in cdir.iterdir()
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default="/path/to/scratch/dspredict-hard",
        help="Download root (should match the data/data/dspredict-hard symlink target).",
    )
    ap.add_argument(
        "--tasks",
        default=str(get_task_path("dspredict") / "hard.json"),
        help="Path to the hard.json task list.",
    )
    args = ap.parse_args()

    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        raise SystemExit(
            "KAGGLE_USERNAME / KAGGLE_KEY must be set (export them from ~/.kaggle/kaggle.json)."
        )

    tasks = json.load(open(args.tasks))
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    d = KaggleChallengeDownloader(download_dir=str(root))

    ok: list[str] = []
    fail: list[tuple[str, str]] = []
    skip: list[str] = []
    n = len(tasks)
    for i, t in enumerate(tasks):
        cn = t["challenge_name"]
        cdir = root / cn
        if _has_data(cdir):
            skip.append(cn)
            print(f"[{i + 1}/{n}] SKIP (have) {cn}", flush=True)
            continue
        print(f"[{i + 1}/{n}] DL {cn} ...", flush=True)
        try:
            r = d.download_competition_data(cn)
            files = (
                [f for f in cdir.iterdir() if f.is_file() and f.name != "description.txt"]
                if cdir.exists()
                else []
            )
            if r.get("status") == "success" and files:
                ok.append(cn)
                print(f"    OK {cn} ({len(files)} files)", flush=True)
            else:
                fail.append((cn, str(r.get("status"))))
                print(
                    f"    FAIL {cn} status={r.get('status')} files={len(files)}",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001 — want to continue the sweep
            fail.append((cn, str(e)[:120]))
            print(f"    EXC {cn}: {e}", flush=True)

    print("\n==== SUMMARY ====", flush=True)
    print(f"ok={len(ok)} skip={len(skip)} fail={len(fail)}", flush=True)
    for cn, why in fail:
        print(f"  FAIL {cn}: {why}", flush=True)
    summary = {"ok": ok, "skip": skip, "fail": fail}
    json.dump(summary, open(root / "_stage_summary.json", "w"), indent=1)
    print(f"\nwrote {root / '_stage_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
