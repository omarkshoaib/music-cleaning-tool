"""Checks the save rules: one engine saves itself, several stage until you choose.

Runs against a temporary output directory so it never touches `clean songs/`.

    python test_flow.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import app
from app import ENGINE_CLEARVOICE, ENGINE_DEMUCS
from test_smoke import build_fixture


def _noop_progress(fraction=0.0, desc=""):
    """Stand-in for gr.Progress, which needs a live request context."""


def _mp3s(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.mp3"))


def _log_records() -> list[dict]:
    if not app.LOG_PATH.exists():
        return []
    return [json.loads(line) for line in app.LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_single_engine_saves_directly(fixture: Path) -> tuple[bool, str]:
    result = app.run_engines(str(fixture), [ENGINE_DEMUCS], False, {}, progress=_noop_progress)
    *players, compare_box, choice, status, download, state = result

    saved = _mp3s(app.OUTPUT_DIR)
    if len(saved) != 1:
        return False, f"expected 1 file in output dir, found {[p.name for p in saved]}"
    if download is None or not Path(download).exists():
        return False, "download slot is empty after a single-engine run"
    if state != {}:
        return False, f"state should be cleared after auto-save, got {state}"
    if compare_box.get("visible") is not False:
        return False, "comparison box should stay hidden for one engine"
    if "demucs" not in saved[0].name:
        return False, f"output not tagged with the engine: {saved[0].name}"

    records = _log_records()
    if not records or records[-1].get("chose") != ENGINE_DEMUCS:
        return False, "save was not written to the log"
    return True, f"auto-saved {saved[0].name}"


def check_multi_engine_stages_then_saves(fixture: Path) -> tuple[bool, str]:
    before = set(_mp3s(app.OUTPUT_DIR))
    engines_wanted = [ENGINE_DEMUCS, ENGINE_CLEARVOICE]

    result = app.run_engines(str(fixture), engines_wanted, False, {}, progress=_noop_progress)
    *players, compare_box, choice, status, download, state = result

    if set(_mp3s(app.OUTPUT_DIR)) != before:
        return False, "a file was written to the output dir before Save was pressed"
    if compare_box.get("visible") is not True:
        return False, "comparison box should be visible for several engines"
    if sorted(choice.get("choices") or []) != sorted(engines_wanted):
        return False, f"radio choices wrong: {choice.get('choices')}"
    if download is not None:
        return False, "download slot should stay empty until Save"
    if len(state.get("renders") or {}) != 2:
        return False, f"expected 2 staged renders, got {state.get('renders')}"

    staged = {engine: Path(path) for engine, path in state["renders"].items()}
    if not all(path.exists() for path in staged.values()):
        return False, "staged renders missing from disk"

    # Now choose the loser on purpose, to prove the choice is honoured.
    picked = ENGINE_CLEARVOICE
    *players2, compare_box2, status2, download2, state2 = app.save_choice(picked, state)

    after = set(_mp3s(app.OUTPUT_DIR)) - before
    if len(after) != 1:
        return False, f"expected exactly 1 new file after Save, got {[p.name for p in after]}"
    kept = next(iter(after))
    if "clearervoice" not in kept.name:
        return False, f"saved the wrong engine: {kept.name}"
    if staged[ENGINE_DEMUCS].exists():
        return False, "the discarded render was not deleted"
    if state2 != {}:
        return False, f"state should be cleared after Save, got {state2}"
    if compare_box2.get("visible") is not False:
        return False, "comparison box should hide after Save"
    if download2 is None or not Path(download2).exists():
        return False, "download slot empty after Save"

    record = _log_records()[-1]
    if record.get("chose") != picked or record.get("discarded") != [ENGINE_DEMUCS]:
        return False, f"log did not record the comparison correctly: {record}"
    return True, f"staged 2, kept {kept.name}, discarded demucs"


def check_rerun_discards_unsaved(fixture: Path) -> tuple[bool, str]:
    before = set(_mp3s(app.OUTPUT_DIR))
    result = app.run_engines(
        str(fixture), [ENGINE_DEMUCS, ENGINE_CLEARVOICE], False, {}, progress=_noop_progress
    )
    orphans = {engine: Path(p) for engine, p in result[-1]["renders"].items()}

    # Start another run without saving; the previous renders must not survive.
    app.run_engines(str(fixture), [ENGINE_DEMUCS], False, result[-1], progress=_noop_progress)

    survivors = [engine for engine, path in orphans.items() if path.exists()]
    if survivors:
        return False, f"abandoned renders left on disk: {survivors}"
    new_files = set(_mp3s(app.OUTPUT_DIR)) - before
    if len(new_files) != 1:
        return False, f"expected only the second run's auto-save, got {[p.name for p in new_files]}"
    return True, "abandoned renders cleaned up on re-run"


def main() -> int:
    checks = [
        ("one engine saves directly", check_single_engine_saves_directly),
        ("several engines stage, Save keeps one", check_multi_engine_stages_then_saves),
        ("re-running discards unsaved renders", check_rerun_discards_unsaved),
    ]

    with tempfile.TemporaryDirectory(prefix="cv_flow_") as tmp:
        sandbox = Path(tmp)
        app.OUTPUT_DIR = sandbox / "clean songs"
        app.LOG_PATH = app.OUTPUT_DIR / "_log.jsonl"
        app.STAGING_DIR = sandbox / "staging"
        app.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        fixture = build_fixture(sandbox / "mix.wav")
        print(f"sandbox: {sandbox}\n")

        failures = 0
        for label, check in checks:
            print(f"  {label} ... ", end="", flush=True)
            try:
                ok, detail = check(fixture)
            except Exception as exc:  # noqa: BLE001 - a raising check is a failure
                import traceback

                traceback.print_exc()
                ok, detail = False, f"raised {type(exc).__name__}: {exc}"
            print(("PASS  " if ok else "FAIL  ") + detail)
            failures += 0 if ok else 1

    print()
    if failures:
        print(f"{failures}/{len(checks)} checks FAILED")
        return 1
    print(f"all {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
