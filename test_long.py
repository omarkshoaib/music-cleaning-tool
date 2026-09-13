"""Checks the windowing path that makes arbitrary-length tracks work.

Two things can silently go wrong with block processing: the crossfade can leave
a dip at every seam, and the reassembly can drop or duplicate samples so the
output no longer matches the input length. Both are checked here.

Engine block sizes are shrunk so a short clip exercises the same multi-block
code path a long song would, without waiting for a real hour of audio.

    python test_long.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import engines
from audio_io import OverlapAdder, iter_chunks
from test_smoke import build_fixture

CLIP_SECONDS = 12.0


def check_flat_signal_reconstructs() -> tuple[bool, str]:
    """A constant signal must come back constant — no scalloping at the seams."""
    worst = 0.0
    for channels in (1, 2):
        total, chunk, overlap = 40_000, 7_000, 1_500
        shape = (total,) if channels == 1 else (total, channels)
        signal = np.ones(shape, dtype=np.float32)

        adder = OverlapAdder(total, overlap, channels=channels)
        for start, end in iter_chunks(total, chunk, overlap):
            adder.add(start, end, signal[start:end])

        deviation = float(np.max(np.abs(adder.result() - 1.0)))
        worst = max(worst, deviation)
        if deviation > 1e-5:
            return False, f"{channels}ch drifts by {deviation:.2e} (seam artefact)"
    return True, f"mono+stereo flat to within {worst:.1e}"


def check_engine_windowing(fixture: Path) -> tuple[bool, str]:
    """With tiny blocks, each engine must still return the full track."""
    saved = (
        engines.CHUNK_SECONDS,
        engines.OVERLAP_SECONDS,
        engines.DEMUCS_BLOCK_SECONDS,
        engines.DEMUCS_OVERLAP_SECONDS,
    )
    engines.CHUNK_SECONDS, engines.OVERLAP_SECONDS = 3.0, 0.5
    engines.DEMUCS_BLOCK_SECONDS, engines.DEMUCS_OVERLAP_SECONDS = 3.0, 0.5

    details = []
    try:
        with tempfile.TemporaryDirectory(prefix="cv_long_") as tmp:
            for engine in (engines.ENGINE_CLEARVOICE, engines.ENGINE_DEMUCS):
                audio, sr = engines.separate(
                    engine, fixture, Path(tmp) / "work", keep_percussion=False
                )
                mono = audio.mean(axis=1) if audio.ndim > 1 else audio
                duration = mono.shape[0] / sr

                if abs(duration - CLIP_SECONDS) > 0.25:
                    return False, f"{engine}: rebuilt {duration:.2f}s, expected {CLIP_SECONDS:.0f}s"
                if not np.isfinite(mono).all():
                    return False, f"{engine}: NaN/inf after reassembly"

                # A dropout at a seam shows up as a run of near-zero samples in
                # the middle of otherwise present audio.
                envelope = np.abs(mono)
                window = int(0.05 * sr)
                smoothed = np.convolve(envelope, np.ones(window) / window, mode="valid")
                interior = smoothed[window:-window] if smoothed.size > 2 * window else smoothed
                if interior.size and float(np.min(interior)) < 1e-5:
                    return False, f"{engine}: dropout at a block seam"

                blocks = len(list(iter_chunks(int(CLIP_SECONDS * sr), int(3.0 * sr), int(0.5 * sr))))
                details.append(f"{engine.split(' ')[0]} {duration:.1f}s/{blocks} blocks")
    finally:
        (
            engines.CHUNK_SECONDS,
            engines.OVERLAP_SECONDS,
            engines.DEMUCS_BLOCK_SECONDS,
            engines.DEMUCS_OVERLAP_SECONDS,
        ) = saved

    return True, ", ".join(details)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cv_long_fx_") as tmp:
        fixture = build_fixture(Path(tmp) / "mix.wav")
        checks = [
            ("flat signal reconstructs (no seam dip)", lambda: check_flat_signal_reconstructs()),
            ("engines rebuild full length from blocks", lambda: check_engine_windowing(fixture)),
        ]
        failures = 0
        for label, check in checks:
            print(f"  {label} ... ", end="", flush=True)
            try:
                ok, detail = check()
            except Exception as exc:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                ok, detail = False, f"raised {type(exc).__name__}: {exc}"
            print(("PASS  " if ok else "FAIL  ") + detail)
            failures += 0 if ok else 1

    print()
    print(f"{failures} failed" if failures else f"all {len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
