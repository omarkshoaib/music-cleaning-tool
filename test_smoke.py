"""Smoke test: prove each engine actually runs and writes real audio.

Builds a music-over-speech mix, pushes it through every engine, and checks the
result is the right length, is not silent, and has less energy at the music
frequencies than the input did. A broken install fails here loudly instead of
quietly writing empty mp3s.

    python test_smoke.py             # all engines
    python test_smoke.py demucs      # substring-matched subset
"""

from __future__ import annotations

import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import engines
from audio_io import TARGET_SR, decode_to_wav, probe, read_wav, write_wav

ROOT = Path(__file__).resolve().parent
CLIP_SECONDS = 12.0
# A major triad standing in for instrumental backing.
MUSIC_HZ = (220.0, 277.2, 329.6)
SPEECH_SOURCE = ROOT.parent / "تحدي الثلاثين 3 ｜ ربع النهائي - المواجهة الثانية [zl_zcYlfPA8].wav"


def _speech_bed(samples: int, sr: int) -> np.ndarray:
    """Real speech if we have it on disk, otherwise a formant-ish synthetic stand-in."""
    if SPEECH_SOURCE.exists():
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "speech.wav"
            cmd = [
                shutil.which("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                "-ss", "60", "-t", str(CLIP_SECONDS), "-i", str(SPEECH_SOURCE),
                "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(clip),
            ]
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                audio, _ = read_wav(clip)
                if audio.size >= samples:
                    return audio[:samples].astype(np.float32)

    t = np.arange(samples, dtype=np.float32) / sr
    envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t)) ** 2
    voice = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate((140.0, 700.0, 1220.0)))
    return (0.4 * envelope * voice).astype(np.float32)


def _band_energy(audio: np.ndarray, sr: int, centre: float, width: float = 12.0) -> float:
    spectrum = np.abs(np.fft.rfft(audio.astype(np.float64)))
    freqs = np.fft.rfftfreq(audio.shape[0], 1.0 / sr)
    band = (freqs > centre - width) & (freqs < centre + width)
    return float(np.sum(spectrum[band] ** 2))


def build_fixture(path: Path) -> Path:
    sr = TARGET_SR
    samples = int(CLIP_SECONDS * sr)
    t = np.arange(samples, dtype=np.float32) / sr

    music = sum(0.22 * np.sin(2 * np.pi * f * t) for f in MUSIC_HZ).astype(np.float32)
    mix = _speech_bed(samples, sr) * 0.8 + music
    mix = (mix / max(float(np.max(np.abs(mix))), 1e-6) * 0.9).astype(np.float32)
    return write_wav(path, mix, sr)


def check(engine: str, fixture: Path, work_dir: Path) -> tuple[bool, str]:
    source, source_sr = read_wav(fixture)
    before = sum(_band_energy(source, source_sr, f) for f in MUSIC_HZ)

    try:
        audio, sr = engines.separate(engine, fixture, work_dir / "run", keep_percussion=False)
    except Exception as exc:  # noqa: BLE001 - a failing engine is a test result
        return False, f"raised {type(exc).__name__}: {exc}"

    mono = audio.mean(axis=1) if audio.ndim > 1 else audio

    if mono.size == 0:
        return False, "produced empty output"
    if not np.isfinite(mono).all():
        return False, "produced NaN/inf samples"

    duration = mono.shape[0] / sr
    if abs(duration - CLIP_SECONDS) > 0.75:
        return False, f"duration {duration:.2f}s, expected ~{CLIP_SECONDS:.0f}s"

    rms = float(np.sqrt(np.mean(mono**2)))
    if rms < 1e-4:
        return False, f"output is effectively silent (rms {rms:.2e})"

    # Compare music-band share of total energy, so overall gain changes don't skew it.
    after = sum(_band_energy(mono, sr, f) for f in MUSIC_HZ)
    before_share = before / max(float(np.sum(source.astype(np.float64) ** 2)), 1e-12)
    after_share = after / max(float(np.sum(mono.astype(np.float64) ** 2)), 1e-12)
    ratio = after_share / max(before_share, 1e-12)
    reduction_db = -10 * np.log10(max(ratio, 1e-12))

    verdict = f"rms {rms:.3f}, {duration:.1f}s, music band -{reduction_db:.1f} dB"
    if reduction_db < 1.0:
        return False, f"{verdict} (music not attenuated)"
    return True, verdict


def main() -> int:
    wanted = sys.argv[1:]
    selected = [
        e for e in engines.ENGINE_CHOICES
        if not wanted or any(w.lower() in e.lower() for w in wanted)
    ]
    if not selected:
        print(f"no engine matches {wanted}; choices: {engines.ENGINE_CHOICES}")
        return 2

    with tempfile.TemporaryDirectory(prefix="cv_smoke_") as tmp:
        work = Path(tmp)
        fixture = build_fixture(work / "mix.wav")
        info = probe(fixture)
        print(f"fixture: {info.duration:.1f}s @ {info.sample_rate} Hz, speech+music mix\n")

        failures = 0
        for engine in selected:
            print(f"  {engine} ... ", end="", flush=True)
            ok, detail = check(engine, fixture, work)
            print(("PASS  " if ok else "FAIL  ") + detail)
            failures += 0 if ok else 1

    print()
    if failures:
        print(f"{failures}/{len(selected)} engines FAILED")
        return 1
    print(f"all {len(selected)} engines passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
