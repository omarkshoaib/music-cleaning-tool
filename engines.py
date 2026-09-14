"""The three separation engines, behind one common interface.

Every engine takes a source file and returns (audio, sample_rate), where audio
is float32 shaped (samples,) for mono or (samples, channels) for stereo.
Models are loaded lazily and cached, so switching engines in the UI does not
pay the load cost twice.

There is no length limit. Both engines window long inputs and reassemble them
with crossfades, reading and writing a window at a time, so peak memory is set
by the window size rather than by the length of the track.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np

from audio_io import (
    TARGET_SR,
    OverlapAdder,
    decode_to_wav,
    frame_count,
    iter_chunks,
    read_block,
    sample_rate_of,
    write_wav,
)

log = logging.getLogger(__name__)

CLEARVOICE_MODEL = "MossFormer2_SE_48K"
DEMUCS_MODEL = "htdemucs"

# MossFormer2 attention cost grows with input length; 20s windows keep memory
# flat regardless of track length while the 1s overlap hides the seams.
CHUNK_SECONDS = 20.0
OVERLAP_SECONDS = 1.0

# Demucs returns all four stems resident on the device (~4.7 GB of VRAM per
# hour of stereo audio), so long tracks are processed a block at a time. Peak
# VRAM is then bounded by the block length, not by the length of the song.
DEMUCS_SR = 44100
DEMUCS_BLOCK_SECONDS = 420.0
DEMUCS_OVERLAP_SECONDS = 2.0

ENGINE_CLEARVOICE = "ClearerVoice (MossFormer2_SE_48K)"
ENGINE_DEMUCS = "Demucs (htdemucs)"
ENGINE_CHAIN = "Demucs → ClearerVoice"

ENGINE_CHOICES = [ENGINE_CLEARVOICE, ENGINE_DEMUCS, ENGINE_CHAIN]
PERCUSSION_ENGINES = {ENGINE_DEMUCS, ENGINE_CHAIN}

_clearvoice_model = None
_demucs_separator = None


class EngineError(RuntimeError):
    """Raised when an engine cannot produce output."""


def _torch():
    import torch

    return torch


def _pick_device(prefer_gpu: bool = True) -> str:
    torch = _torch()
    return "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"


def _free_gpu() -> None:
    torch = _torch()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text


# --------------------------------------------------------------------------
# ClearerVoice
# --------------------------------------------------------------------------

def load_clearvoice():
    """Load and cache the ClearerVoice speech-enhancement model."""
    global _clearvoice_model
    if _clearvoice_model is None:
        from clearvoice import ClearVoice

        log.info("loading ClearerVoice %s", CLEARVOICE_MODEL)
        _clearvoice_model = ClearVoice(task="speech_enhancement", model_names=[CLEARVOICE_MODEL])
    return _clearvoice_model


def _as_mono_array(result) -> np.ndarray:
    """Normalize whatever ClearVoice hands back into a 1-D float32 array.

    The package returns a bare array for a single model but a name-keyed dict
    when several are requested; accept both rather than depend on the shape.
    """
    if isinstance(result, dict):
        if not result:
            raise EngineError("ClearerVoice returned no output")
        result = next(iter(result.values()))

    audio = np.squeeze(np.asarray(result, dtype=np.float32))
    if audio.ndim > 1:
        axis = 0 if audio.shape[0] < audio.shape[-1] else 1
        audio = audio.mean(axis=axis)
    return np.ascontiguousarray(audio, dtype=np.float32)


def run_clearvoice(wav_path: Path, work_dir: Path, progress=None) -> tuple[np.ndarray, int]:
    """Enhance a mono 48k wav, windowing long inputs and crossfading the seams."""
    model = load_clearvoice()
    sample_rate = sample_rate_of(wav_path)
    total = frame_count(wav_path)

    chunk = int(CHUNK_SECONDS * sample_rate)
    overlap = int(OVERLAP_SECONDS * sample_rate)

    if total <= chunk:
        return _as_mono_array(model(input_path=str(wav_path), online_write=False)), sample_rate

    windows = list(iter_chunks(total, chunk, overlap))
    chunk_dir = work_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    adder = OverlapAdder(total, overlap, channels=1)

    for index, (start, end) in enumerate(windows):
        if progress is not None:
            progress(index / len(windows), desc=f"ClearerVoice window {index + 1}/{len(windows)}")
        block = read_block(wav_path, start, end - start)
        if block.ndim > 1:
            block = block.mean(axis=1)
        piece = write_wav(chunk_dir / f"chunk_{index:04d}.wav", block, sample_rate)
        adder.add(start, end, _as_mono_array(model(input_path=str(piece), online_write=False)))
        piece.unlink(missing_ok=True)

    return adder.result(), sample_rate


# --------------------------------------------------------------------------
# Demucs
# --------------------------------------------------------------------------

def load_demucs(device: str | None = None):
    """Load and cache the Demucs separator."""
    global _demucs_separator
    device = device or _pick_device()
    if _demucs_separator is None or getattr(_demucs_separator, "_endpoint_device", None) != device:
        from demucs.api import Separator

        log.info("loading Demucs %s on %s", DEMUCS_MODEL, device)
        _demucs_separator = Separator(model=DEMUCS_MODEL, device=device)
        _demucs_separator._endpoint_device = device
    return _demucs_separator


def _separate_file(src: Path):
    """Run Demucs on one file, dropping to CPU if the GPU cannot hold it."""
    try:
        separator = load_demucs()
        return separator, separator.separate_audio_file(str(src))[1]
    except Exception as exc:  # noqa: BLE001 - retried on CPU below
        if not _is_oom(exc):
            raise EngineError(f"Demucs failed: {exc}") from exc
        log.warning("Demucs hit CUDA OOM, retrying on CPU")
        _free_gpu()
        separator = load_demucs(device="cpu")
        return separator, separator.separate_audio_file(str(src))[1]


def _mix_stems(stems, keep_percussion: bool) -> np.ndarray:
    """Vocals, optionally with the drums stem folded back in."""
    if "vocals" not in stems:
        raise EngineError(f"Demucs produced no vocals stem (got {sorted(stems)})")
    mix = stems["vocals"].clone()
    if keep_percussion and "drums" in stems:
        mix += stems["drums"]
    audio = mix.detach().cpu().numpy().astype(np.float32)
    if audio.ndim == 2:  # (channels, samples) -> (samples, channels)
        audio = audio.T
    return np.ascontiguousarray(audio)


def run_demucs(
    src: Path,
    keep_percussion: bool = False,
    work_dir: Path | None = None,
    progress=None,
) -> tuple[np.ndarray, int]:
    """Separate stems and return the vocal mix, blockwise for long tracks."""
    work_dir = Path(work_dir) if work_dir else src.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    decoded = decode_to_wav(src, work_dir / "demucs_input.wav", DEMUCS_SR, mono=False)
    total = frame_count(decoded)
    block = int(DEMUCS_BLOCK_SECONDS * DEMUCS_SR)
    overlap = int(DEMUCS_OVERLAP_SECONDS * DEMUCS_SR)

    if total <= block:
        separator, stems = _separate_file(decoded)
        audio = _mix_stems(stems, keep_percussion)
        decoded.unlink(missing_ok=True)
        return audio, int(separator.samplerate)

    windows = list(iter_chunks(total, block, overlap))
    log.info("demucs: %.1f min in %d blocks", total / DEMUCS_SR / 60, len(windows))
    block_dir = work_dir / "demucs_blocks"
    block_dir.mkdir(parents=True, exist_ok=True)
    adder = OverlapAdder(total, overlap, channels=2)

    for index, (start, end) in enumerate(windows):
        if progress is not None:
            progress(index / len(windows), desc=f"Demucs block {index + 1}/{len(windows)}")
        piece = write_wav(
            block_dir / f"block_{index:04d}.wav", read_block(decoded, start, end - start), DEMUCS_SR
        )
        _, stems = _separate_file(piece)
        adder.add(start, end, _mix_stems(stems, keep_percussion))
        piece.unlink(missing_ok=True)
        _free_gpu()

    decoded.unlink(missing_ok=True)
    return adder.result(), DEMUCS_SR


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

def separate(
    engine: str,
    src: Path,
    work_dir: Path,
    keep_percussion: bool = False,
    progress=None,
) -> tuple[np.ndarray, int]:
    """Run `src` through the named engine and return (audio, sample_rate)."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if engine == ENGINE_CLEARVOICE:
        wav = decode_to_wav(src, work_dir / "input_48k_mono.wav", TARGET_SR, mono=True)
        return _with_cpu_fallback(lambda: run_clearvoice(wav, work_dir, progress))

    if engine == ENGINE_DEMUCS:
        return run_demucs(src, keep_percussion, work_dir, progress)

    if engine == ENGINE_CHAIN:
        if progress is not None:
            progress(0.0, desc="Demucs: isolating vocals")
        audio, sr = run_demucs(src, keep_percussion, work_dir, progress)
        staged = write_wav(work_dir / "demucs_vocals.wav", audio, sr)
        del audio
        # Round-trip through ffmpeg so ClearerVoice always sees mono 48k.
        wav = decode_to_wav(staged, work_dir / "chain_48k_mono.wav", TARGET_SR, mono=True)
        staged.unlink(missing_ok=True)
        _free_gpu()
        if progress is not None:
            progress(0.5, desc="ClearerVoice: scrubbing residual bleed")
        return _with_cpu_fallback(lambda: run_clearvoice(wav, work_dir, progress))

    raise EngineError(f"unknown engine: {engine!r}")


def _with_cpu_fallback(call):
    """Run `call`, and on CUDA OOM drop the cached model and retry on CPU."""
    global _clearvoice_model
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - deliberate OOM recovery
        if not _is_oom(exc):
            raise
        log.warning("CUDA OOM, retrying on CPU (slower)")
        _clearvoice_model = None
        _free_gpu()
        import torch

        original = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            return call()
        finally:
            torch.cuda.is_available = original
