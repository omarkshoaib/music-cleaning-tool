"""Audio decode/encode helpers built on ffmpeg, plus chunking for long tracks."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 48000
MP3_BITRATE = "192k"


class AudioError(RuntimeError):
    """Raised when an input file is not decodable audio."""


@dataclass
class Probe:
    duration: float
    sample_rate: int
    channels: int
    codec: str


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise AudioError(f"{tool} not found on PATH; install ffmpeg")
    return path


def probe(path: Path) -> Probe:
    """Validate that `path` holds decodable audio and report its parameters."""
    cmd = [
        _require("ffprobe"), "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,codec_name:format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioError(f"could not read '{path.name}' as audio: {result.stderr.strip()[:200]}")

    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise AudioError(f"'{path.name}' contains no audio stream")

    stream = streams[0]
    duration = float(payload.get("format", {}).get("duration") or 0.0)
    if duration <= 0:
        raise AudioError(f"'{path.name}' has zero-length audio")

    return Probe(
        duration=duration,
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        codec=str(stream.get("codec_name") or "unknown"),
    )


def decode_to_wav(src: Path, dst: Path, sample_rate: int = TARGET_SR, mono: bool = True) -> Path:
    """Decode any ffmpeg-readable input into a float-friendly 16-bit PCM wav."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _require("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vn",
        "-ac", "1" if mono else "2",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioError(f"ffmpeg failed to decode '{src.name}': {result.stderr.strip()[:300]}")
    return dst


def encode_mp3(src: Path, dst: Path, bitrate: str = MP3_BITRATE) -> Path:
    """Encode a wav to mp3. Writes via a temp file so a crash never leaves a partial mp3."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp3", dir=str(dst.parent), delete=False) as handle:
        staging = Path(handle.name)
    try:
        cmd = [
            _require("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-c:a", "libmp3lame", "-b:a", bitrate,
            str(staging),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise AudioError(f"mp3 encode failed: {result.stderr.strip()[:300]}")
        staging.replace(dst)
    finally:
        staging.unlink(missing_ok=True)
    return dst


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a wav as float32. Shape is (samples,) for mono, (samples, channels) otherwise."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return data, sr


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)
    return path


def peak_normalize(audio: np.ndarray, target_peak: float = 0.97) -> np.ndarray:
    """Scale up to `target_peak`. Never amplifies pure silence into noise."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-6:
        return audio
    return (audio * (target_peak / peak)).astype(np.float32)


def iter_chunks(total_samples: int, chunk: int, overlap: int):
    """Yield (start, end) windows covering `total_samples`, each overlapping the last."""
    if chunk <= overlap:
        raise ValueError("chunk must exceed overlap")
    step = chunk - overlap
    start = 0
    while start < total_samples:
        end = min(start + chunk, total_samples)
        yield start, end
        if end >= total_samples:
            break
        start += step


def frame_count(path: Path) -> int:
    """Number of sample frames in a wav, without reading it into memory."""
    return int(sf.info(str(path)).frames)


def sample_rate_of(path: Path) -> int:
    """Sample rate of a wav, without decoding it."""
    return int(sf.info(str(path)).samplerate)


def read_block(path: Path, start: int, frames: int) -> np.ndarray:
    """Read a slice of a wav as float32, leaving the rest on disk.

    Lets arbitrarily long tracks be processed a window at a time instead of
    holding the whole decoded file in RAM.
    """
    data, _ = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=False)
    return data


class OverlapAdder:
    """Streaming crossfade reassembly.

    Chunks are added as they are produced rather than collected in a list, so
    peak memory is one output buffer instead of two copies of the track.
    Overlapping regions blend with a linear ramp and the accumulated weights
    are divided out, so a constant signal reconstructs to itself rather than to
    a scalloped envelope.
    """

    def __init__(self, total_samples: int, overlap: int, channels: int = 1):
        shape = (total_samples,) if channels == 1 else (total_samples, channels)
        self.out = np.zeros(shape, dtype=np.float32)
        self.weights = np.zeros(total_samples, dtype=np.float32)
        self.total = total_samples
        self.overlap = overlap

    def add(self, start: int, end: int, audio: np.ndarray) -> None:
        span = end - start
        audio = audio[:span]
        if audio.shape[0] < span:
            pad = [(0, span - audio.shape[0])] + [(0, 0)] * (audio.ndim - 1)
            audio = np.pad(audio, pad)

        window = np.ones(span, dtype=np.float32)
        ramp = min(self.overlap, span)
        if ramp > 1:
            fade = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
            if start > 0:
                window[:ramp] = fade
            if end < self.total:
                window[-ramp:] = fade[::-1]

        shaped = window[:, None] if audio.ndim > 1 else window
        self.out[start:end] += audio * shaped
        self.weights[start:end] += window

    def result(self) -> np.ndarray:
        weights = self.weights[:, None] if self.out.ndim > 1 else self.weights
        np.divide(self.out, weights, out=self.out, where=weights > 1e-6)
        return self.out
