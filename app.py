"""Gradio front end: drop in a song, get back the vocals without the music.

Pick one engine and the result is saved straight away. Pick several and they
all run, you compare them side by side, and only the one you choose is kept —
the rest are deleted when you press Save.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

import engines
from audio_io import (
    AudioError,
    encode_mp3,
    peak_normalize,
    probe,
    write_wav,
)
from app_sam import separate_shoutout

from engines import (
    ENGINE_CHOICES,
    ENGINE_CLEARVOICE,
    EngineError,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("clearer_voice_endpoint")

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "clean songs"
WORK_DIR = ROOT / ".work"
STAGING_DIR = WORK_DIR / "staging"
LOG_PATH = OUTPUT_DIR / "_log.jsonl"

# 7860/7861 are Gradio defaults and get squatted by other users on this shared
# box; 8731 is outside that range. Override with CV_PORT if it ever collides.
PORT = int(os.environ.get("CV_PORT", "8731"))

ENGINE_SLUGS = {ENGINE_CLEARVOICE: "clearervoice"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _safe_stem(name: str) -> str:
    """Reduce a filename to something safe to write, keeping Arabic characters."""
    stem = Path(name).stem.strip()
    stem = re.sub(r"[/\\\x00]", "_", stem)
    stem = re.sub(r"\s+", " ", stem)
    return stem[:120] or "song"


def _unique_path(path: Path) -> Path:
    """Never silently overwrite a previous save; append a counter instead."""
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem} ({index}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise EngineError(f"too many existing versions of {path.name}")


def _append_log(record: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _output_name(stem: str, engine: str, keep_percussion: bool) -> str:
    suffix = ENGINE_SLUGS.get(engine, "clean")
    if keep_percussion and engine in PERCUSSION_ENGINES:
        suffix += "+duff"
    return f"{stem}__{suffix}.mp3"


def _commit(staged: Path, filename: str) -> Path:
    """Move a staged render into `clean songs/`."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = _unique_path(OUTPUT_DIR / filename)
    shutil.move(str(staged), str(destination))
    return destination


def _discard(state: dict, keep: str | None = None) -> list[str]:
    """Delete every staged render except `keep`. Returns what was dropped."""
    dropped = []
    for engine, path in (state.get("renders") or {}).items():
        if engine == keep:
            continue
        Path(path).unlink(missing_ok=True)
        dropped.append(engine)
    return dropped


# --------------------------------------------------------------------------
# UI callbacks
# --------------------------------------------------------------------------

def toggle_percussion(selected):
    """The duff checkbox only means something when Demucs is in the pipeline."""
    active = bool(set(selected or []) & PERCUSSION_ENGINES)
    return gr.update(
        interactive=active,
        value=None if active else False,
        info=(
            "Folds the drums stem back in instead of discarding it."
            if active
            else "Needs a Demucs engine — ClearerVoice alone cannot separate percussion."
        ),
    )


def _render_one(engine, src, keep_percussion, stem, staging, progress, position, total):
    """Run a single engine and encode its result into the staging area."""
    def report(fraction, desc=""):
        span = 1.0 / max(total, 1)
        progress(min((position + fraction) * span, 0.99), desc=f"[{position + 1}/{total}] {desc}")

    report(0.05, f"loading {engine}")
    audio, sample_rate = engines.separate(
        engine=engine,
        src=src,
        work_dir=staging / f"work_{ENGINE_SLUGS.get(engine, 'x')}",
        keep_percussion=keep_percussion,
        progress=lambda f, desc="": report(0.05 + 0.85 * float(f), desc),
    )
    report(0.92, "encoding mp3")
    staged_wav = write_wav(staging / f"{ENGINE_SLUGS[engine]}.wav", peak_normalize(audio), sample_rate)
    staged_mp3 = encode_mp3(staged_wav, staging / _output_name(stem, engine, keep_percussion))
    staged_wav.unlink(missing_ok=True)
    return staged_mp3


def run_engines(audio_path, selected, keep_percussion, state, progress=gr.Progress()):
    """Run every selected engine. One engine saves immediately; several stage for comparison."""
    selected = [e for e in ENGINE_CHOICES if e in (selected or [])]  # keep canonical order
    if not audio_path:
        raise gr.Error("Upload an audio file first.")
    if not selected:
        raise gr.Error("Tick at least one engine to run.")

    src = Path(audio_path)
    if not src.exists():
        raise gr.Error(f"Uploaded file vanished: {src.name}")

    try:
        info = probe(src)
    except AudioError as exc:
        raise gr.Error(str(exc)) from exc

    # A new run abandons anything staged but never saved.
    _discard(state or {})
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    staging = STAGING_DIR / str(int(time.time() * 1000))
    staging.mkdir(parents=True, exist_ok=True)

    keep_percussion = bool(keep_percussion)
    stem = _safe_stem(src.name)
    renders: dict[str, str] = {}
    failures: dict[str, str] = {}
    started = time.perf_counter()

    for position, engine in enumerate(selected):
        try:
            path = _render_one(
                engine, src, keep_percussion, stem, staging, progress, position, len(selected)
            )
            renders[engine] = str(path)
        except (EngineError, AudioError) as exc:
            failures[engine] = str(exc)
            log.error("%s failed: %s", engine, exc)
        except Exception as exc:  # noqa: BLE001 - one engine failing must not sink the rest
            failures[engine] = f"{type(exc).__name__}: {exc}"
            log.exception("%s failed", engine)

    elapsed = time.perf_counter() - started

    if not renders:
        detail = "; ".join(f"{e}: {m}" for e, m in failures.items())
        raise gr.Error(f"Every engine failed. {detail}")

    problems = "".join(f"\n⚠ {engine} failed — {message}" for engine, message in failures.items())
    new_state = {
        "renders": renders,
        "stem": stem,
        "keep_percussion": keep_percussion,
        "input": src.name,
        "input_seconds": round(info.duration, 2),
        "elapsed_seconds": round(elapsed, 2),
    }

    players = [
        gr.update(
            value=renders.get(engine),
            visible=engine in renders,
            label=f"{engine}{' + duff' if keep_percussion and engine in PERCUSSION_ENGINES else ''}",
        )
        for engine in ENGINE_CHOICES
    ]

    # Exactly one result: nothing to choose between, so commit it now.
    if len(renders) == 1:
        engine, staged = next(iter(renders.items()))
        saved = _commit(Path(staged), _output_name(stem, engine, keep_percussion))
        _append_log({**new_state, "renders": [engine], "chose": engine, "compared": [], "output": str(saved)})
        players = [
            gr.update(value=str(saved) if e == engine else None, visible=e == engine, label=e)
            for e in ENGINE_CHOICES
        ]
        status = (
            f"Saved — {engine}\n"
            f"{saved}\n"
            f"{elapsed:.0f}s for {info.duration / 60:.1f} min of audio "
            f"({info.duration / max(elapsed, 1e-6):.1f}x realtime).{problems}"
        )
        return (*players, gr.update(visible=False), gr.update(choices=[], value=None),
                status, str(saved), {})

    ranked = [e for e in ENGINE_CHOICES if e in renders]
    status = (
        f"{len(renders)} versions ready in {elapsed:.0f}s. Listen, pick the best, press Save.\n"
        f"Nothing is written to 'clean songs' until you do — the others are deleted.{problems}"
    )
    return (
        *players,
        gr.update(visible=True),
        gr.update(choices=ranked, value=ranked[0]),
        status,
        None,
        new_state,
    )


def save_choice(choice, state):
    """Commit the chosen render and delete the others."""
    renders = (state or {}).get("renders") or {}
    if not renders:
        raise gr.Error("Nothing staged to save. Run the engines first.")
    if not choice:
        raise gr.Error("Pick which version to keep.")
    staged = renders.get(choice)
    if not staged or not Path(staged).exists():
        raise gr.Error(f"The staged file for {choice} is gone. Run it again.")

    keep_percussion = bool(state.get("keep_percussion"))
    saved = _commit(Path(staged), _output_name(state.get("stem", "song"), choice, keep_percussion))
    dropped = _discard(state, keep=choice)

    _append_log(
        {
            "input": state.get("input"),
            "input_seconds": state.get("input_seconds"),
            "elapsed_seconds": state.get("elapsed_seconds"),
            "keep_percussion": keep_percussion,
            "compared": sorted(renders),
            "chose": choice,
            "discarded": sorted(dropped),
            "output": str(saved),
        }
    )
    shutil.rmtree(STAGING_DIR, ignore_errors=True)

    players = [
        gr.update(value=str(saved) if e == choice else None, visible=e == choice, label=e)
        for e in ENGINE_CHOICES
    ]
    status = (
        f"Saved — {choice}\n{saved}\n"
        f"Discarded: {', '.join(dropped) if dropped else 'nothing'}."
    )
    return (*players, gr.update(visible=False), status, str(saved), {})


def list_clearvoice_outputs():
    return sorted(str(p) for p in OUTPUT_DIR.glob("*.mp3") if "clearervoice" in p.name)

def run_shoutout(selected_path, uploaded_path, prompt):
    audio_path = selected_path or uploaded_path
    if not audio_path: raise gr.Error("Select or upload a ClearerVoice output first.")
    if not prompt or not prompt.strip(): raise gr.Error("Describe the shoutout sound.")
    try:
        target, residual = separate_shoutout(Path(audio_path), prompt, WORK_DIR / "shoutouts")
        return str(target), str(residual), f"Saved isolated shoutout and residual audio."
    except Exception as exc:
        log.exception("SAM-Audio failed")
        raise gr.Error(f"SAM-Audio failed: {exc}") from exc


DESCRIPTION = """
# Clearer Voice — vocals without the music

Drop in a song, tick the engines you want to try, get back the voice on its own.

Tick **one** engine and the result is saved to `clean songs/` straight away.
Tick **several** and they all run so you can compare — then only the version you
pick gets saved, and the rest are deleted.

ClearerVoice removes music while preserving speech.

Use SAM-Audio below to isolate producer names and shoutouts from a saved ClearerVoice output.
""".strip()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Clearer Voice — clean songs") as demo:
        gr.Markdown(DESCRIPTION)
        staged_state = gr.State({})
        percussion_state = gr.State(False)

        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Audio(label="Song", type="filepath", sources=["upload", "microphone"])
                engine_select = gr.CheckboxGroup(
                    choices=ENGINE_CHOICES,
                    value=[ENGINE_CLEARVOICE],
                    label="Engine",
                    info="ClearerVoice processes any length in blocks.",
                )
                run = gr.Button("Remove the music", variant="primary")
                status = gr.Textbox(label="Status", lines=4, interactive=False)
                download = gr.File(label="Saved file", interactive=False)
                gr.Markdown("## Find producer names and shoutouts")
                shoutout_source = gr.Dropdown(choices=list_clearvoice_outputs(), label="ClearerVoice output", allow_custom_value=True)
                shoutout_upload = gr.Audio(label="Or upload audio", type="filepath", sources=["upload"])
                shoutout_prompt = gr.Textbox(value="producer name or shoutout", label="Sound to isolate")
                shoutout_run = gr.Button("Isolate shoutouts", variant="secondary")
                shoutout_target = gr.Audio(label="Isolated shoutouts", interactive=False)
                shoutout_residual = gr.Audio(label="Audio without shoutouts", interactive=False)
                shoutout_status = gr.Textbox(label="SAM-Audio status", interactive=False)

            with gr.Column(scale=1):
                players = [
                    gr.Audio(label=engine, type="filepath", interactive=False, visible=False)
                    for engine in ENGINE_CHOICES
                ]
                with gr.Group(visible=False) as compare_box:
                    gr.Markdown("### Choose the one to keep")
                    choice = gr.Radio(choices=[], label="Best version", interactive=True)
                    save = gr.Button("Save this one, delete the rest", variant="primary")

        shoutout_source.change(lambda p: p, inputs=shoutout_source, outputs=shoutout_source)
        shoutout_run.click(run_shoutout, inputs=[shoutout_source, shoutout_upload, shoutout_prompt], outputs=[shoutout_target, shoutout_residual, shoutout_status])
        run.click(
            run_engines,
            inputs=[source, engine_select, percussion_state, staged_state],
            outputs=[*players, compare_box, choice, status, download, staged_state],
        )
        save.click(
            save_choice,
            inputs=[choice, staged_state],
            outputs=[*players, compare_box, status, download, staged_state],
        )

    return demo


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    build_ui().queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=PORT,
        show_error=True,
        theme=gr.themes.Soft(),  # Gradio 6 takes the theme here, not on Blocks
    )
