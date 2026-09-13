# Clearer Voice endpoint

A small Gradio app that takes a song and gives back the vocals with the music
removed. Outputs land in `clean songs/`.

## Set up on another PC

The shell scripts target Linux. On a Windows home PC, use Ubuntu under WSL2
and run the commands below in its terminal. Install Git, Conda (Miniconda or
Anaconda), and an NVIDIA driver that supports CUDA in your chosen environment.
For WSL2, install the NVIDIA driver on Windows, not a Linux display driver
inside WSL. See the [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
Confirm that `nvidia-smi` sees your GPU before proceeding.

```bash
git clone git@github.com:omarkshoaib/music-cleaning-tool.git
cd music-cleaning-tool
conda install -y -n base -c conda-forge ffmpeg
./setup_env.sh
./run.sh
```

If this PC does not have a GitHub SSH key configured, clone with HTTPS instead:

```bash
git clone https://github.com/omarkshoaib/music-cleaning-tool.git
```

`setup_env.sh` creates the `clearvoice_endpoint` Conda environment with Python
3.10 and installs the Python dependencies. FFmpeg and FFprobe must be on PATH;
the Conda command above installs both. The setup script pins PyTorch and
Torchaudio to 2.4.1 with CUDA 12.1. Verify GPU access inside the environment:

```bash
conda activate clearvoice_endpoint
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

On the home PC, confirm this prints `CUDA: True` and your RTX 4060 Ti.
This repository has not yet been tested on that PC. If an engine runs out of
GPU memory, the application retries on CPU, which is slower.

Model weights download on first use, so the first run needs internet access.
Weights, generated audio in `clean songs/`, scratch files, and logs are not
stored in Git. Copy existing audio separately if you want it on both PCs.

## Run

```bash
./run.sh
```

Open **http://localhost:8731** in your browser and drop in an mp3.
The server binds to `0.0.0.0`, so other machines can use your PC's IP address
if its firewall allows access. The app has no login; keep access to a trusted
network. Override the port with `CV_PORT=9000 ./run.sh` if needed.

A fresh clone does not install any background service or cron jobs.

**Tick one engine** → it runs and the result is saved to `clean songs/`
immediately.

**Tick several** → they all run and appear side by side as players. Nothing is
written to `clean songs/` yet. Listen, choose the best in *Choose the one to
keep*, and press **Save this one, delete the rest** — the winner is saved and
the other renders are deleted from staging.

Unsaved renders live in `.work/staging/` and are wiped when you start another
run or restart the app, so an abandoned comparison never leaks disk.

## Updating either PC

Stop the foreground app with Ctrl+C, then run:

```bash
git pull --ff-only
./run.sh
```

Run `./setup_env.sh` again if dependencies change. To share code changes from
either PC, commit and push them before pulling on the other PC. Generated
songs remain local and are not synchronized by Git.

## Optional background service (Linux)

`keepalive.sh` can start the app and restart it after repeated failed health
checks. It expects Conda at `~/miniconda3` and requires Linux tools including
`flock`, `ss`, and `curl`. Use `./run.sh` for normal desktop use.

```bash
./keepalive.sh --now
tail -f logs/keepalive.log
```

For automatic checks, use `crontab -e` and add entries using the absolute path
to your clone:

```cron
@reboot /absolute/path/to/music-cleaning-tool/keepalive.sh --now
*/2 * * * * /absolute/path/to/music-cleaning-tool/keepalive.sh
```

These entries are optional and must be installed separately on each machine.
Remove only these entries with `crontab -e` when disabling the watchdog.
Server logs are pruned after 14 days. A restart discards unsaved comparisons;
audio already saved to `clean songs/` remains intact.

## Engines

| Engine | What it does | Notes |
| --- | --- | --- |
| `ClearerVoice (MossFormer2_SE_48K)` | Speech enhancement — treats instruments as noise to suppress | Full-band 48 kHz, mono out. 20 s windows, 1 s crossfades. |
| `Demucs (htdemucs)` | Music source separation — pulls the vocal stem out | Built for music; usually the better result on real songs. Stereo out at 44.1 kHz. 7 min blocks, 2 s crossfades. |
| `Demucs → ClearerVoice` | Demucs isolates, ClearerVoice scrubs the leftover bleed | Slowest, cleanest. Mono out. |

**Keep percussion (duff)** folds the drums stem back into the mix instead of
discarding it. It only applies to the two Demucs engines — ClearerVoice on its
own has no notion of separate stems, so the checkbox greys out.

## Output

Files are named `<original name>__<engine>[+duff].mp3` at 192 kbps, so
comparisons of the same song sit next to each other. An existing file is never
overwritten — a `(2)`, `(3)`, … counter is appended instead.

Each save appends a line to `clean songs/_log.jsonl` recording the input, which
engines were compared, which one you chose, which were discarded, durations and
the output path. Over time that tells you which engine actually wins on your
material.

## Testing

```bash
conda activate clearvoice_endpoint
python test_smoke.py            # every engine end to end
python test_smoke.py demucs     # substring match
python test_flow.py             # the save rules
python test_long.py             # the windowing that removes the length limit
```

`test_smoke.py` mixes a synthetic chord over a speech clip, runs it through
each engine, and checks the output is the right length, not silent, free of
NaNs, and has measurably less energy in the music band than the input.

`test_flow.py` covers the save behaviour: one engine auto-saves, several stage
without writing anything, and pressing Save keeps exactly the chosen render and
deletes the others. It deliberately picks the second engine so that "discard
the rest" cannot pass while quietly keeping everything.

`test_long.py` covers block reassembly. It shrinks the block sizes so a short
clip exercises the same multi-block path an hour-long song would, then checks a
constant signal reconstructs flat (no dip at the seams) and that both engines
return the full track length with no dropouts. It runs in seconds rather than
needing a real hour of audio.

## Limits and caveats

- No length limit. Long tracks are processed in windows and crossfaded back
  together: ClearerVoice in 20 s windows, Demucs in 7 min blocks. Peak memory
  is set by the window size, not the length of the song, so an hour-long file
  uses about as much VRAM as a three-minute one.
- Time scales with length, though. Budget roughly the same wall-clock per
  minute of audio whatever the track length, and remember that ticking all
  three engines runs them in sequence.
- Neither model is perfect. Dense mixes leave artifacts, and vocals recorded
  with heavy reverb keep some of that reverb tail — the effect sits on the
  voice, so separating it out is not something these models can do.
- Jobs run one at a time (`default_concurrency_limit=1`) to avoid two models
  competing for the same GPU. Selecting all three engines runs them in
  sequence, so expect roughly the sum of their individual times.
- If one engine fails mid-comparison the others still finish; the failure is
  reported in the status box rather than sinking the whole run.
- On CUDA OOM the pipeline retries on CPU automatically. It finishes, just far
  slower.

## Layout

```
app.py         Gradio UI and job orchestration
engines.py     the three engines behind one `separate()` call
audio_io.py    ffmpeg decode/encode, chunking, overlap-add crossfade
test_smoke.py  end-to-end check of every engine
setup_env.sh   conda env creation and dependency install
run.sh         activate env and launch
clean songs/   outputs + _log.jsonl
.work/         scratch space, cleaned up per job
```
