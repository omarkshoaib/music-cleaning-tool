"""SAM-Audio text-prompted speech isolation."""
from pathlib import Path
import torch

_model = _processor = None
MODEL = "facebook/sam-audio-large-tv"

def separate_shoutout(src: Path, prompt: str, out_dir: Path):
    global _model, _processor
    from sam_audio import SAMAudio, SAMAudioProcessor
    if _model is None:
        _model = SAMAudio.from_pretrained(MODEL).eval().cuda()
        _processor = SAMAudioProcessor.from_pretrained(MODEL)
    batch = _processor(audios=[str(src)], descriptions=[prompt.lower().strip()]).to("cuda")
    with torch.inference_mode():
        result = _model.separate(batch, predict_spans=True, reranking_candidates=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = _processor.audio_sampling_rate
    import torchaudio
    target = out_dir / f"{src.stem}__shoutouts.wav"
    residual = out_dir / f"{src.stem}__without_shoutouts.wav"
    torchaudio.save(str(target), result.target[0].detach().cpu().unsqueeze(0), sr)
    torchaudio.save(str(residual), result.residual[0].detach().cpu().unsqueeze(0), sr)
    return target, residual
