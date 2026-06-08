import subprocess
import shutil
import tempfile
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

SILENCE_THRESH_DBFS = -40
MIN_SILENCE_MS = 650
MIN_CHUNK_MS = 1000
# Coefficient of variation of pitch: below this = stable = singing
SINGING_CV_THRESHOLD = 0.6

DEBUG = True


def _debug(*args) -> None:
    if DEBUG:
        print(*args)


def _remove_background(input_path: Path, output_path: Path) -> None:
    """Use demucs to strip background music, keeping only vocals."""
    demucs_out = output_path.parent / "_demucs_tmp"
    subprocess.run(
        ["python", "-m", "demucs", "--two-stems=vocals", "-o", str(demucs_out), str(input_path)],
        check=True,
    )
    vocals_path = demucs_out / "htdemucs" / input_path.stem / "vocals.wav"
    shutil.move(str(vocals_path), str(output_path))
    shutil.rmtree(demucs_out)


def _pitch_cv(segment: AudioSegment) -> float:
    """Coefficient of variation of pitch over voiced frames.

    Low CV = stable pitch = singing. High CV = variable pitch = speech.
    """
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if segment.channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)

    sr = segment.frame_rate
    win = int(sr * 0.05)       # 50 ms analysis window
    hop = win // 2
    min_period = int(sr / 1000)  # cap at 1000 Hz
    max_period = int(sr / 70)    # floor at 70 Hz

    pitches = []
    for i in range(0, len(samples) - win, hop):
        w = samples[i : i + win]
        corr = np.correlate(w, w, mode="full")[len(w):]
        if max_period >= len(corr):
            continue
        peak = int(np.argmax(corr[min_period:max_period])) + min_period
        if corr[peak] > 0.15 * corr[0]:  # voiced frame check
            pitches.append(sr / peak)

    if len(pitches) < 5:
        return 1.0  # too few voiced frames → treat as speech

    arr = np.array(pitches)
    return float(np.std(arr) / (np.mean(arr) + 1e-8))


def _classify(segment: AudioSegment) -> str:
    cv = _pitch_cv(segment)
    label = "singing" if cv < SINGING_CV_THRESHOLD else "speech"
    _debug(f"  cv={cv:.3f} -> {label}")
    return label


def process(input_path: Path, output_base: Path) -> dict[str, list[Path]]:
    """Remove background music, split by silence, classify speech vs singing.

    Outputs to output_base/speech/ and output_base/singing/.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        vocals_path = Path(f.name)
    try:
        _remove_background(input_path, vocals_path)

        audio = AudioSegment.from_file(vocals_path)
        ranges = detect_nonsilent(
            audio,
            min_silence_len=MIN_SILENCE_MS,
            silence_thresh=SILENCE_THRESH_DBFS,
        )

        stem = input_path.stem
        result: dict[str, list[Path]] = {}
        for i, (start_ms, end_ms) in enumerate(ranges):
            duration_ms = end_ms - start_ms
            if duration_ms < MIN_CHUNK_MS:
                _debug(f"  chunk {i:04d} {start_ms/1000:.2f}s-{end_ms/1000:.2f}s ({duration_ms}ms) -> skipped (too short)")
                continue
            _debug(f"  chunk {i:04d} {start_ms/1000:.2f}s-{end_ms/1000:.2f}s ({duration_ms}ms)")
            chunk = audio[start_ms:end_ms]
            label = _classify(chunk)
            out_dir = output_base / label
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}_{i:04d}.wav"
            chunk.export(out_path, format="wav")
            result.setdefault(label, []).append(out_path)

        return result
    finally:
        vocals_path.unlink(missing_ok=True)
