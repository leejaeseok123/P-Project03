#!/usr/bin/env python3
"""Local FSS Korean STT: uv run python data/fss_transcriber.py --limit 1.

Ubuntu WSL2 / NVIDIA CUDA FP16 only. No source media or collector edits.
The model downloads from Hugging Face; media and transcripts never leave this host.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import inspect
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "voice_phishing/fss"
MODEL_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
MEDIA_EXTS = {".mp4", ".mp3", ".wav", ".m4a", ".wma", ".wmv", ".avi", ".mov", ".webm", ".ogg", ".aac", ".flac", ".mkv"}
LOG = logging.getLogger("fss_transcriber")
# Hard timestamp compensation inserts silence for gaps/leading offsets. Soft
# compensation is disabled: no time stretching, denoising, gain, VAD or trimming.
AUDIO_FILTER = "aresample=16000:async=1:max_soft_comp=0:first_pts=0"
SCHEMA_VERSION = 1


class EnvironmentFailure(RuntimeError):
    """Abort the entire run; never fall back to CPU."""


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def atomic_text(path: Path, value: str):
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(args, timeout=60):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{args[0]} exit {result.returncode}: {result.stderr[-4000:].strip()}")
    return result.stdout


def probe(path):
    return json.loads(command(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))


def positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def audio_duration(info, path=None):
    stream = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
    if stream is None:
        raise ValueError("No audio stream")
    origin = float(info["format"].get("start_time") or 0)
    offset = float(stream.get("start_time") or origin) - origin
    duration = stream.get("duration")
    if duration not in (None, "N/A"):
        end = float(duration) + offset
    elif path is not None:
        # Container duration may include a longer video track. Use packet ends
        # when the audio stream does not expose its own duration (e.g. Matroska).
        packets = json.loads(command(["ffprobe", "-v", "error", "-select_streams", "a:0",
                                      "-show_packets", "-show_entries", "packet=pts_time,duration_time",
                                      "-of", "json", str(path)]))["packets"]
        ends = [float(p["pts_time"]) + float(p.get("duration_time") or 0) - origin
                for p in packets if p.get("pts_time") not in (None, "N/A")]
        end = max(ends, default=0)
    else:
        raise ValueError("Audio stream has no duration")
    if not positive(end):
        raise ValueError("Invalid audio duration")
    return end


def validate_wav(path, expected_duration):
    info = probe(path)
    streams = info.get("streams", [])
    if len(streams) != 1:
        raise ValueError("WAV must contain exactly one audio stream")
    stream = streams[0]
    if (stream.get("codec_name"), stream.get("sample_rate"), stream.get("channels"), stream.get("bits_per_sample")) != ("pcm_s16le", "16000", 1, 16):
        raise ValueError("WAV must be 16000 Hz mono PCM s16le")
    duration = audio_duration(info)
    if abs(duration - expected_duration) > 0.25:
        raise ValueError(f"Audio timeline mismatch: expected {expected_duration:.6f}s, got {duration:.6f}s")
    return duration


def extract(video, root, force, ffmpeg_version):
    started = time.perf_counter()
    wav = root / "audio" / (video.stem + ".wav")
    sidecar = wav.with_suffix(".json")
    source_hash = sha256(video)
    expected = audio_duration(probe(video), video)
    settings = {"filter": AUDIO_FILTER, "copyts": True, "start_at_zero": True,
                "map": "0:a:0", "codec": "pcm_s16le", "sample_rate": 16000,
                "channels": 1, "ffmpeg_version": ffmpeg_version}
    if not force:
        try:
            saved = read_json(sidecar)
            if saved["source_sha256"] == source_hash and saved["settings"] == settings and saved["audio_sha256"] == sha256(wav):
                duration = validate_wav(wav, expected)
                return dict(path=wav, source_sha256=source_hash, audio_sha256=saved["audio_sha256"],
                            duration_sec=duration, reused=True, extraction_time_sec=time.perf_counter() - started)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            pass
    fd, name = tempfile.mkstemp(prefix=f".{video.stem}.", suffix=".wav", dir=wav.parent)
    os.close(fd)
    temp = Path(name)
    try:
        command(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-threads", "1",
                 "-copyts", "-start_at_zero", "-i", str(video), "-map", "0:a:0", "-vn",
                 "-af", AUDIO_FILTER, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(temp)], timeout=3600)
        duration = validate_wav(temp, expected)
        wav_hash = sha256(temp)
        with temp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp, wav)
        atomic_text(sidecar, json_text(dict(source_sha256=source_hash, audio_sha256=wav_hash,
                                           duration_sec=duration, settings=settings)))
        return dict(path=wav, source_sha256=source_hash, audio_sha256=wav_hash,
                    duration_sec=duration, reused=False, extraction_time_sec=time.perf_counter() - started)
    finally:
        temp.unlink(missing_ok=True)


def prepare_cuda_libraries(reexec_script=None):
    """The dynamic loader needs the pip NVIDIA paths before Python imports CT2."""
    import nvidia.cublas.lib
    import nvidia.cudnn.lib
    paths = [str(Path(next(iter(mod.__path__))).resolve()) for mod in (nvidia.cublas.lib, nvidia.cudnn.lib)]
    old = os.environ.get("LD_LIBRARY_PATH", "")
    wanted = ":".join(paths)
    if not old.startswith(wanted):
        env = dict(os.environ, LD_LIBRARY_PATH=wanted + (":" + old if old else ""),
                   HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
        script = Path(reexec_script or __file__).resolve()
        os.execve(sys.executable, [sys.executable, str(script), *sys.argv[1:]], env)


def gpu_sample():
    line = command(["nvidia-smi", "--id=0", "--query-gpu=name,driver_version,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], timeout=10).strip()
    name, driver, memory, util = [p.strip() for p in line.split(",")]
    def number(value):
        try:
            return float(value)
        except ValueError:
            return None
    return dict(name=name, driver_version=driver, memory_used_mib=number(memory), utilization_percent=number(util))


class GpuMonitor:
    """Device-wide samples include Windows/other processes; not process allocation."""
    def __init__(self):
        self.stop = threading.Event()
        self.max_memory = None
        self.max_util = None
        self.samples = 0
        self.errors = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        try:
            value = gpu_sample()
            self.samples += 1
            if value["memory_used_mib"] is not None:
                self.max_memory = max(self.max_memory or 0, value["memory_used_mib"])
            if value["utilization_percent"] is not None:
                self.max_util = max(self.max_util or 0, value["utilization_percent"])
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
            self.errors += 1

    def run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(0.5)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=12)
        self.sample()


def environment():
    import ctranslate2
    result = {"python": sys.version.split()[0], "gpu": gpu_sample(), "packages": {
        name: version(name) for name in ("faster-whisper", "ctranslate2", "nvidia-cublas-cu12", "nvidia-cudnn-cu12", "av")}}
    if ctranslate2.get_cuda_device_count() < 1 or "float16" not in ctranslate2.get_supported_compute_types("cuda", 0):
        raise EnvironmentFailure("CTranslate2 CUDA FP16 is unavailable")
    # Force dependent libraries to load before native inference can abort on dlopen.
    for name in ("libcublasLt.so.12", "libcublas.so.12", "libcudnn.so.9", "libcudnn_ops.so.9", "libcudnn_cnn.so.9", "libcudnn_adv.so.9"):
        ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
    driver = ctypes.CDLL("libcuda.so.1")
    v = ctypes.c_int()
    if driver.cuInit(0) != 0 or driver.cuDriverGetVersion(ctypes.byref(v)) != 0:
        raise EnvironmentFailure("CUDA driver initialization failed")
    result["cuda_driver_api_version"] = v.value
    cudnn = ctypes.CDLL("libcudnn.so.9")
    cudnn.cudnnGetVersion.restype = ctypes.c_size_t
    result["cudnn_runtime_version"] = cudnn.cudnnGetVersion()
    result["cuda_toolkit_nvcc"] = command(["nvcc", "--version"]).strip() if shutil.which("nvcc") else None
    return result


def stt_settings(vad):
    return dict(language="ko", task="transcribe", beam_size=5, temperature=0.0,
                word_timestamps=True, vad_filter=vad, condition_on_previous_text=True,
                initial_prompt=None)


def signature(settings, env):
    return dict(schema_version=SCHEMA_VERSION, model="large-v3", model_repository="Systran/faster-whisper-large-v3",
                model_revision=MODEL_REVISION, backend="faster-whisper", device="cuda", device_index=0,
                compute_type="float16", settings=settings, packages=env["packages"])


def validate_transcript(data, text, duration):
    warnings = []
    if not isinstance(data["text"], str) or text != data["text"]:
        raise ValueError("JSON/TXT mismatch")
    if data["text"] != "".join(s["text"] for s in data["segments"]):
        raise ValueError("Full text does not match raw segment text")
    if not positive(data["processing_time_sec"]) or not math.isclose(data["rtf"], data["processing_time_sec"] / duration, rel_tol=1e-9):
        raise ValueError("Invalid inference timing/RTF")
    if not math.isclose(data["duration_sec"], duration, abs_tol=1/16000):
        raise ValueError("Duration mismatch")
    previous = 0.0
    for segment in data["segments"]:
        for field in ("id", "text", "avg_logprob", "no_speech_prob", "compression_ratio", "words"):
            if field not in segment:
                raise ValueError(f"Missing segment field: {field}")
        start, end = segment["start"], segment["end"]
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end)) or not 0 <= start <= end <= duration + 1/16000:
            raise ValueError(f"Segment timestamp outside audio: {start}, {end}; duration={duration}")
        if start < previous:
            raise ValueError("Segment timestamps out of order")
        previous = start
        word_previous = start
        for word in segment["words"]:
            a, b = word["start"], word["end"]
            if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (a, b)) or not 0 <= a <= b <= duration + 1/16000:
                raise ValueError(f"Word timestamp outside audio: {a}, {b}")
            if a < word_previous - 0.02 or a < start - 0.02 or b > end + 0.02:
                warnings.append(dict(segment_id=segment["id"], word_start=a, word_end=b,
                                     reason="Word overlaps segment boundary or previous word; raw output preserved"))
            word_previous = a
            if not isinstance(word["word"], str) or "probability" not in word:
                raise ValueError("Invalid word record")
    # Strict finite numeric JSON, including optional model metrics.
    json_text(data)
    return warnings


def reusable(json_path, txt_path, audio, sig, video, record):
    try:
        if record.get("json_sha256") != sha256(json_path) or record.get("txt_sha256") != sha256(txt_path):
            return None
        data = read_json(json_path)
        if (data["source_sha256"] != audio["source_sha256"] or data["audio_sha256"] != audio["audio_sha256"]
                or data["reproducibility"] != sig or data["id"] != video.stem
                or data["video_path"] != "raw/" + video.name or data["audio_path"] != "audio/" + audio["path"].name):
            return None
        validate_transcript(data, txt_path.read_text(encoding="utf-8"), audio["duration_sec"])
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_model(settings):
    from faster_whisper import WhisperModel
    if not set(settings).issubset(inspect.signature(WhisperModel.transcribe).parameters):
        raise EnvironmentFailure("Installed faster-whisper transcribe API is incompatible")
    LOG.info("Loading large-v3 once: CUDA device 0, float16, revision %s", MODEL_REVISION)
    started = time.perf_counter()
    try:
        model = WhisperModel("large-v3", device="cuda", device_index=0, compute_type="float16",
                             revision=MODEL_REVISION, num_workers=1)
        if model.model.device != "cuda" or model.model.compute_type != "float16":
            raise EnvironmentFailure("Model did not initialize as CUDA FP16")
        return model, time.perf_counter() - started
    except Exception as exc:
        raise EnvironmentFailure(f"GPU/model initialization failed: {exc}") from exc


def transcribe(model, audio, video, sig):
    started = time.perf_counter()
    segments, info = model.transcribe(str(audio["path"]), **sig["settings"])
    # Consume the lazy generator inside the inference timer. Preserve raw strings.
    output = []
    for s in segments:
        output.append(dict(id=s.id, start=s.start, end=s.end, text=s.text,
                           avg_logprob=getattr(s, "avg_logprob", None),
                           no_speech_prob=getattr(s, "no_speech_prob", None),
                           compression_ratio=getattr(s, "compression_ratio", None),
                           words=[dict(start=w.start, end=w.end, word=w.word, probability=getattr(w, "probability", None)) for w in (s.words or [])]))
    elapsed = time.perf_counter() - started
    data = dict(id=video.stem, source="FSS", video_path="raw/" + video.name,
                audio_path="audio/" + audio["path"].name, language=info.language,
                model="large-v3", backend="faster-whisper", device="cuda", compute_type="float16",
                duration_sec=audio["duration_sec"], processing_time_sec=elapsed,
                rtf=elapsed / audio["duration_sec"], text="".join(s["text"] for s in output),
                segments=output, source_sha256=audio["source_sha256"], audio_sha256=audio["audio_sha256"],
                reproducibility=sig, empty_transcript=not any(s["text"].strip() for s in output),
                created_at=datetime.now(timezone.utc).isoformat())
    # Invalid timestamps are retained for diagnosis, never clamped or corrected.
    return data


def gpu_failure(exc):
    message = str(exc).lower()
    return any(s in message for s in ("cuda", "cublas", "cudnn", "gpu", "out of memory", "driver", "kernel", "device-side", "allocate memory"))


def discover(root):
    raw = root / "raw"
    if not raw.is_dir():
        raise ValueError(f"Missing input raw directory: {raw}")
    videos = sorted(p for p in raw.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTS and not p.name.startswith("."))
    if len({p.stem for p in videos}) != len(videos):
        raise ValueError("Duplicate source stems: cannot use stems as unique IDs")
    # Metadata is inspected only to preserve collector order. It is never a prompt.
    order = {}
    manifest = root / "metadata.jsonl"
    if manifest.exists():
        for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
                if row.get("status") == "ok" and row.get("file_path"):
                    order.setdefault(Path(row["file_path"]).name, len(order))
            except (ValueError, TypeError):
                LOG.warning("Malformed metadata row %d", line_no)
    return sorted(videos, key=lambda p: (order.get(p.name, len(order)), p.name))


def run(args, videos, total, root):
    started = time.perf_counter()
    summary = dict(started_at=datetime.now(timezone.utc).isoformat(), status="running", input_path=str(root),
                   total_videos=total, selected_videos=len(videos), audio_success=0, audio_failed=0, audio_reused=0,
                   stt_success=0, stt_failed=0, transcripts_reused=0, total_audio_duration_sec=0.0,
                   total_inference_time_sec=0.0, inferred_audio_duration_sec=0.0, model_load_time_sec=0.0,
                   total_segments=0, total_characters=0, empty_transcripts=0, failures=[], files=[],
                   audio_only=args.audio_only, workers=args.workers)
    monitor = None
    pool = None
    futures = []
    rows = {}
    model = None
    exit_code = 0
    path = root / "transcription_manifest.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
                rows[value["id"]] = value
            except (ValueError, KeyError, TypeError):
                LOG.warning("Ignoring malformed transcription manifest row")

    def save_row(row):
        row["updated_at"] = datetime.now(timezone.utc).isoformat()
        rows[row["id"]] = row
        atomic_text(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows.values()))

    try:
        for tool in ("ffmpeg", "ffprobe"):
            if not shutil.which(tool):
                raise EnvironmentFailure(f"Required tool missing: {tool}")
        ffmpeg_version = command(["ffmpeg", "-version"]).splitlines()[0]
        summary["ffmpeg_version"] = ffmpeg_version
        if not args.audio_only:
            summary["environment"] = environment()
            sig = signature(stt_settings(args.vad_filter), summary["environment"])
            summary["reproducibility"] = sig
            monitor = GpuMonitor()
            monitor.thread.start()
        pool = ThreadPoolExecutor(max_workers=args.workers)
        # Only extraction is concurrent; all inference uses one model on this thread.
        futures = [pool.submit(extract, video, root, args.force, ffmpeg_version) for video in videos]
        for video, future in zip(videos, futures):
            row = dict(id=video.stem, video_path="raw/" + video.name)
            try:
                audio = future.result()
            except Exception as exc:
                if isinstance(exc, (PermissionError, FileNotFoundError)) or (isinstance(exc, OSError) and exc.errno in (28, 5)):
                    raise EnvironmentFailure(f"Filesystem/tool failure: {exc}") from exc
                row.update(status="audio_failed", error=str(exc))
                summary["audio_failed"] += 1
                summary["failures"].append(dict(row))
                summary["files"].append(row)
                save_row(row)
                LOG.error("%s audio failed: %s", video.stem, exc)
                continue
            summary["audio_success"] += 1
            summary["audio_reused"] += int(audio["reused"])
            summary["total_audio_duration_sec"] += audio["duration_sec"]
            row.update(audio_path="audio/" + audio["path"].name, source_sha256=audio["source_sha256"],
                       audio_sha256=audio["audio_sha256"], duration_sec=audio["duration_sec"],
                       audio_reused=audio["reused"], extraction_time_sec=audio["extraction_time_sec"])
            if args.audio_only:
                row["status"] = "audio_ok"
                old = rows.get(video.stem, {})
                if old.get("source_sha256") == audio["source_sha256"] and old.get("audio_sha256") == audio["audio_sha256"]:
                    for key in ("json_sha256", "txt_sha256"):
                        if key in old:
                            row[key] = old[key]
            else:
                json_path = root / "transcripts" / (video.stem + ".json")
                txt_path = json_path.with_suffix(".txt")
                data = None if args.force else reusable(json_path, txt_path, audio, sig, video, rows.get(video.stem, {}))
                try:
                    if data is not None:
                        summary["transcripts_reused"] += 1
                        row["status"] = "reused"
                    else:
                        if model is None:
                            model, summary["model_load_time_sec"] = load_model(sig["settings"])
                        LOG.info("Transcribing %s (%.3fs)", video.stem, audio["duration_sec"])
                        data = transcribe(model, audio, video, sig)
                        summary["total_inference_time_sec"] += data["processing_time_sec"]
                        summary["inferred_audio_duration_sec"] += audio["duration_sec"]
                        try:
                            data["timestamp_warnings"] = validate_transcript(data, data["text"], audio["duration_sec"])
                        except (ValueError, KeyError, TypeError) as exc:
                            atomic_text(json_path.with_suffix(".invalid.json"), json_text(dict(data, validation_error=str(exc))))
                            raise
                        # Commit TXT then JSON. Interrupted pairs fail validation on resume.
                        atomic_text(txt_path, data["text"])
                        atomic_text(json_path, json_text(data))
                        validate_transcript(read_json(json_path), txt_path.read_text(encoding="utf-8"), audio["duration_sec"])
                        summary["stt_success"] += 1
                        row["status"] = "ok"
                    row.update(processing_time_sec=data["processing_time_sec"], rtf=data["rtf"],
                               segments=len(data["segments"]), characters=len(data["text"]), empty_transcript=data["empty_transcript"],
                               reproducibility=sig, json_sha256=sha256(json_path), txt_sha256=sha256(txt_path),
                               timestamp_warning_count=len(data.get("timestamp_warnings", [])))
                    summary["total_segments"] += row["segments"]
                    summary["total_characters"] += row["characters"]
                    summary["empty_transcripts"] += int(row["empty_transcript"])
                    LOG.info("%s %s: %.3fs, RTF %.4f, %d segments, %d characters%s", video.stem, row["status"], row["processing_time_sec"], row["rtf"], row["segments"], row["characters"], " EMPTY" if row["empty_transcript"] else "")
                except Exception as exc:
                    row.update(status="stt_failed", error=str(exc))
                    summary["stt_failed"] += 1
                    summary["failures"].append(dict(row))
                    save_row(row)
                    if isinstance(exc, EnvironmentFailure) or gpu_failure(exc) or isinstance(exc, OSError):
                        summary["files"].append(row)
                        raise EnvironmentFailure(f"STT environment failure: {exc}") from exc
                    LOG.error("%s STT failed: %s", video.stem, exc)
            summary["files"].append(row)
            save_row(row)
        summary["status"] = "completed_with_errors" if summary["failures"] else "completed"
        exit_code = 1 if summary["failures"] else 0
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        exit_code = 130
        LOG.warning("Interrupted; completed files can be reused")
    except Exception as exc:
        summary.update(status="aborted", fatal_error=f"{type(exc).__name__}: {exc}")
        LOG.error("Aborting without CPU fallback: %s", summary["fatal_error"])
        exit_code = 2
    finally:
        for future in futures:
            future.cancel()
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        if monitor:
            monitor.close()
            summary.update(gpu_max_memory_used_mib=monitor.max_memory, gpu_max_utilization_percent=monitor.max_util,
                           gpu_samples=monitor.samples, gpu_sampling_errors=monitor.errors,
                           gpu_measurement_scope="device 0 total including other processes, sampled every 0.5s")
        summary["pipeline_time_sec"] = time.perf_counter() - started
        duration = summary["inferred_audio_duration_sec"]
        summary["overall_rtf"] = summary["total_inference_time_sec"] / duration if duration else None
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_text(root / "transcription_summary.json", json_text(summary))
        # Preserve run statistics across the cache verification run.
        run_id = summary["started_at"].replace(":", "").replace("+", "_")
        atomic_text(root / "runs" / (run_id + ".json"), json_text(summary))
        LOG.info("%s: audio=%d failed=%d, STT=%d failed=%d reused=%d, inference=%.3fs, RTF=%s, pipeline=%.3fs",
                 summary["status"], summary["audio_success"], summary["audio_failed"], summary["stt_success"],
                 summary["stt_failed"], summary["transcripts_reused"], summary["total_inference_time_sec"],
                 summary["overall_rtf"], summary["pipeline_time_sec"])
    return exit_code


def main():
    parser = argparse.ArgumentParser(description="FSS local Korean large-v3 CUDA FP16 transcription")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="FSS directory containing raw/ and metadata.jsonl")
    parser.add_argument("--limit", type=int, help="Process only the first N media files in collector order")
    parser.add_argument("--workers", type=int, default=2, help="FFmpeg extraction workers (1-8), STT remains sequential")
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-extract audio and transcribe even if valid cached outputs exist")
    parser.add_argument("--dry-run", action="store_true", help="List selected inputs without writing or loading a model")
    parser.add_argument("--vad-filter", action="store_true", help="Enable VAD for comparison; default is disabled")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    root = args.input.expanduser().resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        videos = discover(root)
        total = len(videos)
        videos = videos[:args.limit]
        if not videos:
            parser.error(f"No media found in {root / 'raw'}")
        if args.dry_run:
            print(json_text(dict(input=str(root), total_videos=total, selected_videos=len(videos),
                                 audio_only=args.audio_only, settings=stt_settings(args.vad_filter),
                                 videos=[str(p.relative_to(root)) for p in videos])), end="")
            return 0
        if not args.audio_only:
            prepare_cuda_libraries()
        for directory in ("audio", "transcripts", "runs"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        # flock is released by the kernel on Ctrl+C, errors or process termination.
        with (root / ".transcription.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise EnvironmentFailure("Another transcriber is already using this dataset")
            handler = logging.FileHandler(root / "transcription.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logging.getLogger().addHandler(handler)
            return run(args, videos, total, root)
    except (OSError, ValueError, EnvironmentFailure, ImportError) as exc:
        LOG.error("Startup failed without CPU fallback: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
