from __future__ import annotations

import hashlib
import csv
import io
import json
import os
import re
import secrets
import threading
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoModel, AutoProcessor

load_dotenv()

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SIGLIP_MODEL_ID = os.getenv("SIGLIP_MODEL", "google/siglip2-base-patch16-224")
SIGLIP_BASE_MODEL_ID = os.getenv("SIGLIP_BASE_MODEL", "google/siglip2-base-patch16-224")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
FRAME_BATCH_SIZE = max(1, int(os.getenv("FRAME_BATCH_SIZE", "16")))
CACHE_VIDEO_COUNT = max(1, int(os.getenv("CACHE_VIDEO_COUNT", "2")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Query-Guided Video Frame Retrieval")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_video_index: dict[str, dict[str, Any]] = {}
_video_index_lock = threading.Lock()
_model_lock = threading.Lock()
_model: Any = None
_processor: Any = None
_embedding_cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
_cache_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_cancel_events: dict[str, threading.Event] = {}
_executor = None


class SearchCancelled(Exception):
    """Raised at safe checkpoints after a user cancels a search."""


def check_cancelled(job_id: str) -> None:
    with _jobs_lock:
        cancel_event = _cancel_events.get(job_id)
    if cancel_event is not None and cancel_event.is_set():
        raise SearchCancelled("Search cancelled by user.")


class SearchRequest(BaseModel):
    video_id: str
    query: str = Field(min_length=1, max_length=1000)
    selected_frame_count: int = Field(default=10, ge=1, le=200)


class FrameExportItem(BaseModel):
    second: int = Field(ge=0)
    rank: int = Field(ge=1)
    score: float


class FrameExportRequest(BaseModel):
    video_id: str
    query: str = Field(max_length=1000)
    frames: list[FrameExportItem] = Field(min_length=1, max_length=200)


def configured_video_roots() -> list[Path]:
    custom = os.getenv("VIDEO_DIR", "").strip()
    candidates = [Path(custom).expanduser()] if custom else []
    candidates.extend(
        [
            Path("/content/drive/MyDrive/UCA_fixed_test_set/videos"),
            Path("/content/ucf_crimes"),
            Path("/content/drive/MyDrive/ucf_crimes"),
            ROOT / "videos",
        ]
    )
    roots, seen = [], set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved not in seen and resolved.is_dir():
            roots.append(resolved)
            seen.add(resolved)
    return roots


def scan_videos() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for root in configured_video_roots():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            resolved = str(path.resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            video_id = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
            try:
                relative_name = str(path.relative_to(root))
            except ValueError:
                relative_name = path.name
            label = f"{root.name} / {relative_name}"
            cap = cv2.VideoCapture(resolved)
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT) if cap.isOpened() else 0
            cap.release()
            duration = count / fps if fps > 0 and count > 0 else None
            entries.append(
                {
                    "id": video_id,
                    "name": path.name,
                    "label": label,
                    "duration_seconds": duration,
                    "path": resolved,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "size": path.stat().st_size,
                }
            )
    entries.sort(key=lambda item: (item["label"].lower(), item["name"].lower()))
    with _video_index_lock:
        _video_index.clear()
        _video_index.update({entry["id"]: entry for entry in entries})
    return entries


def get_video(video_id: str) -> dict[str, Any]:
    with _video_index_lock:
        entry = _video_index.get(video_id)
    if entry is None:
        scan_videos()
        with _video_index_lock:
            entry = _video_index.get(video_id)
    if entry is None:
        raise KeyError("Video not found. Refresh the video list and select it again.")
    if not Path(entry["path"]).is_file():
        raise KeyError("The video file is no longer available. Refresh the video list.")
    return entry


def load_siglip() -> tuple[Any, Any]:
    global _model, _processor
    with _model_lock:
        if _model is None or _processor is None:
            checkpoint_path = Path(SIGLIP_MODEL_ID).expanduser()
            if checkpoint_path.suffix.lower() == ".safetensors":
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(f"Fine-tuned weights not found: {checkpoint_path}")
                _processor = AutoProcessor.from_pretrained(SIGLIP_BASE_MODEL_ID)
                _model = AutoModel.from_pretrained(SIGLIP_BASE_MODEL_ID)
                try:
                    from safetensors.torch import load_model
                except ImportError as exc:
                    raise RuntimeError("Install the safetensors package to load this checkpoint.") from exc
                load_model(_model, str(checkpoint_path), strict=True, device="cpu")
                _model = _model.to(DEVICE).eval()
            else:
                _processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
                _model = AutoModel.from_pretrained(SIGLIP_MODEL_ID).to(DEVICE).eval()
        return _model, _processor


def pooled_features(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden.mean(dim=1)
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Unsupported SigLIP feature output: {type(output).__name__}")


def encode_image_batch(model: Any, processor: Any, images: list[Image.Image]) -> torch.Tensor:
    batch = processor(images=images, return_tensors="pt")
    batch = {key: value.to(DEVICE) for key, value in batch.items()}
    with torch.inference_mode():
        if DEVICE.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.get_image_features(**batch)
        else:
            output = model.get_image_features(**batch)
        features = F.normalize(pooled_features(output).float(), dim=-1)
    return features.cpu()


def extract_video_embeddings(
    video: dict[str, Any], model: Any, processor: Any, progress_callback=None, cancel_check=None
) -> dict[str, Any]:
    path = Path(video["path"])
    cache_key = (str(path), video["mtime_ns"], video["size"], SIGLIP_MODEL_ID)
    with _cache_lock:
        cached = _embedding_cache.get(cache_key)
        if cached is not None:
            _embedding_cache.move_to_end(cache_key)
            return cached

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path.name}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0
    if duration <= 0:
        cap.release()
        raise RuntimeError("Could not read the video duration.")

    sample_seconds = [sec for sec in range(int(np.ceil(duration))) if sec < duration]
    times: list[float] = []
    chunks: list[torch.Tensor] = []
    image_batch: list[Image.Image] = []
    time_batch: list[float] = []
    try:
        for index, second in enumerate(sample_seconds, start=1):
            if cancel_check:
                cancel_check()
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
            ok, frame = cap.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_batch.append(Image.fromarray(rgb))
                time_batch.append(float(second))
            if len(image_batch) >= FRAME_BATCH_SIZE:
                chunks.append(encode_image_batch(model, processor, image_batch))
                times.extend(time_batch)
                image_batch, time_batch = [], []
                if cancel_check:
                    cancel_check()
            if progress_callback and (index % 5 == 0 or index == len(sample_seconds)):
                progress_callback(index, len(sample_seconds))
        if image_batch:
            if cancel_check:
                cancel_check()
            chunks.append(encode_image_batch(model, processor, image_batch))
            times.extend(time_batch)
    finally:
        cap.release()

    if not chunks:
        raise RuntimeError("No frames could be decoded from this video.")
    result = {
        "times": np.asarray(times, dtype=np.float32),
        "features": torch.cat(chunks, dim=0),
        "duration_seconds": float(duration),
    }
    with _cache_lock:
        _embedding_cache[cache_key] = result
        _embedding_cache.move_to_end(cache_key)
        while len(_embedding_cache) > CACHE_VIDEO_COUNT:
            _embedding_cache.popitem(last=False)
    return result


def expand_query_with_gemini(query: str) -> tuple[list[str], bool, str | None]:
    if not GEMINI_API_KEY:
        return [query.strip()], False, "GEMINI_API_KEY is not set; searched the original query."

    prompt = f"""Rewrite this CCTV video-search request into 3 or 4 short English visual descriptions that can be matched against individual video frames.
Describe only visible people, objects, actions, and scene details. Keep the user's intent. Do not decide whether a crime occurred, invent details, or add explanations. Include the original wording as one query if useful.
Return JSON only in this shape: {{"queries": ["..."]}}

User request: {query.strip()}"""
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_ID}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "queries": {"type": "ARRAY", "items": {"type": "STRING"}}
                },
                "required": ["queries"],
            },
        },
    }
    try:
        response = requests.post(
            endpoint,
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json=payload,
            timeout=35,
        )
        response.raise_for_status()
        body = response.json()
        text = body["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
        candidates = parsed.get("queries", [])
        unique: list[str] = []
        seen: set[str] = set()
        for item in [query.strip(), *candidates]:
            phrase = str(item).strip()
            key = phrase.casefold()
            if phrase and key not in seen:
                unique.append(phrase)
                seen.add(key)
            if len(unique) >= 5:
                break
        return unique or [query.strip()], True, None
    except Exception as exc:
        return [query.strip()], False, f"Gemini expansion failed; searched the original query. ({exc})"


def update_job(job_id: str, **values: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            current_status = _jobs[job_id].get("status")
            if current_status == "cancelled" and values.get("status") != "cancelled":
                return
            if current_status == "cancelling" and values.get("status") not in {"cancelling", "cancelled"}:
                return
            _jobs[job_id].update(values)


def run_search_job(job_id: str, request_data: SearchRequest) -> None:
    try:
        check_cancelled(job_id)
        update_job(job_id, status="running", phase="Expanding query with Gemini", progress=0.0)
        phrases, expanded, note = expand_query_with_gemini(request_data.query)
        check_cancelled(job_id)
        video = get_video(request_data.video_id)
        update_job(
            job_id,
            phase="Loading SigLIP2",
            progress=0.0,
            progress_label="First load downloads model weights; this can take a few minutes on CPU.",
        )
        model, processor = load_siglip()
        check_cancelled(job_id)
        update_job(
            job_id,
            phase="Scoring sampled frames",
            progress=0.0,
            progress_label="Sampling the video at 1 frame per second.",
        )

        def on_progress(done: int, total: int) -> None:
            update_job(
                job_id,
                progress=done / max(total, 1),
                progress_label=f"Encoded through {done:,} / {total:,} sampled seconds",
            )

        encoded = extract_video_embeddings(video, model, processor, on_progress, lambda: check_cancelled(job_id))
        check_cancelled(job_id)
        text_batch = processor(
            text=phrases,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        text_batch = {key: value.to(DEVICE) for key, value in text_batch.items()}
        with torch.inference_mode():
            text_output = model.get_text_features(**text_batch)
            text_features = F.normalize(pooled_features(text_output).float(), dim=-1).cpu()
        check_cancelled(job_id)
        scores_by_phrase = text_features @ encoded["features"].T
        scores = scores_by_phrase.mean(dim=0).numpy()
        order = np.argsort(-scores)
        ranks = np.empty(len(order), dtype=np.int32)
        ranks[order] = np.arange(1, len(order) + 1)
        selected_count = min(request_data.selected_frame_count, len(order))
        frames = [
            {
                "second": float(encoded["times"][index]),
                "score": float(scores[index]),
                "rank": int(ranks[index]),
            }
            for index in range(len(scores))
        ]
        result = {
            "video": {k: video[k] for k in ("id", "name", "label")},
            "original_query": request_data.query.strip(),
            "queries": phrases,
            "expanded_by_gemini": expanded,
            "note": note,
            "duration_seconds": encoded["duration_seconds"],
            "frame_count": len(frames),
            "selected_frame_count": selected_count,
            "top_frames": [frames[index] for index in order[:selected_count]],
            "frames": frames,
        }
        check_cancelled(job_id)
        update_job(job_id, status="done", phase="Complete", progress=1.0, result=result)
    except SearchCancelled:
        update_job(
            job_id,
            status="cancelled",
            phase="Cancelled",
            progress_label="Search cancelled. You can start another search.",
        )
    except Exception as exc:
        try:
            check_cancelled(job_id)
        except SearchCancelled:
            update_job(
                job_id,
                status="cancelled",
                phase="Cancelled",
                progress_label="Search cancelled. You can start another search.",
            )
        else:
            update_job(job_id, status="error", phase="Error", error=f"{type(exc).__name__}: {exc}")


def get_executor():
    global _executor
    if _executor is None:
        from concurrent.futures import ThreadPoolExecutor

        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="frame-search")
    return _executor


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def api_config():
    gpu_name = torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else None
    return {
        "device": str(DEVICE),
        "device_label": f"GPU · {gpu_name}" if gpu_name else "CPU",
        "gpu_name": gpu_name,
        "siglip_model": SIGLIP_MODEL_ID,
        "gemini_model": GEMINI_MODEL_ID,
        "gemini_configured": bool(GEMINI_API_KEY),
    }


@app.get("/api/videos")
def api_videos():
    videos = scan_videos()
    return {
        "video_root": os.getenv("VIDEO_DIR") or "auto-detected folders",
        "videos": [
            {k: item[k] for k in ("id", "name", "label", "duration_seconds")}
            for item in videos
        ],
    }


@app.post("/api/search")
def api_search(body: SearchRequest):
    try:
        get_video(body.video_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    job_id = secrets.token_urlsafe(12)
    with _jobs_lock:
        active_job = next(
            (
                job
                for job in reversed(list(_jobs.values()))
                if job.get("status") in {"queued", "running", "cancelling"}
            ),
            None,
        )
        if active_job is not None:
            raise HTTPException(
                status_code=409,
                detail="A search is already running. Wait for it to finish or cancel it first.",
            )
        _cancel_events[job_id] = threading.Event()
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "phase": "Queued",
            "progress": 0.0,
            "progress_label": "Waiting to start",
            "video_id": body.video_id,
            "video_name": get_video(body.video_id)["name"],
            "query": body.query.strip(),
        }
    get_executor().submit(run_search_job, job_id, body)
    return {"job_id": job_id}


@app.get("/api/jobs/active")
def api_active_job():
    with _jobs_lock:
        active_job = next(
            (
                dict(job)
                for job in reversed(list(_jobs.values()))
                if job.get("status") in {"queued", "running", "cancelling"}
            ),
            None,
        )
    return {"job": active_job}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job = dict(job)
    if job is None:
        raise HTTPException(status_code=404, detail="Search job not found.")
    return job


@app.delete("/api/jobs/{job_id}")
def api_cancel_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Search job not found.")
        if job.get("status") in {"done", "error", "cancelled"}:
            return dict(job)
        cancel_event = _cancel_events.get(job_id)
        if cancel_event is not None:
            cancel_event.set()
        job.update(
            status="cancelling",
            phase="Cancelling",
            progress_label="Stopping after the current frame batch…",
        )
        return dict(job)


@app.post("/api/export-frames")
def api_export_frames(body: FrameExportRequest):
    try:
        video = get_video(body.video_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    cap = cv2.VideoCapture(video["path"])
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=404, detail="Could not open the selected video.")

    buffer = io.BytesIO()
    manifest = io.StringIO(newline="")
    writer = csv.writer(manifest)
    writer.writerow(["query", "rank", "second", "timestamp", "score", "filename"])
    try:
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(body.frames, key=lambda frame: frame.rank):
                cap.set(cv2.CAP_PROP_POS_MSEC, item.second * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Could not decode the selected frame at {item.second} seconds.",
                    )
                height, width = frame.shape[:2]
                scale = min(1.0, 384 / max(height, width))
                if scale < 1.0:
                    frame = cv2.resize(
                        frame,
                        (int(width * scale), int(height * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    raise HTTPException(status_code=500, detail="Could not encode a selected frame.")
                filename = f"rank_{item.rank:03d}_second_{item.second:06d}.jpg"
                archive.writestr(filename, encoded.tobytes())
                hours, remainder = divmod(item.second, 3600)
                minutes, seconds = divmod(remainder, 60)
                timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                writer.writerow([body.query, item.rank, item.second, timestamp, f"{item.score:.6f}", filename])
            archive.writestr("manifest.csv", manifest.getvalue().encode("utf-8-sig"))
    finally:
        cap.release()

    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(video["name"]).stem).strip("_") or "video"
    archive_name = f"{stem}_selected_{len(body.frames)}_frames.zip"
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{archive_name}"'},
    )


@app.get("/api/frame/{video_id}/{second}")
def api_frame(video_id: str, second: int):
    try:
        video = get_video(video_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if second < 0 or (video.get("duration_seconds") and second >= video["duration_seconds"]):
        raise HTTPException(status_code=404, detail="Frame second is outside this video.")
    cap = cv2.VideoCapture(video["path"])
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=404, detail="Could not open video.")
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(status_code=404, detail="Could not decode this frame.")
    h, w = frame.shape[:2]
    max_dim = 384
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode preview frame.")
    return Response(content=encoded.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})

