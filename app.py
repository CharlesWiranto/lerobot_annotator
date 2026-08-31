"""
LeRobot v2.1 instruction-segment annotator.

Run from this folder:
    python app.py

Then open http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATIC = HERE / "static"
PROGRESS_NAME = "annotation_progress.json"
BROKEN_NAME = "broken.json"

PREFERRED_CAMERAS = [
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
]
CAMERA_LABELS = {
    "observation.images.top_head": "Head",
    "observation.images.hand_left": "Left wrist",
    "observation.images.hand_right": "Right wrist",
}
REQUIRED_STATE = (
    "state/joint/position",
    "state/left_effector/position",
    "state/right_effector/position",
)
REQUIRED_ACTION = (
    "action/joint/position",
    "action/left_effector/position",
    "action/right_effector/position",
)

app = FastAPI(title="LeRobot Annotator", version="1.0")


# ---------------------------------------------------------------------------
# Frame extraction (OpenCV POS_FRAMES is exact on these CFR H.264 files)
# ---------------------------------------------------------------------------

class FrameExtractor:
    def __init__(self, max_open: int = 12, cache_size: int = 768):
        self.max_open = max_open
        self.cache_size = cache_size
        self._guard = threading.Lock()
        self._sessions: OrderedDict[str, "_Session"] = OrderedDict()
        self._jpeg: OrderedDict[tuple, bytes] = OrderedDict()

    class _Session:
        def __init__(self, path: str):
            self.lock = threading.Lock()
            self.path = path
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video: {path}")
            self.last = -1
            self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        def read(self, idx: int) -> np.ndarray:
            with self.lock:
                if self.count <= 0:
                    raise RuntimeError(f"Empty video: {self.path}")
                idx = max(0, min(int(idx), self.count - 1))
                if idx != self.last + 1:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = self.cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Failed to read frame {idx} from {self.path}")
                self.last = idx
                return frame

        def close(self) -> None:
            try:
                self.cap.release()
            except Exception:
                pass

    def _session(self, path: str) -> "_Session":
        with self._guard:
            sess = self._sessions.get(path)
            if sess is not None:
                self._sessions.move_to_end(path)
                return sess
            sess = FrameExtractor._Session(path)
            self._sessions[path] = sess
            while len(self._sessions) > self.max_open:
                _, old = self._sessions.popitem(last=False)
                old.close()
            return sess

    def frame_count(self, path: str) -> int:
        return self._session(path).count

    def jpeg(self, path: str, idx: int, max_w: int = 640, quality: int = 80) -> tuple[bytes, int]:
        key = (path, int(idx), int(max_w), int(quality))
        with self._guard:
            cached = self._jpeg.get(key)
            if cached is not None:
                self._jpeg.move_to_end(key)
                return cached, int(idx)
        sess = self._session(path)
        clamped = max(0, min(int(idx), max(sess.count - 1, 0)))
        frame = sess.read(clamped)
        h, w = frame.shape[:2]
        if max_w and w > max_w:
            nh = max(1, int(round(h * (max_w / w))))
            frame = cv2.resize(frame, (max_w, nh), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        data = buf.tobytes()
        with self._guard:
            self._jpeg[key] = data
            self._jpeg.move_to_end(key)
            while len(self._jpeg) > self.cache_size:
                self._jpeg.popitem(last=False)
        return data, clamped


FRAMES = FrameExtractor()
_INFO_CACHE: dict[str, tuple[float, dict]] = {}
_INFO_LOCK = threading.Lock()


def _info(ds: Path) -> dict:
    path = ds / "meta" / "info.json"
    mtime = path.stat().st_mtime
    key = str(path)
    with _INFO_LOCK:
        hit = _INFO_CACHE.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    data = _read_json(path)
    with _INFO_LOCK:
        _INFO_CACHE[key] = (mtime, data)
    return data


def _invalidate_info(ds: Path) -> None:
    key = str(ds / "meta" / "info.json")
    with _INFO_LOCK:
        _INFO_CACHE.pop(key, None)


# ---------------------------------------------------------------------------
# Dataset IO
# ---------------------------------------------------------------------------

def _safe_dataset(name: str) -> Path:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise HTTPException(400, "Invalid dataset name")
    path = (ROOT / name).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(400, "Invalid dataset name") from exc
    if not path.is_dir() or not (path / "meta" / "info.json").is_file():
        raise HTTPException(404, f"Dataset not found: {name}")
    return path


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _discover_datasets() -> list[str]:
    names = []
    for child in sorted(ROOT.iterdir()):
        if child.is_dir() and (child / "meta" / "info.json").is_file():
            names.append(child.name)
    return names


def _camera_label(key: str) -> str:
    if key in CAMERA_LABELS:
        return CAMERA_LABELS[key]
    return key.replace("observation.images.", "").replace("_", " ")


def _cameras(ds: Path, info: dict) -> list[dict]:
    keys: list[str] = []
    features = info.get("features") or {}
    for key, spec in features.items():
        if isinstance(spec, dict) and spec.get("dtype") == "video":
            keys.append(key)
    videos = ds / "videos"
    if videos.is_dir():
        disk = set()
        for chunk in videos.glob("chunk-*"):
            if not chunk.is_dir():
                continue
            for cam in chunk.iterdir():
                if cam.is_dir():
                    disk.add(cam.name)
        for k in disk:
            if k not in keys:
                keys.append(k)

    def sort_key(k: str) -> tuple[int, str]:
        return (PREFERRED_CAMERAS.index(k) if k in PREFERRED_CAMERAS else 100, k)

    keys = sorted(set(keys), key=sort_key)
    out = []
    for k in keys:
        spec = features.get(k) or {}
        out.append(
            {
                "key": k,
                "label": _camera_label(k),
                "shape": spec.get("shape"),
                "fps": ((spec.get("video_info") or {}).get("video.fps") or info.get("fps") or 30),
            }
        )
    return out


def _chunk_index(episode: int, chunks_size: int) -> int:
    size = int(chunks_size) if chunks_size else 1000
    return int(episode) // size


def _video_file(ds: Path, info: dict, camera: str, episode: int) -> Path:
    if not camera or ".." in camera or "/" in camera or "\\" in camera:
        raise HTTPException(400, "Invalid camera key")
    chunks_size = int(info.get("chunks_size") or 1000)
    chunk = _chunk_index(episode, chunks_size)
    template = info.get("video_path") or (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    rel = template.format(
        episode_chunk=chunk, video_key=camera, episode_index=int(episode)
    )
    path = (ds / rel).resolve()
    try:
        path.relative_to(ds.resolve())
    except ValueError as exc:
        raise HTTPException(400, "Invalid camera path") from exc
    return path


def _field_names(info: dict, feature: str) -> list[str]:
    spec = (info.get("features") or {}).get(feature) or {}
    fd = spec.get("field_descriptions") or {}
    return list(fd.keys()) if isinstance(fd, dict) else []


def _health(info: dict, cameras: list[dict]) -> dict:
    state_fields = set(_field_names(info, "observation.state"))
    action_fields = set(_field_names(info, "action"))
    missing_state = [f for f in REQUIRED_STATE if f not in state_fields]
    missing_action = [f for f in REQUIRED_ACTION if f not in action_fields]
    cam_keys = {c["key"] for c in cameras}
    missing_cams = [k for k in PREFERRED_CAMERAS if k not in cam_keys]
    fps = info.get("fps") or 0
    segs = info.get("instruction_segments") or {}
    hl = info.get("high_level_instruction") or {}
    total_ep = int(info.get("total_episodes") or 0)
    missing_seg = [i for i in range(total_ep) if str(i) not in segs or not segs[str(i)]]
    hl_set = 0
    for i in range(total_ep):
        rec = hl.get(str(i)) or {}
        text = rec.get("high_level_instruction") if isinstance(rec, dict) else rec
        if text:
            hl_set += 1
    return {
        "fps": fps,
        "fps_ok": abs(float(fps) - 30.0) < 1e-6,
        "codebase_version": info.get("codebase_version"),
        "missing_state_fields": missing_state,
        "missing_action_fields": missing_action,
        "field_descriptions_ok": not missing_state and not missing_action,
        "missing_cameras": missing_cams,
        "episodes_missing_segments": missing_seg[:50],
        "episodes_missing_segments_count": len(missing_seg),
        "high_level_instruction_count": hl_set,
        "hl_overrides_subtasks": hl_set > 0,
        "has_waist": "state/waist/position" in state_fields,
    }


def _progress_path(ds: Path) -> Path:
    return ds / "meta" / PROGRESS_NAME


def _broken_path(ds: Path) -> Path:
    return ds / BROKEN_NAME


def _load_broken(ds: Path) -> list[int]:
    path = _broken_path(ds)
    if not path.is_file():
        return []
    try:
        data = _read_json(path)
        if isinstance(data, list):
            return sorted({int(x) for x in data})
    except Exception:
        return []
    return []


def _save_broken(ds: Path, episodes: list[int]) -> list[int]:
    cleaned = sorted({int(x) for x in episodes if int(x) >= 0})
    path = _broken_path(ds)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(cleaned, f)
        f.write("\n")
    tmp.replace(path)
    return cleaned


def _load_progress(ds: Path) -> dict:
    path = _progress_path(ds)
    if path.is_file():
        try:
            return _read_json(path)
        except Exception:
            return {}
    return {}


def _save_progress(ds: Path, progress: dict) -> None:
    path = _progress_path(ds)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def _backup_info(path: Path) -> None:
    bak = path.with_suffix(".json.bak")
    if path.is_file():
        shutil.copy2(path, bak)


def _replace_top_level_key(path: Path, key: str, value: Any) -> None:
    """Rewrite one top-level JSON key without reformatting the rest of info.json."""
    text = path.read_text(encoding="utf-8")
    dumped = json.dumps(value, indent=2, ensure_ascii=False)
    dumped = dumped.replace("\n", "\n  ")
    pattern = re.compile(rf'\n([ \t]*)"{re.escape(key)}"\s*:')
    match = pattern.search(text)
    if match:
        start_val = match.end()
        while start_val < len(text) and text[start_val] in " \t\r\n":
            start_val += 1
        decoder = json.JSONDecoder()
        try:
            _, end_idx = decoder.raw_decode(text, start_val)
        except json.JSONDecodeError as exc:
            raise HTTPException(500, f"Could not parse existing {key}") from exc
        text = text[:start_val] + dumped + text[end_idx:]
    else:
        stripped = text.rstrip()
        if not stripped.endswith("}"):
            raise HTTPException(500, "info.json does not end with }")
        indent = "  "
        insertion = f',\n{indent}"{key}": {dumped}\n'
        text = stripped[:-1].rstrip() + insertion + "}\n"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def _normalize_segment(seg: dict) -> dict:
    out = {
        "instruction": str(seg.get("instruction") or ""),
        "instruction_augmentation": seg.get("instruction_augmentation")
        if isinstance(seg.get("instruction_augmentation"), dict)
        else {},
        "start_frame_index": int(seg["start_frame_index"]),
        "success_frame_index": int(seg["success_frame_index"]),
        "end_frame_index": int(seg["end_frame_index"]),
    }
    if seg.get("track"):
        out["track"] = str(seg["track"])
    return out


def _ordered_episode_map(raw: dict, n_episodes: int) -> dict:
    out: dict[str, Any] = {}
    if not isinstance(raw, dict):
        return out
    for i in range(n_episodes):
        k = str(i)
        if k in raw:
            out[k] = raw[k]
    for k, v in raw.items():
        if k not in out:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class SaveBody(BaseModel):
    instruction_segments: dict[str, list[dict]] = Field(default_factory=dict)
    high_level_instruction: dict[str, dict] = Field(default_factory=dict)
    progress: dict | None = None
    broken: list[int] | None = None


class BrokenBody(BaseModel):
    episodes: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/datasets")
def api_list_datasets() -> dict:
    items = []
    for name in _discover_datasets():
        ds = ROOT / name
        try:
            info = _info(ds)
        except Exception:
            continue
        episodes = _read_jsonl(ds / "meta" / "episodes.jsonl")
        progress = _load_progress(ds)
        reviewed = progress.get("reviewed") or {}
        segs = info.get("instruction_segments") or {}
        n = int(info.get("total_episodes") or len(episodes) or 0)
        annotated = sum(1 for i in range(n) if segs.get(str(i)))
        broken = _load_broken(ds)
        items.append(
            {
                "name": name,
                "robot_type": info.get("robot_type"),
                "codebase_version": info.get("codebase_version"),
                "fps": info.get("fps") or 30,
                "total_episodes": n,
                "total_frames": info.get("total_frames"),
                "total_tasks": info.get("total_tasks"),
                "cameras": [c["key"] for c in _cameras(ds, info)],
                "annotated_episodes": annotated,
                "reviewed_episodes": sum(1 for i in range(n) if reviewed.get(str(i))),
                "broken": broken,
            }
        )
    return {"root": str(ROOT), "datasets": items}


@app.get("/api/datasets/{name}")
def api_dataset(name: str) -> dict:
    ds = _safe_dataset(name)
    info = _info(ds)
    episodes = _read_jsonl(ds / "meta" / "episodes.jsonl")
    tasks = _read_jsonl(ds / "meta" / "tasks.jsonl")
    cameras = _cameras(ds, info)
    progress = _load_progress(ds)
    reviewed = progress.get("reviewed") or {}
    broken = _load_broken(ds)
    broken_set = set(broken)
    segs = info.get("instruction_segments") or {}
    hl = info.get("high_level_instruction") or {}
    ep_out = []
    for row in episodes:
        idx = int(row["episode_index"])
        key = str(idx)
        ep_segs = segs.get(key) or []
        ep_out.append(
            {
                "episode_index": idx,
                "length": int(row.get("length") or 0),
                "tasks": row.get("tasks") or [],
                "n_segments": len(ep_segs) if isinstance(ep_segs, list) else 0,
                "reviewed": bool(reviewed.get(key)),
                "broken": idx in broken_set,
            }
        )
    return {
        "name": name,
        "fps": float(info.get("fps") or 30),
        "robot_type": info.get("robot_type"),
        "codebase_version": info.get("codebase_version"),
        "data_version": info.get("data_version"),
        "chunks_size": int(info.get("chunks_size") or 1000),
        "total_episodes": int(info.get("total_episodes") or len(episodes)),
        "total_frames": info.get("total_frames"),
        "cameras": cameras,
        "episodes": ep_out,
        "tasks": tasks,
        "instruction_segments": segs,
        "high_level_instruction": hl,
        "health": _health(info, cameras),
        "progress": progress,
        "broken": broken,
    }


@app.get("/api/datasets/{name}/episodes/{episode}")
def api_episode(name: str, episode: int) -> dict:
    ds = _safe_dataset(name)
    info = _info(ds)
    cameras = _cameras(ds, info)
    counts = {}
    missing = []
    for cam in cameras:
        path = _video_file(ds, info, cam["key"], episode)
        if not path.is_file():
            missing.append(cam["key"])
            counts[cam["key"]] = 0
            continue
        try:
            counts[cam["key"]] = FRAMES.frame_count(str(path))
        except Exception:
            counts[cam["key"]] = 0
            missing.append(cam["key"])
    episodes = _read_jsonl(ds / "meta" / "episodes.jsonl")
    row = next((r for r in episodes if int(r["episode_index"]) == int(episode)), None)
    length = int(row["length"]) if row else 0
    return {
        "episode_index": int(episode),
        "length": length,
        "video_frames": counts,
        "missing_videos": missing,
        "frame_count_mismatch": any(c and c != length for c in counts.values()),
    }


@app.get("/api/datasets/{name}/frame")
def api_frame(
    name: str,
    episode: int,
    camera: str,
    frame: int,
    max_w: int = 640,
    q: int = 80,
) -> Response:
    ds = _safe_dataset(name)
    info = _info(ds)
    path = _video_file(ds, info, camera, episode)
    if not path.is_file():
        raise HTTPException(404, f"Video not found: {path.name}")
    try:
        data, actual = FRAMES.jpeg(str(path), int(frame), max_w=int(max_w), quality=int(q))
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=86400, immutable",
            "X-Frame-Index": str(actual),
            "X-Requested-Frame": str(int(frame)),
        },
    )


@app.post("/api/datasets/{name}/save")
def api_save(name: str, body: SaveBody) -> dict:
    ds = _safe_dataset(name)
    info_path = ds / "meta" / "info.json"
    info = _info(ds)
    episodes = _read_jsonl(ds / "meta" / "episodes.jsonl")
    lengths = {int(r["episode_index"]): int(r["length"]) for r in episodes}
    n_ep = int(info.get("total_episodes") or len(episodes))

    warnings: list[str] = []
    norm_segs: dict[str, list[dict]] = {}
    for key, segs in (body.instruction_segments or {}).items():
        try:
            ei = int(key)
        except ValueError:
            warnings.append(f"Skipping non-integer episode key {key!r}")
            continue
        length = lengths.get(ei)
        if not isinstance(segs, list) or not segs:
            warnings.append(f"Episode {ei} has no segments and will be dropped from the epoch.")
            continue
        cleaned = []
        for j, seg in enumerate(segs):
            try:
                cleaned.append(_normalize_segment(seg))
            except Exception:
                warnings.append(f"Episode {ei} segment {j} is invalid")
                continue
        for j, seg in enumerate(cleaned):
            s, suc, e = (
                seg["start_frame_index"],
                seg["success_frame_index"],
                seg["end_frame_index"],
            )
            if s >= e:
                warnings.append(f"Episode {ei} segment {j}: start ({s}) >= end ({e})")
            if length is not None:
                if s < 0 or e > length:
                    warnings.append(
                        f"Episode {ei} segment {j}: frames [{s},{e}) outside length {length}"
                    )
            if not (s <= suc < e):
                warnings.append(
                    f"Episode {ei} segment {j}: success {suc} not in [{s}, {e})"
                )
            low = (seg["instruction"] or "").lower()
            if any(w in low for w in ("reset", "return", "default")) and (suc - s) > 90:
                warnings.append(
                    f"Episode {ei} segment {j}: instruction contains reset/return/default "
                    "and interval > 90 frames (sampler will cut it to 45)."
                )
        if cleaned:
            if cleaned[0]["start_frame_index"] != 0:
                warnings.append(
                    f"Episode {ei}: first start is {cleaned[0]['start_frame_index']} "
                    "(prompt lookup forces 0 at runtime; sampler still uses this start)."
                )
            if length is not None and cleaned[-1]["end_frame_index"] != length:
                warnings.append(
                    f"Episode {ei}: last end is {cleaned[-1]['end_frame_index']}, length is {length}"
                )
            for j in range(len(cleaned) - 1):
                if cleaned[j]["end_frame_index"] != cleaned[j + 1]["start_frame_index"]:
                    warnings.append(
                        f"Episode {ei}: gap/overlap between segments {j} and {j + 1}"
                    )
        norm_segs[str(ei)] = cleaned

    norm_segs = _ordered_episode_map(norm_segs, n_ep)

    norm_hl: dict[str, dict] = {}
    for key, rec in (body.high_level_instruction or {}).items():
        try:
            ei = int(key)
        except ValueError:
            continue
        if isinstance(rec, dict):
            text = rec.get("high_level_instruction") or ""
        else:
            text = str(rec or "")
        norm_hl[str(ei)] = {"high_level_instruction": str(text)}
    norm_hl = _ordered_episode_map(norm_hl, n_ep)

    _backup_info(info_path)
    _replace_top_level_key(info_path, "instruction_segments", norm_segs)
    _replace_top_level_key(info_path, "high_level_instruction", norm_hl)
    _invalidate_info(ds)

    if body.progress is not None:
        _save_progress(ds, body.progress)
    if body.broken is not None:
        _save_broken(ds, body.broken)

    return {
        "ok": True,
        "episodes_written": len(norm_segs),
        "warnings": warnings,
        "backup": str(info_path.with_suffix(".json.bak")),
        "broken": _load_broken(ds),
    }


@app.post("/api/datasets/{name}/progress")
def api_progress(name: str, body: dict) -> dict:
    ds = _safe_dataset(name)
    _save_progress(ds, body)
    return {"ok": True}


@app.post("/api/datasets/{name}/broken")
def api_broken(name: str, body: BrokenBody) -> dict:
    ds = _safe_dataset(name)
    cleaned = _save_broken(ds, body.episodes)
    return {"ok": True, "broken": cleaned}


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn

    print(f"Dataset root: {ROOT}")
    print("Open http://127.0.0.1:8765")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
