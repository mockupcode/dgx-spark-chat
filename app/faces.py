"""Face enrolment and recognition with OpenCV's YuNet detector and SFace embedder.

Both networks are small (a few ms per frame), so recognition runs while the
user is still talking and adds nothing to the reply latency. Only embeddings
are stored, never images.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
EMBEDDER_FILE = "face_recognition_sface_2021dec.onnx"
MAX_SAMPLES_PER_PERSON = 8


@dataclass
class SeenFace:
    name: str | None  # None means "someone not enrolled"
    score: float
    area: float


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def best_match(embedding: np.ndarray, db: dict[str, list[np.ndarray]], threshold: float) -> tuple[str | None, float]:
    best_name, best_score = None, -1.0
    for name, samples in db.items():
        for sample in samples:
            score = cosine(embedding, sample)
            if score > best_score:
                best_name, best_score = name, score
    if best_score >= threshold:
        return best_name, best_score
    return None, best_score


def describe(faces: list[SeenFace]) -> str:
    """A short Thai note the LLM gets alongside the user's words."""
    if not faces:
        return ""
    known = []
    for face in faces:
        if face.name and face.name not in known:
            known.append(face.name)
    unknown = sum(1 for f in faces if f.name is None)
    parts = []
    if known:
        parts.append("เห็น " + ", ".join(f"คุณ{name}" for name in known))
    if unknown:
        parts.append(f"คนที่ยังไม่รู้จัก {unknown} คน")
    return "(หมายเหตุจากระบบจดจำใบหน้า: " + " และ".join(parts) + ")"


class FaceBook:
    def __init__(self, model_dir: Path, db_path: Path, threshold: float) -> None:
        import cv2

        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # OpenCV 5 warns about unused DNN targets
        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN.create(str(model_dir / DETECTOR_FILE), "", (320, 320), 0.8, 0.3, 20)
        self._embedder = cv2.FaceRecognizerSF.create(str(model_dir / EMBEDDER_FILE), "")
        self._db_path = db_path
        self._threshold = threshold
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()  # the OpenCV nets keep per-call state
        self._db: dict[str, list[np.ndarray]] = {}
        self._load()

    @property
    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._db)

    def _load(self) -> None:
        if not self._db_path.exists():
            return
        raw = json.loads(self._db_path.read_text(encoding="utf-8"))
        self._db = {
            name: [np.asarray(v, dtype=np.float32) for v in entry["embeddings"]]
            for name, entry in raw.get("people", {}).items()
        }

    def _save(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "people": {
                name: {"embeddings": [v.round(6).tolist() for v in samples], "updated": int(time.time())}
                for name, samples in self._db.items()
            }
        }
        tmp = self._db_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._db_path)

    def _decode(self, jpeg: bytes) -> np.ndarray | None:
        cv2 = self._cv2
        return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    def _embed_all(self, image: np.ndarray) -> list[tuple[np.ndarray, float]]:
        h, w = image.shape[:2]
        out = []
        with self._infer_lock:
            self._detector.setInputSize((w, h))
            _, detections = self._detector.detect(image)
            for det in detections if detections is not None else []:
                aligned = self._embedder.alignCrop(image, det)
                feature = self._embedder.feature(aligned).reshape(-1).astype(np.float32)
                out.append((feature, float(det[2] * det[3])))
        return out

    def recognize(self, jpeg: bytes) -> list[SeenFace]:
        image = self._decode(jpeg)
        if image is None:
            return []
        faces = self._embed_all(image)
        with self._lock:
            db = dict(self._db)
        seen = []
        for feature, area in faces:
            name, score = best_match(feature, db, self._threshold)
            seen.append(SeenFace(name, score, area))
        seen.sort(key=lambda f: f.area, reverse=True)
        return seen

    def enroll(self, name: str, jpeg: bytes) -> bool:
        """Add the largest face in the frame under `name`."""
        image = self._decode(jpeg)
        if image is None:
            return False
        faces = self._embed_all(image)
        if not faces:
            return False
        feature, _ = max(faces, key=lambda f: f[1])
        with self._lock:
            samples = self._db.setdefault(name, [])
            samples.append(feature)
            del samples[:-MAX_SAMPLES_PER_PERSON]
            self._save()
        return True

    def forget(self, name: str) -> bool:
        with self._lock:
            match = next((n for n in self._db if n.casefold() == name.casefold()), None)
            if match is None:
                return False
            del self._db[match]
            self._save()
            return True
