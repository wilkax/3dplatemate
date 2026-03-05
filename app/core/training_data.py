"""
Training data storage.

Every completed prepare → generate-stl pair produces one labelled sample:

  platemate-training-data/
  └── training/20260304_143022_a3f1/
      ├── corrected.jpg   perspective-corrected plate (pre-enhancement)
      ├── enhanced.jpg    CLAHE + sharpened version shown to the user
      └── mask.png        user-painted binary mask  (white = dirty spots)

When BLOB_READ_WRITE_TOKEN is set the files are uploaded to Vercel Blob
(platemate-training-data store).  Without it they fall back to a local
`training_data/` directory — useful during local development.
"""

from __future__ import annotations

import logging
import os
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_LOCAL_ROOT  = Path("training_data")
_BLOB_UPLOAD = "https://blob.vercel-storage.com"
_BLOB_PREFIX = "training"   # top-level folder inside the store


# ── helpers ───────────────────────────────────────────────────────────────────

def make_session_id() -> str:
    """Return a unique, human-readable session identifier."""
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}_{short}"


def _token() -> str | None:
    return os.environ.get("BLOB_READ_WRITE_TOKEN")


def _blob_put(pathname: str, data: bytes, content_type: str) -> None:
    """PUT `data` to Vercel Blob at `pathname` relative to the store root."""
    url = f"{_BLOB_UPLOAD}/{pathname}"
    req = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers={
            "Authorization": f"Bearer {_token()}",
            "Content-Type":  content_type,
            "x-vercel-blob-store-id": "platemate-training-data",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()   # consume body so the connection is released
    logger.debug("Blob uploaded: %s", pathname)


# ── public API ────────────────────────────────────────────────────────────────

def save_plate_images(
    session_id: str,
    corrected_img: np.ndarray,
    enhanced_bytes: bytes,
) -> None:
    """
    Persist the plate images produced by the prepare step.

    Parameters
    ----------
    session_id    : identifier returned to the frontend and later passed back
                    with the mask so both files land in the same folder.
    corrected_img : perspective-corrected BGR numpy array (pre-enhancement).
    enhanced_bytes: CLAHE-enhanced JPEG bytes sent to the browser canvas.
    """
    _, buf = cv2.imencode(".jpg", corrected_img)
    corrected_bytes = buf.tobytes()

    if _token():
        try:
            _blob_put(f"{_BLOB_PREFIX}/{session_id}/corrected.jpg", corrected_bytes, "image/jpeg")
            _blob_put(f"{_BLOB_PREFIX}/{session_id}/enhanced.jpg",  enhanced_bytes,  "image/jpeg")
            logger.info("Training data — uploaded to blob: %s", session_id)
        except Exception as exc:
            logger.warning("Training data — blob upload failed: %s", exc)
    else:
        run_dir = _LOCAL_ROOT / session_id
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "corrected.jpg").write_bytes(corrected_bytes)
            (run_dir / "enhanced.jpg").write_bytes(enhanced_bytes)
            logger.info("Training data — saved locally: %s", run_dir)
        except Exception as exc:
            logger.warning("Training data — local save failed: %s", exc)


def save_mask(session_id: str, mask_bytes: bytes) -> None:
    """
    Persist the user-painted mask produced by the generate-stl step.

    Parameters
    ----------
    session_id : must match the value returned by the prepare endpoint.
    mask_bytes : raw PNG bytes of the binary mask (white = dirty spots).
    """
    if _token():
        try:
            _blob_put(f"{_BLOB_PREFIX}/{session_id}/mask.png", mask_bytes, "image/png")
            logger.info("Training data — mask uploaded to blob: %s", session_id)
        except Exception as exc:
            logger.warning("Training data — blob mask upload failed: %s", exc)
    else:
        run_dir = _LOCAL_ROOT / session_id
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "mask.png").write_bytes(mask_bytes)
            logger.info("Training data — mask saved locally: %s", run_dir / "mask.png")
        except Exception as exc:
            logger.warning("Training data — local mask save failed: %s", exc)

