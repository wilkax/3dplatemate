"""
Training data storage.

Every completed prepare → generate-stl pair produces one labelled sample:

  training_data/
  └── 20260304_143022_a3f1/
      ├── corrected.jpg   perspective-corrected plate (pre-enhancement)
      ├── enhanced.jpg    CLAHE + sharpened version shown to the user
      └── mask.png        user-painted binary mask  (white = dirty spots)

These image/mask pairs can be used directly to fine-tune a segmentation
model (e.g. U-Net) without any additional labelling work.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_TRAINING_ROOT = Path("training_data")


def make_session_id() -> str:
    """Return a unique, human-readable session identifier."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}_{short}"


def save_plate_images(
    session_id: str,
    corrected_img: np.ndarray,
    enhanced_bytes: bytes,
) -> Path:
    """
    Persist the plate images produced by the prepare step.

    Parameters
    ----------
    session_id    : identifier returned to the frontend and later passed back
                    with the mask so both files land in the same folder.
    corrected_img : perspective-corrected BGR numpy array (pre-enhancement).
    enhanced_bytes: CLAHE-enhanced JPEG bytes sent to the browser canvas.

    Returns
    -------
    Path to the session directory (for logging).
    """
    run_dir = _TRAINING_ROOT / session_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        cv2.imwrite(str(run_dir / "corrected.jpg"), corrected_img)
        (run_dir / "enhanced.jpg").write_bytes(enhanced_bytes)
        logger.info("Training data — plate images saved: %s", run_dir)
    except Exception as exc:
        logger.warning("Training data — failed to save plate images: %s", exc)

    return run_dir


def save_mask(session_id: str, mask_bytes: bytes) -> None:
    """
    Persist the user-painted mask produced by the generate-stl step.

    If the session directory does not exist yet (e.g. the prepare step was
    skipped in a test), it is created so the mask is never silently lost.

    Parameters
    ----------
    session_id : must match the value returned by the prepare endpoint.
    mask_bytes : raw PNG bytes of the binary mask (white = dirty spots).
    """
    run_dir = _TRAINING_ROOT / session_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        (run_dir / "mask.png").write_bytes(mask_bytes)
        logger.info("Training data — mask saved: %s", run_dir / "mask.png")
    except Exception as exc:
        logger.warning("Training data — failed to save mask: %s", exc)

