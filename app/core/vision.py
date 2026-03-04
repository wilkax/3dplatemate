"""
Image enhancement for the build plate paint workflow.

enhance_for_detection():
    Applies CLAHE + unsharp masking to the perspective-corrected plate image
    so subtle residue is easier to see before the user paints dirty spots.
"""

import cv2
import numpy as np


def enhance_for_detection(image_bytes: bytes) -> bytes:
    """
    Pre-process the corrected plate image to make residue more visible.

    Steps:
      1. CLAHE on the L channel (LAB) — boosts local contrast without blowing
         out highlights; makes faint film and footprints stand out.
      2. Unsharp mask — sharpens edges of residue blobs.

    Returns JPEG bytes of the enhanced image at high quality.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # CLAHE in LAB colour space (operates only on luminance)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Unsharp mask: sharpened = original * 1.5 - blurred * 0.5
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()

