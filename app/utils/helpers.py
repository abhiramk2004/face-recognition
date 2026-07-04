"""
helpers.py

Utility helper functions.
"""

import cv2


def crop_person(frame, x1, y1, x2, y2):
    """
    Safely crop a person from a frame.

    Returns
    -------
    numpy.ndarray | None
    """

    h, w = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]