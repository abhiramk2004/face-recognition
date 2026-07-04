"""
face_model.py

Initializes the InsightFace model.
"""

from insightface.app import FaceAnalysis

from app.config import (
    FACE_MODEL_NAME,
    FACE_DETECTION_SIZE,
)


def initialize_face_model():
    """
    Load and initialize the InsightFace model.

    Returns
    -------
    FaceAnalysis
        Initialized InsightFace object.
    """

    print("Loading InsightFace...")

    face_app = FaceAnalysis(
        name=FACE_MODEL_NAME,
        providers=["CPUExecutionProvider"]
    )

    face_app.prepare(
        ctx_id=0,
        det_size=FACE_DETECTION_SIZE
    )

    print("InsightFace ready.\n")

    return face_app
