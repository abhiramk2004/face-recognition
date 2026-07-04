"""
recognizer.py

Runtime face recognition utilities.
"""

import numpy as np

from app.config import SIMILARITY_THRESHOLD


# ==========================================================
# COSINE SIMILARITY
# ==========================================================

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two embeddings.
    """
    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


# ==========================================================
# RECOGNIZE FACE
# ==========================================================

def recognize_face(
    face_app,
    image,
    known_faces,
    threshold=SIMILARITY_THRESHOLD
):
    """
    Recognize the largest face inside an image.

    Parameters
    ----------
    face_app : FaceAnalysis
        Initialized InsightFace model.

    image : ndarray
        Person crop.

    known_faces : dict
        Dictionary of enrolled identities.

    threshold : float
        Minimum cosine similarity.

    Returns
    -------
    (name, score)
    """

    faces = face_app.get(image)

    if len(faces) == 0:
        return "Unknown", 0.0

    face = max(
        faces,
        key=lambda f:
        (f.bbox[2] - f.bbox[0]) *
        (f.bbox[3] - f.bbox[1])
    )

    embedding = face.embedding

    embedding = (
        embedding /
        np.linalg.norm(embedding)
    )

    best_name = "Unknown"
    best_score = -1

    for name, known_embedding in known_faces.items():

        score = cosine_similarity(
            embedding,
            known_embedding
        )

        if score > best_score:

            best_score = score
            best_name = name

    if best_score < threshold:
        return "Unknown", float(best_score)

    return best_name, float(best_score)
