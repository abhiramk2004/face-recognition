"""
enrollment.py

Loads the enrolled face database and generates
one normalized embedding for each identity.
"""

import os
import cv2
import numpy as np

from app.config import FACE_DATABASE_PATH


# ==========================================================
# COSINE NORMALIZATION
# ==========================================================

def normalize_embedding(embedding):
    """
    Normalize an embedding to unit length.
    """
    return embedding / np.linalg.norm(embedding)


# ==========================================================
# EXTRACT LARGEST FACE EMBEDDING
# ==========================================================

def extract_embedding(face_app, image):
    """
    Detect the largest face in an image and return
    its embedding.

    Returns
    -------
    numpy.ndarray | None
    """

    faces = face_app.get(image)

    if len(faces) == 0:
        return None

    face = max(
        faces,
        key=lambda f:
        (f.bbox[2] - f.bbox[0]) *
        (f.bbox[3] - f.bbox[1])
    )

    return face.embedding


# ==========================================================
# LOAD SINGLE PERSON
# ==========================================================

def load_person(face_app, person_dir):

    embeddings = []

    for image_name in sorted(os.listdir(person_dir)):

        image_path = os.path.join(
            person_dir,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        embedding = extract_embedding(
            face_app,
            image
        )

        if embedding is None:
            continue

        embeddings.append(embedding)

    return embeddings


# ==========================================================
# LOAD ENTIRE DATABASE
# ==========================================================

def load_known_faces(face_app):

    known_faces = {}

    if not os.path.exists(FACE_DATABASE_PATH):

        raise FileNotFoundError(
            f"Missing directory: {FACE_DATABASE_PATH}"
        )

    print("Loading enrolled faces...\n")

    identity_count = 0

    for person_name in sorted(os.listdir(FACE_DATABASE_PATH)):

        person_dir = os.path.join(
            FACE_DATABASE_PATH,
            person_name
        )

        if not os.path.isdir(person_dir):
            continue

        embeddings = load_person(
            face_app,
            person_dir
        )

        if len(embeddings) == 0:

            print(f"[SKIPPED] {person_name}")

            continue

        mean_embedding = np.mean(
            embeddings,
            axis=0
        )

        mean_embedding = normalize_embedding(
            mean_embedding
        )

        known_faces[person_name] = mean_embedding

        identity_count += 1

        print(
            f"[ENROLL] "
            f"{person_name} "
            f"({len(embeddings)} images)"
        )

    print(
        f"\nLoaded "
        f"{identity_count} identities.\n"
    )

    return known_faces
