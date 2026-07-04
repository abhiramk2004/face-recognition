"""
service.py

High-level face recognition service.
"""

from app.recognition.face_model import initialize_face_model
from app.recognition.enrollment import load_known_faces
from app.recognition.recognizer import recognize_face


class RecognitionService:

    def __init__(self):

        self.face_app = initialize_face_model()

        self.known_faces = load_known_faces(
            self.face_app
        )

    def recognize(self, image):

        return recognize_face(
            self.face_app,
            image,
            self.known_faces
        )