"""
detector.py

YOLO11 object detector.
"""

from ultralytics import YOLO

from app.config import (
    YOLO_MODEL,
    DEVICE,
    PERSON_CLASS_ID,
    TRACKER_CONFIG
)


class YOLODetector:

    def __init__(self):

        print("Loading YOLO model...")

        self.model = YOLO(YOLO_MODEL)

        self.model.to(DEVICE)

        print("YOLO ready.\n")

    def detect(self, frame):

        """
        Run YOLO + ByteTrack.

        Returns
        -------
        Ultralytics Results
        """

        results = self.model.track(

            source=frame,

            persist=True,

            tracker=TRACKER_CONFIG,

            classes=[PERSON_CLASS_ID],

            device=DEVICE,

            verbose=False
        )

        return results[0]