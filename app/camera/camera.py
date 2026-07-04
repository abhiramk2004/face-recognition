"""
camera.py

Video source abstraction.
"""

import cv2

from app.config import (
    USE_RTSP,
    RTSP_URL,
    WEBCAM_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT
)


class Camera:

    def __init__(self):

        self.cap = None

    def open(self):

        if USE_RTSP:

            self.cap = cv2.VideoCapture(
                RTSP_URL,
                cv2.CAP_FFMPEG
            )

            self.cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

        else:

            self.cap = cv2.VideoCapture(
                WEBCAM_INDEX
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                FRAME_WIDTH
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                FRAME_HEIGHT
            )

        if not self.cap.isOpened():

            raise RuntimeError(
                "Unable to open video source."
            )

        print("Camera opened successfully.")

    def read(self):

        ret, frame = self.cap.read()

        if not ret:

            return None

        frame = cv2.resize(
            frame,
            (
                FRAME_WIDTH,
                FRAME_HEIGHT
            )
        )

        return frame

    def release(self):

        if self.cap:

            self.cap.release()