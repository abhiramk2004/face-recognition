"""
renderer.py

Responsible for all visualization.
"""

import cv2

from app.config import (
    WINDOW_NAME,
    FONT,
    FONT_SCALE,
    FONT_THICKNESS,
    KNOWN_COLOR,
    UNKNOWN_COLOR,
    FPS_COLOR,
    PERSON_COUNT_COLOR,
    TEXT_COLOR
)


class Renderer:

    def draw_person(
        self,
        frame,
        x1,
        y1,
        x2,
        y2,
        track_id,
        name,
        score
    ):

        color = (
            UNKNOWN_COLOR
            if name == "Unknown"
            else KNOWN_COLOR
        )

        label = (
            f"ID:{track_id} "
            f"{name} "
            f"{score:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            FONT,
            FONT_SCALE,
            color,
            FONT_THICKNESS
        )


    def draw_stats(
        self,
        frame,
        fps,
        person_count,
        track_count,
        recognized_count,
        unknown_count
    ):

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10,30),
            FONT,
            0.75,
            FPS_COLOR,
            2
        )

        cv2.putText(
            frame,
            f"Persons: {person_count}",
            (10,60),
            FONT,
            0.75,
            PERSON_COUNT_COLOR,
            2
        )

        cv2.putText(
            frame,
            f"Tracks: {track_count}",
            (10,90),
            FONT,
            0.75,
            TEXT_COLOR,
            2
        )

        cv2.putText(
            frame,
            f"Known: {recognized_count}",
            (10,120),
            FONT,
            0.75,
            KNOWN_COLOR,
            2
        )

        cv2.putText(
            frame,
            f"Unknown: {unknown_count}",
            (10,150),
            FONT,
            0.75,
            UNKNOWN_COLOR,
            2
        )


    def show(self, frame):

        cv2.imshow(
            WINDOW_NAME,
            frame
        )