"""
pipeline.py

Main Vision Pipeline.
"""

import cv2
import time

from app.camera.camera import Camera

from app.detection.detector import YOLODetector
from app.detection.tracker import TrackManager

from app.recognition.service import RecognitionService

from app.visualization.renderer import Renderer

from app.utils.metrics import Metrics
from app.utils.helpers import crop_person

from app.config import (
    RECHECK_INTERVAL,
    MIN_PERSON_WIDTH,
    MIN_PERSON_HEIGHT
)


class VisionPipeline:

    def __init__(self):

        print("Initializing Vision Pipeline...\n")

        self.recognizer = RecognitionService()

        self.camera = Camera()

        self.detector = YOLODetector()

        self.tracker = TrackManager()

        self.renderer = Renderer()

        self.metrics = Metrics()

    def run(self):

        self.camera.open()

        print("Tracking + Recognition started.")
        print("Press 'q' to quit.\n")

        while True:

            frame = self.camera.read()

            if frame is None:

                print("WARNING: Failed to read frame.")

                break

            infer_start = time.perf_counter()

            result = self.detector.detect(frame)

            infer_end = time.perf_counter()

            self.metrics.update_inference(

                (infer_end - infer_start) * 1000

            )

            visible_ids = set()

            person_count = 0
            recognized_count = 0
            unknown_count = 0

            current_time = time.time()

            if result.boxes is not None:

                for box in result.boxes:

                    if box.id is None:
                        continue

                    track_id = int(box.id.item())

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    visible_ids.add(track_id)

                    person_count += 1

                    is_new = self.tracker.update_track(
                        track_id
                    )

                    if is_new:

                        print(
                            f"[NEW TRACK] ID={track_id}"
                        )

                    width = x2 - x1
                    height = y2 - y1

                    if (
                        width < MIN_PERSON_WIDTH
                        or
                        height < MIN_PERSON_HEIGHT
                    ):
                        continue

                    cached = self.tracker.get_recognition(
                        track_id
                    )

                    need_recognition = False

                    if cached is None:

                        need_recognition = True

                    elif (
                        current_time
                        -
                        cached["last_check"]
                        >
                        RECHECK_INTERVAL
                    ):

                        need_recognition = True

                    if need_recognition:

                        crop = crop_person(
                            frame,
                            x1,
                            y1,
                            x2,
                            y2
                        )

                        if (
                            crop is not None
                            and
                            crop.size > 0
                        ):

                            rec_start = (
                                time.perf_counter()
                            )

                            name, score = self.recognizer.recognize(crop)

                            rec_end = (
                                time.perf_counter()
                            )

                            self.metrics.update_recognition(

                                (rec_end - rec_start)
                                * 1000

                            )

                            old = (
                                cached["name"]
                                if cached
                                else None
                            )

                            self.tracker.set_recognition(

                                track_id,

                                name,

                                score

                            )

                            if old != name:

                                print(

                                    f"[RECOGNIZED] "

                                    f"Track={track_id} "

                                    f"Name={name} "

                                    f"Score={score:.3f}"

                                )

                    cached = self.tracker.get_recognition(
                        track_id
                    )

                    if cached:

                        name = cached["name"]

                        score = cached["score"]

                    else:

                        name = "Unknown"

                        score = 0.0

                    if name == "Unknown":
                        unknown_count += 1
                    else:
                        recognized_count += 1

                    self.renderer.draw_person(

                        frame,

                        x1,
                        y1,
                        x2,
                        y2,

                        track_id,

                        name,

                        score

                    )

            self.tracker.cleanup(visible_ids)

            fps = self.metrics.fps()

            self.renderer.draw_stats(

                frame,

                fps,

                person_count,

                self.tracker.active_count(),

                recognized_count,

                unknown_count

            )

            self.renderer.show(frame)

            self.metrics.tick()

            if self.metrics.should_print():

                self.metrics.print_stats(

                    self.tracker.active_count()

                )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

        self.camera.release()

        cv2.destroyAllWindows()