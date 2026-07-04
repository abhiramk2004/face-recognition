"""
metrics.py

Performance metrics.
"""

import time


class Metrics:

    def __init__(self):

        self.frame_count = 0

        self.inference_times = []

        self.recognition_times = []

        self.stats_start = time.time()

        self.prev_frame = time.time()

    def update_inference(self, value):

        self.inference_times.append(value)

    def update_recognition(self, value):

        self.recognition_times.append(value)

    def fps(self):

        now = time.time()

        fps = 1.0 / max(now - self.prev_frame, 1e-6)

        self.prev_frame = now

        return fps

    def tick(self):

        self.frame_count += 1

    def should_print(self):

        return (time.time() - self.stats_start) >= 5

    def print_stats(self, track_count):

        elapsed = time.time() - self.stats_start

        avg_fps = self.frame_count / elapsed

        avg_inference = (
            sum(self.inference_times)
            /
            max(len(self.inference_times), 1)
        )

        avg_recognition = (
            sum(self.recognition_times)
            /
            max(len(self.recognition_times), 1)
        )

        print(

            f"[5s Stats] "

            f"FPS={avg_fps:.2f} | "

            f"YOLO={avg_inference:.2f}ms | "

            f"Face={avg_recognition:.2f}ms | "

            f"Tracks={track_count}"

        )

        self.reset()

    def reset(self):

        self.frame_count = 0

        self.inference_times.clear()

        self.recognition_times.clear()

        self.stats_start = time.time()