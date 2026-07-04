"""
tracker.py

Application-level track management.
"""

import time

from app.config import TRACK_TIMEOUT


class TrackManager:

    def __init__(self):

        self.active_tracks = {}

        self.track_recognition = {}

    def update_track(self, track_id):

        now = time.time()

        is_new = track_id not in self.active_tracks

        self.active_tracks[track_id] = now

        return is_new

    def cleanup(self, visible_ids):

        now = time.time()

        expired = []

        for track_id, last_seen in self.active_tracks.items():

            if track_id in visible_ids:
                continue

            if now - last_seen > TRACK_TIMEOUT:

                expired.append(track_id)

        for track_id in expired:

            print(f"[TRACK LOST] ID={track_id}")

            del self.active_tracks[track_id]

            self.track_recognition.pop(track_id, None)

        return expired
    def has_recognition(self, track_id):
        return track_id in self.track_recognition


    def get_recognition(self, track_id):
        return self.track_recognition.get(track_id)


    def set_recognition(self, track_id, name, score):

        self.track_recognition[track_id] = {

            "name": name,

            "score": score,

            "last_check": time.time()
        }
    def active_count(self):
        """
        Return the number of currently active tracks.
        """
        return len(self.active_tracks)
