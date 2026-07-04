import os
import cv2
import time
import numpy as np

from ultralytics import YOLO
from insightface.app import FaceAnalysis
from app.recognition.face_model import initialize_face_model


# ==========================================================
# CONFIGURATION
# ==========================================================

YOLO_MODEL = "yolo11n.pt"
FACES_DIR = "faces"

TARGET_WIDTH = 704
TARGET_HEIGHT = 480

PERSON_CLASS = 0

TRACK_TIMEOUT = 3.0
RECHECK_INTERVAL = 5.0

SIMILARITY_THRESHOLD = 0.60

MIN_PERSON_WIDTH = 80
MIN_PERSON_HEIGHT = 80

# ==========================================================
# INSIGHTFACE INITIALIZATION
# ==========================================================


face_app = initialize_face_model()

# ==========================================================
# COSINE SIMILARITY
# ==========================================================

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# ==========================================================
# LOAD ENROLLED FACES
# ==========================================================

def load_known_faces():

    known_faces = {}

    if not os.path.exists(FACES_DIR):
        raise FileNotFoundError(
            f"Missing directory: {FACES_DIR}"
        )

    print("Loading enrolled faces...\n")

    identity_count = 0

    for person_name in sorted(os.listdir(FACES_DIR)):

        person_dir = os.path.join(
            FACES_DIR,
            person_name
        )

        if not os.path.isdir(person_dir):
            continue

        embeddings = []

        for image_name in sorted(os.listdir(person_dir)):

            image_path = os.path.join(
                person_dir,
                image_name
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            faces = face_app.get(image)

            if len(faces) == 0:
                continue

            face = max(
                faces,
                key=lambda f:
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            )

            embeddings.append(
                face.embedding
            )

        if len(embeddings) == 0:

            print(
                f"[SKIPPED] {person_name}"
            )

            continue

        mean_embedding = np.mean(
            embeddings,
            axis=0
        )

        mean_embedding = (
            mean_embedding /
            np.linalg.norm(mean_embedding)
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


# ==========================================================
# FACE RECOGNITION
# ==========================================================

def recognize_face(
    image,
    known_faces,
    threshold=SIMILARITY_THRESHOLD
):

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


# ==========================================================
# SAFE PERSON CROP
# ==========================================================

def crop_person(
    frame,
    x1,
    y1,
    x2,
    y2
):

    h, w = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ------------------------------------------------------
    # Load enrolled identities
    # ------------------------------------------------------
    known_faces = load_known_faces()

    # ------------------------------------------------------
    # Load YOLO
    # ------------------------------------------------------
    print("Loading YOLO11n model...")

    model = YOLO(YOLO_MODEL)
    model.to("cpu")

    print("YOLO ready.\n")
    url="rtsp://192.168.10.100:8554/webcam"
    # ------------------------------------------------------
    # Webcam
    # ------------------------------------------------------
    cap = cv2.VideoCapture(
    url,cv2.CAP_FFMPEG
)

    if not cap.isOpened():

        print("ERROR: Unable to access webcam.")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("Webcam opened successfully.")
    print("Tracking + Recognition started.")
    print("Press 'q' to quit.\n")

    # ------------------------------------------------------
    # Tracking state
    # ------------------------------------------------------
    active_tracks = {}

    track_recognition = {}

    # ------------------------------------------------------
    # Performance
    # ------------------------------------------------------
    frame_count = 0

    inference_times = []
    recognition_times = []

    stats_start_time = time.time()
    prev_frame_time = time.time()

    # ------------------------------------------------------
    # Main loop
    # ------------------------------------------------------
    while True:

        ret, frame = cap.read()

        if not ret:
            print("WARNING: Failed to read frame.")
            break

        frame = cv2.resize(
            frame,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )

        # --------------------------------------------------
        # YOLO + ByteTrack
        # --------------------------------------------------
        infer_start = time.perf_counter()

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[PERSON_CLASS],
            device="cpu",
            verbose=False
        )

        infer_end = time.perf_counter()

        inference_time_ms = (
            infer_end - infer_start
        ) * 1000

        inference_times.append(
            inference_time_ms
        )

        # --------------------------------------------------
        # Parse results
        # --------------------------------------------------
        result = results[0]

        visible_ids = set()

        person_count = 0
        recognized_count = 0
        unknown_count = 0

        current_time = time.time()

        if result.boxes is not None:

            for box in result.boxes:

                if box.id is None:
                    continue

                track_id = int(
                    box.id.item()
                )

                confidence = float(
                    box.conf.item()
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                visible_ids.add(
                    track_id
                )

                person_count += 1

                # ------------------------------------------
                # New track
                # ------------------------------------------
                if track_id not in active_tracks:

                    print(
                        f"[NEW TRACK] "
                        f"ID={track_id}"
                    )

                active_tracks[track_id] = (
                    current_time
                )

                # ------------------------------------------
                # Skip tiny people
                # ------------------------------------------
                width = x2 - x1
                height = y2 - y1

                if (
                    width < MIN_PERSON_WIDTH
                    or
                    height < MIN_PERSON_HEIGHT
                ):
                    continue

                # ------------------------------------------
                # Recognition trigger
                # ------------------------------------------
                need_recognition = False

                if track_id not in track_recognition:

                    need_recognition = True

                elif (
                    current_time
                    -
                    track_recognition[track_id][
                        "last_check"
                    ]
                    >
                    RECHECK_INTERVAL
                ):

                    need_recognition = True

                # ------------------------------------------
                # Run recognition
                # ------------------------------------------
                if need_recognition:

                    person_crop = crop_person(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    if (
                        person_crop is not None
                        and
                        person_crop.size > 0
                    ):

                        rec_start = (
                            time.perf_counter()
                        )

                        name, score = (
                            recognize_face(
                                person_crop,
                                known_faces
                            )
                        )

                        rec_end = (
                            time.perf_counter()
                        )

                        recognition_times.append(
                            (
                                rec_end
                                -
                                rec_start
                            )
                            * 1000
                        )

                        old_name = (
                            track_recognition[
                                track_id
                            ]["name"]
                            if track_id
                            in track_recognition
                            else None
                        )

                        track_recognition[
                            track_id
                        ] = {

                            "name": name,

                            "score": score,

                            "last_check":
                            current_time
                        }

                        if old_name != name:

                            print(
                                f"[RECOGNIZED] "
                                f"Track={track_id} "
                                f"Name={name} "
                                f"Score={score:.3f}"
                            )

                # ------------------------------------------
                # Cached result
                # ------------------------------------------
                if track_id in track_recognition:

                    name = (
                        track_recognition[
                            track_id
                        ]["name"]
                    )

                    score = (
                        track_recognition[
                            track_id
                        ]["score"]
                    )

                else:

                    name = "Unknown"
                    score = 0.0

                if name == "Unknown":
                    unknown_count += 1
                else:
                    recognized_count += 1

                # ------------------------------------------
                # Draw box
                # ------------------------------------------
                if name == "Unknown":

                    color = (
                        0,
                        0,
                        255
                    )

                else:

                    color = (
                        0,
                        255,
                        0
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
                    (
                        x1,
                        max(
                            y1 - 10,
                            20
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

        # --------------------------------------------------
        # Lost tracks
        # --------------------------------------------------
        expired_tracks = []

        for (
            track_id,
            last_seen
        ) in active_tracks.items():

            if track_id in visible_ids:
                continue

            if (
                current_time
                -
                last_seen
                >
                TRACK_TIMEOUT
            ):

                expired_tracks.append(
                    track_id
                )

        for track_id in expired_tracks:

            print(
                f"[TRACK LOST] "
                f"ID={track_id}"
            )

            del active_tracks[
                track_id
            ]

            track_recognition.pop(
                track_id,
                None
            )

        # --------------------------------------------------
        # FPS
        # --------------------------------------------------
        now = time.time()

        fps = (
            1.0 /
            max(
                now - prev_frame_time,
                1e-6
            )
        )

        prev_frame_time = now

        # --------------------------------------------------
        # Overlay
        # --------------------------------------------------
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Persons: {person_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Tracks: {len(active_tracks)}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Known: {recognized_count}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Unknown: {unknown_count}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )

        # --------------------------------------------------
        # Show frame
        # --------------------------------------------------
        cv2.imshow(
            "Phase 3 - YOLO11n + ByteTrack + InsightFace",
            frame
        )

        frame_count += 1

        # --------------------------------------------------
        # Stats every 5 sec
        # --------------------------------------------------
        elapsed = (
            time.time()
            -
            stats_start_time
        )

        if elapsed >= 5:

            avg_inference = (
                sum(inference_times)
                /
                max(
                    len(
                        inference_times
                    ),
                    1
                )
            )

            avg_recognition = (
                sum(recognition_times)
                /
                max(
                    len(
                        recognition_times
                    ),
                    1
                )
            )

            avg_fps = (
                frame_count
                /
                elapsed
            )

            print(
                f"[5s Stats] "
                f"FPS={avg_fps:.2f} | "
                f"YOLO={avg_inference:.2f}ms | "
                f"Face={avg_recognition:.2f}ms | "
                f"Tracks={len(active_tracks)}"
            )

            frame_count = 0

            inference_times.clear()
            recognition_times.clear()

            stats_start_time = time.time()

        # --------------------------------------------------
        # Quit
        # --------------------------------------------------
        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord('q'):

            print(
                "Exiting..."
            )

            break

    # ------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------
    cap.release()

    cv2.destroyAllWindows()


# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":
    main()
