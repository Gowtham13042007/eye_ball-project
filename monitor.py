import base64
import os
import statistics
import time
import threading
from collections import deque
import copy

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Configuration (Aligned with Technical Documentation Specs)
# ---------------------------------------------------------------------------
FACE_MODEL_PATH = os.environ.get(
    "FACE_LANDMARKER_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task"),
)

WARNING_LIMIT = int(os.environ.get("WARNING_LIMIT", "5"))

# Documented baseline thresholds (Symmetric extensions applied where implied)
DEFAULT_GAZE_H_RANGE = (0.42, 0.58)    # Left < 0.42, Right > 0.58
DEFAULT_GAZE_V_RANGE = (0.40, 0.60)    # Baseline window for vertical gaze
DEFAULT_HEAD_H_RANGE = (0.45, 0.55)    # Left < 0.45, Right > 0.55
DEFAULT_HEAD_V_RANGE = (0.40, 0.60)    # Up < 0.40, Down > 0.60

EYE_CLOSED_RATIO = float(os.environ.get("EYE_CLOSED_RATIO", "0.18"))
EYE_CLOSED_DURATION_SECONDS = float(os.environ.get("EYE_CLOSED_DURATION_SECONDS", "1.2"))
WARNING_INTERVAL_SECONDS = float(os.environ.get("WARNING_INTERVAL_SECONDS", "4"))

GRACE_PERIOD_SECONDS = float(os.environ.get("GRACE_PERIOD_SECONDS", "3"))
CALIBRATION_MIN_SAMPLES = 15
CALIBRATION_TOLERANCE_GAZE = float(os.environ.get("CALIBRATION_TOLERANCE_GAZE", "0.08"))
CALIBRATION_TOLERANCE_HEAD_H = float(os.environ.get("CALIBRATION_TOLERANCE_HEAD_H", "0.05"))
CALIBRATION_TOLERANCE_HEAD_V = float(os.environ.get("CALIBRATION_TOLERANCE_HEAD_V", "0.10"))

FACE_COUNT_CONFIRM_FRAMES = int(os.environ.get("FACE_COUNT_CONFIRM_FRAMES", "6"))
REASON_VOTE_WINDOW = int(os.environ.get("REASON_VOTE_WINDOW", "10"))
REASON_VOTE_RATIO = float(os.environ.get("REASON_VOTE_RATIO", "0.6"))

REASON_INTERVAL_MULTIPLIERS = {
    "Multiple faces detected": 0.35,
    "No face detected": 0.6,
    "Gaze deviation": 1.0,
    "Head turned away": 1.0,
    "Looking down": 1.4,
    "Eyes closed": 1.4,
}

WEBCAM_OPEN_RETRIES = int(os.environ.get("WEBCAM_OPEN_RETRIES", "3"))
BUFFER_SIZE = 8
VOTED_REASONS = ("Gaze deviation", "Head turned away", "Looking down")


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


class AttentionMonitor:
    def __init__(self, model_path: str = FACE_MODEL_PATH, warning_limit: int = WARNING_LIMIT):
        self.model_path = model_path
        self.warning_limit = warning_limit

        self._lock = threading.Lock()
        self._thread = None
        self.running = False

        self.warnings = 0
        self.disqualified = False
        self.history = []
        self.last_error = None
        self.webcam_ok = True
        self.calibrated = False

        self._buffers = {"gaze_h": [], "gaze_v": [], "head_h": [], "head_v": []}
        self._latest_jpeg = None
        self._current_reasons = []
        
        # State tracking strings matching documentation outputs
        self.current_gaze_state = "CENTER"
        self.current_face_state = "FRONT"

        self._violation_start = None
        self._streak_warnings_issued = 0
        self.violation_seconds = 0.0
        self.seconds_to_next_warning = None

        self._start_time = None
        self._reset_detection_state()

    def _reset_detection_state(self):
        self._calibration_samples = {"gaze_h": [], "gaze_v": [], "head_h": [], "head_v": []}
        self._gaze_h_range = DEFAULT_GAZE_H_RANGE
        self._gaze_v_range = DEFAULT_GAZE_V_RANGE
        self._head_h_range = DEFAULT_HEAD_H_RANGE
        self._head_v_range = DEFAULT_HEAD_V_RANGE
        self._eyes_closed_since = None
        self._face_miss_streak = 0
        self._face_multi_streak = 0
        self._reason_votes = {name: deque(maxlen=REASON_VOTE_WINDOW) for name in VOTED_REASONS}

    def _smooth(self, val, key):
        buf = self._buffers[key]
        buf.append(val)
        if len(buf) > BUFFER_SIZE:
            buf.pop(0)
        return sum(buf) / len(buf)

    def _vote(self, reason, is_present_raw):
        votes = self._reason_votes[reason]
        votes.append(1 if is_present_raw else 0)
        if len(votes) < max(3, REASON_VOTE_WINDOW // 2):
            return False
        return (sum(votes) / len(votes)) >= REASON_VOTE_RATIO

    def _maybe_calibrate(self, gaze_h, gaze_v, head_h, head_v):
        samples = self._calibration_samples
        samples["gaze_h"].append(gaze_h)
        samples["gaze_v"].append(gaze_v)
        samples["head_h"].append(head_h)
        samples["head_v"].append(head_v)

    def _finish_calibration(self):
        samples = self._calibration_samples
        if len(samples["gaze_h"]) >= CALIBRATION_MIN_SAMPLES:
            gh_mid = statistics.median(samples["gaze_h"])
            gv_mid = statistics.median(samples["gaze_v"])
            hh_mid = statistics.median(samples["head_h"])
            hv_mid = statistics.median(samples["head_v"])

            self._gaze_h_range = (
                _clamp(gh_mid - CALIBRATION_TOLERANCE_GAZE, 0.05, 0.95),
                _clamp(gh_mid + CALIBRATION_TOLERANCE_GAZE, 0.05, 0.95),
            )
            self._gaze_v_range = (
                _clamp(gv_mid - CALIBRATION_TOLERANCE_GAZE, 0.05, 0.95),
                _clamp(gv_mid + CALIBRATION_TOLERANCE_GAZE, 0.05, 0.95),
            )
            self._head_h_range = (
                _clamp(hh_mid - CALIBRATION_TOLERANCE_HEAD_H, 0.05, 0.95),
                _clamp(hh_mid + CALIBRATION_TOLERANCE_HEAD_H, 0.05, 0.95),
            )
            self._head_v_range = (
                _clamp(hv_mid - CALIBRATION_TOLERANCE_HEAD_V, 0.05, 0.95),
                _clamp(hv_mid + CALIBRATION_TOLERANCE_HEAD_V, 0.05, 0.95),
            )
            self.calibrated = True
        else:
            self._gaze_h_range = DEFAULT_GAZE_H_RANGE
            self._gaze_v_range = DEFAULT_GAZE_V_RANGE
            self._head_h_range = DEFAULT_HEAD_H_RANGE
            self._head_v_range = DEFAULT_HEAD_V_RANGE
            self.calibrated = False

    def start(self):
        with self._lock:
            if self.running:
                return
            self.running = True
            self.warnings = 0
            self.disqualified = False
            self.history = []
            self.last_error = None
            self.webcam_ok = True
            self.calibrated = False
            self._buffers = {"gaze_h": [], "gaze_v": [], "head_h": [], "head_v": []}
            self._latest_jpeg = None
            self._current_reasons = []
            self.current_gaze_state = "CENTER"
            self.current_face_state = "FRONT"
            self._violation_start = None
            self._streak_warnings_issued = 0
            self.violation_seconds = 0.0
            self.seconds_to_next_warning = None
            self._start_time = time.time()
            self._reset_detection_state()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        with self._lock:
            self.running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def report_external_violation(self, reason: str):
        with self._lock:
            if not self.running or self.disqualified:
                return self._status_locked()

            self.warnings += 1
            self.history.append({
                "warning": self.warnings,
                "reasons": [reason],
                "time": time.ctime(),
                "snapshot": None,
            })
            if self.warnings >= self.warning_limit:
                self.disqualified = True
            return self._status_locked()

    def _open_webcam(self):
        for _ in range(WEBCAM_OPEN_RETRIES):
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                return cap
            cap.release()
            time.sleep(0.5)
        return None

    def _run(self):
        cap = None
        detector = None
        was_in_grace = True
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=2)
            detector = vision.FaceLandmarker.create_from_options(options)

            cap = self._open_webcam()
            if cap is None:
                with self._lock:
                    self.last_error = "Could not open webcam (device 0) after retries."
                    self.webcam_ok = False
                    self.running = False
                return

            consecutive_read_failures = 0

            while True:
                with self._lock:
                    if not self.running:
                        break

                success, frame = cap.read()
                if not success:
                    consecutive_read_failures += 1
                    with self._lock:
                        self.webcam_ok = consecutive_read_failures < 30
                    if consecutive_read_failures >= 90:
                        with self._lock:
                            self.last_error = "Webcam stopped returning frames."
                            self.running = False
                        break
                    time.sleep(0.03)
                    continue
                consecutive_read_failures = 0

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = detector.detect(mp_image)

                in_grace = (
                    self._start_time is not None
                    and (time.time() - self._start_time) < GRACE_PERIOD_SECONDS
                )
                if was_in_grace and not in_grace:
                    self._finish_calibration()
                was_in_grace = in_grace

                raw_reasons = []
                h, w, _ = frame.shape
                iris_points_px = []
                
                gaze_state = "OUT OF FRAME"
                face_state = "OUT OF FRAME"

                num_faces = len(result.face_landmarks) if result.face_landmarks else 0

                if num_faces == 0:
                    self._face_miss_streak += 1
                    self._face_multi_streak = 0
                    self._eyes_closed_since = None
                    if self._face_miss_streak >= FACE_COUNT_CONFIRM_FRAMES:
                        raw_reasons.append("No face detected")
                elif num_faces > 1:
                    self._face_multi_streak += 1
                    self._face_miss_streak = 0
                    self._eyes_closed_since = None
                    if self._face_multi_streak >= FACE_COUNT_CONFIRM_FRAMES:
                        raw_reasons.append("Multiple faces detected")
                    face_state = "MULTIPLE"
                else:
                    self._face_miss_streak = 0
                    self._face_multi_streak = 0
                    lm = result.face_landmarks[0]

                    # 1. Gaze Deviation Analysis
                    l_iris, l_low, l_high = lm[468].x, lm[33].x, lm[133].x
                    r_iris, r_low, r_high = lm[473].x, lm[362].x, lm[263].x
                    
                    gaze_h = self._smooth(
                        ((l_iris - l_low) / max((l_high - l_low), 1e-6) + 
                         (r_iris - r_low) / max((r_high - r_low), 1e-6)) / 2,
                        "gaze_h",
                    )
                    
                    # Vertical Gaze Ratio Tracking
                    l_iris_y, l_top_y, l_bot_y = lm[468].y, lm[159].y, lm[145].y
                    r_iris_y, r_top_y, r_bot_y = lm[473].y, lm[386].y, lm[374].y
                    gaze_v = self._smooth(
                        ((l_iris_y - l_top_y) / max((l_bot_y - l_top_y), 1e-6) + 
                         (r_iris_y - r_top_y) / max((r_bot_y - r_top_y), 1e-6)) / 2,
                        "gaze_v"
                    )

                    # 2. Stable Head Pose Analysis (Yaw / Pitch)
                    nx, ny = lm[1].x, lm[1].y
                    lb, rb = lm[234].x, lm[454].x
                    tb, bb = lm[10].y, lm[152].y
                    head_h = self._smooth((nx - lb) / max((rb - lb), 1e-6), "head_h")
                    head_v = self._smooth((ny - tb) / max((bb - tb), 1e-6), "head_v")

                    if in_grace:
                        self._maybe_calibrate(gaze_h, gaze_v, head_h, head_v)

                    # State Assignment Logic Matching Documentation Output Options
                    gaze_lo_h, gaze_hi_h = self._gaze_h_range
                    gaze_lo_v, gaze_hi_v = self._gaze_v_range
                    head_lo_h, head_hi_h = self._head_h_range
                    head_lo_v, head_hi_v = self._head_v_range

                    # Calculate Explicit Gaze Direction
                    if gaze_h < gaze_lo_h:
                        gaze_state = "LEFT"
                    elif gaze_h > gaze_hi_h:
                        gaze_state = "RIGHT"
                    elif gaze_v < gaze_lo_v:
                        gaze_state = "UP"
                    elif gaze_v > gaze_hi_v:
                        gaze_state = "DOWN"
                    else:
                        gaze_state = "CENTER"

                    # Calculate Explicit Face Orientation Direction
                    if head_h < head_lo_h:
                        face_state = "LEFT"
                    elif head_h > head_hi_h:
                        face_state = "RIGHT"
                    elif head_v < head_lo_v:
                        face_state = "UP"
                    elif head_v > head_hi_v:
                        face_state = "DOWN"
                    else:
                        face_state = "FRONT"

                    # Flag Determinations
                    gaze_raw_flag = gaze_state != "CENTER"
                    head_h_raw_flag = face_state in ("LEFT", "RIGHT")
                    head_v_raw_flag = face_state == "DOWN"  # Specific "Looking Down" validation

                    if self._vote("Gaze deviation", gaze_raw_flag):
                        raw_reasons.append("Gaze deviation")
                    if self._vote("Head turned away", head_h_raw_flag):
                        raw_reasons.append("Head turned away")
                    if self._vote("Looking down", head_v_raw_flag):
                        raw_reasons.append("Looking down")

                    # 3. Drowsiness / Eyes-closed Check
                    def px(i):
                        return (lm[i].x * w, lm[i].y * h)

                    left_ear = _dist(px(159), px(145)) / max(_dist(px(33), px(133)), 1e-6)
                    right_ear = _dist(px(386), px(374)) / max(_dist(px(362), px(263)), 1e-6)
                    ear = (left_ear + right_ear) / 2

                    now_ts = time.time()
                    if ear < EYE_CLOSED_RATIO:
                        if self._eyes_closed_since is None:
                            self._eyes_closed_since = now_ts
                        elif now_ts - self._eyes_closed_since >= EYE_CLOSED_DURATION_SECONDS:
                            raw_reasons.append("Eyes closed")
                    else:
                        self._eyes_closed_since = None

                    iris_points_px = [
                        (int(l_iris * w), int(lm[468].y * h)),
                        (int(r_iris * w), int(lm[473].y * h)),
                    ]

                reasons = raw_reasons
                focused = len(reasons) == 0

                # Focus Feedback Color Rule: Green if Focus metrics clear, Orange/Red when anomalous
                if in_grace:
                    status_color = (255, 200, 0)      # Orange
                    status_text = "CALIBRATING..."
                elif focused and face_state == "FRONT" and gaze_state == "CENTER":
                    status_color = (86, 199, 52)      # Green Focus Feedback
                    status_text = "FOCUSED"
                else:
                    status_color = (60, 60, 255)      # Red Alert Feedback
                    countdown = self.seconds_to_next_warning
                    countdown_str = f" ({countdown:.1f}s to warning)" if countdown is not None else ""
                    status_text = " / ".join(r.upper() for r in reasons) + countdown_str

                # Render Modern Semi-Transparent HUD Box
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (420, 115), (28, 28, 28), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

                # HUD Metrics Display Layout
                cv2.putText(frame, f"WARNINGS: {self.warnings} / {self.warning_limit}",
                            (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(frame, f"FACE: {face_state}   |   GAZE: {gaze_state}",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame, status_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

                # Pupil Indicators (Visual Confirmation Red Dots)
                for (px_, py_) in iris_points_px:
                    cv2.circle(frame, (px_, py_), 4, (0, 0, 255), -1)

                if not focused and not in_grace:
                    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), status_color, 4)

                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                encoded_frame = buf.tobytes() if ok else None

                now = time.time()
                with self._lock:
                    if encoded_frame is not None:
                        self._latest_jpeg = encoded_frame
                    self._current_reasons = reasons
                    self.current_gaze_state = gaze_state
                    self.current_face_state = face_state
                    self.webcam_ok = True

                    if reasons and not in_grace:
                        if self._violation_start is None:
                            self._violation_start = now
                            self._streak_warnings_issued = 0

                        self.violation_seconds = now - self._violation_start

                        effective_interval = WARNING_INTERVAL_SECONDS * min(
                            REASON_INTERVAL_MULTIPLIERS.get(r, 1.0) for r in reasons
                        )
                        due = int(self.violation_seconds // effective_interval)

                        if due > self._streak_warnings_issued:
                            new_warnings = due - self._streak_warnings_issued
                            self._streak_warnings_issued = due
                            self.warnings += new_warnings

                            snapshot_b64 = (
                                base64.b64encode(encoded_frame).decode("ascii")
                                if encoded_frame is not None
                                else None
                            )
                            self.history.append({
                                "warning": self.warnings,
                                "reasons": sorted(set(reasons)),
                                "time": time.ctime(),
                                "snapshot": snapshot_b64,
                            })
                            if self.warnings >= self.warning_limit:
                                self.disqualified = True

                        remaining = effective_interval - (self.violation_seconds % effective_interval)
                        self.seconds_to_next_warning = remaining
                    else:
                        self._violation_start = None
                        self._streak_warnings_issued = 0
                        self.violation_seconds = 0.0
                        self.seconds_to_next_warning = None

                time.sleep(0.03)

        except Exception as exc:
            with self._lock:
                self.last_error = str(exc)
                self.webcam_ok = False
        finally:
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass
            if cap is not None:
                cap.release()
            with self._lock:
                self.running = False

    def _status_locked(self):
        return {
            "running": self.running,
            "warnings": self.warnings,
            "limit": self.warning_limit,
            "disqualified": self.disqualified,
            "history": copy.deepcopy(self.history),
            "error": self.last_error,
            "webcam_ok": self.webcam_ok,
            "calibrated": self.calibrated,
            "current_reasons": list(self._current_reasons),
            "gaze_state": self.current_gaze_state,
            "face_state": self.current_face_state,
            "violation_seconds": round(self.violation_seconds, 1),
            "seconds_to_next_warning": (
                round(self.seconds_to_next_warning, 1) if self.seconds_to_next_warning is not None else None
            ),
        }

    def get_status(self):
        with self._lock:
            return self._status_locked()

    def get_frame(self):
        with self._lock:
            return self._latest_jpeg

monitor = AttentionMonitor()