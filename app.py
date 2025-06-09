import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from threading import Thread

# ——— Initialize MediaPipe solutions ———
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)

st.set_page_config(page_title="Live Webcam Analytics", layout="wide")
st.title("Live Face & Posture Analytics")

# Placeholder elements
video_placeholder = st.empty()
status_col1, status_col2, status_col3 = st.columns(3)
status_col1.metric("Face Detected", "❌")
status_col2.metric("Looking at Camera", "❌")
status_col3.metric("Centered Posture", "❌")

# Threaded video capture for non-blocking UI
class WebcamStreamer(Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            ret, img = self.cap.read()
            if ret:
                self.frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def stop(self):
        self.running = False
        self.cap.release()

streamer = WebcamStreamer()
streamer.start()

try:
    while True:
        frame = streamer.frame
        if frame is None:
            continue

        # ——— Analytics Pipeline ———
        img = frame.copy()
        results_face = mp_face.process(img)
        face_detected = False
        looking = False
        centered = False

        h, w, _ = img.shape
        if results_face.detections:
            face_detected = True
            for detection in results_face.detections:
                bbox = detection.location_data.relative_bounding_box
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

            # Gaze: use face_mesh landmarks 33 (right eye), 263 (left eye)
            mesh = mp_mesh.process(img)
            if mesh.multi_face_landmarks:
                lm = mesh.multi_face_landmarks[0].landmark
                eye_center = np.mean([
                    [lm[33].x * w, lm[33].y * h],
                    [lm[263].x * w, lm[263].y * h]
                ], axis=0)
                dx = abs((w/2) - eye_center[0])
                looking = dx < w * 0.1  # within 10% of center

            # Posture: use pose landmarks 11 (shoulder L), 12 (shoulder R)
            pose = mp_pose.process(img)
            if pose.pose_landmarks:
                l_sh, r_sh = pose.pose_landmarks.landmark[11], pose.pose_landmarks.landmark[12]
                torso_center = ((l_sh.x + r_sh.x) / 2 * w)
                centered = abs((w/2) - torso_center) < w * 0.1

        # ——— Update UI ———
        status_col1.metric("Face Detected", "✅" if face_detected else "❌")
        status_col2.metric("Looking at Camera", "✅" if looking else "❌")
        status_col3.metric("Centered Posture", "✅" if centered else "❌")
        video_placeholder.image(img, channels="RGB")

        # throttle to ~10 FPS
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    streamer.stop()
