import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime
import tensorflow as tf
import screeninfo

# Configuration
IDLE_THRESHOLD = 10  # seconds of inactivity
LOG_FILE = "activity_log.csv"
MODEL_PATH = "D:/Worker/model/movenet_singlepose_lightning.tflite"

# Detection Parameters
MIN_KEYPOINTS = 5
KEYPOINT_CONFIDENCE = 0.3
UPPER_BODY_CONFIDENCE = 0.4
MOTION_THRESHOLD = 1000
CONSECUTIVE_FRAMES = 5



screen = screeninfo.get_monitors()[0]
monitor_width = screen.width
monitor_height = screen.height



import screeninfo

class WorkerMonitor:
    def __init__(self):
        # Get monitor size
        screen = screeninfo.get_monitors()[0]
        self.monitor_width = screen.width
        self.monitor_height = screen.height

        # Initialize model
        self.interpreter = tf.lite.Interpreter(MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Setup camera
        #self.cap = cv2.VideoCapture("D:\Worker\Assembly line worker.mp4")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        # Set frame size to monitor size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.monitor_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.monitor_height)

        # Let user draw ROI
        self.roi = self.select_roi()

        # 🧠 Add these missing initializations:
        self.prev_frame = None
        self.last_active = time.time()
        self.state = "No Human"
        self.human_buffer = []

    def select_roi(self):
        """Let user draw ROI on the first frame."""
        print("Draw ROI and press ENTER or SPACE. Press ESC to cancel.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI")

            if roi != (0, 0, 0, 0):
                x, y, w, h = roi
                return [x, y, x + w, y + h]
            else:
                print("Invalid ROI selected. Try again.")

    def log_activity(self, state):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([timestamp, state, self.roi])
        print(f"[{timestamp}] {state}")

    def detect_pose(self, frame):
        """Process frame through MoveNet"""
        img = cv2.resize(frame, (192, 192))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img.astype(np.uint8), axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

    def is_human_present(self, keypoints):
        """Enhanced human verification"""
        confidences = keypoints[0, 0, :, 2]
        upper_body = confidences[5:11]

        has_min_keypoints = np.sum(confidences > KEYPOINT_CONFIDENCE) >= MIN_KEYPOINTS
        has_upper_body = np.any(upper_body > UPPER_BODY_CONFIDENCE)

        return has_min_keypoints and has_upper_body

    def update_human_buffer(self, current_status):
        """Temporal filtering for reliable detection"""
        self.human_buffer.append(current_status)
        if len(self.human_buffer) > CONSECUTIVE_FRAMES:
            self.human_buffer.pop(0)

        return sum(self.human_buffer) > (CONSECUTIVE_FRAMES // 2)

    def calculate_motion(self, current_gray):
        """Quantify movement in ROI"""
        if self.prev_frame is None:
            return 0

        diff = cv2.absdiff(self.prev_frame, current_gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        return np.sum(thresh) / 255

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                x1, y1, x2, y2 = self.roi
                roi_frame = frame[y1:y2, x1:x2]

                if roi_frame.size == 0:
                    continue

                keypoints = self.detect_pose(roi_frame)
                current_detection = self.is_human_present(keypoints)
                human_confirmed = self.update_human_buffer(current_detection)

                gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                motion = self.calculate_motion(gray)
                self.prev_frame = gray

                new_state = self.state
                if human_confirmed:
                    if motion > MOTION_THRESHOLD:
                        new_state = "Working"
                        self.last_active = time.time()
                    elif time.time() - self.last_active > IDLE_THRESHOLD:
                        new_state = "Idle"
                    else:
                        new_state = "Working"
                else:
                    new_state = "No Human"
                    self.last_active = time.time()

                if new_state != self.state:
                    self.log_activity(new_state)
                    self.state = new_state

                self.display_status(frame, roi_frame, keypoints, motion)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def display_status(self, frame, roi_frame, keypoints, motion):
        x1, y1, x2, y2 = self.roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        status_text = f"{self.state} | Motion: {motion:.0f}"
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if keypoints is not None:
            conf = np.max(keypoints[0, 0, :, 2])
            cv2.putText(frame, f"Conf: {conf:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Worker Monitor", frame)


if __name__ == "__main__":
    monitor = WorkerMonitor()
    monitor.run()
