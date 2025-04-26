
# 👷‍♂️ SmartVision - Worker Activity Monitoring System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)  
![License](https://img.shields.io/badge/License-MIT-green.svg)  
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red.svg)  
![TensorFlow Lite](https://img.shields.io/badge/Model-TensorFlow%20Lite-orange.svg)

---

## 📄 Overview
The **Worker Activity Monitoring System** is a lightweight, real-time vision-based solution designed to detect if a worker is actively engaged in tasks.  
It is optimized for **low-cost hardware** 💻, auto-starts on boot 🔥, and works independently without manual intervention.

---

## 🎯 Objective
- Capture live video feed 📹.
- Detect presence and activity of workers 👷.
- Log working and idle states ⏱️.
- Run efficiently on startup without any user interaction 🚀.

---

## 🛠️ System Architecture

- **Camera Input**: Real-time frame capture.
- **Core Application**:
  - Built in **Python 3** 🐍.
  - Uses **OpenCV**, **TensorFlow Lite**, and **NumPy**.
- **Activity Detection**:
  - Detects human presence and movement using pose estimation 🧍.
- **Logging**:
  - Saves timestamps 🕒, worker state, activity flag, and bounding box coordinates.
- **Auto Startup**:
  - Configured with **systemd service** on Linux systems.

---

## 🧰 Tech Stack
- **Programming Language**: Python 3
- **Libraries**: OpenCV, NumPy, TensorFlow Lite
- **Optimization Tools**: TFLite, ONNX Runtime (optional)

---

## 🧠 How It Works

1. **Frame Capture**:
   - Capture frames from the camera in real time 🎥.
2. **Preprocessing**:
   - Resize and normalize frames 📐.
3. **Region of Interest (ROI)**:
   - Focus only on relevant work zones 🔍.
4. **Detection and Classification**:
   - Detect human presence 🧍‍♂️.
   - Use lightweight pose models or motion heuristics 🏃‍♂️.
5. **Decision Logic**:
   - **Working** if activity is detected ✅.
   - **Idle/No Human** if no movement ❌.

---

## 🗃️ Example Log Output

| Timestamp           | State      | Activity Detected | Bounding Box         |
|---------------------|------------|-------------------|-----------------------|
| 4/25/2025 20:14     | Working    | TRUE              | [100, 100, 400, 400]   |
| 4/25/2025 20:14     | No Human   | FALSE             | [100, 100, 400, 400]   |

---

## ⚙️ Auto-Start Setup

We configure a **systemd** service to automatically launch the system at boot:

```bash
sudo nano /etc/systemd/system/worker-monitor.service
```

Add the following:
```ini
[Unit]
Description=Worker Activity Monitoring Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 /path/to/your/script.py
WorkingDirectory=/path/to/your/
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

Activate service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable worker-monitor.service
sudo systemctl start worker-monitor.service
```

---

## 📂 Project Structure

```
SmartVision-Worker-Monitor/
├── model/
│   └── movenet_singlepose_lightning.tflite
├── scripts/
│   └── worker_monitor.py
├── logs/
│   └── activity_log.csv
├── systemd/
│   └── worker-monitor.service
└── README.md
```

---

## 🚀 Future Improvements
- Integrate alerts 📢 when worker is idle for too long.
- Dashboard visualization 📊 for management.
- Multi-worker tracking 👥.
- Use ONNX or TensorRT for even faster inference ⚡.

---

## 👏 Acknowledgments
- TensorFlow Lite Pose Estimation 📚
- OpenCV for real-time video processing 🎥

---

> **Made with ❤️ for industrial automation.**

---
