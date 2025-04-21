# ♻️ AeroSweep – Autonomous Trash Collecting Drone using Jetson Nano 🚁🧠

AeroSweep is an **intelligent autonomous drone** powered by **NVIDIA Jetson Nano** that detects and locates **plastic trash (like water bottles)** using computer vision and deep learning. It’s designed to **identify non-biodegradable waste**, mark GPS coordinates, and can potentially integrate with pickup mechanisms in future iterations.

> 💡 Developed as a mini project for environmental sustainability using AI + Drones!

---

## 📸 Project Preview

![Front View of Setup](./1.jpg)
![Top View of Drone Setup](./2.jpg)

---

## ⚙️ Components Used

| Component               | Description                             |
|------------------------|-----------------------------------------|
| **Jetson Nano**        | Onboard AI processing and camera input  |
| **Webcam (USB)**       | For real-time trash detection           |
| **Flight Controller**  | Mamba MK4 F722                          |
| **ESC**                | Mamba F65A 128k 4-in-1                  |
| **Motors**             | 2207 Super Nova                         |
| **Frame**              | Manta SE 5-inch                         |
| **Receiver**           | Radiomaster RP3                         |
| **WiFi Antennas**      | Dual for connectivity and communication |
| **Power Source**       | 4S LiPo Battery                         |

---

## 🧠 AI Model – Trash Detection

- **Model**: MobileNetV2 (trained to detect recyclable trash – plastic bottles)
- **Framework**: TensorFlow / Keras
- **Training Dataset**: Custom collected images of plastic bottles + background
- **Accuracy**: ~95% on validation set
- **Input**: Webcam feed
- **Output**: Bounding box around trash + confidence

---

## 🖥️ Software Stack

- Python
- OpenCV
- TensorFlow / Keras
- Flask (for interface)
- JetPack on Jetson Nano

---

## 🔧 How it Works

1. Drone hovers or moves low-altitude over land.
2. Camera captures real-time video.
3. Jetson Nano processes each frame using MobileNetV2.
4. If **trash (like plastic bottle)** is detected:
   - Marks location
   - Sends alert or GPS coordinate
   - (Future: Triggers robotic arm to pick it up)

---

## 🚀 Setup Instructions

1. **Set up Jetson Nano**
   - Flash SD card with JetPack.
   - Connect webcam, WiFi dongle, HDMI monitor.

2. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-opencv
   pip3 install tensorflow flask numpy
