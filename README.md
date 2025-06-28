# 👕 Garment Size Detector using MoveNet & MediaPipe

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-blue)
[![IJIRT Publication](https://img.shields.io/badge/Published_in-IJIRT-red?logo=readthedocs&logoColor=white)](https://ijirt.org/publishedpaper/IJIRT172021_PAPER.pdf)

---

## 📌 Overview

This project implements an automated **garment size measurement** tool using state-of-the-art keypoint detection models — **MediaPipe** and **MoveNet** — to extract accurate body measurements (shoulder width, sleeve length, leg length) in real time from videos or webcam input.

> 🔬 Originally designed for **e-commerce and retail automation**, this system has strong applications in **healthcare** as well, where accurate body sizing is critical. Built to reduce manual errors and hallucinations, the framework was validated using real-time videos and reference images.

---

## 🏗️ Architecture

![Architecture](assets/architecture.png)

---

## 🎥 Demo

![Demo](assets/demo.gif)

---

## 📦 Features

- 📏 Pixel-to-centimeter calibration using reference object
- 🎯 Keypoint detection with MoveNet Thunder
- 🕺 Full-body MediaPipe pose landmarks
- 🧠 Real-time frame analysis with OpenCV
- 📊 Displays measurements on video overlay

---

## 🔍 Key Measurements

| Measurement Type | Accuracy (%) | Ground Truth (cm) | Estimated (cm) | MAE (cm) |
|------------------|--------------|--------------------|----------------|----------|
| Shoulder Width   | 95.99%       | 37.89              | 36.37          | 0.8      |
| Sleeve Length    | 94.99%       | 64.72              | 61.48          | 1.2      |
| Waist Width      | 96.01%       | 21.07              | 20.23          | 0.84     |
| Pant Length      | 95.00%       | 94.20              | 89.49          | 1.5      |

---

## 🛠️ Setup Instructions

```bash
# 1. Clone the repository
https://github.com/yourusername/garment-size-detector.git
cd garment-size-detector

# 2. Install dependencies
pip install -r requirements.txt
```

---

## ✅ Requirements

```txt
opencv-python
mediapipe
numpy
tensorflow
dnnlib
```

---

## 🚀 Run the Tool
Update the following variables in `src/garment_measurement.py`:

```python
reference_image_path = 'path/to/reference.jpg'   # object with known width
movenet_model_path = 'path/to/movenet.pb'
video_source = 'path/to/video.mp4'  # or 0 for webcam
```

Then run:
```bash
python src/garment_measurement.py
```

Press `q` to quit the video.

---

## 📚 Citation

If you use this project in your research, please cite:

```
Macha Naga Sai Vignesh, Saraswathi D, Kanumuri Nitin Varma, Naveen Nishal S.
"Human Pose Keypoint Detection for Apparel Sizing: Accurate Measurement Estimation using MediaPipe and MoveNet"
Vellore Institute of Technology, Chennai.
```

---

## 📄 License

Apache License 2.0 — Free to use, modify, and distribute with attribution.

---
