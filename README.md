# Human Pose Keypoint Detection for Apparel Sizing 👕

<div align="center">
  <img src="https://kemtai.com/blog/the-complete-guide-to-human-pose-estimation" alt="Apparel Sizing with Pose Estimation" width="800">
  
  <h1>Automated Apparel Sizing via Pose Keypoint Detection</h1>
  
  <p>
    <strong>MoveNet + MediaPipe implementation for accurate garment measurements</strong>
  </p>

  <p>
    <a href="https://ijirt.org/publishedpaper/IJIRT172021_PAPER.pdf">
      <img alt="Paper PDF" src="https://img.shields.io/badge/Paper-PDF-blue">
    </a>
    <a href="LICENSE">
      <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg">
    </a>
  </p>
</div>

## 🔍 Project Overview

Official implementation for the paper **"Human Pose Keypoint Detection for Apparel Sizing: Accurate Measurement Estimation using MediaPipe and MoveNet"** (Published in IJIRT | Volume 11 Issue 8 | January 2025)

**Key Features:**
- Real-time body measurement extraction (sleeve length, shoulder width, pant length)
- Pixel-to-centimeter calibration system
- MoveNet Thunder + MediaPipe integration
- Average MAE of 1.17cm across measurements
- 41.7 FPS processing speed

## 🛠️ Technical Architecture

graph TD
A[Input Image/Video] --> B[MoveNet Keypoint Detection]
B --> C[MediaPipe Pose Refinement]
C --> D[Pixel-to-CM Calibration]
D --> E[Measurement Calculation]
E --> F[Sleeve Length]
E --> G[Shoulder Width]
E --> H[Pant Length]

## 🚀 Quick Start

### Installation

git clone https://github.com/yourusername/apparel-sizing.git
cd apparel-sizing

Create virtual environment
python -m venv venv
source venv/bin/activate # Linux/Mac

venv\Scripts\activate # Windows
Install dependencies
pip install -r requirements.txt



### Basic Usage
from core.pipeline import ApparelSizingSystem

Initialize system
sizing_system = ApparelSizingSystem(
movenet_thunder="models/movenet_thunder.tflite",
ref_height_cm=175 # Reference height for calibration
)

Process image
measurements = sizing_system.estimate("data/raw/test_image.jpg")

print(f"""
Shoulder Width: {measurements['shoulder_width']:.1f} cm
Sleeve Length: {measurements['sleeve_length']:.1f} cm
Pant Length: {measurements['pant_length']:.1f} cm
""")


## 📊 Key Technical Components

### 1. Measurement Algorithms
**Shoulder Width Calculation:**

def calculate_shoulder_width(left_shoulder, right_shoulder, scale):
return np.linalg.norm(left_shoulder - right_shoulder) * scale


**Pixel-to-Centimeter Calibration:**

def get_scale_factor(ref_height_px, ref_height_cm=175):
return ref_height_cm / ref_height_px # From neck to hip distance



### 2. Performance Metrics
| Measurement       | MAE (cm) | PCK@0.5 (%) |
|--------------------|----------|-------------|
| Sleeve Length      | 1.2      | 95.2        |
| Shoulder Width     | 0.8      | 97.8        |
| Pant Length        | 1.5      | 94.9        |

## 🤝 Contributors
- Macha Naga Sai Vignesh
- Kanumuri Nitin Varma
- Naveen Nishal Singuru

<div align="center">
  <img src="figures/measurement_demo.gif" alt="Demo" width="600">
  <p><em>Real-time measurement demonstration</em></p>
</div>
