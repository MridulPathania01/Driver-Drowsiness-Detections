# 🚗 Driver Drowsiness Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

> A real-time computer vision system capable of detecting driver fatigue using deep learning, achieving **96.7% accuracy** with a custom CNN architecture.

---

### 👨‍💻 Project By
**Mridul Pathania** | **Midhun KP** | **Ashutosh Chauhan**
*Department of Computer Science & Engineering, Lovely Professional University*

---

## 📌 Overview
Driver fatigue is a leading cause of road accidents globally. This project aims to mitigate this risk by developing a robust **Drowsiness Detection System**. By leveraging **Deep Learning** and **OpenCV**, the system monitors the driver's facial features (specifically eye closure and yawning) through a webcam to classify their state as either **Alert** or **Drowsy**.

The system is optimized for CPU performance, running at **~22 FPS**, making it suitable for deployment in non-GPU environments.

## ✨ Key Features
* **Real-Time Monitoring:** Continuous live-stream analysis via webcam.
* **High Accuracy:** The custom CNN model achieves **96.7% accuracy** on the test set.
* **Imbalance Handling:** Implements class weighting to handle dataset disparities between 'Drowsy' and 'Natural' states.
* **Robustness:** Data augmentation (rotation, zoom, shift, flip) ensures the model performs well under various head poses.
* **Lightweight:** Designed to run efficiently on standard CPUs without heavy hardware requirements.

---

## 🛠️ Technologies Used

| Domain | Tech Stack |
| :--- | :--- |
| **Deep Learning** | TensorFlow, Keras |
| **Computer Vision** | OpenCV (Haar Cascades) |
| **Data Processing** | NumPy, Matplotlib |
| **Language** | Python |

---

## 🧠 Model Architectures & Performance

We implemented and benchmarked three different deep learning architectures to find the optimal balance between speed and accuracy.

| Model | Architecture Type | Accuracy | Status |
| :--- | :--- | :--- | :--- |
| **Baseline CNN** | **Custom Conv2D (3 Blocks)** | **96.7%** | 🏆 **Deployed** |
| **MobileNetV2** | Transfer Learning (Frozen Base) | 93.5% | Good Alternative |
| **EfficientNetB0** | Transfer Learning | 51.0% | Convergence Issues |

### Why Baseline CNN?
Our custom CNN (3 Convolutional Blocks + MaxPool + BatchNorm + Dropout) outperformed the transfer learning models on this specific dataset. While MobileNetV2 was competitive, the lightweight custom CNN provided the best trade-off for real-time inference.

---

## 📂 Dataset Details

The model was trained on **Yashar Jebraeily’s Drowsy Detection Dataset** sourced from Kaggle.

* **Total Images:** ~7,300
    * *Training:* 5,859
    * *Testing:* 1,483
* **Classes:** `DROWSY` vs. `NATURAL` (Alert)
* **Preprocessing:**
    * Resizing: $64 \times 64$ (CNN) / $128 \times 128$ (Transfer Learning)
    * Normalization: Pixel values scaled to $[0, 1]$

---

## 🚀 Real-Time Pipeline

The live detection system follows this pipeline using OpenCV:

1.  **Capture:** Frame captured from the webcam.
2.  **Face Detection:** Haar Cascade Classifier identifies the face.
3.  **ROI Extraction:** The region of interest (ROI) is cropped.
4.  **Preprocessing:** ROI is resized and normalized to match model input.
5.  **Inference:** The trained model predicts the class probabilities.
6.  **Alerting:** If the "Drowsy" probability exceeds the threshold, a visual warning is displayed.

---

## 💻 Installation & Usage

### Prerequisites
Ensure you have Python installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/MridulPathania01/Driver-Drowsiness-Detections.git
cd drowsiness-detection
pip install tensorflow opencv-python matplotlib numpy
