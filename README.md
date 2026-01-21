# EVEYES 360 TRACKING 🏥ENG
**Advanced AI-Integrated Hospital Safety & Patient Monitoring Ecosystem**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker: Ready](https://img.shields.io/badge/Docker-Ready-brightgreen.svg)](https://www.docker.com/)

## 🌟 Overview
EVEYES 360 TRACKING is a comprehensive, AI-driven digital twin and monitoring system designed for modern healthcare facilities. It merges computer vision, NLP-based translation, and vital signal processing into a unified "Master Control" interface to ensure patient safety, staff security, and operational efficiency.

## 🚀 Core Features

### 🛡️ 1. Intelligent Security (Staff Protection)
- **Aggression Detection:** Real-time analysis of physical movements and audio decibels to detect fights or assaults.
- **Automated Alerts:** Instant triggering of "Code White" (Security Emergency) via the Master Control panel.

### 🌡️ 2. Clinical Monitoring (Patient Safety)
- **Contactless Vitals:** Monitoring of Heart Rate, Body Temperature, and SpO2 using mmWave radar and thermal imaging simulations.
- **Fall Detection:** Pose estimation (MediaPipe) to detect falls or collapse within <1.5 seconds.
- **Postural Analysis:** Continuous tracking of patient mobility and bed-exit attempts.

### 🌍 3. Universal Communication (Medical Translator)
- **Real-time Translation:** Instant STT (Speech-to-Text) and translation in 15+ languages to eliminate the language barrier between doctors and foreign patients.
- **Medical Terminology Support:** Optimized for ICD-10 and clinical vocabulary.

### 🚨 4. Emergency Management
- **Smart Evacuation:** Dynamic route planning based on real-time fire/smoke detection.
- **Hospital Code Integration:** Full support for Code Blue, Code Red, and Code White protocols.

## 🛠️ Tech Stack
- **Backend:** Python 3.9, FastAPI
- **AI/ML:** OpenCV, MediaPipe, Face-Recognition
- **NLP:** SpeechRecognition, Deep-Translator
- **DevOps:** Docker, Uvicorn
- **Architecture:** Multithreaded Edge-Computing approach

## 📦 Installation & Deployment

### Prerequisites
- Docker (Recommended) or Python 3.9+
- Webcam (for Vision modules)
- Microphone (for Translation modules)

### Using Docker
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/eveyes360.git](https://github.com/yourusername/eveyes360.git)
   cd eveyes360

# EVEYES 360 TRACKING 🏥 TR
AI-Powered Hospital Safety & Patient Monitoring Ecosystem

## Özellikler
- 🛡️ **Güvenlik:** Kavga ve Saldırı Tespiti (MediaPipe).
- 🌡️ **Klinik:** Temassız Vital Takip (Nabız/Ateş Simülasyonu).
- 🌍 **İletişim:** 15+ Dilde Anlık Tıbbi Tercüman.
- 🚨 **Kriz Yönetimi:** Beyaz/Kırmızı/Mavi Kod Entegrasyonu.

## Kurulum
1. `docker build -t eveyes360 .`
2. `docker run -p 8000:8000 eveyes360`
