
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import cv2
# Mediapipe'ı hata almayacak şekilde yükle
try:
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    
    # Model başlatma
    pose_tracker = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    st.success("AI Modülü Başarıyla Yüklendi")
except ImportError as e:
    st.error(f"Kütüphane yükleme hatası: {e}")
except AttributeError as e:
    st.error(f"Özellik hatası (AttributeError): {e}. Lütfen requirements.txt dosyasını kontrol edin.")
# Mediapipe'ı hata almayacak şekilde yükle
try:
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    
    # Model başlatma
    pose_tracker = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    st.success("AI Modülü Başarıyla Yüklendi")
except ImportError as e:
    st.error(f"Kütüphane yükleme hatası: {e}")
except AttributeError as e:
    st.error(f"Özellik hatası (AttributeError): {e}. Lütfen requirements.txt dosyasını kontrol edin.")

# Sayfa Konfigürasyonu
st.set_page_config(page_title="EVEYES 360 Dashboard", layout="wide")
st.title("🏥 EVEYES 360 - AI Agent Home-Hospital Tracking")

# Sidebar - Sistem Kontrolü
st.sidebar.title("System Control")
run_ai = st.sidebar.checkbox("Start AI Monitoring", value=True)
patient_id = st.sidebar.text_input("Patient ID", "P-104")

# Veri Füzyonu Simülasyonu (Vitals)
def get_vitals():
    return {
        "temp": round(36.5 + np.random.uniform(-0.5, 2.0), 1),
        "hr": np.random.randint(60, 110),
        "spo2": np.random.randint(94, 100)
    }

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Real-Time AI Analysis")
    frame_placeholder = st.empty()

with col2:
    st.subheader("🚨 Live Alerts & Vitals")
    vitals = get_vitals()
    st.metric("Body Temp", f"{vitals['temp']} °C", delta="Normal" if vitals['temp'] < 38 else "High")
    st.metric("Heart Rate", f"{vitals['hr']} BPM")
    
    alert_box = st.empty()
    st.write("---")
    st.subheader("📝 Activity Log")
    log_data = pd.DataFrame(columns=["Time", "Event", "Severity"])
    st.table(log_data)

# Video İşleme (Webcam)
if run_ai:
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # AI Analizi (Pose)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            nose_y = results.pose_landmarks.landmark[0].y
            if nose_y > 0.75:
                alert_box.error(f"CRITICAL: FALL DETECTED - {datetime.now().strftime('%H:%M:%S')}")
        
        frame_placeholder.image(frame, channels="BGR")
        if not run_ai: break
    cap.release()
