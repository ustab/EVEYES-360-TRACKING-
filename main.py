import cv2
import face_recognition
import numpy as np

# 1. ADIM: Bilinen Yüzlerin Sisteme Kaydedilmesi (Veri Tabanı Ön Hazırlığı)
# Burada örnek olarak bir doktorun fotoğrafını yüklüyoruz.
# 'doktor_foto.jpg' dosyasının projenizle aynı klasörde olduğundan emin olun.

try:
   doktor_image = face_recognition.load_image_file("doktor_foto.jpg")
   doktor_face_encoding = face_recognition.face_encodings(doktor_image)[0]
except IndexError:
    #print("HATA: Fotoğrafta yüz bulunamadı veya dosya yok.")

# Tanınan yüzlerin listesi ve isimleri
known_face_encodings = [doktor_face_encoding]
known_face_names = ["Dr. Ahmet Yilmaz"]

# 2. ADIM: Kamera Akışını Başlatma
video_capture = cv2.VideoCapture(0) # '0' varsayılan kameradır.

print("EVEYES 360 Sistemi Başlatıldı. Gözlem Yapılıyor...")

while True:
    # Kameradan tek bir kare (frame) yakala
    ret, frame = video_capture.read()
    
    # İşleme hızını artırmak için görüntüyü 1/4 oranında küçültelim
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # BGR (OpenCV formatı) görüntüyü RGB (face_recognition formatı) formatına çevir
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Kare içindeki tüm yüzleri ve kodlamalarını bul
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        # Mevcut yüzün 'bilinen yüzler' ile eşleşip eşleşmediğine bak
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Bilinmeyen Hasta" # Varsayılan değer

        # En yakın eşleşmeyi bul
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Sonucu ekranda göster (Burada analiz sonuçları Master Kontrol'e gidecek)
        print(f"Tespit Edilen Kişi: {name}")

    # Görüntü penceresini göster (Geliştirme aşaması için)
    cv2.imshow('EVEYES 360 - Kamera Akisi', frame)

    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sistemi kapat
video_capture.release()
cv2.destroyAllWindows()
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import os

def translate_and_speak(text, target_lang='tr'):
    """Metni hedef dile çevirir ve seslendirir."""
    # 1. Çeviri İşlemi
    translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
    print(f"Orijinal: {text}")
    print(f"Çeviri ({target_lang}): {translated}")
    
    # 2. Seslendirme (Opsiyonel - AI Agent Cevap Veriyor)
    # tts = gTTS(text=translated, lang=target_lang)
    # tts.save("response.mp3")
    # os.system("start response.mp3") # Windows için çalma komutu
    
    return translated

def listen_and_translate():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("\nEVEYES 360 Dinliyor... (Lütfen konuşun)")
        # Arka plan gürültüsünü ayarla
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        # Sesi önce orijinal dilinde tanı (Otomatik algılama için başlangıçta İngilizce/Global seçilebilir)
        original_text = recognizer.recognize_google(audio, language="en-US")
        
        # Tanınan metni Türkçe'ye (Doktorun dili) çevir
        translate_and_speak(original_text, target_lang='tr')
        
    except sr.UnknownValueError:
        print("EVEYES Sesi Anlayamadı.")
    except sr.RequestError:
        print("Servis Hatası: İnternet bağlantınızı kontrol edin.")

# Test için çalıştıralım
if __name__ == "__main__":
    listen_and_translate()

    import cv2
import mediapipe as mp
import math

# MediaPipe Pose (Poz) modülünü hazırla
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """Üç nokta arasındaki açıyı hesaplar (Örn: Diz veya Dirsek açısı)"""
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians*180.0/math.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

# Kamera akışını başlat (Adım 1'deki video_capture'ı kullanabiliriz)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü RGB'ye çevir (MediaPipe RGB bekler)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Görüntüyü tekrar BGR'ye çevir (Ekranda göstermek için)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # İskelet çizimini yap
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Örnek Analiz: Omuz ve Kalça noktalarını al
        landmarks = results.pose_landmarks.landmark
        
        # 1. DÜŞME TESPİTİ (Basit Mantık: Baş noktası kalça noktasının çok altına inerse)
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
        hip_y = landmarks[mp_pose.PoseLandmark.HIP_LEFT].y
        
        if nose_y > hip_y + 0.2: # Kişi yatay pozisyona yakınsa
            cv2.putText(image, "UYARI: DUSME TESPITI!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            print("Kritik Durum: Hasta yere düşmüş olabilir!")

        # 2. SALDIRI / KAVGA TESPİTİ (Hızlı kol hareketleri veya elin omuz hizasından yukarıda olması)
        wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        if wrist_y < shoulder_y: # El omuzdan yukarıdaysa (Saldırı veya yardım isteme pozu)
            cv2.putText(image, "AGRESIF HAREKET / YARDIM CAGRISI", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('EVEYES 360 - Postur ve Hareket Analizi', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import time
import random

class VitalMonitor:
    def __init__(self):
        # Normal değer aralıkları (Eşik değerler)
        self.thresholds = {
            "temp": {"min": 36.0, "max": 38.0},
            "heart_rate": {"min": 60, "max": 100},
            "spO2": {"min": 95, "max": 100}
        }

    def read_from_sensors(self):
        """
        Gerçek senaryoda burada seri porttan veya IP üzerinden 
        termal kamera/radar verisi okunur.
        Şimdilik sistemi test etmek için simülasyon verisi üretiyoruz.
        """
        # Sensörden gelen ham veri simülasyonu
        current_data = {
            "temp": round(random.uniform(36.2, 39.5), 1),
            "heart_rate": random.randint(55, 110),
            "spO2": random.randint(90, 100)
        }
        return current_data

    def analyze_vitals(self, data):
        """Verileri analiz eder ve kritik bir durum varsa alarm üretir."""
        alerts = []
        
        # Ateş Kontrolü
        if data["temp"] > self.thresholds["temp"]["max"]:
            alerts.append(f"YÜKSEK ATEŞ TESPİTİ: {data['temp']}°C")
        
        # Nabız Kontrolü
        if data["heart_rate"] > self.thresholds["heart_rate"]["max"]:
            alerts.append(f"TAŞİKARDİ RİSKİ: {data['heart_rate']} BPM")
        elif data["heart_rate"] < self.thresholds["heart_rate"]["min"]:
            alerts.append(f"BRADİKARDİ RİSKİ: {data['heart_rate']} BPM")

        # Oksijen Kontrolü
        if data["spO2"] < self.thresholds["spO2"]["min"]:
            alerts.append(f"DÜŞÜK OKSİJEN (SpO2): %{data['spO2']}")

        return alerts

# --- ANA DÖNGÜYE ENTEGRASYON TESTİ ---
monitor = VitalMonitor()

print("EVEYES 360 Vital Takip Modülü Aktif...")

try:
    while True:
        # 1. Veriyi Oku
        vitals = monitor.read_from_sensors()
        
        # 2. Analiz Et
        critical_alerts = monitor.analyze_vitals(vitals)
        
        # 3. Master Kontrol Paneline Yazdır
        print(f"\r[TAKİP] Ateş: {vitals['temp']}°C | Nabız: {vitals['heart_rate']} | SpO2: %{vitals['spO2']}", end="")
        
        if critical_alerts:
            print("\n" + "!"*30)
            for alert in critical_alerts:
                print(f"[KRİTİK UYARI]: {alert}")
            print("!"*30)
            # Burada Beyaz Kod veya Mavi Kod tetiklenebilir
        
        time.sleep(2) # 2 saniyede bir güncelle (Hastane standardı)

except KeyboardInterrupt:
    print("\nVital Takip Durduruldu.")

    import json
from datetime import datetime

class ReportManager:
    def __init__(self, filename="hastane_log.json"):
        self.filename = filename
        self.logs = []

    def log_event(self, patient_name, event_type, details, severity="Normal"):
        """Her olayı zaman damgasıyla kaydeder."""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient": patient_name,
            "type": event_type, # Vital, Hareket, Tercüme, Güvenlik
            "details": details,
            "severity": severity # Normal, Uyari, Kritik
        }
        self.logs.append(entry)
        self.save_to_file()

    def save_to_file(self):
        """Verileri kalıcı olarak dosyaya yazar."""
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=4)

    def generate_doctor_report(self, patient_name):
        """Doktor için gün sonu veya anlık özet raporu oluşturur."""
        patient_logs = [log for log in self.logs if log["patient"] == patient_name]
        
        print(f"\n--- DR. RAPORU: {patient_name} ---")
        print(f"Rapor Tarihi: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        
        # Kritik olayları filtrele
        criticals = [l for l in patient_logs if l["severity"] == "Kritik"]
        
        if criticals:
            print(f"\n[!] DİKKAT: Son 24 saatte {len(criticals)} kritik olay saptandı.")
            for c in criticals:
                print(f"- {c['timestamp']}: {c['details']}")
        else:
            print("\n[+] Hasta stabil. Kritik anomali saptanmadı.")
        
        # Hareket ve Postür Özeti
        movements = [l for l in patient_logs if l["type"] == "Hareket"]
        if movements:
            print(f"- Hareket Analizi: {len(movements)} adet postür değişikliği kaydedildi.")

# --- SİSTEM ENTEGRASYON TESTİ ---
report_system = ReportManager()

# ÖRNEK SENARYO: Sistem çalışırken verileri kaydeder
# 1. Adım: Yüz Tanındı
patient = "Ali Veli"

# 2. Adım: Vital Ölçüldü (Kritik bir durum oluştu)
report_system.log_event(patient, "Vital", "Nabız: 115 (Taşikardi Saptandı)", "Kritik")

# 3. Adım: Postür Analizi
report_system.log_event(patient, "Hareket", "Yataktan kalkış denemesi - Dengeli", "Normal")

# 4. Adım: Tercüme Yapıldı
report_system.log_event(patient, "Tercüme", "Hasta 'Göğsümde baskı var' dedi (Rusça -> Türkçe)", "Uyari")

# 5. Adım: Doktor Raporu İstendi
report_system.generate_doctor_report("Ali Veli")


import threading
import time

class EveyesSecuritySystem:
    def __init__(self):
        self.emergency_mode = False
        self.lock = threading.Lock()

    def camera_security_monitor(self):
        """Kamera verisinden kavga/saldırı analizi yapar (3. Adım Entegrasyonu)"""
        while not self.emergency_mode:
            # Burada AI modelinden gelen 'saldırı_olasiligi' verisi işlenir
            # Örnek simülasyon:
            aggression_score = random.randint(0, 100)
            if aggression_score > 90:
                self.trigger_alarm("GÜVENLİK: Fiziksel Saldırı/Kavga Saptandı!", "BEYAZ KOD")
            time.sleep(1)

    def audio_security_monitor(self):
        """Ses verisinden bağırma/cam kırılması analizi yapar (2. Adım Entegrasyonu)"""
        while not self.emergency_mode:
            # Burada 'imdat', 'help' veya yüksek desibel analizi yapılır
            # Örnek simülasyon:
            noise_level = random.randint(30, 110)
            if noise_level > 100:
                self.trigger_alarm("SES: Yüksek Desibel / Yardım Çığlığı Saptandı!", "BEYAZ KOD")
            time.sleep(1)

    def fire_thermal_monitor(self):
        """Termal kameradan yangın/duman analizi yapar (4. Adım Entegrasyonu)"""
        while not self.emergency_mode:
            # Termal ısı eşiği kontrolü
            temp_spot = random.uniform(20.0, 100.0)
            if temp_spot > 80.0:
                self.trigger_alarm("YANGIN: Kritik Isı Artışı Saptandı!", "KIRMIZI KOD")
            time.sleep(1)

    def trigger_alarm(self, reason, code):
        """Tüm sistemi kriz moduna sokar ve Master Kontrol'e bildiri gönderir."""
        with self.lock:
            if not self.emergency_mode:
                self.emergency_mode = True
                print(f"\n\n{'!'*50}")
                print(f"!!! ACİL DURUM AKTİVE EDİLDİ !!!")
                print(f"NEDEN: {reason}")
                print(f"SİSTEM KODU: {code}")
                print(f"AKSİYON: Güvenlik birimleri yönlendirildi. Tahliye rotaları açıldı.")
                print(f"{'!'*50}\n")
                
                # Burada donanımsal tetiklemeler yapılabilir:
                # smart_locks.unlock_all()
                # sirens.start()

# --- SİSTEMİ EŞ ZAMANLI ÇALIŞTIRALIM ---
security = EveyesSecuritySystem()

# Her sensör analizini ayrı bir 'iş parçacığı' (thread) olarak başlatıyoruz
t1 = threading.Thread(target=security.camera_security_monitor)
t2 = threading.Thread(target=security.audio_security_monitor)
t3 = threading.Thread(target=security.fire_thermal_monitor)

print("EVEYES 360 Master Kontrol: Güvenlik Katmanı Başlatıldı...")

t1.start()
t2.start()
t3.start()

# Ana programın kapanmaması için bekletiyoruz
t1.join()

from fastapi import FastAPI
import threading
import random

app = FastAPI(title="EVEYES 360 Master Control API")

# Global sistem durumu (Modüllerden gelen veriler burada toplanacak)
system_status = {
    "security": {"status": "GÜVENLİ", "last_alert": "Yok", "code": "YEŞİL"},
    "patients": [
        {"id": 1, "name": "Ali Veli", "temp": 36.5, "hr": 75, "status": "Stabil", "lang": "TR"},
        {"id": 2, "name": "Hans Müller", "temp": 38.2, "hr": 95, "status": "Gözlem Altında", "lang": "DE"}
    ],
    "environment": {"fire_risk": "Düşük", "noise_level": "Normal"}
}

# --- ENDPOINT'LER (Arayüzün veri çekeceği kapılar) ---

@app.get("/")
def read_root():
    return {"message": "EVEYES 360 Master Control Merkezi Aktif"}

@app.get("/dashboard")
def get_dashboard_data():
    """Tüm hastane durumunu tek seferde döner."""
    return system_status

@app.get("/patient/{patient_id}")
def get_patient_detail(patient_id: int):
    """Belirli bir hastanın AI-Agent analizlerini getirir."""
    patient = next((p for p in system_status["patients"] if p["id"] == patient_id), None)
    if patient:
        return {
            "patient_info": patient,
            "ai_insights": [
                "Postür Analizi: Normal",
                "Tercüme İhtiyacı: Düşük",
                "Tahmini Taburcu: 2 Gün"
            ]
        }
    return {"error": "Hasta bulunamadı"}

@app.post("/trigger-alarm")
def manual_alarm(code: str, reason: str):
    """Güvenlik amirinin manuel alarm başlatmasını sağlar."""
    system_status["security"]["status"] = "ACİL DURUM"
    system_status["security"]["code"] = code
    system_status["security"]["last_alert"] = reason
    return {"status": "Alarm Aktive Edildi", "code": code}

# --- SİSTEMİ BAŞLATMA ---
if __name__ == "__main__":
    import uvicorn
    # Bu komutla API sunucusu 8000 portunda başlar
    uvicorn.run(app, host="0.0.0.0", port=8000)


    import cv2
import threading
from queue import Queue

class SmartCameraStream:
    def __init__(self, rtsp_url):
        # NVIDIA Jetson için donanım hızlandırmalı GStreamer pipeline'ı
        # Bu satır, CPU'yu yormadan GPU üzerinden görüntüyü çözer
        self.pipeline = (
            f"rtspsrc location={rtsp_url} latency=0 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink"
        )
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        self.q = Queue()
        self.stopped = False

    def _reader(self):
        """Kameradan gelen kareleri sürekli oku ve kuyruğa at (Frame Buffer)"""
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait() # Eski kareyi at (Gerçek zamanlılık için)
                except:
                    pass
            self.q.put(frame)

    def get_frame(self):
        return self.q.get()

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- ENTEGRASYON ---
# Örn: Acil Servis Kamerası
camera_1 = SmartCameraStream("rtsp://admin:password@192.168.1.50:554/stream")
t = threading.Thread(target=camera_1._reader)
t.start()

while True:
    frame = camera_1.get_frame()
    
    # Burada daha önce yazdığımız AI modüllerini çağırıyoruz:
    # 1. Yüzü bul -> recognize_faces(frame)
    # 2. İskeleti çıkar -> detect_pose(frame)
    # 3. Şiddet var mı? -> analyze_aggression(frame)
    
    cv2.imshow("EVEYES 360 Live - Emergency Ward", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera_1.stop()
import multiprocessing as mp
import time

def thermal_sensor_worker(data_queue):
    """Termal sensörden bağımsız bir çekirdekte veri okur."""
    while True:
        # Gerçek sensör okuma kodu burada olacak
        simulated_temp = 36.6 + (time.time() % 2) 
        data_queue.put({"type": "thermal", "val": simulated_temp})
        time.sleep(0.125) # 8 Hz

def camera_ai_worker(data_queue):
    """NVIDIA TensorRT kullanarak GPU üzerinden görüntü işler."""
    while True:
        # AI Analiz sonuçları
        # Örn: detect_fall() -> True
        data_queue.put({"type": "vision", "alert": "Fall Detected"})
        time.sleep(0.033) # 30 FPS

if __name__ == "__main__":
    shared_queue = mp.Queue()
    
    # Süreçleri başlat
    p1 = mp.Process(target=thermal_sensor_worker, args=(shared_queue,))
    p2 = mp.Process(target=camera_ai_worker, args=(shared_queue,))
    
    p1.start()
    p2.start()

    # MASTER ANALİZÖR: Gelen verileri birleştirir (Data Fusion)
    while True:
        if not shared_queue.empty():
            data = shared_queue.get()
            if data['type'] == 'thermal' and data['val'] > 39.0:
                print(f"KRİTİK: Hastanın ateşi yükseldi! ({data['val']}°C)")
            if data['type'] == 'vision' and data['alert'] == 'Fall Detected':
                print("ACİL: Düşme algılandı! Kameralar odaya odaklanıyor.")


# Farklı sensörlerden gelen veriyi 'Timestamp' (Zaman Damgası) ile eşleştirme
def sync_sensor_data(vision_data, vital_data):
    # Eğer vizyon düşme diyorsa VE nabız aniden yükselmişse (STRESS DETECTED)
    if vision_data['action'] == "fallen" and vital_data['heart_rate'] > 110:
        return "CRITICAL ALERT: FALL + HIGH DISTRESS"
    return "MONITORING"
# Modeli Jetson üzerinde GPU ile çalıştırmak için örnek motor başlatma
import tensorrt as trt

def load_engine(engine_file_path):
    # Bu fonksiyon, AI modelini donanımın mimarisine göre optimize eder
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# AI işlemlerini 'Asenkron' (Async) yaparak kameranın donmasını engelliyoruz
