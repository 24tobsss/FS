import face_recognition
import cv2
import os
import numpy as np
import threading
import time
import pygame
import sqlite3
import pickle
import json
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from collections import defaultdict
import psutil
from pathlib import Path
import requests
from io import BytesIO
import logging
from logging.handlers import RotatingFileHandler
import platform
import argparse
import RPi.GPIO as GPIO
import signal
import sys
from twilio.rest import Client

# -----------------------
# GPIO Setup - ANGEPASSTE PINS
# -----------------------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Buzzer Pins - ANGEPASST
BUZZER_ACTIVE = 7    # GPIO 7 (Pin 26)
BUZZER_PASSIVE = 25  # GPIO 25 (Pin 22)

# RGB LED Pins 
LED_RED = 17    # GPIO 17 (Pin 11)
LED_GREEN = 27  # GPIO 27 (Pin 13) 
LED_BLUE = 22   # GPIO 22 (Pin 15)

# Servo Pins - ANGEPASST
SERVO_HORIZONTAL = 6  # GPIO 6 (Pin 31) - Für horizontale Bewegung

# Setup
GPIO.setup(BUZZER_ACTIVE, GPIO.OUT)
GPIO.setup(BUZZER_PASSIVE, GPIO.OUT)
GPIO.setup(LED_RED, GPIO.OUT)
GPIO.setup(LED_GREEN, GPIO.OUT)
GPIO.setup(LED_BLUE, GPIO.OUT)
GPIO.setup(SERVO_HORIZONTAL, GPIO.OUT)

# Initialzustand
GPIO.output(BUZZER_ACTIVE, GPIO.LOW)
GPIO.output(BUZZER_PASSIVE, GPIO.LOW)
GPIO.output(LED_RED, GPIO.LOW)
GPIO.output(LED_GREEN, GPIO.LOW)
GPIO.output(LED_BLUE, GPIO.LOW)

# Servo PWM initialisieren
servo_horizontal = GPIO.PWM(SERVO_HORIZONTAL, 50)  # 50Hz PWM
servo_horizontal.start(7.5)  # Neutralposition (90°)
time.sleep(0.5)
servo_horizontal.ChangeDutyCycle(0)  # PWM ausschalten

# OLED (vereinfacht)
OLED_AVAILABLE = False

# -----------------------
# Konfiguration
# -----------------------
CONFIG_FILE = "config.json"
KNOWN_FACES_DIR = "faces"
UNKNOWN_FACES_DIR = "unknown_faces"
DB_FILE = "face_db.sqlite"
LOG_FILE = "face_recognition.log"
TEMP_DIR = "temp"

# Standardkonfiguration
DEFAULT_CONFIG = {
    "camera_index": "auto",
    "model": "hog",
    "tolerance": 0.6,  # Höhere Toleranz für Fernerkennung
    "alert_duration": 3,
    "frame_skip": 3,  # Weniger Frames für bessere Performance
    "min_face_size": 50,  # Kleinere Gesichter erkennen (für Ferne)
    "email_alerts": False,
    "sms_alerts": False,
    "whatsapp_alerts": False,
    "alarm_sound": True,
    "resolution": [320, 240],  # Niedrigere Auflösung
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_username": "",
    "smtp_password": "",
    "admin_email": "",
    "twilio_sid": "",
    "twilio_token": "",
    "twilio_whatsapp_from": "",
    "twilio_whatsapp_to": "",
    "flask_enabled": False,
    "flask_port": 5000,
    "flask_host": "0.0.0.0",
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "auto_backup": True,
    "backup_interval_hours": 24,
    "privacy_mode": False,
    "data_retention_days": 30,
    "language": "de",
    "sound_effects": True,
    "debug_mode": False,
    "servo_tracking": True,
    "servo_speed": 2,
    "buzzer_type": "active",
    "oled_display": False,
    "servo_horizontal_enabled": True,
    "enhanced_distance_detection": True,  # Neue Option für Fernerkennung
    "upsample_factor": 2  # Für bessere Fernerkennung
}

# Ordner erstellen
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# -----------------------
# Logging einrichten
# -----------------------
def setup_logging():
    logger = logging.getLogger('FaceRecognition')
    logger.setLevel(logging.DEBUG)

    # Log-Datei (max. 5MB, 3 Backup-Dateien)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setLevel(logging.DEBUG)

    # Console-Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

# -----------------------
# Signal Handler für graceful shutdown
# -----------------------
def signal_handler(sig, frame):
    """Behandelt Ctrl+C graceful"""
    logger.info("Ctrl+C empfangen, beende Anwendung...")
    global running
    running = False
    time.sleep(1)
    # Aufräumen
    if 'cap' in globals() and cap is not None:
        cap.release()
    if 'servo_horizontal' in globals():
        servo_horizontal.stop()
    GPIO.cleanup()
    if 'pygame' in sys.modules:
        pygame.quit()
    sys.exit(0)

# -----------------------
# Hardware Funktionen
# -----------------------
def set_rgb_color(red, green, blue):
    """Setzt die RGB LED Farbe"""
    GPIO.output(LED_RED, red)
    GPIO.output(LED_GREEN, green)
    GPIO.output(LED_BLUE, blue)

def buzzer_beep(duration=0.1, frequency=1000):
    """Aktiviert den Buzzer"""
    try:
        if config.get("buzzer_type") == "passive":
            # Passive Buzzer mit PWM
            pwm = GPIO.PWM(BUZZER_PASSIVE, frequency)
            pwm.start(50)
            time.sleep(duration)
            pwm.stop()
        else:
            # Aktiver Buzzer
            GPIO.output(BUZZER_ACTIVE, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(BUZZER_ACTIVE, GPIO.LOW)
    except Exception as e:
        logger.error(f"Buzzer Fehler: {e}")

def set_servo_angle(servo_pwm, angle):
    """Setzt den Servo auf einen Winkel (0-180°)"""
    try:
        duty = angle / 18 + 2
        servo_pwm.ChangeDutyCycle(duty)
        time.sleep(0.1)
        servo_pwm.ChangeDutyCycle(0)
    except Exception as e:
        logger.error(f"Servo Fehler: {e}")

def track_face_with_servo(face_x, face_y, frame_width, frame_height):
    """Verfolgt Gesichter mit Servo für 360° Bewegung"""
    if not config.get("servo_tracking", True):
        return
        
    try:
        # Horizontale Bewegung (360°)
        if config.get("servo_horizontal_enabled", True):
            center_x = face_x + (frame_width / 2)
            angle_horizontal = (center_x / frame_width) * 180
            angle_horizontal = max(0, min(180, angle_horizontal))
            set_servo_angle(servo_horizontal, angle_horizontal)
            
    except Exception as e:
        logger.error(f"Servo Tracking Fehler: {e}")

def update_oled_display(text_lines):
    """Einfache Konsolenausgabe für OLED"""
    if config.get("debug_mode", False):
        print("=== OLED DISPLAY ===")
        for i, line in enumerate(text_lines):
            print(f"{i+1}. {line}")
        print("====================")

def alert_sequence():
    """Aktiviert alle Alarm-Komponenten"""
    # RGB LED rot blinken
    for _ in range(5):
        set_rgb_color(1, 0, 0)  # Rot
        time.sleep(0.2)
        set_rgb_color(0, 0, 0)  # Aus
        time.sleep(0.2)
    
    # Buzzer aktivieren
    buzzer_beep(0.5, 1000)
    
    # OLED Alarm anzeigen
    update_oled_display(["ALARM!", "Unbekannte", "Person erkannt!"])

def welcome_sequence(name):
    """Begrüßungs-Sequenz für bekannte Personen"""
    # RGB LED grün
    set_rgb_color(0, 1, 0)
    
    # Kurzer freundlicher Ton
    buzzer_beep(0.1, 800)
    
    # OLED Willkommensnachricht
    update_oled_display(["Willkommen", name, ""])
    
    # Nach 2 Sekunden ausschalten
    time.sleep(2)
    set_rgb_color(0, 0, 0)

# -----------------------
# Messaging Funktionen
# -----------------------
def send_gmail_alert(image_path, timestamp, config):
    """Sendet Gmail-Benachrichtigung bei unbekannten Gesichtern"""
    if not config["email_alerts"] or not config["smtp_username"]:
        return

    try:
        msg = MIMEMultipart()
        msg['Subject'] = '🔴 Sicherheitsalarm: Unbekannte Person erkannt'
        msg['From'] = config["smtp_username"]
        msg['To'] = config["admin_email"]

        # Textteil
        body = f"""🚨 SICHERHEITSALARM 🚨

Unbekannte Person erkannt um: {time.ctime(timestamp)}
Ort: Dein Zimmer
System: Face Recognition Security

Bitte überprüfe die Anlage für Details.
"""
        msg.attach(MIMEText(body, 'plain'))

        # Bild anhängen
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img_data = f.read()
            image = MIMEImage(img_data, name=os.path.basename(image_path))
            msg.attach(image)

        # E-Mail senden
        with smtplib.SMTP(config["smtp_server"], config["smtp_port"]) as server:
            server.starttls()
            server.login(config["smtp_username"], config["smtp_password"])
            server.send_message(msg)
        
        logger.info("Gmail-Alarm gesendet")
        
    except Exception as e:
        logger.error(f"Gmail konnte nicht gesendet werden: {e}")

def send_whatsapp_alert(image_path, timestamp, config):
    """Sendet WhatsApp-Benachrichtigung über Twilio"""
    if not config["whatsapp_alerts"] or not config["twilio_sid"]:
        return

    try:
        client = Client(config["twilio_sid"], config["twilio_token"])
        
        # Nachricht erstellen
        message = f"""🚨 *SICHERHEITSALARM* 🚨

Unbekannte Person erkannt!
• Zeit: {time.ctime(timestamp)}
• Ort: Dein Zimmer
• System: Face Recognition

Bitte sofort überprüfen!"""

        # WhatsApp Nachricht senden
        if config["twilio_whatsapp_from"] and config["twilio_whatsapp_to"]:
            client.messages.create(
                body=message,
                from_=f"whatsapp:{config['twilio_whatsapp_from']}",
                to=f"whatsapp:{config['twilio_whatsapp_to']}"
            )
            logger.info("WhatsApp-Alarm gesendet")
            
        # Bild per MMS falls gewünscht (Twilio unterstützt auch MMS)
        if os.path.exists(image_path):
            try:
                with open(image_path, 'rb') as img_file:
                    img_data = img_file.read()
                
                # Hier könntest du das Bild auch per Twilio MMS senden
                # (Twilio MMS erfordert US-Nummern, daher alternative Lösung)
                pass
                
            except Exception as img_error:
                logger.error(f"Bild konnte nicht für WhatsApp vorbereitet werden: {img_error}")
                
    except Exception as e:
        logger.error(f"WhatsApp-Nachricht konnte nicht gesendet werden: {e}")

def send_telegram_alert(image_path, message, config):
    """Sendet eine Telegram-Benachrichtigung"""
    if not config["telegram_bot_token"] or not config["telegram_chat_id"]:
        return

    try:
        url = f"https://api.telegram.org/bot{config['telegram_bot_token']}/sendPhoto"

        with open(image_path, 'rb') as photo:
            files = {'photo': photo}
            data = {
                'chat_id': config['telegram_chat_id'], 
                'caption': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            logger.info("Telegram-Alarm gesendet")
        else:
            logger.error(f"Telegram-API-Fehler: {response.status_code}")
    except Exception as e:
        logger.error(f"Telegram-Nachricht konnte nicht gesendet werden: {e}")

# -----------------------
# Erweiterte Gesichtserkennung für Ferne
# -----------------------
def enhance_detection_for_distance(frame, config):
    """Verbessert die Erkennung für weitere Entfernungen"""
    enhanced_frame = frame.copy()
    
    if config.get("enhanced_distance_detection", True):
        # Kontrastverbesserung
        alpha = 1.5  # Kontrast (1.0-3.0)
        beta = 0     # Helligkeit (0-100)
        enhanced_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        # Schärfen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced_frame = cv2.filter2D(enhanced_frame, -1, kernel)
        
        # Rauschreduzierung
        enhanced_frame = cv2.medianBlur(enhanced_frame, 3)
    
    return enhanced_frame

def detect_faces_at_distance(rgb_frame, config):
    """Erkennt Gesichter in weiteren Entfernungen"""
    # Upsampling für bessere Fernerkennung
    upsample_factor = config.get("upsample_factor", 2)
    
    # Gesichter erkennen mit optimierten Parametern
    face_locations = face_recognition.face_locations(
        rgb_frame, 
        model=config["model"], 
        number_of_times_to_upsample=upsample_factor
    )
    
    # Gesichter encodieren
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    return face_locations, face_encodings

# -----------------------
# Datenbank-Hilfsfunktionen
# -----------------------
def get_db_connection():
    """Erstellt eine neue Datenbankverbindung für den aktuellen Thread"""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    return conn

def init_database():
    """Initialisiert die SQLite-Datenbank"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS known_faces
                 (id INTEGER PRIMARY KEY, name TEXT, encoding BLOB, created_date TEXT, updated_date TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS unknown_faces
                 (id INTEGER PRIMARY KEY, timestamp INTEGER, encoding BLOB, image_path TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS access_log
                 (id INTEGER PRIMARY KEY, timestamp TEXT, name TEXT, status TEXT,
                 confidence REAL, face_id INTEGER, distance_estimate REAL)''')  # NEU: Entfernungsschätzung
    c.execute('''CREATE TABLE IF NOT EXISTS system_metrics
                 (id INTEGER PRIMARY KEY, timestamp TEXT, cpu REAL, memory REAL, disk REAL, fps REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY, timestamp TEXT, type TEXT, message TEXT,
                 image_path TEXT, resolved INTEGER)''')

    # Indizes für bessere Performance
    c.execute('''CREATE INDEX IF NOT EXISTS idx_access_log_timestamp ON access_log(timestamp)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_unknown_faces_timestamp ON unknown_faces(timestamp)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_known_faces_name ON known_faces(name)''')

    conn.commit()
    conn.close()
    logger.info("Datenbank initialisiert")

def save_known_face_db(name, encoding):
    """Speichert ein bekanntes Gesicht in der Datenbank"""
    conn = get_db_connection()
    c = conn.cursor()
    now = datetime.now().isoformat()

    # Prüfen, ob Gesicht bereits existiert
    c.execute("SELECT id FROM known_faces WHERE name = ?", (name,))
    existing = c.fetchone()

    if existing:
        c.execute("UPDATE known_faces SET encoding = ?, updated_date = ? WHERE name = ?",
                  (sqlite3.Binary(pickle.dumps(encoding)), now, name))
        logger.info(f"Gesicht von {name} in Datenbank aktualisiert")
    else:
        c.execute("INSERT INTO known_faces (name, encoding, created_date, updated_date) VALUES (?, ?, ?, ?)",
                  (name, sqlite3.Binary(pickle.dumps(encoding)), now, now))
        logger.info(f"Gesicht von {name} in Datenbank gespeichert")

    conn.commit()
    conn.close()

def load_known_faces_db():
    """Lädt bekannte Gesichter aus der Datenbank"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT name, encoding FROM known_faces")
    names, encodings = [], []
    for name, encoding_blob in c.fetchall():
        names.append(name)
        encodings.append(pickle.loads(encoding_blob))

    conn.close()
    logger.info(f"{len(names)} bekannte Gesichter aus Datenbank geladen")
    return encodings, names

def log_access_db(name, status, confidence=0.0, face_id=-1, distance_estimate=0.0):
    """Protokolliert Zugriffe in der Datenbank mit Entfernungsschätzung"""
    conn = get_db_connection()
    timestamp = datetime.now().isoformat()
    c = conn.cursor()
    c.execute("INSERT INTO access_log (timestamp, name, status, confidence, face_id, distance_estimate) VALUES (?, ?, ?, ?, ?, ?)",
              (timestamp, name, status, confidence, face_id, distance_estimate))
    conn.commit()
    conn.close()

def log_alert_db(alert_type, message, image_path=None):
    """Protokolliert Alarme in der Datenbank"""
    conn = get_db_connection()
    timestamp = datetime.now().isoformat()
    c = conn.cursor()
    c.execute("INSERT INTO alerts (timestamp, type, message, image_path, resolved) VALUES (?, ?, ?, ?, ?)",
              (timestamp, alert_type, message, image_path, 0))
    conn.commit()
    conn.close()
    logger.warning(f"Alarm protokolliert: {alert_type} - {message}")

def log_system_metrics_db(cpu, memory, disk, fps=None):
    """Protokolliert Systemmetriken in der Datenbank"""
    conn = get_db_connection()
    timestamp = datetime.now().isoformat()
    c = conn.cursor()
    c.execute("INSERT INTO system_metrics (timestamp, cpu, memory, disk, fps) VALUES (?, ?, ?, ?, ?)",
              (timestamp, cpu, memory, disk, fps))
    conn.commit()
    conn.close()

def cleanup_old_data(config):
    """Bereinigt alte Daten basierend auf Aufbewahrungsrichtlinien"""
    if config["data_retention_days"] <= 0:
        return

    conn = get_db_connection()
    cutoff_date = (datetime.now() - timedelta(days=config["data_retention_days"])).isoformat()
    c = conn.cursor()

    # Alte Zugriffslogs löschen
    c.execute("DELETE FROM access_log WHERE timestamp < ?", (cutoff_date,))
    access_deleted = c.rowcount

    # Alte Systemmetriken löschen
    c.execute("DELETE FROM system_metrics WHERE timestamp < ?", (cutoff_date,))
    metrics_deleted = c.rowcount

    # Alte aufgelöste Alarme löschen
    c.execute("DELETE FROM alerts WHERE timestamp < ? AND resolved = 1", (cutoff_date,))
    alerts_deleted = c.rowcount

    conn.commit()
    conn.close()

    if access_deleted > 0 or metrics_deleted > 0 or alerts_deleted > 0:
        logger.info(f"Bereinigung: {access_deleted} Zugriffslogs, {metrics_deleted} Metriken, {alerts_deleted} Alarme gelöscht")

# -----------------------
# Weitere Hilfsfunktionen
# -----------------------
def load_config():
    """Lädt die Konfiguration aus der JSON-Datei"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            logger.info("Konfiguration geladen")
            return {**DEFAULT_CONFIG, **config}
    except FileNotFoundError:
        logger.info("Konfigurationsdatei nicht gefunden, verwende Standardwerte")
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG
    except json.JSONDecodeError:
        logger.error("Konfigurationsdatei ist beschädigt, verwende Standardwerte")
        return DEFAULT_CONFIG

def find_camera():
    """Findet automatisch die Kamera"""
    # Zuerst USB-Kameras probieren
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            ret, _ = cap.read()
            if ret:
                logger.info(f"Kamera gefunden: Index {i}")
                cap.release()
                return i
            cap.release()

    # Fallback: Versuch mit OpenCV Default
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        logger.info("Verwende Standardkamera (Index 0)")
        cap.release()
        return 0

    raise RuntimeError("Keine Kamera gefunden!")

def estimate_distance(face_height_pixels, frame_height, known_face_height_meters=0.25):
    """Schätzt die Entfernung basierend auf der Gesichtsgröße"""
    # Annahme: Durchschnittliche Gesichtshöhe = 0.25m
    if face_height_pixels <= 0:
        return 0.0
    
    # Einfache Entfernungsschätzung: distance = (known_height * focal_length) / pixel_height
    # Vereinfacht: distance ≈ (known_height * frame_height) / (2 * face_height_pixels * tan(0.5 * FOV))
    # Für typische Webcam: FOV ≈ 60°, also vereinfachte Formel:
    focal_length_approx = frame_height / (2 * np.tan(np.radians(30)))
    distance = (known_face_height_meters * focal_length_approx) / face_height_pixels
    
    return round(distance, 2)

def apply_privacy_blur(frame, faces):
    """Wendet Privacy-Blur auf Gesichter an"""
    blurred_frame = frame.copy()
    for (top, right, bottom, left, name, known, face_id, confidence, distance) in faces:
        face_roi = blurred_frame[top:bottom, left:right]
        if face_roi.size > 0:
            face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
            blurred_frame[top:bottom, left:right] = face_roi
    return blurred_frame

def speak_text(text, language="de"):
    """Spricht Text laut vor (Text-to-Speech)"""
    try:
        if language == "de":
            os.system(f"espeak -v german '{text}' 2>/dev/null")
        else:
            os.system(f"espeak '{text}' 2>/dev/null")
    except:
        logger.warning("Text-to-Speech nicht verfügbar")

# -----------------------
# Gesichtsverwaltung-Funktionen
# -----------------------
def rename_face(face_id, old_name, new_name):
    """Benennt ein Gesicht um"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE known_faces SET name = ?, updated_date = ? WHERE id = ?",
              (new_name, datetime.now().isoformat(), face_id))
    conn.commit()
    conn.close()
    logger.info(f"Gesicht umbenannt: {old_name} -> {new_name}")

def delete_face(face_id):
    """Löscht ein Gesicht aus der Datenbank"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM known_faces WHERE id = ?", (face_id,))
    conn.commit()
    conn.close()
    logger.info(f"Gesicht mit ID {face_id} gelöscht")

def capture_new_face():
    """Erfasst ein neues Gesicht von der Kamera"""
    global cap, screen, font, screen_width, screen_height, config
    
    try:
        # Display vorbereiten
        screen.fill((0, 0, 0))
        title_text = font.render("NEUES GESICHT ERFASSEN", True, (255, 255, 0))
        screen.blit(title_text, (screen_width // 2 - 120, 20))

        info_text = font.render("Bitte in die Kamera schauen...", True, (255, 255, 255))
        screen.blit(info_text, (screen_width // 2 - 100, 60))
        pygame.display.update()

        # Kurze Verzögerung für Vorbereitung
        time.sleep(2)

        # Frame von der Kamera erfassen
        ret, frame = cap.read()

        if not ret:
            error_text = font.render("Kamera-Fehler!", True, (255, 0, 0))
            screen.blit(error_text, (screen_width // 2 - 60, 100))
            pygame.display.update()
            time.sleep(2)
            return

        # Bild für Fernerkennung optimieren
        enhanced_frame = enhance_detection_for_distance(frame, config)
        small_frame = cv2.resize(enhanced_frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Gesichtserkennung mit optimierter Fernerkennung
        face_locations, face_encodings = detect_faces_at_distance(rgb_small_frame, config)

        if not face_encodings:
            error_text = font.render("Kein Gesicht erkannt!", True, (255, 0, 0))
            screen.blit(error_text, (screen_width // 2 - 80, 100))
            pygame.display.update()
            time.sleep(2)
            return

        # Namen eingeben lassen
        name = text_input("Name eingeben:", "")
        if not name:
            return

        # Gesicht speichern
        save_known_face_db(name, face_encodings[0])
        
        success_text = font.render(f"Gesicht '{name}' gespeichert!", True, (0, 255, 0))
        screen.blit(success_text, (screen_width // 2 - 100, 100))
        pygame.display.update()
        time.sleep(2)
        
    except Exception as e:
        error_text = font.render(f"Fehler: {str(e)}", True, (255, 0, 0))
        screen.blit(error_text, (screen_width // 2 - 100, 100))
        pygame.display.update()
        time.sleep(2)
        logger.error(f"Fehler in capture_new_face: {e}")

def text_input(prompt, default_text=""):
    """Einfache Texteingabe"""
    input_text = default_text
    
    screen.fill((0, 0, 0))
    prompt_text = font.render(prompt, True, (255, 255, 255))
    screen.blit(prompt_text, (50, 50))
    pygame.display.update()
    
    while True:
        screen.fill((0, 0, 0))
        
        # Prompt
        prompt_text = font.render(prompt, True, (255, 255, 255))
        screen.blit(prompt_text, (50, 50))
        
        # Eingabefeld
        pygame.draw.rect(screen, (50, 50, 50), (50, 100, screen_width-100, 40))
        
        # Text mit Blink-Cursor
        display_text = input_text
        if time.time() % 1.0 > 0.5:  # Cursor blinkt
            display_text += "|"
        
        input_render = font.render(display_text, True, (255, 255, 255))
        screen.blit(input_render, (60, 105))
        
        # Beschreibung
        help_text = font.render("Enter: Bestätigen, ESC: Abbrechen", True, (150, 150, 150))
        screen.blit(help_text, (50, 160))
        
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return input_text
                elif event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    if event.unicode.isprintable() and len(input_text) < 30:
                        input_text += event.unicode
        
        time.sleep(0.05)

# ... (Der Rest des Codes bleibt ähnlich, aber mit den neuen Funktionen integriert)

# -----------------------
# Hauptprogramm
# -----------------------
def main():
    global cap, screen, font, screen_width, screen_height, config, running
    
    # Signal Handler registrieren
    signal.signal(signal.SIGINT, signal_handler)
    
    # Argumentparser für Kommandozeilenoptionen
    parser = argparse.ArgumentParser(description='Face Recognition Security System')
    parser.add_argument('--config', help='Pfad zur Konfigurationsdatei')
    parser.add_argument('--no-gui', action='store_true', help='Startet ohne GUI')
    parser.add_argument('--debug', action='store_true', help='Aktiviert Debug-Modus')
    args = parser.parse_args()

    # Konfiguration laden
    config_path = args.config if args.config else CONFIG_FILE
    config = load_config()

    if args.debug:
        config["debug_mode"] = True
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug-Modus aktiviert")

    # Hardware initialisieren
    try:
        # Servo initialisieren
        set_servo_angle(servo_horizontal, 90)  # Mittige Position
        
        # LED ausschalten
        set_rgb_color(0, 0, 0)
        
        # OLED Display initialisieren
        update_oled_display(["System", "wird gestartet...", ""])
            
    except Exception as e:
        logger.error(f"Hardware Initialisierungsfehler: {e}")

    # Datenbank initialisieren
    init_database()

    # Bekannte Gesichter laden
    known_face_encodings, known_face_names = load_known_faces_db()
    unknown_face_encodings = []
    unknown_face_paths = []

    # Thread-sichere Datenstrukturen
    frame_lock = threading.Lock()
    frame_for_display = None
    faces_to_draw = []
    alert_flag = False
    alert_timer = 0
    frame_counter = 0
    last_cleanup_time = time.time()

    # Gesichtsnachverfolgung
    face_tracker = defaultdict(dict)
    next_face_id = 0

    # Kamera initialisieren
    try:
        camera_index = config["camera_index"] if config["camera_index"] != "auto" else find_camera()
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["resolution"][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["resolution"][1])

        logger.info(f"Kamera mit Index {camera_index} initialisiert")
    except Exception as e:
        logger.error(f"Kamera konnte nicht initialisiert werden: {e}")
        return

    # Pygame initialisieren (falls GUI aktiviert)
    if not args.no_gui:
        try:
            pygame.init()
            screen_width, screen_height = 480, 320
            screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Face Tracker - Security Mode")
            clock = pygame.time.Clock()
            font = pygame.font.SysFont(None, 24)
            large_font = pygame.font.SysFont(None, 36)

            # Alarm-Sound laden
            alarm_sound = None
            if config["alarm_sound"]:
                try:
                    alarm_sound = pygame.mixer.Sound("alarm.wav")
                except:
                    logger.warning("Alarm-Sound konnte nicht geladen werden")
        except Exception as e:
            logger.error(f"Pygame konnte nicht initialisiert werden: {e}")
            args.no_gui = True
    else:
        logger.info("GUI deaktiviert")

    # -----------------------
    # Face Recognition Thread
    # -----------------------
    def recognition_thread():
        nonlocal frame_for_display, faces_to_draw, alert_flag, alert_timer
        nonlocal next_face_id, face_tracker, unknown_face_encodings, unknown_face_paths

        frame_skip_counter = 0
        last_servo_update = time.time()

        while running:
            with frame_lock:
                if frame_for_display is None:
                    time.sleep(0.01)
                    continue
                frame = frame_for_display.copy()

            # Frame-Skipping für bessere Performance
            frame_skip_counter += 1
            if frame_skip_counter % config["frame_skip"] != 0:
                time.sleep(0.01)
                continue

            try:
                # Bild für Fernerkennung optimieren
                enhanced_frame = enhance_detection_for_distance(frame, config)
                small_frame = cv2.resize(enhanced_frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Gesichter erkennen mit optimierter Fernerkennung
                face_locations, face_encodings = detect_faces_at_distance(rgb_small_frame, config)

                # Gesichter nachverfolgen
                face_ids = track_faces(face_locations, face_encodings)

                temp_faces = []
                alert_triggered = False
                known_person_detected = False

                for i, ((top, right, bottom, left), face_encoding) in enumerate(zip(face_locations, face_encodings)):
                    # Koordinaten zurück auf Originalgröße skalieren
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2

                    # Gesichtsgröße prüfen und Entfernung schätzen
                    face_height = bottom - top
                    face_width = right - left
                    
                    # Entfernung schätzen
                    distance_estimate = estimate_distance(face_height, frame.shape[0])
                    
                    if face_height < config["min_face_size"] or face_width < config["min_face_size"]:
                        continue

                    # Gesichter vergleichen mit höherer Toleranz für Ferne
                    matches_known = face_recognition.compare_faces(
                        known_face_encodings, face_encoding, tolerance=config["tolerance"]
                    )
                    name = "Unbekannt"
                    known = False
                    face_id = face_ids[i] if i < len(face_ids) else -1

                    if True in matches_known:
                        first_match_index = matches_known.index(True)
                        name = known_face_names[first_match_index]
                        known = True
                        known_person_detected = True
                        confidence = 1.0 - face_recognition.face_distance(
                            [known_face_encodings[first_match_index]], face_encoding
                        )[0]
                        
                        # Servo-Verfolgung für 360° Bewegung
                        if time.time() - last_servo_update > 0.5:
                            track_face_with_servo(left, top, frame.shape[1], frame.shape[0])
                            last_servo_update = time.time()
                            
                    else:
                        # Mit unbekannten Gesichtern vergleichen
                        if unknown_face_encodings:
                            matches_unknown = face_recognition.compare_faces(
                                unknown_face_encodings, face_encoding, tolerance=config["tolerance"]
                            )
                            if True in matches_unknown:
                                name = "Bekannt (Unbekannt)"
                                confidence = 0.5
                            else:
                                # Neues unbekanntes Gesicht speichern
                                save_unknown_face(frame, top, right, bottom, left, face_encoding)
                                alert_triggered = True
                                confidence = 0.0
                        else:
                            # Erste unbekannte Person
                            save_unknown_face(frame, top, right, bottom, left, face_encoding)
                            alert_triggered = True
                            confidence = 0.0

                    # Zugriff protokollieren mit Entfernungsschätzung
                    status = "KNOWN" if known else "UNKNOWN"
                    log_access_db(name, status, confidence, face_id, distance_estimate)

                    temp_faces.append((top, right, bottom, left, name, known, face_id, confidence, distance_estimate))

                faces_to_draw = temp_faces

                # Hardware-Reaktionen
                if known_person_detected:
                    welcome_sequence("Gast")
                elif alert_triggered:
                    alert_flag = True
                    alert_timer = time.time()
                    alert_sequence()
                    
                    if not args.no_gui and alarm_sound and config["alarm_sound"]:
                        alarm_sound.play()

                    # Sprachalarm
                    if config["sound_effects"]:
                        speak_text("Unbekannte Person erkannt", config["language"])

                # OLED Status aktualisieren
                status_lines = [
                    f"Bekannt: {len(known_face_names)}",
                    f"Unbekannt: {len(unknown_face_encodings)}",
                    "Bereit" if not alert_flag else "ALARM!"
                ]
                update_oled_display(status_lines)

            except Exception as e:
                logger.error(f"Fehler in recognition_thread: {e}")

            time.sleep(0.03)

    def track_faces(face_locations, face_encodings):
        """Verfolgt Gesichter über mehrere Frames hinweg"""
        nonlocal next_face_id, face_tracker

        current_face_ids = []

        for i, (location, encoding) in enumerate(zip(face_locations, face_encodings)):
            face_matched = False
            center = ((location[3] + location[1]) // 2, (location[0] + location[2]) // 2)

            for face_id, data in list(face_tracker.items()):
                # Distanz zum vorherigen Position berechnen
                prev_center = data['last_center']
                distance = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)

                if distance < 50:  # Schwellenwert für Bewegung
                    face_tracker[face_id]['last_center'] = center
                    face_tracker[face_id]['last_seen'] = time.time()
                    current_face_ids.append(face_id)
                    face_matched = True
                    break

            if not face_matched:
                # Neues Gesicht
                face_id = next_face_id
                next_face_id += 1
                face_tracker[face_id] = {
                    'last_center': center,
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'encoding': encoding
                }
                current_face_ids.append(face_id)

        # Alte Gesichter bereinigen
        for face_id in list(face_tracker.keys()):
            if face_id not in current_face_ids:
                if time.time() - face_tracker[face_id]['last_seen'] > 5:  # 5 Sekunden nicht gesehen
                    del face_tracker[face_id]

        return current_face_ids

    def save_unknown_face(frame, top, right, bottom, left, encoding):
        """Speichert ein unbekanntes Gesicht"""
        timestamp = int(time.time())
        rand = np.random.randint(10000)
        unknown_path = os.path.join(UNKNOWN_FACES_DIR, f"unknown_{timestamp}_{rand}.jpg")

        # Gesicht ausschneiden und speichern
        face_image = frame[top:bottom, left:right]

        if face_image.size > 0:
            cv2.imwrite(unknown_path, face_image)
            unknown_face_encodings.append(encoding)
            unknown_face_paths.append(unknown_path)

            # In Datenbank speichern
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("INSERT INTO unknown_faces (timestamp, encoding, image_path) VALUES (?, ?, ?)",
                      (timestamp, sqlite3.Binary(pickle.dumps(encoding)), unknown_path))
            conn.commit()
            conn.close()

            logger.warning(f"Neues unbekanntes Gesicht: {unknown_path}")
            log_alert_db("unknown_face", "Unbekannte Person erkannt", unknown_path)

            # Entfernung schätzen für Benachrichtigungen
            face_height = bottom - top
            distance_estimate = estimate_distance(face_height, frame.shape[0])
            
            # Alle Benachrichtigungen senden
            send_gmail_alert(unknown_path, timestamp, config)
            send_whatsapp_alert(unknown_path, timestamp, config)
            
            telegram_msg = f"🚨 Unbekannte Person erkannt um {time.ctime(timestamp)} - Entfernung: {distance_estimate}m"
            send_telegram_alert(unknown_path, telegram_msg, config)

        return {}

    # -----------------------
    # System Monitoring Thread
    # -----------------------
    def system_monitor_thread():
        """Überwacht Systemressourcen"""
        nonlocal last_cleanup_time

        while running:
            try:
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                # FPS berechnen
                fps = clock.get_fps() if not args.no_gui else 0

                # Systemmetriken protokollieren
                log_system_metrics_db(cpu_percent, memory.percent, disk.percent, fps)

                # Warnung bei hoher Auslastung
                if cpu_percent > 90:
                    logger.warning(f"Hohe CPU-Auslastung: {cpu_percent}%")

                if memory.percent > 90:
                    logger.warning(f"Hohe Speicherauslastung: {memory.percent}%")

                # Regelmäßige Wartungsaufgaben
                current_time = time.time()

                # Datenbereinigung
                if current_time - last_cleanup_time > 3600:  # Stündlich
                    cleanup_old_data(config)
                    last_cleanup_time = current_time

            except Exception as e:
                logger.error(f"Fehler im Systemmonitor: {e}")

            time.sleep(60)  # Alle 60 Sekunden prüfen

    # -----------------------
    # Threads starten
    # -----------------------
    recognition_thread = threading.Thread(target=recognition_thread, daemon=True)
    recognition_thread.start()

    monitor_thread = threading.Thread(target=system_monitor_thread, daemon=True)
    monitor_thread.start()

    # Flask-Server in eigenem Thread starten
    if config["flask_enabled"]:
        flask_thread = threading.Thread(
            target=run_flask_server,
            args=(config, frame_lock, frame_for_display, known_face_names, unknown_face_encodings, alert_flag),
            daemon=True
        )
        flask_thread.start()

    # -----------------------
    # Hauptloop (Pygame oder Headless)
    # -----------------------
    running = True
    menu_active = False

    if args.no_gui:
        logger.info("Headless-Modus aktiviert")
        try:
            while running:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Frame konnte nicht gelesen werden")
                    time.sleep(1)
                    continue

                # Frame für Erkennungsthread speichern
                with frame_lock:
                    frame_for_display = frame.copy()

                # Kurze Pause, um CPU-Last zu reduzieren
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Anwendung durch Benutzer beendet")
            running = False
    else:
        # Pygame GUI Loop
        while running:
            ret, frame = cap.read()
            if not ret:
                logger.error("Frame konnte nicht gelesen werden")
                time.sleep(1)
                continue

            # Frame für Display vorbereiten
            display_frame = frame.copy()

            # Frame für Erkennungsthread speichern
            with frame_lock:
                frame_for_display = frame.copy()

            # Privacy-Modus anwenden
            if config["privacy_mode"]:
                display_frame = apply_privacy_blur(display_frame, faces_to_draw)

            # Gesichter einzeichnen (wenn nicht im Privacy-Modus)
            if not config["privacy_mode"]:
                for (top, right, bottom, left, name, known, face_id, confidence, distance) in faces_to_draw:
                    color = (0, 255, 0) if known else (0, 0, 255)  # Grün = bekannt, Rot = unbekannt
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)

                    # Label mit Konfidenz und Entfernung anzeigen
                    label = f"{name} ({confidence:.2f}) {distance}m" if confidence > 0 else f"{name} {distance}m"
                    cv2.putText(display_frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Alarm-Overlay
            if alert_flag:
                if time.time() - alert_timer < config["alert_duration"]:
                    # Blinkeffekt
                    if int(time.time() * 5) % 2 == 0:  # 5 Mal pro Sekunde blinken
                        overlay = display_frame.copy()
                        overlay[:] = (0, 0, 255)  # Rot
                        alpha = 0.4
                        cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

                        # Warntext
                        cv2.putText(display_frame, "UNBEKANNTE PERSON ERKANNT!", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    alert_flag = False

            # Statusinformationen
            cv2.putText(display_frame, f"Bekannt: {len(known_face_names)} | Unbekannt: {len(unknown_face_encodings)}",
                        (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Für Pygame anpassen
            display_frame = cv2.resize(display_frame, (screen_width, screen_height))
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            display_frame = np.rot90(display_frame)

            # In Pygame Surface konvertieren
            frame_surface = pygame.surfarray.make_surface(display_frame)
            screen.blit(frame_surface, (0, 0))

            # UI-Overlay zeichnen
            draw_ui_overlay(screen, known_face_names, unknown_face_encodings, alert_flag, clock, font, screen_width, screen_height)

            pygame.display.update()
            clock.tick(30)

            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_m:
                        menu_active = not menu_active
                        if menu_active:
                            show_menu()
                    elif event.key == pygame.K_d:
                        # Debug-Modus umschalten
                        config["debug_mode"] = not config["debug_mode"]
                        logger.info(f"Debug-Modus: {'aktiviert' if config['debug_mode'] else 'deaktiviert'}")
                    elif event.key == pygame.K_p:
                        # Privacy-Modus umschalten
                        config["privacy_mode"] = not config["privacy_mode"]
                        logger.info(f"Privacy-Modus: {'aktiviert' if config['privacy_mode'] else 'deaktiviert'}")
                    elif event.key == pygame.K_c:
                        # Neues Gesicht erfassen
                        capture_new_face()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Touch-Event für Menü-Button
                    x, y = event.pos
                    if screen_width - 100 <= x <= screen_width - 10 and screen_height - 40 <= y <= screen_height - 10:
                        menu_active = not menu_active
                        if menu_active:
                            show_menu()

    # Aufräumen
    cap.release()
    servo_horizontal.stop()
    GPIO.cleanup()

    if not args.no_gui:
        pygame.quit()

    logger.info("Anwendung beendet")

# ... (Rest der UI-Funktionen bleiben ähnlich)

if __name__ == "__main__":
    main()
