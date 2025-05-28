import os
import platform
import time
import threading
import uuid
import datetime
import traceback
import re
import requests
import json
import numpy as np
import random

import cv2
import fitz  # PyMuPDF
import speech_recognition as sr
import pyttsx3
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
from PIL import Image, ImageTk

try:
    import ttkbootstrap as tb
    USE_BOOTSTRAP = True
except ImportError:
    USE_BOOTSTRAP = False

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ---------------------------------------------------------
# 1) UI / Theme Constants
# ---------------------------------------------------------
APP_TITLE = "Futuristic AI Interview Coach"
MAIN_BG = "#101826"
MAIN_FG = "#FFFFFF"
ACCENT_COLOR = "#31F4C7"
BUTTON_BG = "#802BB1"
BUTTON_FG = "#FFFFFF"
GRADIENT_START = "#802BB1"
GRADIENT_END   = "#1CD8D2"
GLASS_BG = "#1F2A3A"
FONT_FAMILY = "Helvetica"
FONT_SIZE_TITLE = 22
FONT_SIZE_NORMAL = 11

# HF Inference
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
HF_AUTH_TOKEN = "hf_token"  # Provide or remove your HF token if needed
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
headers = {"Authorization": f"Bearer {HF_AUTH_TOKEN}"}

CONVO_MODEL_NAME = "facebook/blenderbot-400M-distill"
ZS_MODEL_NAME = "facebook/bart-large-mnli"

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 1.0)
recognizer = sr.Recognizer()

import cv2.data
CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ---------------------------------------------------------
# DNN Face Detector Initialization
# ---------------------------------------------------------
DNN_PROTO = "deploy.prototxt"
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
try:
    face_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
    print("DNN face detector loaded successfully.")
except Exception as e:
    print(f"Could not load DNN face detector: {e}")
    face_net = None

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
session_id = str(uuid.uuid4())[:8]
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
log_filename = os.path.join(LOG_DIR, f"InterviewLog_{session_id}.txt")

def log_event(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"Log write error: {e}")

# ---------------------------------------------------------
# 2) Global Vars
# ---------------------------------------------------------
speaking_lock = threading.Lock()
is_speaking = False
interview_running = False
cap = None
candidate_name = "Candidate"
job_role = ""
interview_context = ""
user_submitted_answer = None
last_input_time = None  # NEW: Track last keystroke activity

# LBPH
lbph_recognizer = None
registered_label = 100
LBPH_THRESHOLD = 70  # Lower = stricter

# YOLO phone detection
yolo_model = None
PHONE_LABELS = {"cell phone", "mobile phone", "phone"}

# STT
is_recording_voice = False
stop_recording_flag = False
recording_thread = None

# HF Models
convo_tokenizer = None
convo_model = None
zero_shot_classifier = None

# Lip sync frames
lip_sync_frames = None

# Multi-face detection counter (for transient warnings vs. persistent error)
multi_face_counter = 0
MULTI_FACE_THRESHOLD = 10

# Phone detection counter to avoid random false positives
phone_detect_counter = 0
PHONE_DETECT_THRESHOLD = 3

# Global error tracking variable (for both phone & face mismatch warnings)
warning_count = 0
previous_warning_message = None

# Tkinter references
root = None
chat_display = None
start_button = None
stop_button = None
resume_entry = None
role_entry = None
record_btn = None
stop_record_btn = None
submit_btn = None
camera_label = None
bot_label = None
answer_entry = None
warning_count_label = None  # Dashboard label for warnings

# ---------------------------------------------------------
# 3) Helper / UI Functions
# ---------------------------------------------------------
def safe_showerror(title, message):
    root.after(0, lambda: messagebox.showerror(title, message))

def safe_showinfo(title, message):
    root.after(0, lambda: messagebox.showinfo(title, message))

def safe_update(widget, func, *args, **kwargs):
    root.after(0, lambda: func(*args, **kwargs))

# --- UPDATED append_transcript: adds extra spacing between messages ---
def append_transcript(widget, text):
    widget.config(state=tk.NORMAL)
    current_content = widget.get("1.0", tk.END)
    if current_content.strip():
        widget.insert(tk.END, "\n" + text + "\n")
    else:
        widget.insert(tk.END, text + "\n")
    widget.see(tk.END)
    widget.config(state=tk.DISABLED)

# ---------------------------------------------------------
# 4) Hugging Face Q&A Functions (with retry)
# ---------------------------------------------------------
def safe_query_hf(payload, max_retries=3):
    """
    A safer query function that retries the HF API call up to max_retries times
    in case of temporary network or service issues.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            log_event(f"HF query error on attempt {attempt}: {e}")
            time.sleep(1)
    return {}

def generate_interview_question_api(_job_role, context=""):
    prompt = (
        f"{context}\n"
        f"You are a professional interviewer. Generate one clear, relevant question "
        f"for a candidate applying for a {_job_role} position. "
        "Output only 'Interviewer: <question>'."
    )
    payload_data = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 100, "temperature": 0.8, "do_sample": True},
        "options": {"wait_for_model": True}
    }
    result = safe_query_hf(payload_data)
    if isinstance(result, list) and "generated_text" in result[0]:
        g = result[0]["generated_text"]
    else:
        g = ""
    question = g.replace(prompt, "").strip()
    if not question:
        question = "Could you please share more about your experience in this role?"
    if not question.lower().startswith("interviewer:"):
        question = f"Interviewer: {question}"
    return question

def evaluate_response(question, response, context=""):
    prompt = (
        f"{context}\n"
        f"Interview Question: {question}\n"
        f"Candidate's Response: {response}\n\n"
        "Provide a short, constructive comment on the candidate's answer (1-2 sentences max)."
    )
    payload_data = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 80, "temperature": 0.7, "do_sample": True},
        "options": {"wait_for_model": True}
    }
    result = safe_query_hf(payload_data)
    if isinstance(result, list) and "generated_text" in result[0]:
        g = result[0]["generated_text"]
    else:
        g = ""
    return g.replace(prompt, "").strip()

def summarize_all_responses(transcript, context=""):
    prompt = (
        f"{context}\n"
        "You are a professional interviewer. Provide a concise overview (2-3 sentences) summarizing the candidate's overall performance.\n\n"
        f"Interview Transcript:\n{transcript}\n"
        "Overall Overview:"
    )
    payload_data = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 150, "temperature": 0.7, "do_sample": True},
        "options": {"wait_for_model": True}
    }
    result = safe_query_hf(payload_data)
    if isinstance(result, list) and "generated_text" in result[0]:
        s = result[0]["generated_text"]
    else:
        s = ""
    return s.replace(prompt, "").strip()

# ---------------------------------------------------------
# 5) Performance Scoring (Based Solely on Candidate Responses)
# ---------------------------------------------------------
def grade_interview_with_breakdown(transcript, context=""):
    """
    Compute performance score purely based on candidate responses.
    For each candidate response, use:
      - <10 words: 10 points
      - 10-19 words: 30 points
      - >=20 words: 50 points
    The average is scaled to 100 and then warning count is subtracted.
    """
    candidate_lines = [line for line in transcript.split("\n") if line.strip().startswith("Candidate:")]
    if candidate_lines:
        total_score = 0
        num_responses = len(candidate_lines)
        for line in candidate_lines:
            response = line.replace("Candidate:", "").strip()
            word_count = len(response.split())
            if word_count < 10:
                candidate_score = 10
            elif word_count < 20:
                candidate_score = 30
            else:
                candidate_score = 50
            total_score += candidate_score
        avg_candidate_score = total_score / num_responses
        base_score = int(min(100, max(0, (avg_candidate_score / 50) * 100)))
        final_score = max(0, base_score - warning_count)
        breakdown = {
            "base_score": base_score,
            "warning_penalty": warning_count,
            "final_score": final_score,
            "explanation": "Score computed based solely on candidate responses word count."
        }
        return final_score, breakdown, "Heuristic scoring used."
    else:
        return 0, {"explanation": "No candidate responses found."}, "No candidate lines found."

# ---------------------------------------------------------
# 6) Resume Parsing Functions
# ---------------------------------------------------------
def parse_resume(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        log_event(f"PDF parse error: {e}")
    return text

def extract_candidate_name(resume_text):
    lines = [ln.strip() for ln in resume_text.split("\n") if ln.strip()]
    if lines:
        first_line = lines[0]
        if re.match(r"^[A-Za-z\s]+$", first_line) and len(first_line.split()) <= 4:
            return first_line.strip()
    return "Candidate"

def build_context(resume_text, role):
    length = 600
    summary = resume_text[:length].replace("\n", " ")
    if len(resume_text) > length:
        summary += "..."
    c_name = extract_candidate_name(resume_text)
    return (
        f"Candidate's Desired Role: {role}\n"
        f"Resume Summary: {summary}\n"
        f"Candidate Name: {c_name}\n\n"
        "You are a seasoned interviewer. Ask professional role-based or scenario-based questions."
    )

# ---------------------------------------------------------
# 7) TTS / STT Functions
# ---------------------------------------------------------
def text_to_speech(text):
    global is_speaking
    with speaking_lock:
        is_speaking = True
    to_speak = text.replace("Interviewer:", "").strip()
    try:
        tts_engine.say(to_speak)
        tts_engine.runAndWait()
    except Exception as e:
        log_event(f"TTS error: {e}")
    with speaking_lock:
        is_speaking = False

def record_audio():
    global is_recording_voice, stop_recording_flag, user_submitted_answer
    safe_update(record_btn, record_btn.config, state=tk.DISABLED)
    safe_update(stop_record_btn, stop_record_btn.config, state=tk.NORMAL)
    is_recording_voice = True
    stop_recording_flag = False

    append_transcript(chat_display, "(Recording in progress...)")
    recognized_segments = []

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        while not stop_recording_flag and interview_running:
            try:
                audio = recognizer.listen(source, phrase_time_limit=10)
                if stop_recording_flag or not interview_running:
                    break
                text_chunk = recognizer.recognize_google(audio)
                recognized_segments.append(text_chunk)
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                log_event(f"STT error: {e}")
                break

    final_text = " ".join(recognized_segments).strip()
    user_submitted_answer = final_text
    if final_text:
        append_transcript(chat_display, f"(Recognized Voice): {final_text}")
    else:
        append_transcript(chat_display, "(No speech recognized.)")

    is_recording_voice = False
    safe_update(stop_record_btn, stop_record_btn.config, state=tk.DISABLED)

def start_recording_voice():
    global recording_thread
    if not interview_running:
        append_transcript(chat_display, "Interview not running. Click 'Start Interview' first.")
        return
    if recording_thread and recording_thread.is_alive():
        append_transcript(chat_display, "Already recording.")
        return
    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()

def stop_recording_voice():
    global stop_recording_flag
    if not interview_running:
        return
    if is_recording_voice:
        stop_recording_flag = True
        append_transcript(chat_display, "(Stopping recording...)")

# ---------------------------------------------------------
# 8) LBPH Face Recognition Functions
# ---------------------------------------------------------
def open_camera_for_windows(index=0):
    if platform.system() == "Windows":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        return cv2.VideoCapture(index)

def capture_face_samples(sample_count=30, delay=0.1):
    faces_collected = []
    cap_temp = open_camera_for_windows(0)
    if not cap_temp.isOpened():
        safe_showerror("Webcam Error", "Cannot open camera for face registration.")
        return []

    collected = 0
    while collected < sample_count:
        ret, frame = cap_temp.read()
        if not ret:
            time.sleep(0.1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80,80))
        if len(found) == 1:
            x, y, w, h = found[0]
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (200, 200))
            faces_collected.append(roi_resized)
            collected += 1
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Registering Face... (Press 'q' to abort)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(delay)

    cap_temp.release()
    cv2.destroyAllWindows()
    return faces_collected

def train_lbph(faces_list, label_id=100):
    global lbph_recognizer
    if lbph_recognizer is None:
        lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()

    if len(faces_list) < 15:
        raise ValueError("Not enough valid face samples captured. Keep face visible longer.")

    labels = [label_id]*len(faces_list)
    labels_array = np.array(labels, dtype=np.int32)
    lbph_recognizer.train(faces_list, labels_array)

def register_candidate_face():
    global lbph_recognizer
    face_imgs = capture_face_samples(sample_count=30, delay=0.1)
    if not face_imgs:
        safe_showerror("Registration Error", "No face samples collected. Check lighting/camera and try again.")
        return False
    try:
        train_lbph(face_imgs, label_id=registered_label)
        safe_showinfo("Face Registration", "Face registered successfully!")
        return True
    except Exception as e:
        safe_showerror("Registration Error", f"LBPH train failed: {e}")
        return False

# ---------------------------------------------------------
# 9) YOLO Phone Detection Functions
# ---------------------------------------------------------
def load_yolo_model():
    global yolo_model
    if YOLO is None:
        log_event("YOLO not installed. Skipping phone detection.")
        return
    try:
        yolo_model = YOLO("yolov8n.pt")
        log_event("YOLO loaded for phone detection.")
    except Exception as e:
        log_event(f"YOLO load error: {e}")
        yolo_model = None

def detect_phone_in_frame(frame):
    if yolo_model is None:
        return False
    results = yolo_model.predict(frame, imgsz=640, verbose=False)
    if not results:
        return False
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            class_id = int(box.cls[0].item())
            class_name = r.names[class_id]
            if class_name.lower() in PHONE_LABELS:
                return True
    return False

# ---------------------------------------------------------
# 10) Updated Monitoring Functions (Webcam & Face Detection)
# ---------------------------------------------------------
def detect_faces_dnn(frame, conf_threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0,
                                 (300,300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            boxes.append((startX, startY, endX - startX, endY - startY, confidence))
    return boxes

def check_same_person_and_phone(frame):
    global lbph_recognizer, multi_face_counter, phone_detect_counter
    # Phone detection: only trigger if detected for consecutive frames.
    if detect_phone_in_frame(frame):
        phone_detect_counter += 1
        if phone_detect_counter >= PHONE_DETECT_THRESHOLD:
            return (False, "Phone detected.")
    else:
        phone_detect_counter = 0

    if lbph_recognizer is None:
        return (False, "No LBPH model found. Register face first.")

    if face_net is not None:
        boxes = detect_faces_dnn(frame, conf_threshold=0.5)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = []
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
        for (x, y, w, h) in detected:
            boxes.append((x, y, w, h, 1.0))
    
    if len(boxes) == 0:
        return (False, "No face detected.")
    elif len(boxes) > 1:
        multi_face_counter += 1
        if multi_face_counter > MULTI_FACE_THRESHOLD:
            return (False, "Critical: Multiple faces detected persistently. Please ensure only you are visible.")
        else:
            return (True, "Warning: Multiple faces detected. Please ensure only you are visible.")
    else:
        multi_face_counter = 0
        x, y, w, h, conf = boxes[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y:y+h, x:x+w]
        try:
            roi_resized = cv2.resize(roi, (200,200))
        except Exception as e:
            log_event(f"ROI resize error: {e}")
            return (False, "Face detected but ROI processing failed.")
        pred_label, confidence = lbph_recognizer.predict(roi_resized)
        # Standardize the face mismatch warning message.
        if pred_label != registered_label or confidence > LBPH_THRESHOLD:
            return (False, "Face mismatch detected.")
    return (True, "OK")

def monitor_webcam():
    global interview_running, cap, warning_count, previous_warning_message
    while interview_running and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        status, reason = check_same_person_and_phone(frame)
        if not status:
            if reason.startswith("Critical"):
                safe_showerror("Security Issue", reason + " Ending interview.")
                log_event(f"Security fail: {reason}")
                interview_running = False
                break
            else:
                # Only show pop-up if this is a new warning (not already active)
                if previous_warning_message != reason:
                    safe_showinfo("Camera Warning", reason)
                    previous_warning_message = reason
                    warning_count += 1
                    safe_update(warning_count_label, warning_count_label.config, text=f"Camera Warnings: {warning_count}")
        else:
            previous_warning_message = None
        time.sleep(0.1)

def update_camera_view():
    global interview_running, cap
    if not interview_running or cap is None or not cap.isOpened():
        return
    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.config(image=imgtk)
        camera_label.image = imgtk
    if interview_running:
        camera_label.after(30, update_camera_view)

# ---------------------------------------------------------
# 11) Bot Animation Functions
# ---------------------------------------------------------
GIF_PATH = "VirtualCoach.gif"
bot_frames = []
frame_index = 0

def load_gif_frames(gif_path):
    frames_ = []
    try:
        pil_img = Image.open(gif_path)
        for i in range(pil_img.n_frames):
            pil_img.seek(i)
            frm = pil_img.copy()
            frames_.append(ImageTk.PhotoImage(frm))
    except Exception as e:
        log_event(f"GIF load error: {e}")
    return frames_

def animate_bot():
    global frame_index, bot_frames, lip_sync_frames, is_speaking
    with speaking_lock:
        speaking_now = is_speaking
    if speaking_now:
        if lip_sync_frames:
            frame_index = (frame_index + 1) % len(lip_sync_frames)
            bot_label.config(image=lip_sync_frames[frame_index])
        elif bot_frames:
            frame_index = (frame_index + 1) % len(bot_frames)
            bot_label.config(image=bot_frames[frame_index])
        bot_label.image = bot_label.cget("image")
    else:
        if bot_frames:
            bot_label.config(image=bot_frames[0])
            bot_label.image = bot_frames[0]
        frame_index = 0
    root.after(100, animate_bot)

# ---------------------------------------------------------
# 12) Interview Conversation Logic (Revised Question Generation)
# ---------------------------------------------------------
def generate_interview_question(role, resume_context="", question_level="basic", candidate_response="", challenge_type=None, category=""):
    """
    Generate an interview question based on the resume context and desired role.
    Optionally focus on a given resume section (category) and include a coding or SQL challenge.
    A random seed is appended to ensure diversity. The generated output is cleaned.
    """
    prompt = f"{resume_context}\n"
    prompt += f"You are a seasoned interviewer interviewing a candidate for a '{role}' position. "
    if category:
        prompt += f"Focus on the candidate's {category} as mentioned in their resume. "
    prompt += ("The candidate's resume includes details on Academic Background, Work Experience, Technical Skills, "
               "Projects, Certifications, Achievements/Awards, Internships, Extracurricular Activities, Leadership Roles, "
               "Publications/Research, Tools and Technologies, Soft Skills, and Career Goals. ")
    
    if candidate_response:
        prompt += f"The candidate's previous answer was: '{candidate_response}'. "
        prompt += "Based on this, ask a clarifying follow-up question that probes further into any ambiguous points. "
    else:
        if question_level == "basic":
            prompt += "Begin with a warm, introductory question that invites the candidate to share their background naturally. "
        elif question_level == "intermediate":
            prompt += "Now, ask a question that explores the candidate’s practical experience and relevant skills. "
        elif question_level in ['proficient', 'advanced']:
            prompt += "Proceed to ask a challenging question that requires critical thinking and deeper insight into their expertise. "

    # Integrate dynamic challenge prompt if challenge_type is provided.
    if challenge_type == "sql":
         sql_prompts = [
             "You are an AI that generates SQL challenge prompts for Data Analysis roles. "
             "Generate a unique SQL challenge prompt that includes a scenario with one or more tables, "
             "describes the context, and instructs the candidate to write a SQL query to solve a specific problem. "
             "Do not use any predefined text; the prompt must be generated dynamically and be completely original."
         ]
         chosen_sql_prompt = random.choice(sql_prompts)
         prompt += f"Additionally, include a SQL challenge: {chosen_sql_prompt} "
    elif challenge_type == "coding":
         coding_prompts = [
             "You are an AI that generates coding challenge prompts for Computer Science and IT roles. "
             "Generate a unique coding challenge prompt that describes a clear problem scenario and asks the candidate "
             "to write a function in a popular programming language (e.g., Python, Java, JavaScript, or C++). "
             "The problem statement must be fully dynamic, original, and contain no predefined content."
         ]
         chosen_coding_prompt = random.choice(coding_prompts)
         prompt += f"Additionally, include a coding challenge: {chosen_coding_prompt} "

    prompt += f" RandomSeed: {random.randint(0, 1000000)}. "
    prompt += "Return only the question in the format: 'Interviewer: <question>' without any extra commentary."

    payload_data = {
         "inputs": prompt,
         "parameters": {"max_new_tokens": 150, "temperature": 0.8, "do_sample": True},
         "options": {"wait_for_model": True}
    }
    result = safe_query_hf(payload_data, max_retries=2)
    if isinstance(result, list) and "generated_text" in result[0]:
         question = result[0]["generated_text"]
    else:
         question = ""
    # Clean up the question output: keep only the first line and ensure proper punctuation.
    question = question.replace(prompt, "").strip()
    question = question.split("\n")[0].strip()
    question = re.sub(r'[\s\W]+$', '', question)
    if not question.lower().startswith("interviewer:"):
         question = f"Interviewer: {question}"
    parts = question.split("Interviewer:")
    if len(parts) > 1:
         body = parts[1].strip()
         if not body.endswith('?'):
              body = body.rstrip(".! ") + "?"
         question = "Interviewer: " + body
    else:
         if not question.endswith('?'):
              question = question.rstrip(".! ") + "?"
    return question

def generate_followup_question(previous_question, candidate_response, resume_context, role):
    """
    Generate a follow-up question based on the candidate's previous response.
    A random seed is appended to ensure diversity and the output is cleaned.
    """
    prompt = f"{resume_context}\n"
    prompt += f"You are a professional interviewer for a '{role}' role. "
    prompt += f"Previously, you asked: '{previous_question.replace('Interviewer:', '').strip()}'. "
    prompt += f"The candidate responded: '{candidate_response}'. "
    prompt += "Since the answer appears unclear or incomplete, ask a follow-up question to gain further insight. "
    prompt += f" RandomSeed: {random.randint(0, 1000000)}. "
    prompt += "Return the question in the format: 'Interviewer: <question>' using fresh, varied language."
    
    payload_data = {
         "inputs": prompt,
         "parameters": {"max_new_tokens": 150, "temperature": 0.8, "do_sample": True},
         "options": {"wait_for_model": True}
    }
    result = safe_query_hf(payload_data, max_retries=2)
    if isinstance(result, list) and "generated_text" in result[0]:
         followup = result[0]["generated_text"]
    else:
         followup = ""
    followup = followup.replace(prompt, "").strip()
    followup = followup.split("\n")[0].strip()
    followup = re.sub(r'[\s\W]+$', '', followup)
    if not followup.lower().startswith("interviewer:"):
         followup = f"Interviewer: {followup}"
    parts = followup.split("Interviewer:")
    if len(parts) > 1:
         body = parts[1].strip()
         if not body.endswith('?'):
             body = body.rstrip(".! ") + "?"
         followup = "Interviewer: " + body
    else:
         if not followup.endswith('?'):
             followup = followup.rstrip(".! ") + "?"
    return followup

def generate_unique_question_list(context, role):
    """
    Generate a unique list of at least 10 interview questions that progressively cover different
    resume sections and role-related topics. Duplicates (by normalized text) are removed.
    Includes a compulsory coding challenge for IT roles and a SQL challenge for Data Analytics roles.
    """
    topics = [
        "Introduction", "Academic Background", "Work Experience", "Technical Skills", "Projects", 
        "Certifications", "Achievements/Awards", "Internships", "Extracurricular Activities", 
        "Leadership Roles", "Publications/Research", "Tools and Technologies", "Soft Skills", 
        "Career Goals", "Role Specific", "Scenario Based", "Technical Concepts", "System Design", 
        "Behavioral Questions", "Situational Questions", "Industry Trends", "Case Studies", 
        "Problem-Solving", "Critical Thinking", "Ethical Dilemmas", "Teamwork", "Time Management", 
        "Conflict Resolution", "Innovation and Creativity", "Adaptability", "Communication Skills", 
        "Cultural Fit", "Stress Management", "Decision-Making Scenarios"
    ]
    random.shuffle(topics)
    questions = []
    # Always start with a basic introductory question.
    intro_q = generate_interview_question(role, context, question_level="basic", category="Introduction")
    questions.append(intro_q)
    
    count = 1
    i = 0
    while count < 10 and i < len(topics):
        topic = topics[i]
        if topic == "Introduction":
            level = "basic"
        elif topic in ["Academic Background", "Work Experience", "Technical Skills", "Projects", "Certifications", "Achievements/Awards"]:
            level = "intermediate"
        else:
            level = "advanced"
        q = generate_interview_question(role, context, question_level=level, category=topic)
        questions.append(q)
        count += 1
        i += 1

    role_lower = role.lower()
    # Add a coding challenge question for IT/Computer Science roles.
    if any(keyword in role_lower for keyword in [
        "it", "computer", "software", "developer", "programmer", "machine",
        "information", "engineer", "coder", "systems", "architect", "devops",
        "cybersecurity", "network", "cloud", "qa", "quality assurance", "sysadmin",
        "administrator", "support", "technical", "data engineer", "machine learning",
        "ai", "artificial intelligence", "deep learning", "nlp", "embedded", "iot",
        "firmware", "blockchain", "cryptography", "game", "gaming", "unreal", "unity",
        "robotics", "automation"
    ]):
         coding_q = generate_interview_question(role, context, question_level="intermediate", challenge_type="coding", category="Coding Challenge")
         questions.append(coding_q)
    # Add compulsory SQL challenge for Data Analytics related roles
    if any(keyword in role_lower for keyword in [
        "data", "analytics", "analyst", "data science", "business intelligence",
        "big data", "data scientist", "data engineer", "quantitative", "statistics",
        "statistic", "data mining", "database", "dba", "database administrator",
        "etl", "data warehousing", "reporting", "bi", "business analyst", "data visualization"
    ]):
         sql_q = generate_interview_question(role, context, question_level="intermediate", challenge_type="sql", category="SQL Challenge")
         questions.append(sql_q)
    
    unique_list = []
    seen = set()
    for q in questions:
         q_core = q.replace("Interviewer:", "").strip().lower()
         q_core_norm = re.sub(r'\W+', ' ', q_core).strip()
         if q_core_norm not in seen and len(q_core_norm) > 0:
             unique_list.append(q)
             seen.add(q_core_norm)
    return unique_list

def interview_loop(chat_display):
    global interview_running, user_submitted_answer, last_input_time
    global candidate_name, job_role, interview_context

    transcript = []
    candidate_responses_count = 0

    try:
        if not interview_running:
            return

        greeting = f"Hello, {candidate_name}! I'm your Virtual Interviewer. Keep your face visible and say 'stop' or 'exit' anytime to end."
        transcript.append(f"Interviewer: {greeting}")
        append_transcript(chat_display, f"Interviewer: {greeting}")
        text_to_speech(greeting)

        q_list = generate_unique_question_list(interview_context, job_role)

        for q in q_list:
            if not interview_running:
                break
            question_text = q.replace("Interviewer:", "").strip()
            current_question = f"Interviewer: {question_text}"
            transcript.append(current_question)
            append_transcript(chat_display, current_question)
            text_to_speech(current_question)

            safe_update(record_btn, record_btn.config, state=tk.NORMAL)
            safe_update(submit_btn, submit_btn.config, state=tk.NORMAL)

            user_submitted_answer = None
            candidate_response = ""
            last_input_time = time.time()
            while interview_running and not candidate_response:
                time.sleep(0.5)
                if user_submitted_answer:
                    candidate_response = user_submitted_answer
                    user_submitted_answer = None
                    break
                if time.time() - last_input_time >= 30:
                    break

            safe_update(record_btn, record_btn.config, state=tk.DISABLED)
            safe_update(submit_btn, submit_btn.config, state=tk.DISABLED)

            if not interview_running:
                break

            if not candidate_response:
                farewell = "No response received within 30 seconds. Ending the interview now."
                transcript.append(f"Interviewer: {farewell}")
                append_transcript(chat_display, f"Interviewer: {farewell}")
                text_to_speech(farewell)
                break

            if "stop" in candidate_response.lower() or "exit" in candidate_response.lower():
                end_msg = "Understood. Ending interview now."
                transcript.append(f"Interviewer: {end_msg}")
                append_transcript(chat_display, f"Interviewer: {end_msg}")
                text_to_speech(end_msg)
                break

            cand_line = f"Candidate: {candidate_response}"
            candidate_responses_count += 1
            transcript.append(cand_line)
            append_transcript(chat_display, cand_line)

            try:
                feedback = evaluate_response(question_text, candidate_response, interview_context)
                if feedback.strip():
                    feed_line = f"Interviewer: {feedback}"
                    transcript.append(feed_line)
                    append_transcript(chat_display, feed_line)
                    text_to_speech(feedback)
            except Exception as e:
                log_event(f"Minor error in evaluate_response: {e}")

            if len(candidate_response.split()) < 10:
                followup_q = generate_followup_question(current_question, candidate_response, interview_context, job_role)
                transcript.append(followup_q)
                append_transcript(chat_display, followup_q)
                text_to_speech(followup_q)

                safe_update(record_btn, record_btn.config, state=tk.NORMAL)
                safe_update(submit_btn, submit_btn.config, state=tk.NORMAL)
                user_submitted_answer = None
                followup_response = ""
                last_input_time = time.time()
                while interview_running and not followup_response:
                    time.sleep(0.5)
                    if user_submitted_answer:
                        followup_response = user_submitted_answer
                        user_submitted_answer = None
                        break
                    if time.time() - last_input_time >= 30:
                        break
                safe_update(record_btn, record_btn.config, state=tk.DISABLED)
                safe_update(submit_btn, submit_btn.config, state=tk.DISABLED)

                if followup_response:
                    cand_line2 = f"Candidate: {followup_response}"
                    candidate_responses_count += 1
                    transcript.append(cand_line2)
                    append_transcript(chat_display, cand_line2)
                    try:
                        followup_feedback = evaluate_response(followup_q, followup_response, interview_context)
                        if followup_feedback.strip():
                            feed_line2 = f"Interviewer: {followup_feedback}"
                            transcript.append(feed_line2)
                            append_transcript(chat_display, feed_line2)
                            text_to_speech(followup_feedback)
                    except Exception as e:
                        log_event(f"Minor error evaluating followup: {e}")

    except Exception as e:
        log_event(f"Interview error: {e}\n{traceback.format_exc()}")
        err_msg = "An unexpected error occurred, but we will proceed to wrap-up."
        transcript.append(f"Interviewer: {err_msg}")
        append_transcript(chat_display, f"Interviewer: {err_msg}")
        text_to_speech(err_msg)

    finally:
        if interview_running:
            interview_running = False
        log_event("Interview finishing... Attempting final summary/scoring.")

        if candidate_responses_count > 0:
            try:
                t_text = "\n".join(transcript)
                overview = summarize_all_responses(t_text, interview_context)
                if overview:
                    ov_line = f"Interviewer: {overview}"
                    transcript.append(ov_line)
                    append_transcript(chat_display, ov_line)
                    text_to_speech(overview)
                sc, breakdown, _ = grade_interview_with_breakdown(t_text, interview_context)
                if sc > 0:
                    final_msg = f"Your final interview score is {sc}/100. Thank you for your time!"
                else:
                    final_msg = "Could not determine a numeric score at this time. Thank you for your time!"
                transcript.append(f"Interviewer: {final_msg}")
                append_transcript(chat_display, f"Interviewer: {final_msg}")
                text_to_speech(final_msg)
            except Exception as e2:
                log_event(f"Error in final summary/scoring: {e2}")
                fallback_msg = "We encountered an error in final scoring. Please review the transcript manually."
                transcript.append(f"Interviewer: {fallback_msg}")
                append_transcript(chat_display, fallback_msg)
        else:
            no_resp_msg = "No candidate responses were recorded. Ending session."
            transcript.append(f"Interviewer: {no_resp_msg}")
            append_transcript(chat_display, f"Interviewer: {no_resp_msg}")
            text_to_speech(no_resp_msg)

        append_transcript(chat_display, "Interview ended. You may close or start a new session.")
        safe_update(start_button, start_button.config, state=tk.NORMAL)
        safe_update(stop_button, stop_button.config, state=tk.DISABLED)
        
        global warning_count, previous_warning_message
        warning_count = 0
        previous_warning_message = None
        safe_update(warning_count_label, warning_count_label.config, text="Camera Warnings: 0")

        log_event("Interview fully finished.")

# ---------------------------------------------------------
# 13) Model Loading and Splash Screen
# ---------------------------------------------------------
def load_lip_sync_model():
    log_event("Lip sync stub loaded.")
    return True

def load_model_splash():
    splash = tk.Toplevel()
    splash.overrideredirect(True)
    splash.configure(bg=MAIN_BG)
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    w, h = 350, 120
    x = (sw - w)//2
    y = (sh - h)//2
    splash.geometry(f"{w}x{h}+{x}+{y}")
    sp_canvas = tk.Canvas(splash, width=w, height=h, highlightthickness=0)
    sp_canvas.pack(fill=tk.BOTH, expand=True)

    def draw_gradient(canv, ww, hh, c1=GRADIENT_START, c2=GRADIENT_END):
        steps = 100
        for i in range(steps):
            ratio = i / steps
            r1, g1, b1 = splash.winfo_rgb(c1)
            r2, g2, b2 = splash.winfo_rgb(c2)
            rr = int(r1 + (r2 - r1)*ratio) >> 8
            gg = int(g1 + (g2 - g1)*ratio) >> 8
            bb = int(b1 + (b2 - b1)*ratio) >> 8
            color = f"#{rr:02x}{gg:02x}{bb:02x}"
            canv.create_line(0, int(hh*ratio), ww, int(hh*ratio), fill=color)

    draw_gradient(sp_canvas, w, h)
    lbl = tk.Label(sp_canvas, text="Loading Models...\nPlease wait.",
                   fg=MAIN_FG, font=(FONT_FAMILY, 14, "bold"), bg=None)
    lbl.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
    pb = ttk.Progressbar(sp_canvas, mode="indeterminate", length=250)
    pb.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
    pb.start()

    def finish_loading():
        global convo_tokenizer, convo_model, zero_shot_classifier
        try:
            convo_tokenizer = AutoTokenizer.from_pretrained(CONVO_MODEL_NAME)
            convo_model = AutoModelForSeq2SeqLM.from_pretrained(CONVO_MODEL_NAME)
            if convo_tokenizer.pad_token is None:
                convo_tokenizer.pad_token = convo_tokenizer.eos_token
            zero_shot_classifier = pipeline("zero-shot-classification", model=ZS_MODEL_NAME)
            load_lip_sync_model()
            load_yolo_model()
        except Exception as e:
            log_event(f"Model load error: {e}")
            safe_showerror("Model Load Error", str(e))
        splash.destroy()

    th = threading.Thread(target=finish_loading)
    th.start()

    def check_thread():
        if th.is_alive():
            root.after(100, check_thread)
        else:
            if splash.winfo_exists():
                splash.destroy()
            main_app()

    check_thread()

# ---------------------------------------------------------
# 14) UI and Main Functions
# ---------------------------------------------------------
def submit_answer():
    global user_submitted_answer
    ans = answer_entry.get("1.0", tk.END).strip()
    if ans:
        user_submitted_answer = ans
        answer_entry.delete("1.0", tk.END)

def on_close():
    global interview_running, cap
    interview_running = False
    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    root.destroy()

def browse_resume():
    f = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")], title="Select Resume PDF")
    if f:
        resume_entry.delete(0, tk.END)
        resume_entry.insert(0, f)

def stop_interview():
    global interview_running
    if interview_running:
        interview_running = False
        safe_showinfo("Interview Stopped", "Interview was stopped manually.")
        safe_update(start_button, start_button.config, state=tk.NORMAL)
        safe_update(stop_button, stop_button.config, state=tk.DISABLED)

def start_interview():
    global interview_running, cap, candidate_name, job_role, interview_context, multi_face_counter
    if lbph_recognizer is None:
        safe_showerror("Face Not Registered", "Please click 'Register Face' first!")
        return

    pdf_path = resume_entry.get().strip()
    role = role_entry.get().strip()
    if not pdf_path or not os.path.isfile(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        safe_showerror("Error", "Please provide a valid PDF resume.")
        return
    if not role:
        safe_showerror("Error", "Please enter a desired role.")
        return

    txt_ = parse_resume(pdf_path)
    if not txt_:
        safe_showerror("Error", "Couldn't parse the resume. Check the PDF or try again.")
        return

    c_name = extract_candidate_name(txt_)
    candidate_name = c_name

    try:
        res = zero_shot_classifier(txt_, candidate_labels=[role])
        sc = res["scores"][0]
        log_event(f"Zero-shot alignment for role '{role}': {sc:.2f}")
    except Exception as e:
        log_event(f"Zero-shot error: {e}")

    job_role = role
    interview_context = build_context(txt_, job_role)

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    cap = open_camera_for_windows()
    if not cap.isOpened():
        safe_showerror("Camera Error", "Cannot open webcam for interview.")
        return

    multi_face_counter = 0
    interview_running = True
    update_camera_view()
    threading.Thread(target=monitor_webcam, daemon=True).start()
    threading.Thread(target=interview_loop, args=(chat_display,), daemon=True).start()

    safe_update(start_button, start_button.config, state=tk.DISABLED)
    safe_update(stop_button, stop_button.config, state=tk.NORMAL)

def main_app():
    global root
    global bot_label, camera_label, chat_display
    global resume_entry, role_entry
    global start_button, stop_button
    global record_btn, stop_record_btn, submit_btn
    global answer_entry, warning_count_label

    root.title(APP_TITLE)
    root.geometry("1280x820")
    root.configure(bg=MAIN_BG)
    apply_theme()
    root.protocol("WM_DELETE_WINDOW", on_close)

    banner_height = 70
    gradient_canvas = tk.Canvas(root, height=banner_height, bd=0, highlightthickness=0)
    gradient_canvas.pack(fill=tk.X)

    def draw_grad(canv, ww, hh, c1=GRADIENT_START, c2=GRADIENT_END):
        steps = 100
        for i in range(steps):
            ratio = i / steps
            r1, g1, b1 = root.winfo_rgb(c1)
            r2, g2, b2 = root.winfo_rgb(c2)
            rr = int(r1 + (r2 - r1)*ratio) >> 8
            gg = int(g1 + (g2 - g1)*ratio) >> 8
            bb = int(b1 + (b2 - b1)*ratio) >> 8
            color = f"#{rr:02x}{gg:02x}{bb:02x}"
            canv.create_line(int(ww*ratio), 0, int(ww*ratio), hh, fill=color)

    def on_resize(e):
        gradient_canvas.delete("all")
        draw_grad(gradient_canvas, e.width, e.height)

    gradient_canvas.bind("<Configure>", on_resize)
    title_label = tk.Label(gradient_canvas, text=APP_TITLE,
                           font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"), fg=MAIN_FG, bg=None)
    title_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    content_frame = tk.Frame(root, bg=MAIN_BG)
    content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    left_frame = tk.Frame(content_frame, bg=MAIN_BG)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

    bot_frame = tk.Frame(left_frame, bg=MAIN_BG)
    bot_frame.pack(side=tk.TOP, pady=10)

    bot_label = tk.Label(bot_frame, bg=MAIN_BG)
    bot_label.pack()

    global bot_frames
    bot_frames = load_gif_frames(GIF_PATH)
    if bot_frames:
        bot_label.config(image=bot_frames[0])
        bot_label.image = bot_frames[0]

    camera_frame = tk.LabelFrame(left_frame, text="Live Camera Feed",
                                 bg=GLASS_BG, fg=ACCENT_COLOR,
                                 font=(FONT_FAMILY, 12, "bold"), bd=3, labelanchor="n")
    camera_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)

    global camera_label
    camera_label = tk.Label(camera_frame, bg="#000000")
    camera_label.pack(fill=tk.BOTH, expand=True)

    right_frame = tk.Frame(content_frame, bg=MAIN_BG)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

    form_frame = tk.LabelFrame(right_frame, text="Candidate Info",
                               bg=GLASS_BG, fg=ACCENT_COLOR,
                               font=(FONT_FAMILY, 12, "bold"), bd=3, labelanchor="n")
    form_frame.pack(pady=10, fill=tk.X)

    lbl_pdf = tk.Label(form_frame, text="Resume (PDF):", fg=MAIN_FG, bg=GLASS_BG,
                       font=(FONT_FAMILY, FONT_SIZE_NORMAL))
    lbl_pdf.grid(row=0, column=0, padx=5, pady=5, sticky=tk.E)

    resume_entry = tk.Entry(form_frame, width=40)
    resume_entry.grid(row=0, column=1, padx=5, pady=5)

    browse_btn = tk.Button(form_frame, text="Browse", command=browse_resume,
                           bg=BUTTON_BG, fg=BUTTON_FG,
                           font=(FONT_FAMILY, 10, "bold"))
    browse_btn.grid(row=0, column=2, padx=5, pady=5)

    lbl_role = tk.Label(form_frame, text="Desired Role:", fg=MAIN_FG, bg=GLASS_BG,
                        font=(FONT_FAMILY, FONT_SIZE_NORMAL))
    lbl_role.grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)

    role_entry = tk.Entry(form_frame, width=40)
    role_entry.grid(row=1, column=1, padx=5, pady=5)

    reg_face_btn = tk.Button(form_frame, text="Register Face", command=register_candidate_face,
                             bg="#5555CC", fg="#FFFFFF",
                             font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"))
    reg_face_btn.grid(row=1, column=2, padx=5, pady=5)

    global start_button, stop_button
    start_button = tk.Button(form_frame, text="Start Interview", command=start_interview,
                             bg=BUTTON_BG, fg=BUTTON_FG,
                             font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"))
    start_button.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

    stop_button = tk.Button(form_frame, text="Stop Interview", command=stop_interview,
                            bg="#CC0000", fg="#FFFFFF",
                            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
                            state=tk.DISABLED)
    stop_button.grid(row=2, column=2, padx=5, pady=5, sticky=tk.E)

    transcript_frame = tk.LabelFrame(right_frame, text="Interview Transcript",
                                     bg=GLASS_BG, fg=ACCENT_COLOR,
                                     font=(FONT_FAMILY, 12, "bold"), bd=3, labelanchor="n")
    transcript_frame.pack(pady=10, fill=tk.BOTH, expand=True)

    global chat_display
    chat_display = scrolledtext.ScrolledText(transcript_frame, wrap=tk.WORD, width=50, height=15,
                                             bg="#222222", fg=ACCENT_COLOR,
                                             font=("Consolas", 11))
    chat_display.pack(fill=tk.BOTH, expand=True)
    chat_display.config(state=tk.DISABLED)

    warning_count_label = tk.Label(right_frame, text="Camera Warnings: 0",
                               fg=ACCENT_COLOR, bg=MAIN_BG,
                               font=(FONT_FAMILY, FONT_SIZE_NORMAL))
    warning_count_label.pack(pady=5)

    ans_frame = tk.LabelFrame(right_frame, text="Your Answer",
                              bg=GLASS_BG, fg=ACCENT_COLOR,
                              font=(FONT_FAMILY, 12, "bold"), bd=3, labelanchor="n")
    ans_frame.pack(pady=10, fill=tk.X)

    global answer_entry
    answer_entry = tk.Text(ans_frame, width=40, height=3, font=(FONT_FAMILY, FONT_SIZE_NORMAL))
    answer_entry.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
    def on_answer_typing(event):
        global last_input_time
        last_input_time = time.time()
    answer_entry.bind("<Key>", on_answer_typing)

    global submit_btn
    submit_btn = tk.Button(ans_frame, text="Submit Answer", command=submit_answer,
                           bg=BUTTON_BG, fg=BUTTON_FG,
                           font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"))
    submit_btn.pack(side=tk.LEFT, padx=5, pady=5)
    submit_btn.config(state=tk.DISABLED)

    global record_btn, stop_record_btn
    record_btn = tk.Button(ans_frame, text="Record", command=start_recording_voice,
                           bg="#00CC66", fg="#FFFFFF",
                           font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"))
    record_btn.pack(side=tk.LEFT, padx=5, pady=5)
    record_btn.config(state=tk.DISABLED)

    stop_record_btn = tk.Button(ans_frame, text="Stop Recording", command=stop_recording_voice,
                                bg="#FF9900", fg="#FFFFFF",
                                font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"))
    stop_record_btn.pack(side=tk.LEFT, padx=5, pady=5)
    stop_record_btn.config(state=tk.DISABLED)

    root.after(100, animate_bot)

def apply_theme():
    if USE_BOOTSTRAP:
        style = tb.Style('darkly')
        style.configure("TButton", font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"))
    else:
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background=MAIN_BG)
        style.configure("TLabel", background=MAIN_BG, foreground=MAIN_FG)
        style.configure("TButton", background=BUTTON_BG, foreground=BUTTON_FG,
                        font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"))

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
root = tk.Tk()
root.title(APP_TITLE)
root.geometry("1280x820")
root.configure(bg=MAIN_BG)
apply_theme()
root.protocol("WM_DELETE_WINDOW", on_close)
load_model_splash()
root.mainloop()
