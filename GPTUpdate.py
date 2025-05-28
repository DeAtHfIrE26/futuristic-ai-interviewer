import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow and absl log warnings

import logging
logging.getLogger("absl").setLevel(logging.ERROR)

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
import io

import cv2
import fitz  # PyMuPDF
import speech_recognition as sr
import pyttsx3
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw

# New imports for eye tracking, voice embedding, and audio file reading
import mediapipe as mp
import librosa
import soundfile as sf

# Import InsightFace for advanced face recognition
import insightface
from insightface.app import FaceAnalysis

# Import Resemblyzer for voice recognition and anti-spoofing
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import webrtcvad
import wave

# Import pyannote.audio for robust speaker embedding extraction.
from pyannote.audio import Inference

# Import FPDF for PDF report generation
from fpdf import FPDF

# Global variables for voice reference and recognition
reference_voice_embedding = None
reference_audio_data = None
voice_reference_recorded = False
cap = None
camera_label = None

HF_TOKEN = "HF_token"  # Replace with your token
try:
    voice_inference = Inference("pyannote/embedding", device="cpu", use_auth_token=HF_TOKEN)
    logging.info("pyannote.audio model loaded for voice matching.")
except Exception as e:
    logging.error(f"Could not load pyannote.audio model: {e}")
    voice_inference = None

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

# Constant for minimum acceptable mouth movement (for lip-sync)
MIN_MOUTH_MOVEMENT_RATIO = 0.025  # Increased from 0.02 for better lip movement detection

# HF Inference
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
headers = {"Authorization": f"Bearer {HF_AUTH_TOKEN}"}

CONVO_MODEL_NAME = "facebook/blenderbot-400M-distill"
ZS_MODEL_NAME = "facebook/bart-large-mnli"

# Voice recognition and anti-spoofing
voice_encoder = None
vad = webrtcvad.Vad(3)  # Aggressiveness mode 3 (highest)
VOICE_SIMILARITY_THRESHOLD = 0.70  # Lowered from 0.75 for more strict voice verification
VOICE_SAMPLE_RATE = 16000  # Sample rate for voice processing
MIN_VOICE_CONFIDENCE = 0.6  # Minimum confidence for voice verification
MIN_LIP_MOVEMENT_FRAMES_RATIO = 0.2  # Lowered from 0.4 to make registration easier
MIN_LIP_MOVEMENT_VARIANCE = 0.00005  # Lowered from 0.0001 to be more permissive
MIN_PROMPT_SIMILARITY = 0.2  # Lowered from 0.3 to accommodate different speaking styles
VOICE_RECORDING_DURATION = 5  # Duration in seconds for voice recording

# Metrics tracking
current_face_similarity = 0.0
current_face_quality = 0.0
current_voice_similarity = 0.0
registered_face_image = None

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
candidate_name = "Candidate"
job_role = ""
interview_context = ""
user_submitted_answer = None
last_input_time = None

# InsightFace face recognition
face_app = None
face_embeddings = {}
registered_face_id = None
FACE_SIMILARITY_THRESHOLD = 0.6  # Adjusted for better accuracy (0.6 is a good balance)
MIN_FACE_SIZE = 100  # Minimum face size in pixels
MAX_FACE_YAW = 45  # Maximum face rotation in degrees
MIN_FACE_QUALITY = 0.6  # Minimum face quality score

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

# Global warning tracking
warning_count = 0
previous_warning_message = None

# Face mismatch
face_mismatch_counter = 0
FACE_MISMATCH_THRESHOLD = 5

# Eye tracking and voice matching
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
latest_frame = None
eye_away_count = 0
total_eye_checks = 0
previous_engagement_warning = None

# Interactive compiler/coding challenge
compiler_active = False
challenge_submitted = False

# For extended code editor waiting logic
code_editor_last_activity_time = None
CODE_EDIT_TIMEOUT = 15

# Tkinter references
root = None
chat_display = None
start_button = None
stop_button = None
record_btn = None
stop_record_btn = None
bot_label = None
warning_count_label = None

# Code Challenge UI
code_editor = None
run_code_btn = None
code_output = None
language_var = None
compiler_instructions_label = None
submit_btn = None

# ---------------------------------------------------------
# 3) Helper / UI Functions
# ---------------------------------------------------------
def safe_showerror(title, message):
    """Shows an error message in a thread-safe way with enhanced details for camera errors"""
    try:
        # Special handling for camera errors to provide better troubleshooting info
        if "camera" in title.lower() or "camera" in message.lower():
            detailed_message = message + "\n\nTroubleshooting steps:\n"
            detailed_message += "1. Ensure no other application is using your camera\n"
            detailed_message += "2. Check your camera is properly connected\n"
            detailed_message += "3. Verify camera permissions in your system settings\n"
            detailed_message += "4. Try restarting the application\n"
            detailed_message += "5. If problems persist, try using a different camera"
            
            message = detailed_message
        
        def show():
            messagebox.showerror(title, message)
        threading.Thread(target=show).start()
    except Exception as e:
        log_event(f"Error showing error message: {e}")

def safe_showinfo(title, message):
    root.after(0, lambda: messagebox.showinfo(title, message))

def safe_update(widget, func, *args, **kwargs):
    root.after(0, lambda: func(*args, **kwargs))

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
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            log_event(f"HF query error on attempt {attempt}: {e}")
            time.sleep(1)
    return {}

def evaluate_response(question, response, context=""):
    """
    Evaluate the candidate's response and provide constructive feedback or compliments.
    
    This function analyzes the candidate's response to determine its quality and relevance,
    then generates appropriate feedback that is encouraging but also highlights areas for improvement.
    
    Parameters:
    - question: The interview question that was asked
    - response: The candidate's response to the question
    - context: The resume and role context
    
    Returns:
    - A short feedback or compliment (1-2 sentences)
    """
    log_event(f"Evaluating response of length {len(response)}")
    
    # Skip evaluation for very short responses
    if len(response.split()) < 3:
        return ""
    
    # Analyze response characteristics
    response_length = len(response.split())
    has_specific_examples = re.search(r'(for example|instance|specifically|in my experience|I handled|I implemented|I developed)', response.lower())
    has_technical_details = re.search(r'(algorithm|framework|library|system|architecture|database|code|function|method|api|interface)', response.lower())
    
    prompt = (
        f"{context}\n"
        f"Interview Question: {question.replace('Interviewer:', '').strip()}\n"
        f"Candidate's Response: {response}\n\n"
    )
    
    # Tailor the evaluation instructions based on response characteristics
    if response_length > 100 and (has_specific_examples or has_technical_details):
        prompt += (
            "The candidate provided a detailed response. Give a brief, positive comment that acknowledges "
            "a specific strength in their answer (e.g., depth of knowledge, relevant examples, clear explanation). "
            "Keep your comment encouraging and specific to what they said."
        )
    elif response_length < 30:
        prompt += (
            "The candidate provided a brief response. Give a gentle, constructive comment that acknowledges "
            "what they said but encourages more detail or depth. Be tactful and supportive, not critical."
        )
    else:
        prompt += (
            "Provide a balanced comment on the candidate's answer. Acknowledge one strength in their response "
            "and subtly indicate one area where more detail or clarity would be helpful. Keep your comment "
            "professional and constructive."
        )
    
    # Add formatting instructions
    prompt += (
        "\nYour feedback must be exactly 1-2 sentences, conversational in tone, and specific to their answer. "
        "Do not use generic phrases like 'Thank you for sharing' or 'That's interesting'. "
        "Focus on the content and quality of their response."
    )
    
    # Call the API to generate the evaluation
    payload_data = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 100, "temperature": 0.7, "do_sample": True},
        "options": {"wait_for_model": True}
    }
    
    try:
        result = safe_query_hf(payload_data)
        if isinstance(result, list) and "generated_text" in result[0]:
            feedback = result[0]["generated_text"].replace(prompt, "").strip()
            
            # Clean up the feedback
            feedback = re.sub(r'\n.*', '', feedback)  # Keep only the first line
            
            # Ensure the feedback is concise
            sentences = re.split(r'(?<=[.!?])\s+', feedback)
            if len(sentences) > 2:
                feedback = ' '.join(sentences[:2])
                
            log_event("Successfully generated response evaluation")
            return feedback
    except Exception as e:
        log_event(f"Error evaluating response: {e}")
    
    # Fallback feedback if API call fails
    fallback_feedback = [
        "I appreciate your response. Could you elaborate a bit more on the specific techniques you used?",
        "That's a good starting point. Can you share a specific example from your experience?",
        "You've made some interesting points. How did you apply these concepts in your previous work?",
        "Thank you for sharing your perspective. What specific outcomes resulted from this approach?"
    ]
    
    log_event("Using fallback response evaluation")
    return random.choice(fallback_feedback)

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

def generate_interview_question(role, resume_context="", question_level="basic", candidate_response="", challenge_type=None, category=""):
    """
    Enhanced question generation with adaptive difficulty, context-awareness, and diverse question types.
    
    Args:
        role (str): The job role for the interview
        resume_context (str): Text extracted from the candidate's resume
        question_level (str): Difficulty level - 'basic', 'intermediate', 'advanced', or 'expert'
        candidate_response (str): Previous response from the candidate to build follow-up questions
        challenge_type (str): Type of challenge - 'technical', 'behavioral', 'scenario', 'coding'
        category (str): Specific category within the role for targeted questions
    
    Returns:
        str: An interview question appropriate for the specified parameters
    """
    log_event(f"Generating {question_level} question for {role} role")
    
    if not role:
        return "Interviewer: Please tell me about your background and experience."
    
    # Define role-specific patterns and keywords
    role_patterns = {
        "software engineer": ["coding", "programming", "software development", "algorithms", "data structures"],
        "data scientist": ["machine learning", "statistics", "data analysis", "data visualization", "big data"],
        "product manager": ["product strategy", "user experience", "market research", "roadmap", "stakeholder"],
        "ux designer": ["user experience", "design thinking", "wireframes", "prototyping", "user research"],
        "devops engineer": ["ci/cd", "infrastructure", "automation", "cloud", "containers"],
        "data analyst": ["sql", "data visualization", "reporting", "business intelligence", "analytics"],
        "web developer": ["frontend", "backend", "full stack", "frameworks", "web technologies"],
        "project manager": ["project management", "agile", "scrum", "kanban", "resource allocation"],
        "business analyst": ["requirements", "documentation", "use cases", "process analysis", "system analysis"],
        "systems administrator": ["infrastructure", "networks", "security", "troubleshooting", "maintenance"]
    }
    
    # Generic role patterns if the specific role is not found
    default_patterns = ["technical skills", "soft skills", "problem solving", "teamwork", "communication"]
    
    # Find the most relevant role pattern
    relevant_patterns = []
    for key, patterns in role_patterns.items():
        if key.lower() in role.lower():
            relevant_patterns = patterns
            break
    
    if not relevant_patterns:
        # Use default patterns if no specific match
        relevant_patterns = default_patterns
    
    # Define question templates for different levels
    templates = {
        "basic": [
            "Tell me about your experience with {topic}.",
            "How would you describe your proficiency in {topic}?",
            "What do you consider your greatest strength in {topic}?",
            "How have you applied {topic} in your previous roles?",
            "What attracted you to working with {topic}?"
        ],
        "intermediate": [
            "Describe a challenging problem you solved related to {topic}.",
            "How do you stay updated with the latest developments in {topic}?",
            "What's your approach to learning new aspects of {topic}?",
            "Compare and contrast different approaches to {topic} based on your experience.",
            "What are some common pitfalls or challenges you've encountered with {topic}?"
        ],
        "advanced": [
            "Explain how you would implement {topic} in a complex system architecture.",
            "What innovative approaches have you taken with {topic} that improved outcomes?",
            "How would you redesign a system using {topic} to improve its scalability and performance?",
            "Describe a situation where conventional wisdom about {topic} failed and how you adapted.",
            "How do you evaluate the tradeoffs between different {topic} methodologies?"
        ],
        "expert": [
            "What are the cutting-edge developments in {topic} that you believe will revolutionize the field?",
            "Design a complex system architecture that leverages {topic} optimally for scale and performance.",
            "How would you address the fundamental limitations of current {topic} approaches?",
            "Compare the theoretical foundations of different {topic} paradigms and their practical implications.",
            "How would you mentor a team to elevate their expertise in {topic} to an expert level?"
        ]
    }
    
    # Generate follow-up question if we have a previous response
    if candidate_response and len(candidate_response) > 10:
        try:
            return generate_followup_question(
                previous_question="", 
                candidate_response=candidate_response, 
                resume_context=resume_context, 
                role=role
            )
        except Exception as e:
            log_event(f"Error generating follow-up: {e}")
            # Continue with regular question generation if follow-up fails
    
    # Extract relevant keywords from resume context
    keywords = []
    if resume_context:
        # Simple keyword extraction logic
        for pattern in relevant_patterns:
            if pattern.lower() in resume_context.lower():
                keywords.append(pattern)
        
        # Extract technologies and skills (simple heuristic)
        tech_matches = re.findall(r'\b([A-Za-z0-9#+\.]+(?:\s[A-Za-z0-9]+)?)\b', resume_context)
        keywords.extend([t for t in tech_matches if len(t) > 2 and t.lower() not in ["the", "and", "for", "with"]])
    
    # Ensure we have some keywords to work with
    if not keywords:
        keywords = relevant_patterns
    
    # Select topic based on role, resume context, and question level
    if challenge_type == "technical" or (challenge_type is None and random.random() < 0.7):
        # Technical questions
        if question_level in ["advanced", "expert"]:
            # For advanced questions, use more specific technical topics
            potential_topics = [k for k in keywords if k.lower() not in ["experience", "skills", "background"]]
            if not potential_topics:
                potential_topics = relevant_patterns
            topic = random.choice(potential_topics)
            
            # For coding roles, add specific algorithm or data structure challenges
            if any(keyword in role.lower() for keyword in ["software", "programmer", "developer", "coding", "engineer"]):
                coding_challenges = [
                    "implement an efficient algorithm for searching in a sorted array",
                    "design a system that handles high-concurrency web requests",
                    "optimize a database query for large datasets",
                    "implement a scalable caching mechanism",
                    "create a distributed processing pipeline"
                ]
                if random.random() < 0.3:  # 30% chance for coding challenge
                    return f"Interviewer: Could you {random.choice(coding_challenges)}? You can use the code editor to implement your solution."
        else:
            # For basic/intermediate questions, use broader topics
            topic = random.choice(relevant_patterns)
    elif challenge_type == "behavioral":
        # Behavioral questions
        behavioral_questions = [
            "Describe a situation where you had to work under pressure to meet a deadline.",
            "Tell me about a time when you had to resolve a conflict within your team.",
            "Share an example of how you've demonstrated leadership in a previous role.",
            "How do you prioritize tasks when you have multiple competing deadlines?",
            "Describe a situation where you had to adapt to significant changes in the workplace."
        ]
        return f"Interviewer: {random.choice(behavioral_questions)}"
    elif challenge_type == "scenario":
        # Scenario-based questions
        scenario_templates = [
            "Imagine you're working on a project and a key team member suddenly leaves. How would you handle this situation?",
            "Your project is behind schedule due to unforeseen technical challenges. What steps would you take?",
            "A client or stakeholder is unhappy with a deliverable. How would you address their concerns?",
            "You've identified a significant flaw in your team's approach midway through a project. What would you do?",
            "You're assigned to lead a team with members who have more experience than you. How would you approach this?"
        ]
        return f"Interviewer: {random.choice(scenario_templates)}"
    elif challenge_type == "coding":
        # Specific coding challenge
        coding_problems = [
            "Write a function to find the longest substring without repeating characters in a given string.",
            "Implement an algorithm to determine if a string has all unique characters.",
            "Write a function to reverse a linked list.",
            "Create a function to determine if two strings are anagrams.",
            "Implement a function to find the first non-repeated character in a string."
        ]
        difficulty_levels = {
            "basic": ["simple", "straightforward"],
            "intermediate": ["moderately complex", "reasonably challenging"],
            "advanced": ["complex", "sophisticated"],
            "expert": ["highly complex", "extremely challenging"]
        }
        difficulty_descriptor = random.choice(difficulty_levels.get(question_level, ["challenging"]))
        return f"Interviewer: Here's a {difficulty_descriptor} coding problem. {random.choice(coding_problems)} Please use the code editor to implement your solution."
    else:
        # Default to using a relevant pattern as the topic
        topic = random.choice(relevant_patterns)
    
    # Select template based on question level
    if question_level not in templates:
        question_level = "intermediate"  # Default to intermediate if level is invalid
    
    template = random.choice(templates[question_level])
    question = template.format(topic=topic)
    
    # For data analysis roles, potentially ask SQL question
    if any(keyword in role.lower() for keyword in ["data", "analyst", "analytics", "bi", "intelligence"]):
        if random.random() < 0.2:  # 20% chance for SQL question
            sql_questions = [
                "write a SQL query to find the top 5 customers by total purchase amount",
                "create a SQL query to identify products that haven't been sold in the last 30 days",
                "develop a SQL query to calculate monthly sales growth percentage",
                "write a SQL query to find overlapping date ranges between two tables",
                "create a SQL query to identify duplicate records in a customer database"
            ]
            return f"Interviewer: Could you {random.choice(sql_questions)}? You can use the code editor to write your SQL query."
    
    # Ensure the question is formatted properly
    if not question.startswith("Interviewer:"):
        question = f"Interviewer: {question}"
    
    return question

def generate_followup_question(previous_question, candidate_response, resume_context, role):
    for _ in range(3):
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
        if not followup.endswith('?'):
            followup = followup.rstrip(".! ") + "?"
        if len(followup.replace("Interviewer:", "").strip().split()) >= 5:
            return followup
    return "Interviewer: Could you please provide more details?"

def similar_questions(q1, q2):
    """Calculate similarity score between two text strings"""
    q1_words = set(q1.lower().replace("interviewer:", "").strip().split())
    q2_words = set(q2.lower().replace("interviewer:", "").strip().split())
    
    # Remove common stop words
    stop_words = {"a", "an", "the", "in", "on", "at", "to", "for", "with", "about", "you", "your", "can", "could", "would", "please"}
    q1_words = q1_words - stop_words
    q2_words = q2_words - stop_words
    
    # Calculate Jaccard similarity
    if not q1_words or not q2_words:
        return 0.0
        
    intersection = len(q1_words.intersection(q2_words))
    union = len(q1_words.union(q2_words))
    similarity = intersection / union
    
    return similarity  # Return the actual similarity score as a float

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
# 5) Performance Scoring
# ---------------------------------------------------------
def grade_interview_with_breakdown(transcript, context=""):
    """
    A robust, multi-factor grading function that measures the candidate's overall performance.
    This approach covers:

      1) Depth of Response          (thoroughness/length)
      2) Clarity & Organization     (filler words, logical connectors)
      3) Domain Relevance          (keywords matching the role/context)
      4) Confidence                (low frequency of hedging words)
      5) Problem-Solving Approach  (presence of solution-oriented terms)
      6) Teamwork/Collaboration    (mentions 'team', 'we', etc.)
      7) Code Quality Indicators   (technical references if applicable)
      8) Engagement (Eye Tracking) (global eye_away_count vs. total_eye_checks)
      9) Warning Penalties         (phone detection, mismatched voice, etc.)

    Returns:
        final_score (int): The computed interview score [0..100].
        breakdown (dict): Detailed scoring breakdown.
        message (str): Text explaining how the score was calculated.
    """
    import re

    global eye_away_count, total_eye_checks
    global warning_count

    candidate_lines = [line for line in transcript.split("\n") if line.strip().startswith("Candidate:")]
    if not candidate_lines:
        return 0, {"explanation": "No candidate responses found."}, "No candidate lines found."

    # ------------ Helper Sub-scores ------------
    def measure_depth(response):
        words = response.split()
        wc = len(words)
        if wc < 10:
            return 0.3
        elif wc < 50:
            return 0.6
        else:
            return 1.0

    def measure_clarity(response):
        resp_lower = response.lower()
        words = response.split()
        wc = len(words) + 1e-6

        filler_words = ["um", "uh", "er", "ah", "uhm", "like ", "basically", "i mean", "you know"]
        filler_count = sum(resp_lower.count(fw) for fw in filler_words)
        filler_ratio = filler_count / wc
        if filler_ratio < 0.02:
            clarity_factor = 1.0
        elif filler_ratio < 0.05:
            clarity_factor = 0.8
        else:
            clarity_factor = 0.5

        # Check for logical connectors
        connectors = ["first", "second", "next", "finally", "therefore", "thus", "because"]
        if any(conn in resp_lower for conn in connectors):
            clarity_factor = min(1.0, clarity_factor + 0.1)

        return clarity_factor

    def measure_domain_relevance(response, ctx):
        ctx_tokens = set(re.findall(r"[a-zA-Z]+", ctx.lower()))
        resp_tokens = set(re.findall(r"[a-zA-Z]+", response.lower()))
        matches = len(ctx_tokens.intersection(resp_tokens))
        return min(1.0, matches / 5.0)  # up to 5 matches => full score

    def measure_confidence(response):
        resp_lower = response.lower()
        words = response.split()
        wc = len(words) + 1e-6
        disclaimers = ["maybe", "not sure", "i guess", "i think", "probably", "might", "perhaps", "sort of"]
        disc_count = sum(resp_lower.count(d) for d in disclaimers)
        disc_ratio = disc_count / wc
        if disc_ratio < 0.02:
            return 1.0
        elif disc_ratio < 0.05:
            return 0.8
        else:
            return 0.5

    def measure_problem_solving(response):
        keywords = ["approach", "solution", "method", "algorithm", "strategy", "plan", "steps", "test"]
        resp_lower = response.lower()
        found = sum(1 for kw in keywords if kw in resp_lower)
        return min(1.0, found / 5.0)

    def measure_teamwork(response):
        teamwork_terms = ["team", "collaborate", "we ", "our ", "together", "partner", "collective"]
        resp_lower = response.lower()
        found = sum(resp_lower.count(tw) for tw in teamwork_terms)
        if found == 0:
            return 0.2
        elif found == 1:
            return 0.5
        elif found == 2:
            return 0.8
        else:
            return 1.0

    def measure_code_quality(response):
        code_terms = ["function", "class", "variable", "data structure", "big-o",
                      "complexity", "optimize", "testing", "compile", "exception"]
        resp_lower = response.lower()
        found = sum(1 for term in code_terms if term in resp_lower)
        return min(1.0, found / 5.0)

    if total_eye_checks == 0:
        engagement_factor = 1.0
    else:
        away_ratio = eye_away_count / float(total_eye_checks)
        engagement_factor = max(0.0, 1.0 - away_ratio)

    total_score = 0.0
    breakdown_per_response = []
    for line in candidate_lines:
        response = line.replace("Candidate:", "").strip()

        depth      = measure_depth(response)
        clarity    = measure_clarity(response)
        domain     = measure_domain_relevance(response, context)
        confidence = measure_confidence(response)
        probsolve  = measure_problem_solving(response)
        teamwork   = measure_teamwork(response)
        codequal   = measure_code_quality(response)

        single_score = (
            depth      * 14.3 +
            clarity    * 14.3 +
            domain     * 14.3 +
            confidence * 14.3 +
            probsolve  * 14.3 +
            teamwork   * 14.3 +
            codequal   * 14.3
        )

        total_score += single_score

        breakdown_per_response.append({
            "response_snippet": response[:60] + ("..." if len(response) > 60 else ""),
            "depth": round(depth, 2),
            "clarity": round(clarity, 2),
            "domain_relevance": round(domain, 2),
            "confidence": round(confidence, 2),
            "problem_solving": round(probsolve, 2),
            "teamwork": round(teamwork, 2),
            "code_quality": round(codequal, 2),
            "response_score": round(single_score, 2)
        })

    num_responses = len(candidate_lines)
    avg_score_before_engagement = total_score / num_responses if num_responses else 0.0

    final_raw_score = avg_score_before_engagement * engagement_factor

    base_score = min(100, max(0, final_raw_score))

    # Subtract warnings
    final_score = max(0, base_score - warning_count)

    explanation = (
        "Multi-factor Scoring:\n"
        "1) Depth of Response\n"
        "2) Clarity & Organization\n"
        "3) Domain Relevance\n"
        "4) Confidence (fewer hedging terms)\n"
        "5) Problem-Solving Approach\n"
        "6) Teamwork Indicators\n"
        "7) Code Quality References\n"
        "8) Engagement (Eye Contact)\n"
        "9) Warning Penalties\n"
        "These factors are combined into a weighted average, then penalties are subtracted."
    )

    breakdown = {
        "depth": round(sum(br["depth"] for br in breakdown_per_response) / num_responses * 14.3, 1),
        "clarity": round(sum(br["clarity"] for br in breakdown_per_response) / num_responses * 14.3, 1),
        "domain_relevance": round(sum(br["domain_relevance"] for br in breakdown_per_response) / num_responses * 14.3, 1),
        "confidence": round(sum(br["confidence"] for br in breakdown_per_response) / num_responses * 14.3, 1),
        "problem_solving": round(sum(br["problem_solving"] for br in breakdown_per_response) / num_responses * 14.3, 1),
        "teamwork": round(sum(br["teamwork"] for br in breakdown_per_response) / num_responses * 14.3, 1),
        "code_quality": round(sum(br["code_quality"] for br in breakdown_per_response) / num_responses * 14.3, 1),
        "engagement": round(engagement_factor * 10, 1),
        "warning_penalty": warning_count,
        "base_score": round(base_score, 1),
        "final_score": round(final_score, 1),
        "explanation": explanation,
        "per_response": breakdown_per_response
    }

    return round(final_score), breakdown, explanation

def get_level_description(score_ratio):
    """Helper function to convert score ratios to descriptive text"""
    if score_ratio >= 0.9:
        return "excellent"
    elif score_ratio >= 0.8:
        return "very good"
    elif score_ratio >= 0.7:
        return "good"
    elif score_ratio >= 0.6:
        return "above average"
    elif score_ratio >= 0.5:
        return "average"
    elif score_ratio >= 0.4:
        return "below average"
    elif score_ratio >= 0.3:
        return "fair"
    elif score_ratio >= 0.2:
        return "poor"
    else:
        return "very poor"

# ---------------------------------------------------------
# 6) Resume Parsing
# ---------------------------------------------------------
def parse_resume(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        log_event(f"Successfully parsed resume with {len(text)} characters")
    except Exception as e:
        log_event(f"PDF parse error: {e}")
    return text

def extract_candidate_name(resume_text):
    lines = [ln.strip() for ln in resume_text.split("\n") if ln.strip()]
    if lines:
        first_line = lines[0]
        if re.match(r"^[A-Za-z\s.'-]+$", first_line) and len(first_line.split()) <= 4:
            return first_line.strip()
    
    # Try to find name patterns if first line approach fails
    name_patterns = [
        r"(?:Name|NAME)[:\s]+([A-Za-z\s.'-]+)",
        r"^([A-Za-z\s.'-]{2,30})$",
        r"^([A-Za-z\s.'-]{2,30})\s*[\|\-•]"
    ]
    
    for pattern in name_patterns:
        matches = re.search(pattern, resume_text[:500])
        if matches:
            return matches.group(1).strip()
            
    return "Candidate"

def extract_resume_sections(resume_text):
    """Extract different sections from the resume for better context building"""
    sections = {}
    
    # Common section headers in resumes
    section_patterns = {
        "education": r"(?:EDUCATION|Education|Academic Background)[\s\:]*?(.*?)(?:EXPERIENCE|Experience|SKILLS|Skills|PROJECTS|Projects|CERTIFICATIONS|Certifications|$)",
        "experience": r"(?:EXPERIENCE|Experience|Work Experience|WORK EXPERIENCE)[\s\:]*?(.*?)(?:EDUCATION|Education|SKILLS|Skills|PROJECTS|Projects|CERTIFICATIONS|Certifications|$)",
        "skills": r"(?:SKILLS|Skills|Technical Skills|TECHNICAL SKILLS)[\s\:]*?(.*?)(?:EDUCATION|Education|EXPERIENCE|Experience|PROJECTS|Projects|CERTIFICATIONS|Certifications|$)",
        "projects": r"(?:PROJECTS|Projects|PERSONAL PROJECTS)[\s\:]*?(.*?)(?:EDUCATION|Education|EXPERIENCE|Experience|SKILLS|Skills|CERTIFICATIONS|Certifications|$)",
        "certifications": r"(?:CERTIFICATIONS|Certifications|CERTIFICATES)[\s\:]*?(.*?)(?:EDUCATION|Education|EXPERIENCE|Experience|SKILLS|Skills|PROJECTS|Projects|$)"
    }
    
    for section_name, pattern in section_patterns.items():
        matches = re.search(pattern, resume_text, re.DOTALL)
        if matches:
            content = matches.group(1).strip()
            # Limit section content to reasonable size
            if len(content) > 500:
                content = content[:500] + "..."
            sections[section_name] = content
    
    return sections

def build_context(resume_text, role):
    """Build a comprehensive context from the resume for better question generation"""
    c_name = extract_candidate_name(resume_text)
    sections = extract_resume_sections(resume_text)
    
    # Create a structured context with resume sections
    context = f"Candidate's Desired Role: {role}\n"
    context += f"Candidate Name: {c_name}\n\n"
    
    # Add each section to the context
    for section_name, content in sections.items():
        if content:
            context += f"{section_name.upper()}:\n{content}\n\n"
    
    # If we couldn't extract structured sections, use a summary
    if not sections:
        length = 800
        summary = resume_text[:length].replace("\n", " ")
        if len(resume_text) > length:
            summary += "..."
        context += f"Resume Summary: {summary}\n\n"
    
    # Add instructions for the interviewer
    context += (
        "You are a professional interviewer conducting a technical interview. "
        "Ask relevant questions based on the candidate's resume and desired role. "
        "Progress from basic to advanced questions based on the candidate's responses. "
        "For technical roles, include scenario-based questions and eventually a coding challenge. "
        "For data-related roles, include SQL or data analysis questions."
    )
    
    log_event(f"Built context with {len(context)} characters")
    return context

# ---------------------------------------------------------
# 7) TTS / STT
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

def compute_voice_embedding(audio_data, sample_rate=16000):
    """
    Compute voice embedding using Resemblyzer for better voice recognition and anti-spoofing.
    Falls back to pyannote.audio and then to librosa if Resemblyzer fails.
    
    Args:
        audio_data: WAV audio data bytes
        sample_rate: Sample rate of the audio
        
    Returns:
        numpy array: Voice embedding vector
    """
    global voice_encoder
    
    # Initialize Resemblyzer if not already done
    if voice_encoder is None:
        try:
            voice_encoder = VoiceEncoder()
            log_event("Resemblyzer voice encoder initialized successfully")
        except Exception as e:
            log_event(f"Failed to initialize Resemblyzer: {e}")
    
    # Try Resemblyzer first (best for voice verification)
    if voice_encoder is not None:
        try:
            # Convert audio bytes to wav numpy array
            wav_bytes = io.BytesIO(audio_data)
            wav, sr = sf.read(wav_bytes)
            
            # Resample if needed
            if sr != sample_rate:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
                
            # Preprocess for Resemblyzer
            wav = preprocess_wav(wav)
            
            # Get embedding
            embedding = voice_encoder.embed_utterance(wav)
            return embedding
        except Exception as e:
            log_event(f"Resemblyzer embedding error: {e}")
    
    # Fall back to pyannote.audio
    if voice_inference is not None:
        try:
            wav_bytes = io.BytesIO(audio_data)
            wav, sr = sf.read(wav_bytes)
            if sr != sample_rate:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
            embedding = voice_inference(wav)
            if embedding.ndim > 1:
                embedding = np.mean(embedding, axis=0)
            return embedding
        except Exception as e:
            log_event(f"Voice embedding error with pyannote.audio: {e}")
    
    # Last resort: use librosa MFCCs
    try:
        y, sr = librosa.load(io.BytesIO(audio_data), sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfcc, axis=1)
    except Exception as e:
        log_event(f"Voice embedding fallback error: {e}")
        return None

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-6)

def detect_eye_gaze(frame, face_mesh):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if not results.multi_face_landmarks:
        return None
    for face_landmarks in results.multi_face_landmarks:
        h, w, _ = frame.shape
        def landmark_to_point(landmark):
            return np.array([landmark.x * w, landmark.y * h])
        left_iris = [landmark_to_point(face_landmarks.landmark[i]) for i in range(468, 473)]
        left_corner_left = landmark_to_point(face_landmarks.landmark[33])
        left_corner_right = landmark_to_point(face_landmarks.landmark[133])
        right_iris = [landmark_to_point(face_landmarks.landmark[i]) for i in range(473, 478)]
        right_corner_left = landmark_to_point(face_landmarks.landmark[263])
        right_corner_right = landmark_to_point(face_landmarks.landmark[362])
        left_iris_center = np.mean(left_iris, axis=0)
        right_iris_center = np.mean(right_iris, axis=0)
        left_ratio = (left_iris_center[0] - left_corner_left[0]) / (left_corner_right[0] - left_corner_left[0] + 1e-6)
        right_ratio = (right_iris_center[0] - right_corner_left[0]) / (right_corner_right[0] - right_corner_left[0] + 1e-6)
        threshold = 0.20
        if abs(left_ratio - 0.5) > threshold or abs(right_ratio - 0.5) > threshold:
            return False
        else:
            return True
    return None

def compute_mouth_opening(frame, face_mesh):
    """
    Enhanced function to compute mouth opening and movement metrics.
    Returns the ratio of mouth opening to face height and additional metrics.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return None
        
    for face_landmarks in results.multi_face_landmarks:
        h, w, _ = frame.shape
        def to_point(landmark): 
            return np.array([landmark.x * w, landmark.y * h])
            
        # Get key mouth landmarks
        upper_lip = to_point(face_landmarks.landmark[13])  # Upper lip
        lower_lip = to_point(face_landmarks.landmark[14])  # Lower lip
        
        # Get additional mouth landmarks for better analysis
        left_mouth = to_point(face_landmarks.landmark[78])   # Left corner of mouth
        right_mouth = to_point(face_landmarks.landmark[308]) # Right corner of mouth
        
        # Face height reference points
        face_top = to_point(face_landmarks.landmark[10])     # Top of face
        face_bottom = to_point(face_landmarks.landmark[152]) # Bottom of face
        
        # Calculate vertical mouth opening
        mouth_opening = np.linalg.norm(upper_lip - lower_lip)
        
        # Calculate horizontal mouth width
        mouth_width = np.linalg.norm(left_mouth - right_mouth)
        
        # Calculate face height for normalization
        face_height = np.linalg.norm(face_top - face_bottom)
        
        # Calculate normalized ratios
        vertical_ratio = mouth_opening / (face_height + 1e-6)
        width_ratio = mouth_width / (face_height + 1e-6)
        
        # Calculate mouth area (approximated as ellipse area)
        mouth_area = np.pi * (mouth_opening/2) * (mouth_width/2)
        face_area = face_height * face_height  # Approximation
        area_ratio = mouth_area / (face_area + 1e-6)
        
        # Draw mouth landmarks on frame for visualization
        cv2.circle(frame, (int(upper_lip[0]), int(upper_lip[1])), 2, (0, 255, 0), -1)
        cv2.circle(frame, (int(lower_lip[0]), int(lower_lip[1])), 2, (0, 255, 0), -1)
        cv2.circle(frame, (int(left_mouth[0]), int(left_mouth[1])), 2, (0, 255, 0), -1)
        cv2.circle(frame, (int(right_mouth[0]), int(right_mouth[1])), 2, (0, 255, 0), -1)
        
        # Draw mouth opening visualization
        cv2.line(frame, 
                (int(upper_lip[0]), int(upper_lip[1])), 
                (int(lower_lip[0]), int(lower_lip[1])), 
                (0, 255, 0), 1)
        
        # Return the vertical ratio as the primary metric (for backward compatibility)
        # In a real implementation, you might want to use a combination of metrics
        return vertical_ratio
        
    return None

def monitor_background_audio():
    global interview_running
    with sr.Microphone() as source:
        while interview_running:
            try:
                audio = recognizer.listen(source, phrase_time_limit=5)
                raw_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                energy = np.sqrt(np.mean(raw_data.astype(np.float32)**2))
                if energy > 1000:
                    safe_showinfo("Background Noise Warning", "Excessive background noise detected. Please ensure a quiet environment.")
            except Exception:
                pass
            time.sleep(5)

# ---------------------------------------------------------
# 8) InsightFace Face Recognition
# ---------------------------------------------------------
def open_camera_for_windows(index=0):
    """
    Opens a camera device with enhanced error handling and fallback options.
    Tries multiple camera indices if the first one fails.
    """
    if platform.system() == "Windows":
        # Try the specified index first
        cap_temp = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap_temp.isOpened():
            return cap_temp
        
        # If primary camera fails, try alternative indices
        for alt_index in [0, 1, 2]:
            if alt_index == index:
                continue  # Skip the already tried index
            try:
                log_event(f"Trying camera at index {alt_index}")
                cap_temp = cv2.VideoCapture(alt_index, cv2.CAP_DSHOW)
                if cap_temp.isOpened():
                    log_event(f"Successfully opened camera at index {alt_index}")
                    return cap_temp
            except Exception as e:
                log_event(f"Failed to open camera at index {alt_index}: {e}")
        
        # If all attempts fail, log the error
        log_event("Failed to open any camera device")
        return None
    else:
        # Non-Windows platforms
        try:
            cap_temp = cv2.VideoCapture(index)
            if not cap_temp.isOpened():
                # Try alternative index
                cap_temp = cv2.VideoCapture(0)
            return cap_temp
        except Exception as e:
            log_event(f"Camera error: {e}")
            return None

def initialize_insightface():
    """Initialize the InsightFace model for face recognition"""
    global face_app
    try:
        # Initialize FaceAnalysis with detection and recognition models
        face_app = FaceAnalysis(
            providers=['CPUExecutionProvider'], 
            allowed_modules=['detection', 'recognition'],
            name='buffalo_l'  # Use the high-quality model
        )
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Test the model with a small blank image to ensure it's working
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            face_app.get(test_img)
            log_event("InsightFace model initialized and tested successfully.")
            return True
        except Exception as e:
            log_event(f"InsightFace model test failed: {e}")
            face_app = None
            return False
            
    except Exception as e:
        log_event(f"InsightFace initialization error: {e}")
        safe_showerror("Model Error", f"Failed to initialize face recognition model: {e}")
        face_app = None
        return False

def cleanup_face_recognition():
    """Cleanup face recognition resources"""
    global face_app, face_embeddings, registered_face_id
    try:
        if face_app is not None:
            # Clear any cached data
            face_app = None
        face_embeddings = {}
        registered_face_id = None
        log_event("Face recognition resources cleaned up.")
    except Exception as e:
        log_event(f"Error during face recognition cleanup: {e}")

def capture_face_samples(sample_count=15, delay=0.1):
    """Capture face samples using InsightFace for registration with enhanced quality checks"""
    global face_app, camera_label, registered_face_image
    
    # Initialize InsightFace if not already done
    if face_app is None:
        if not initialize_insightface():
            return [], []
    
    face_embeddings = []
    face_images = []  # Store face images for report
    cap_temp = open_camera_for_windows(0)
    if not cap_temp.isOpened():
        safe_showerror("Webcam Error", "Cannot open camera for face registration.")
        return [], []
    
    collected = 0
    start_time = time.time()
    
    # Create a progress window
    progress_window = tk.Toplevel()
    progress_window.title("Face Registration")
    progress_window.geometry("500x400")
    progress_window.configure(bg=MAIN_BG)
    
    # Add progress label
    progress_label = tk.Label(progress_window, 
                            text=f"Registering face... (0/{sample_count})",
                            bg=MAIN_BG, fg=MAIN_FG,
                            font=(FONT_FAMILY, 12))
    progress_label.pack(pady=10)
    
    # Add instructions
    instructions_label = tk.Label(progress_window,
                                text="Please look directly at the camera.\nEnsure good lighting and remove glasses if possible.",
                                bg=MAIN_BG, fg=ACCENT_COLOR,
                                font=(FONT_FAMILY, 10))
    instructions_label.pack(pady=5)
    
    # Add quality feedback label
    quality_label = tk.Label(progress_window,
                            text="Face quality: Waiting for face...",
                            bg=MAIN_BG, fg=MAIN_FG,
                            font=(FONT_FAMILY, 10))
    quality_label.pack(pady=5)
    
    # Add preview area
    preview_label = tk.Label(progress_window, bg="black")
    preview_label.pack(pady=10)
    
    # Add cancel button
    def cancel_registration():
        nonlocal cancelled
        cancelled = True
        progress_window.destroy()
    
    cancelled = False
    cancel_btn = tk.Button(progress_window, text="Cancel", 
                        command=cancel_registration,
                        bg=BUTTON_BG, fg=BUTTON_FG)
    cancel_btn.pack(pady=10)
    
    prev_time = time.time()
    face_quality_threshold = 0.65  # Threshold for face quality (higher = better)
    min_face_size = 120  # Minimum face size for quality sample
    
    while collected < sample_count and time.time() - start_time < 60 and not cancelled:
        ret, frame = cap_temp.read()
        if not ret:
            continue
            
        # Flip horizontally for natural mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame with InsightFace
        try:
            faces = face_app.get(frame)
            
            if len(faces) == 1:
                face = faces[0]
                bbox = face.bbox.astype(np.int32)
                
                # Calculate face size and quality metrics
                face_size = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
                face_quality = face.det_score  # Detection confidence as quality metric
                
                # Draw rectangle around face
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Display quality information
                quality_text = f"Face quality: {face_quality:.2f} (min: {face_quality_threshold:.2f})"
                quality_label.config(text=quality_text)
                
                # Update UI with current status
                if face_size < min_face_size:
                    cv2.putText(frame, "Move closer", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    quality_label.config(text=f"Face too small - Please move closer to the camera")
                elif face_quality < face_quality_threshold:
                    cv2.putText(frame, "Improve lighting", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    quality_label.config(text=f"Low quality - Check lighting and face position")
                else:
                    # Only collect samples at regular intervals
                    current_time = time.time()
                    if current_time - prev_time >= delay and face_quality >= face_quality_threshold and face_size >= min_face_size:
                        # Extract embedding and save it
                        embedding = face.embedding
                        face_embeddings.append(embedding)
                        
                        # Save the face image for reference
                        face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
                        face_images.append(face_img)
                        
                        # If this is the first face, save as registered face image
                        if collected == 0:
                            registered_face_image = face_img.copy()
                            
                        collected += 1
                        prev_time = current_time
                        
                        # Update progress
                        progress_label.config(text=f"Registering face... ({collected}/{sample_count})")
                        cv2.putText(frame, f"Sample {collected}/{sample_count}", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif len(faces) > 1:
                cv2.putText(frame, "Multiple faces detected!", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                quality_label.config(text="Multiple faces detected - Please ensure only you are in the frame")
            else:
                cv2.putText(frame, "No face detected", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                quality_label.config(text="No face detected - Please look at the camera")
        
        except Exception as e:
            log_event(f"Error during face registration: {e}")
            
        # Show the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((320, 240))
        imgtk = ImageTk.PhotoImage(image=img)
        preview_label.imgtk = imgtk
        preview_label.configure(image=imgtk)
        
        # Update window
        progress_window.update()
            
    # Clean up
    cap_temp.release()
    progress_window.destroy()
    
    # Save face embeddings to file if enough were collected
    if collected >= sample_count * 0.8:  # At least 80% of requested samples
        face_dir = "face_samples"
        if not os.path.exists(face_dir):
            os.makedirs(face_dir)
            
        # Save mean embedding
        if face_embeddings:
            mean_embedding = np.mean(face_embeddings, axis=0)
            np.save(os.path.join(face_dir, "reference_face.npy"), mean_embedding)
            
            # Save a representative face image for the report
            if registered_face_image is not None:
                cv2.imwrite(os.path.join(face_dir, "reference_face.jpg"), registered_face_image)
                
        log_event(f"Collected {collected} face samples successfully")
        return face_embeddings, face_images
    else:
        if not cancelled:
            safe_showerror("Registration Failed", 
                        f"Could only collect {collected}/{sample_count} face samples. Please try again with better lighting and positioning.")
        log_event(f"Face registration failed, only collected {collected}/{sample_count} samples")
        return [], []

def register_candidate_face():
    """Register the candidate's face using InsightFace"""
    global face_embeddings, registered_face_id, registered_face_image
    
    # Generate a unique ID for this registration
    registered_face_id = str(uuid.uuid4())
    
    # Capture face samples
    embeddings, face_images = capture_face_samples(sample_count=10, delay=0.1)
    if not embeddings:
        safe_showerror("Registration Error", "No face samples collected. Check lighting/camera and try again.")
        return False
    
    try:
        # Store the embeddings
        if len(embeddings) > 0:
            # Calculate the average embedding for more robust recognition
            avg_embedding = np.mean(embeddings, axis=0)
            face_embeddings[registered_face_id] = avg_embedding
            
            # Store the best face image for the report
            if face_images:
                # Select the largest face image as the best one
                best_image_idx = np.argmax([img.shape[0] * img.shape[1] for img in face_images])
                registered_face_image = face_images[best_image_idx]
                
                # Save the face image for later use in the report
                face_dir = "face_samples"
                if not os.path.exists(face_dir):
                    os.makedirs(face_dir)
                face_path = os.path.join(face_dir, f"{registered_face_id}.jpg")
                cv2.imwrite(face_path, registered_face_image)
            
            safe_showinfo("Face Registration", "Face registered successfully!")
            log_event("Face registered successfully with InsightFace.")
            return True
        else:
            raise ValueError("No valid face embeddings captured")
    except Exception as e:
        safe_showerror("Registration Error", f"Face registration failed: {e}")
        log_event(f"Face registration error: {e}")
        return False

# ---------------------------------------------------------
# 9) YOLO Phone Detection
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
# 10) Monitoring (Webcam, Face & Eye Tracking)
# ---------------------------------------------------------
def check_same_person_and_phone(frame):
    """
    Enhanced comprehensive monitoring system for interview integrity:
    1) Face identity verification using InsightFace
    2) Phone and electronic device detection using YOLO
    3) Gaze tracking to ensure candidate is looking at the camera
    4) Environment quality assessment
    5) Suspicious behavior detection
    
    Returns a tuple of (is_valid, warning_message, detailed_metrics)
    """
    global face_embeddings, registered_face_id, multi_face_counter, phone_detect_counter
    global current_face_similarity, current_face_quality
    global warning_count, warning_labels

    # Initialize metrics dictionary to track all assessment factors
    metrics = {
        "face_detected": False,
        "face_count": 0,
        "is_registered_person": False,
        "face_similarity": 0.0,
        "face_quality": 0.0,
        "phone_detected": False,
        "looking_away": False,
        "poor_lighting": False,
        "suspicious_activity": False,
        "warnings": []
    }
    
    # 1) Environment quality check
    try:
        # Calculate the average brightness of the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 50:  # Threshold for dark environments
            metrics["poor_lighting"] = True
            metrics["warnings"].append("Poor lighting conditions")
    except Exception as e:
        log_event(f"Brightness check error: {e}")

    # 2) Phone detection
    if detect_phone_in_frame(frame):
        phone_detect_counter += 1
        if phone_detect_counter >= PHONE_DETECT_THRESHOLD:
            metrics["phone_detected"] = True
            metrics["warnings"].append("Phone or electronic device detected")
    else:
        phone_detect_counter = max(0, phone_detect_counter - 1)  # Gradual reduction

    # 3) Make sure InsightFace is ready
    if face_app is None:
        if not initialize_insightface():
            metrics["warnings"].append("Face recognition model not initialized")
            return (False, "Face recognition model not initialized", metrics)
    
    # Skip identity check if no reference face is registered
    if not registered_face_id or registered_face_id not in face_embeddings:
        metrics["warnings"].append("No face registered")
        return (False, "No face registered. Register face first.", metrics)

    # 4) Face detection and recognition
    try:
        faces = face_app.get(frame)
        metrics["face_count"] = len(faces)
        
        # Handle case with no faces
        if len(faces) == 0:
            current_face_similarity = 0.0
            current_face_quality = 0.0
            metrics["warnings"].append("No face detected")
            return (False, "No face detected.", metrics)
            
        # Handle multiple faces
        if len(faces) > 1:
            multi_face_counter += 1
            if multi_face_counter >= MULTI_FACE_THRESHOLD:
                metrics["warnings"].append("Multiple faces detected")
                return (False, "Multiple people detected in frame.", metrics)
        else:
            multi_face_counter = 0
            metrics["face_detected"] = True
            
        # Get the best matching face
        registered_embedding = face_embeddings[registered_face_id]
        best_similarity = 0
        best_quality = 0
        best_face = None
        
        for face in faces:
            # Calculate similarity with registered face
            similarity = cosine_similarity(face.embedding, registered_embedding)
            quality = face.det_score  # Detection confidence
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_quality = quality
                best_face = face
        
        # Update metrics
        current_face_similarity = best_similarity
        current_face_quality = best_quality
        metrics["face_similarity"] = best_similarity
        metrics["face_quality"] = best_quality
        
        # Check if it's the same person (with adaptive threshold based on quality)
        # Lower quality detections require higher similarity for confidence
        similarity_threshold = 0.7 if best_quality < 0.65 else 0.65
        
        if best_similarity < similarity_threshold:
            metrics["warnings"].append(f"Person identity verification failed (similarity: {best_similarity:.2f})")
            return (False, "Different person detected.", metrics)
        
        metrics["is_registered_person"] = True
        
        # 5) Gaze tracking with MediaPipe
        if best_face is not None:
            try:
                mp_face_mesh = mp.solutions.face_mesh
                with mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                ) as face_mesh:
                    # Check if person is looking at camera
                    gaze_result = detect_eye_gaze(frame, face_mesh)
                    if gaze_result is False:  # Only if explicitly looking away
                        metrics["looking_away"] = True
                        metrics["warnings"].append("Looking away from camera")
                    
                    # Check for mouth movement if recording
                    if is_recording_voice:
                        mouth_metrics = compute_mouth_opening(frame, face_mesh)
                        if mouth_metrics is not None:
                            mouth_ratio, _ = mouth_metrics
                            if mouth_ratio < MIN_MOUTH_MOVEMENT_RATIO:
                                metrics["suspicious_activity"] = True
                                metrics["warnings"].append("Insufficient lip movement while speaking")
            except Exception as e:
                log_event(f"Gaze tracking error: {e}")
    
    except Exception as e:
        log_event(f"Face recognition error: {e}")
        metrics["warnings"].append(f"Face recognition error: {e}")
        return (False, "Face recognition error occurred.", metrics)
    
    # 6) Final decision
    is_valid = (
        metrics["face_detected"] and 
        metrics["is_registered_person"] and 
        not metrics["phone_detected"] and 
        not metrics["suspicious_activity"]
    )
    
    # Update warning count if needed
    if not is_valid and metrics["warnings"]:
        # Only count each type of warning once per detection cycle
        unique_warnings = set(metrics["warnings"])
        for warning in unique_warnings:
            if warning not in warning_labels:
                warning_count += 1
                warning_labels.append(warning)
                log_event(f"Warning #{warning_count}: {warning}")
    
    message = "Verification successful." if is_valid else metrics["warnings"][0] if metrics["warnings"] else "Verification failed."
    return (is_valid, message, metrics)

def monitor_webcam():
    """
    Enhanced webcam monitoring with comprehensive security and identity verification.
    Includes:
    - Face identity verification
    - Phone detection
    - Eye gaze tracking
    - Behavioral anomaly detection
    - Visual warning indicators
    """
    global cap, warning_count, camera_label, metrics_label, warning_count_label, warning_labels
    global current_face_similarity, current_face_quality, registered_face_image
    global multi_face_counter, phone_detect_counter
    
    # Initialize counters for consecutive detections
    multi_face_counter = 0
    phone_detect_counter = 0
    warning_labels = []  # Store unique warning labels
    
    # Set up warning display format
    if warning_count_label is not None:
        warning_count_label.config(text="Warnings: 0")
    
    if cap is None or not cap.isOpened():
        try:
            cap = open_camera_for_windows()
            if cap is None or not cap.isOpened():
                log_event("Failed to open webcam for monitoring")
                if camera_label:
                    camera_label.config(image="")
                    camera_label.image = None
                return
        except Exception as e:
            log_event(f"Camera error: {e}")
            return
    
    log_event("Webcam monitoring started")
    
    try:
        while interview_running and cap is not None and cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Process frame for verification
                status, reason, metrics = check_same_person_and_phone(frame)
                
                # Create visual indicators for the frame
                # Draw face similarity meter
                frame_height, frame_width = frame.shape[:2]
                meter_width = int(frame_width * 0.2)
                meter_height = 15
                meter_x = 10
                meter_y = frame_height - meter_height - 10
                
                # Base meter background
                cv2.rectangle(frame, 
                    (meter_x, meter_y), 
                    (meter_x + meter_width, meter_y + meter_height), 
                    (50, 50, 50), -1)
                
                # Face similarity fill
                fill_width = int(meter_width * metrics["face_similarity"])
                color = (0, 255, 0) if metrics["face_similarity"] >= 0.65 else (0, 165, 255) if metrics["face_similarity"] >= 0.5 else (0, 0, 255)
                cv2.rectangle(frame, 
                    (meter_x, meter_y), 
                    (meter_x + fill_width, meter_y + meter_height), 
                    color, -1)
                
                # Label
                cv2.putText(frame, f"Identity: {metrics['face_similarity']:.2f}", 
                    (meter_x, meter_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display warnings on frame
                y_offset = 30
                if metrics["warnings"]:
                    for i, warning in enumerate(metrics["warnings"][:3]):  # Show up to 3 warnings
                        cv2.putText(frame, warning, 
                            (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        y_offset += 25
                
                # Convert frame to format for Tkinter
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                img = ImageTk.PhotoImage(image=img)
                
                # Update camera display
                if camera_label is not None:
                    camera_label.config(image=img)
                    camera_label.image = img
                
                # Update metrics display with comprehensive information
                if metrics_label is not None:
                    metrics_text = (
                        f"Identity Match: {metrics['face_similarity']:.2f}\n"
                        f"Face Quality: {metrics['face_quality']:.2f}\n"
                        f"Status: {'✓ Verified' if status else '✗ Failed'}"
                    )
                    metrics_label.config(text=metrics_text)
                
                # Update warning count display
                if warning_count_label is not None:
                    warning_count_label.config(text=f"Warnings: {warning_count}")
                    
                # Brief pause to reduce CPU usage
                time.sleep(0.03)  # ~30 fps
                
            except Exception as e:
                log_event(f"Error in webcam monitoring: {e}")
                time.sleep(0.1)
                
    except Exception as e:
        log_event(f"Webcam monitor thread error: {e}")
    finally:
        log_event("Webcam monitoring stopped")

def update_camera_view():
    global interview_running, cap, latest_frame
    if not interview_running or cap is None or not cap.isOpened():
        return
    ret, frame = cap.read()
    if ret:
        latest_frame = frame.copy()
        
        # Detect faces using InsightFace and draw bounding boxes
        if face_app is not None:
            try:
                faces = face_app.get(frame)
                for face in faces:
                    bbox = face.bbox.astype(int)
                    # Draw rectangle around face
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    
                    # If this is the registered user, add a label
                    if registered_face_id in face_embeddings:
                        registered_embedding = face_embeddings[registered_face_id]
                        embedding = face.embedding
                        similarity = np.dot(embedding, registered_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(registered_embedding))
                        
                        if similarity >= FACE_SIMILARITY_THRESHOLD:
                            # Draw label for recognized user
                            label = f"Registered ({similarity:.2f})"
                            color = (0, 255, 0)  # Green
                        else:
                            # Draw label for unrecognized face
                            label = f"Unknown ({similarity:.2f})"
                            color = (0, 0, 255)  # Red
                            
                        # Add label with background for better visibility
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, 
                                    (bbox[0], bbox[1] - 20), 
                                    (bbox[0] + label_size[0], bbox[1]), 
                                    color, 
                                    -1)
                        cv2.putText(frame, 
                                  label,
                                  (bbox[0], bbox[1] - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5,
                                  (255, 255, 255),  # White text
                                  2)
            except Exception as e:
                log_event(f"Face detection visualization error: {e}")
        
        # Convert to RGB for tkinter
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.config(image=imgtk)
        camera_label.image = imgtk  # Keep a reference to prevent garbage collection
    
    # Schedule the next update
    if interview_running:
        root.after(33, update_camera_view)  # ~30 FPS

# ---------------------------------------------------------
# 11) Bot Animation
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
    """
    Enhanced animation for the AI interviewer that simulates natural movement
    and lip syncing when speaking. Includes subtle idle animations and
    more fluid transitions between speaking and non-speaking states.
    """
    global frame_index, bot_frames, lip_sync_frames, is_speaking
    global animation_state, idle_timer, bot_label
    
    # Initialize animation state variables if they don't exist
    if 'animation_state' not in globals():
        global animation_state, idle_timer, blink_counter
        animation_state = "idle"
        idle_timer = 0
        blink_counter = 0
    
    # Check if speaking
    with speaking_lock:
        speaking_now = is_speaking
    
    # Load default frames if not loaded
    if not bot_frames:
        bot_frames = load_gif_frames(GIF_PATH)
        if not bot_frames:
            # Fallback if GIF not available
            fallback_img = Image.new('RGB', (320, 240), color='black')
            draw = ImageDraw.Draw(fallback_img)
            draw.text((10, 120), "AI Interviewer", fill=(200, 200, 200))
            bot_frames = [ImageTk.PhotoImage(fallback_img)]
    
    # Animation state machine
    if speaking_now:
        # Active speaking animation
        animation_state = "speaking"
        idle_timer = 0
        
        # Use lip sync frames if available, otherwise use animated frames
        if lip_sync_frames and len(lip_sync_frames) > 0:
            # Dynamically adjust speaking animation based on speech content
            frame_index = (frame_index + 1) % len(lip_sync_frames)
            bot_label.config(image=lip_sync_frames[frame_index])
            bot_label.image = lip_sync_frames[frame_index]
        elif bot_frames and len(bot_frames) > 1:
            # Use regular animation as fallback
            frame_index = (frame_index + 1) % len(bot_frames)
            bot_label.config(image=bot_frames[frame_index])
            bot_label.image = bot_frames[frame_index]
    else:
        # Idle animations - occasional movements to appear more lifelike
        idle_timer += 1
        
        if animation_state == "speaking":
            # Transition from speaking to idle
            animation_state = "idle"
            frame_index = 0
            if bot_frames:
                bot_label.config(image=bot_frames[0])
                bot_label.image = bot_frames[0]
        
        elif animation_state == "idle":
            # Occasional random movements during idle state
            if idle_timer > 30:  # Every ~3 seconds
                # 20% chance to blink or make a small movement
                if random.random() < 0.2:
                    animation_state = "blink"
                    blink_counter = 0
                idle_timer = 0
        
        elif animation_state == "blink":
            # Quick blink animation (3 frames)
            blink_counter += 1
            if blink_counter >= 3:
                animation_state = "idle"
                # Reset to neutral frame
                if bot_frames:
                    bot_label.config(image=bot_frames[0])
                    bot_label.image = bot_frames[0]
            else:
                # Show blink frame if available, otherwise skip
                if len(bot_frames) > 1:
                    bot_label.config(image=bot_frames[1])  # Use second frame for blink
                    bot_label.image = bot_frames[1]
    
    # Adjust animation speed based on state
    if animation_state == "speaking":
        delay = 100  # Normal speaking animation (10 FPS)
    elif animation_state == "blink":
        delay = 50   # Faster for blink (20 FPS)
    else:
        delay = 200  # Slower for idle (5 FPS)
    
    # Schedule next frame
    root.after(delay, animate_bot)

# ---------------------------------------------------------
# 12) Interview Conversation Logic
# ---------------------------------------------------------
def generate_unique_question_list(context, role):
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

# ---------------------------------------------------------
# 13) PDF Report Generation
# ---------------------------------------------------------
def generate_pdf_report(transcript, final_score, breakdown):
    """
    Generate a comprehensive PDF report of the interview with detailed metrics,
    candidate information, and personalized feedback.
    """
    report_dir = "reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # Initialize PDF with proper settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Add title and header
    pdf.set_font("Helvetica", 'B', 18)
    pdf.set_text_color(0, 51, 102)  # Navy blue
    pdf.cell(0, 10, "AI Interview Assessment Report", ln=True, align="C")
    
    # Add date and candidate information
    pdf.set_font("Helvetica", 'I', 10)
    pdf.set_text_color(100, 100, 100)  # Gray
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 5, f"Generated on: {current_date}", ln=True, align="C")
    pdf.ln(5)
    
    # Candidate information section
    pdf.set_font("Helvetica", 'B', 14)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, "Candidate Information", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())  # Add a horizontal line
    pdf.ln(3)
    
    # Add candidate details
    pdf.set_font("Helvetica", '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Name: {candidate_name}", ln=True)
    pdf.cell(0, 8, f"Position: {job_role}", ln=True)
    
    # Try to add candidate photo if available
    if registered_face_image is not None:
        try:
            # Save temporary image file
            temp_img_path = os.path.join(report_dir, "temp_face.jpg")
            cv2.imwrite(temp_img_path, registered_face_image)
            
            # Add to PDF
            if os.path.exists(temp_img_path):
                pdf.image(temp_img_path, x=160, y=35, w=30, h=30)
                # Clean up temp file
                try:
                    os.remove(temp_img_path)
                except:
                    pass
        except Exception as e:
            log_event(f"Error adding photo to PDF: {e}")
    
    pdf.ln(5)
    
    # Performance summary section
    pdf.set_font("Helvetica", 'B', 14)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, "Performance Summary", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    # Final score with color-coding
    pdf.set_font("Helvetica", 'B', 16)
    if final_score >= 80:
        pdf.set_text_color(0, 128, 0)  # Green for high scores
    elif final_score >= 60:
        pdf.set_text_color(255, 165, 0)  # Orange for medium scores
    else:
        pdf.set_text_color(220, 20, 60)  # Crimson for low scores
    
    pdf.cell(0, 10, f"Final Score: {final_score}/100", ln=True, align="C")
    
    # Performance level
    level_desc = get_level_description(final_score/100)
    pdf.set_font("Helvetica", 'I', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Performance Level: {level_desc.title()}", ln=True, align="C")
    
    # Add warning summary if there were warnings
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(0, 8, f"Total Warnings: {warning_count}", ln=True)
    
    if warning_count > 0 and 'warning_labels' in globals() and warning_labels:
        pdf.set_font("Helvetica", '', 10)
        pdf.set_text_color(220, 20, 60)  # Red for warnings
        unique_warnings = set(warning_labels)
        for warning in unique_warnings:
            pdf.cell(0, 6, f"• {warning}", ln=True)
    
    pdf.ln(5)
    
    # Score breakdown section with table
    pdf.set_font("Helvetica", 'B', 14)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, "Score Breakdown", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    # Create table for score breakdown
    if isinstance(breakdown, dict):
        # Table header
        pdf.set_font("Helvetica", 'B', 12)
        pdf.set_text_color(255, 255, 255)
        pdf.set_fill_color(0, 51, 102)
        pdf.cell(130, 8, "Category", 1, 0, 'L', True)
        pdf.cell(50, 8, "Score", 1, 1, 'C', True)
        
        # Table rows
        pdf.set_font("Helvetica", '', 11)
        pdf.set_text_color(0, 0, 0)
        
        for category, score in breakdown.items():
            if isinstance(score, (int, float)) and category not in ["base_score", "warning_penalty"]:
                category_name = category.replace('_', ' ').title()
                
                # Alternate row coloring for better readability
                fill = False
                pdf.set_fill_color(240, 240, 240)
                
                # Set text color based on score
                if score >= 8:
                    pdf.set_text_color(0, 128, 0)  # Green for high scores
                elif score >= 6:
                    pdf.set_text_color(0, 0, 0)  # Black for average scores
                else:
                    pdf.set_text_color(220, 20, 60)  # Red for low scores
                
                pdf.cell(130, 8, category_name, 1, 0, 'L', fill)
                pdf.cell(50, 8, f"{score:.1f}/10", 1, 1, 'C', fill)
                
                # Reset text color
                pdf.set_text_color(0, 0, 0)
                fill = not fill
        
        # Add warning penalty if applicable
        if "warning_penalty" in breakdown:
            pdf.set_text_color(220, 20, 60)  # Red for penalty
            pdf.cell(130, 8, "Warning Penalty", 1, 0, 'L')
            pdf.cell(50, 8, f"-{breakdown['warning_penalty']:.1f}", 1, 1, 'C')
            pdf.set_text_color(0, 0, 0)  # Reset text color
    
    pdf.ln(10)
    
    # Start a new page for transcript
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 14)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, "Interview Transcript", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Format the transcript with proper styling
    pdf.set_font("Helvetica", '', 10)
    pdf.set_text_color(0, 0, 0)
    
    # Process transcript line by line
    for line in transcript.split('\n'):
        if line.strip():
            if line.startswith("Interviewer:"):
                # Style for interviewer questions
                pdf.set_font("Helvetica", 'B', 10)
                pdf.set_text_color(0, 51, 102)
            else:
                # Style for candidate responses
                pdf.set_font("Helvetica", '', 10)
                pdf.set_text_color(0, 0, 0)
            
            # Add the line with proper wrapping
            pdf.multi_cell(0, 6, line)
            pdf.ln(1)
    
    # Add recommendations section
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 14)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, "Recommendations & Next Steps", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Generate customized recommendations based on performance
    pdf.set_font("Helvetica", '', 11)
    pdf.set_text_color(0, 0, 0)
    
    # Basic recommendations based on score
    if final_score >= 85:
        recommendations = [
            "Excellent performance! You demonstrated strong qualifications for this role.",
            "Your answers showed depth of knowledge and clear communication.",
            "Consider highlighting these strengths in future interviews."
        ]
    elif final_score >= 70:
        recommendations = [
            "Good overall performance with some areas for improvement.",
            "You demonstrated solid understanding of key concepts.",
            "Consider preparing more concrete examples for technical questions."
        ]
    elif final_score >= 50:
        recommendations = [
            "Average performance with significant room for improvement.",
            "Focus on developing deeper technical knowledge in key areas.",
            "Practice explaining complex concepts more clearly."
        ]
    else:
        recommendations = [
            "Further preparation recommended before pursuing similar roles.",
            "Focus on strengthening fundamentals in your field.",
            "Consider additional training or courses to build expertise."
        ]
    
    # Add specific recommendations based on breakdown
    if isinstance(breakdown, dict):
        # Find the lowest scoring categories
        weakness_categories = []
        for category, score in breakdown.items():
            if isinstance(score, (int, float)) and category not in ["base_score", "warning_penalty"]:
                if score < 6:  # Consider scores below 6 as areas for improvement
                    weakness_categories.append((category, score))
        
        # Sort by score (ascending)
        weakness_categories.sort(key=lambda x: x[1])
        
        if weakness_categories:
            recommendations.append("\nFocus on improving these specific areas:")
            for category, score in weakness_categories[:3]:  # Top 3 weaknesses
                category_name = category.replace('_', ' ').title()
                if "technical" in category.lower():
                    recommendations.append(f"• {category_name}: Strengthen your technical knowledge through practice and study.")
                elif "communication" in category.lower() or "clarity" in category.lower():
                    recommendations.append(f"• {category_name}: Practice explaining complex concepts clearly and concisely.")
                elif "problem" in category.lower():
                    recommendations.append(f"• {category_name}: Work on structured problem-solving approaches.")
                else:
                    recommendations.append(f"• {category_name}: Dedicate time to improving this skill.")
    
    # Add recommendations to PDF
    for recommendation in recommendations:
        if recommendation.startswith("•"):
            pdf.set_x(15)  # Indent bullet points
        elif recommendation == "":
            pdf.ln(3)  # Add space for empty lines
            continue
        else:
            pdf.set_x(10)  # Reset margin for regular text
        
        pdf.multi_cell(0, 6, recommendation)
        pdf.ln(2)
    
    # Add footer
    pdf.set_y(-25)
    pdf.set_font("Helvetica", 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, "This report was generated automatically by the AI Interview Coach system.", ln=True, align="C")
    pdf.cell(0, 5, f"© {datetime.datetime.now().year} AI Interview Coach", ln=True, align="C")
    
    # Generate filename with candidate name
    safe_name = ''.join(c if c.isalnum() else '_' for c in candidate_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{report_dir}/Interview_Report_{safe_name}_{timestamp}.pdf"
    
    # Generate PDF
    pdf.output(filename)
    log_event(f"Enhanced PDF report generated: {filename}")
    return filename

# ---------------------------------------------------------
# 14) TTS / STT (Fixed is_speaking)
# ---------------------------------------------------------
def record_audio():
    """
    Improved audio recording with better state management and error handling.
    """
    global is_recording_voice, stop_recording_flag, user_submitted_answer
    global reference_voice_embedding, warning_count, is_speaking

    if not interview_running:
        return

    safe_update(record_btn, record_btn.config, state=tk.DISABLED)
    safe_update(stop_record_btn, stop_record_btn.config, state=tk.NORMAL)

    is_speaking = True
    append_transcript(chat_display, "(Recording in progress...)")
    recognized_segments = []
    
    try:
        with sr.Microphone() as source:
            # Quick ambient noise adjustment
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            recognizer.energy_threshold = 150  # Lower threshold to detect quieter speech
            
            # Main recording loop with timeout
            start_time = time.time()
            silent_duration = 0
            last_recognition_time = time.time()
            
            while not stop_recording_flag and interview_running:
                if time.time() - start_time > 120:  # 2-minute max recording
                    break
                    
                try:
                    # Use shorter phrase time limit to get more frequent updates
                    audio = recognizer.listen(source, phrase_time_limit=3, timeout=1)
                    
                    if stop_recording_flag or not interview_running:
                        break
                        
                    text = recognizer.recognize_google(audio)
                    if text.strip():
                        recognized_segments.append(text)
                        # Show intermediate recognition
                        append_transcript(chat_display, f"(Recognizing: {text})")
                        # Store partial results in case of early termination
                        user_submitted_answer = " ".join(recognized_segments).strip()
                        last_recognition_time = time.time()
                        silent_duration = 0
                        
                except sr.UnknownValueError:
                    # Track silence duration
                    current_time = time.time()
                    silent_duration = current_time - last_recognition_time
                    # Auto-stop after 5 seconds of silence if we have recognized something
                    if len(recognized_segments) > 0 and silent_duration > 5:
                        log_event("Auto-stopping recording after 5 seconds of silence")
                        break
                    continue
                except sr.RequestError as e:
                    log_event(f"Speech recognition error: {e}")
                    break
                    
    except Exception as e:
        log_event(f"Recording error: {e}")
    finally:
        # Ensure proper cleanup
        is_speaking = False
        is_recording_voice = False
        stop_recording_flag = False
        
        safe_update(stop_record_btn, stop_record_btn.config, state=tk.DISABLED)
        safe_update(record_btn, record_btn.config, state=tk.NORMAL)
        
        final_text = " ".join(recognized_segments).strip()
        if final_text:
            user_submitted_answer = final_text
            append_transcript(chat_display, f"(Final Recognition): {final_text}")
        else:
            # Don't set to None if we have partial segments - keep what we've got
            if not user_submitted_answer:
                user_submitted_answer = None
                append_transcript(chat_display, "(No speech recognized)")
            else:
                append_transcript(chat_display, f"(Final Recognition): {user_submitted_answer}")

def start_recording_voice():
    """
    Improved recording start with state validation
    """
    global recording_thread, is_recording_voice, stop_recording_flag
    
    if not interview_running:
        append_transcript(chat_display, "Interview not running. Click 'Start Interview' first.")
        return
        
    if is_recording_voice:
        append_transcript(chat_display, "Already recording.")
        return
        
    is_recording_voice = True
    stop_recording_flag = False
    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()

def stop_recording_voice():
    """
    Improved recording stop with proper cleanup
    """
    global stop_recording_flag, is_recording_voice
    
    if not is_recording_voice:
        return
        
    stop_recording_flag = True
    append_transcript(chat_display, "(Stopping recording...)")
    
    # Wait briefly for the recording thread to clean up
    time.sleep(0.5)
    is_recording_voice = False

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-6)

def detect_voice_spoofing(audio_data, reference_embedding=None):
    """
    Enhanced voice spoofing detection with similarity metrics and additional checks
    for audio quality and potential playback detection.
    """
    global voice_encoder, reference_voice_embedding, current_voice_similarity
    
    try:
        # Convert audio data to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Normalize audio
        audio_np = audio_np.astype(np.float32) / 32768.0
        
        # Check for audio quality issues
        if len(audio_np) < 8000:  # Less than 0.5 seconds at 16kHz
            return True, 0.9, "Audio sample too short"
            
        # Check for silence or very low volume
        rms = np.sqrt(np.mean(np.square(audio_np)))
        if rms < 0.01:  # Very low volume
            return True, 0.9, f"Audio volume too low (RMS: {rms:.4f})"
            
        # Check for clipping (potential playback from speakers)
        clip_ratio = np.sum(np.abs(audio_np) > 0.95) / len(audio_np)
        if clip_ratio > 0.05:  # More than 5% of samples are clipping
            return True, 0.8, f"Audio may be from playback device (clipping ratio: {clip_ratio:.4f})"
        
        # Compute embedding for current audio
        current_embedding = voice_encoder.embed_utterance(audio_np)
        
        # Use provided reference or global reference
        if reference_embedding is None:
            if reference_voice_embedding is None:
                return False, 0.0, "No reference embedding available"
            reference_embedding = reference_voice_embedding
        
        # Calculate similarity
        similarity = cosine_similarity(current_embedding, reference_embedding)
        current_voice_similarity = similarity  # Update global metric
        
        # Log the similarity for debugging
        log_event(f"Voice similarity: {similarity:.4f} (threshold: {VOICE_SIMILARITY_THRESHOLD})")
        
        # Check for spoofing based on similarity threshold
        if similarity < VOICE_SIMILARITY_THRESHOLD:
            confidence = 1.0 - similarity
            return True, confidence, f"Voice doesn't match reference (similarity: {similarity:.2f})"
        
        # Additional check for spectral flatness (can indicate synthetic speech)
        # This is a simplified version - a real implementation would use more sophisticated methods
        spectral_flatness = np.exp(np.mean(np.log(np.abs(np.fft.rfft(audio_np)) + 1e-10))) / np.mean(np.abs(np.fft.rfft(audio_np)))
        if spectral_flatness > 0.5:  # Higher values can indicate synthetic speech
            return True, 0.7, f"Possible synthetic speech detected (flatness: {spectral_flatness:.4f})"
        
        return False, 0.0, "Voice verified successfully"
        
    except Exception as e:
        log_event(f"Voice spoofing detection error: {e}")
        return False, 0.0, f"Error in voice verification: {e}"

# Voice prompts for reference recording
VOICE_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, how are you doing today?",
    "Please read this sentence clearly.",
    "I am ready for the interview process.",
    "Today is a great opportunity to showcase my abilities."
]

# Function to generate a random sentence
def generate_random_sentence():
    """Generate a random sentence for voice verification."""
    subjects = ["The candidate", "A professional", "An applicant", "The interviewee", "A person", "The individual"]
    verbs = ["should", "must", "can", "will", "might", "needs to"]
    actions = ["demonstrate", "showcase", "present", "explain", "discuss", "highlight", "emphasize"]
    qualities = ["communication skills", "problem-solving abilities", "technical knowledge", "creative thinking", 
                "leadership potential", "teamwork experience", "analytical skills", "attention to detail"]
    connectors = ["while", "as", "when", "because", "although", "since"]
    contexts = ["interviewing for a position", "seeking new opportunities", "presenting qualifications", 
               "discussing prior experience", "answering technical questions", "sharing career goals"]
    
    # Create a random sentence structure
    sentence = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(actions)} {random.choice(qualities)} {random.choice(connectors)} {random.choice(contexts)}."
    return sentence

# Global variable to help with voice registration
_voice_registration_tmp = None

def record_voice_reference():
    """
    Advanced voice reference recording with enhanced spoofing protection, lip sync verification,
    and quality checks.
    """
    global reference_voice_embedding, reference_audio_data, voice_reference_recorded, cap, _voice_registration_tmp
    
    # Declare variables that will be used by nested functions
    recording_complete = False
    is_recording = False
    recording_thread = None
    audio_data = None
    countdown_timer = None
    recording_length = 8  # seconds
    lip_movement_frames = []
    lip_movement_values = []
    process_running = True
    preview_running = True
    frames_with_lips_moving = 0
    total_frames = 0
    recording_start_time = 0  # Initialize the recording start time
    
    # Enhanced variables for lip sync detection
    lip_sync_score = 0
    face_detected_consistently = False
    potential_spoofing_detected = False
    audio_energy_values = []
    audio_frames = []
    
    # Define process_result function early
    def process_result():
        """Process recording results and update UI"""
        nonlocal recording_complete, potential_spoofing_detected, face_detected_consistently, lip_sync_score
        
        log_event(f"Processing recording results (recording_complete={recording_complete}, voice_reference_recorded={voice_reference_recorded})")
        
        if recording_complete:
            log_event("Results already processed, skipping")
            return
            
        recording_complete = True
        
        # Re-enable record button for retry
        record_button.config(state=tk.NORMAL)
        
        # Check for spoofing or verification issues
        if potential_spoofing_detected:
            log_event("Voice registration failed: Potential spoofing detected")
            status_label.config(text="Voice verification failed: Inconsistent lip sync detected. Please try again.", fg="#FF0000")
            voice_reference_recorded = False
            return
            
        if not face_detected_consistently:
            log_event("Voice registration failed: Face not consistently detected")
            status_label.config(text="Voice verification failed: Face not consistently visible. Please try again.", fg="#FF0000")
            voice_reference_recorded = False
            return
            
        if lip_sync_score < 0.3:  # Less than 30% of frames had adequate lip movement
            log_event(f"Voice registration failed: Low lip sync score {lip_sync_score:.2f}")
            status_label.config(text="Voice verification failed: Insufficient lip movement detected. Please speak clearly.", fg="#FF0000")
            voice_reference_recorded = False
            return
        
        # Update UI based on recording success
        if voice_reference_recorded and reference_voice_embedding is not None:
            log_event("Voice registration successful, closing window")
            status_label.config(text="Voice registration successful!", fg="#00FF00")
            
            # Close window after a delay
            voice_window.after(2000, voice_window.destroy)
        else:
            log_event("Voice registration failed")
            status_label.config(text="Voice registration failed. Please try again.", fg="#FF0000")
    
    # Save process_result to global for later reference
    _voice_registration_tmp = process_result
    
    # Initialize face mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize Resemblyzer voice encoder if not already done
    global voice_encoder
    if voice_encoder is None:
        try:
            voice_encoder = VoiceEncoder()
            log_event("Voice encoder loaded")
        except Exception as e:
            log_event(f"Error loading voice encoder: {e}")
            safe_showerror("Voice Registration Error", 
                          "Unable to initialize voice recognition system. Please check your installation.")
            return False
    
    # Initialize webcam - try multiple methods to ensure it opens
    if cap is None or not cap.isOpened():
        try:
            log_event("Attempting to open webcam for voice registration...")
            
            # Try multiple camera indices (0, 1, 2) with DirectShow backend on Windows
            for cam_index in range(3):
                try:
                    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
                    if cap is not None and cap.isOpened():
                        log_event(f"Successfully opened camera at index {cam_index}")
                        break
                except Exception as e:
                    log_event(f"Failed to open camera at index {cam_index}: {e}")
            
            # If still no camera, try one more time with default settings
            if cap is None or not cap.isOpened():
                try:
                    cap = cv2.VideoCapture(0)
                except Exception as e:
                    log_event(f"Final attempt to open camera failed: {e}")
            
            # Check if camera opened successfully
            if cap is None or not cap.isOpened():
                log_event("Could not open webcam after multiple attempts")
                safe_showerror("Camera Error", 
                             "Cannot access webcam. Please check your camera connections and permissions.")
                return False
            else:
                # Log camera properties
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                log_event(f"Camera opened successfully. Resolution: {width}x{height}, FPS: {fps}")
                # Check if we can actually read frames
                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    log_event("Camera opened but can't read frames")
                    safe_showerror("Camera Error", "Camera connected but cannot read frames. Please try a different camera.")
                    return False
                else:
                    log_event(f"Successfully read test frame of size {test_frame.shape}")
                
        except Exception as e:
            log_event(f"Error initializing camera: {e}")
            safe_showerror("Camera Error", f"Error initializing camera: {str(e)}")
            return False
    
    log_event("Starting voice reference recording process")
    
    # Set up UI
    voice_window = tk.Toplevel(root)
    voice_window.title("Voice Reference Registration")
    voice_window.geometry("640x720")
    voice_window.configure(bg=MAIN_BG)
    voice_window.grab_set()  # Make window modal
    
    # Add a title and instructions
    title_label = tk.Label(
        voice_window, 
        text="Voice Registration", 
        font=(FONT_FAMILY, 18, "bold"),
        bg=MAIN_BG, fg=ACCENT_COLOR
    )
    title_label.pack(pady=(20, 10))
    
    # Instructions with clear text
    instruction_frame = tk.Frame(voice_window, bg=GLASS_BG, padx=15, pady=15)
    instruction_frame.pack(fill=tk.X, padx=20, pady=10)
    
    # Generate a random prompt instead of using predefined ones
    selected_prompt = generate_random_sentence()
    
    instruction_text = (
        "Please read the following text aloud when recording starts:\n\n"
        f'"{selected_prompt}"\n\n'
        "Speak clearly and at a normal pace. This will be used to verify your identity."
    )
    
    instructions = tk.Label(
        instruction_frame, 
        text=instruction_text,
        wraplength=500,
        justify=tk.LEFT,
        font=(FONT_FAMILY, 12),
        bg=MAIN_BG, fg=MAIN_FG
    )
    instructions.pack(pady=10, padx=20, fill=tk.X)
    
    # Security notice
    security_notice = tk.Label(
        voice_window,
        text="For security purposes, we'll verify that your lip movements match what you're saying.\nThe system will detect if someone else is speaking while you pretend to talk.",
        wraplength=600,
        font=(FONT_FAMILY, 10, "italic"),
        bg=MAIN_BG, fg=ACCENT_COLOR
    )
    security_notice.pack(pady=(5, 20), padx=20, fill=tk.X)
    
    # Camera requirement notice - add this new warning
    camera_requirement = tk.Label(
        voice_window,
        text="A working webcam is REQUIRED for voice registration to prevent spoofing.\n"
             "If you see a camera error, please verify that:\n"
             "1. Your camera is connected and working\n"
             "2. Your browser/OS permissions allow camera access\n"
             "3. No other application is using your camera",
        wraplength=600,
        font=(FONT_FAMILY, 10),
        bg=MAIN_BG, fg="#FF9900"  # Warning color
    )
    camera_requirement.pack(pady=(0, 10), padx=20, fill=tk.X)
    
    # Add a video preview
    preview_label = tk.Label(voice_window, bg="black")
    preview_label.pack(pady=10)
    
    # Status indicator
    status_label = tk.Label(
        voice_window,
        text="Ready to record",
        font=(FONT_FAMILY, 12),
        bg=MAIN_BG, fg=MAIN_FG
    )
    status_label.pack(pady=10)
    
    # Progress bar
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(
        voice_window,
        variable=progress_var,
        maximum=100,
        length=500,
        mode='determinate'
    )
    progress_bar.pack(pady=10)
    
    # Countdown label
    countdown_label = tk.Label(
        voice_window,
        text="",
        font=(FONT_FAMILY, 24, "bold"),
        bg=MAIN_BG, fg=ACCENT_COLOR
    )
    countdown_label.pack(pady=10)
    
    # Buttons frame
    buttons_frame = tk.Frame(voice_window, bg=MAIN_BG)
    buttons_frame.pack(pady=20)
    
    # Define essential functions before using them
    def cancel_recording():
        """Cancel recording process"""
        nonlocal process_running, preview_running, is_recording
        process_running = False
        preview_running = False
        if is_recording:
            process_recording(force_complete=True)
        voice_window.destroy()
    
    def check_recording_timeout():
        """Check if recording has timed out"""
        nonlocal recording_thread, is_recording, recording_start_time
        if is_recording and time.time() - recording_start_time > recording_length + 5:
            process_recording(force_complete=True)
    
    def record_audio_thread():
        """Thread to record audio for voice verification"""
        nonlocal audio_data, is_recording, audio_energy_values, audio_frames
        global reference_voice_embedding, reference_audio_data, voice_reference_recorded
        
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                r.energy_threshold = 150  # Lower threshold for easier detection
                status_label.config(text="Recording... Please read the prompt text clearly")
                
                # Use chunked recording to get energy values over time
                chunk_size = 0.1  # seconds per chunk
                chunks = []
                start_time = time.time()
                
                # Record in chunks to analyze energy over time
                while is_recording and time.time() - start_time < recording_length:
                    chunk = r.record(source, duration=chunk_size)
                    chunk_data = np.frombuffer(chunk.get_raw_data(), dtype=np.int16)
                    chunks.append(chunk_data)
                    
                    # Calculate energy for this chunk
                    if len(chunk_data) > 0:
                        chunk_energy = np.sqrt(np.mean(chunk_data.astype(np.float32)**2))
                        audio_energy_values.append(chunk_energy)
                        audio_frames.append(time.time() - start_time)
                
                # Combine all chunks
                audio_data = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
                
                if len(audio_data) > 0:
                    try:
                        # Calculate audio energy to ensure we have actual speech
                        rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                        log_event(f"Recorded audio with RMS energy: {rms}")
                        
                        if rms < 50:  # Very quiet recording
                            log_event("Warning: Very quiet audio recording")
                            status_label.config(text="Audio too quiet. Please speak louder and try again.", fg="#FF0000")
                            is_recording = False
                            return
                            
                        sample_rate = 16000  # Fixed sample rate for consistency
                        audio_data = audio_data.astype(np.float32) / 32768.0
                        
                        # Try to recognize some speech to validate recording
                        try:
                            recognized_text = r.recognize_google(sr.AudioData(audio_data.tobytes(), sample_rate, 2))
                            log_event(f"Recognized text from recording: {recognized_text}")
                            
                            # Calculate similarity between recognized text and prompt
                            from difflib import SequenceMatcher
                            text_similarity = SequenceMatcher(None, recognized_text.lower(), selected_prompt.lower()).ratio()
                            log_event(f"Text similarity with prompt: {text_similarity:.2f}")
                            
                            if text_similarity < 0.3:  # Very low similarity might indicate wrong text was read
                                log_event("Warning: Recognized text doesn't match the prompt")
                                status_label.config(text="Please make sure to read the provided text. Try again.", fg="#FFA500")
                        except Exception:
                            log_event("Could not recognize speech, but continuing with embedding calculation")
                            
                        log_event("Computing voice embedding...")
                        reference_voice_embedding = compute_voice_embedding(audio_data, sample_rate)
                        
                        if reference_voice_embedding is not None:
                            log_event("Voice embedding computed successfully")
                            reference_audio_data = audio_data
                            voice_reference_recorded = True
                        else:
                            log_event("Failed to compute voice embedding")
                            status_label.config(text="Voice processing failed. Please try again.", fg="#FF0000")
                    except Exception as e:
                        log_event(f"Error processing audio: {str(e)}")
                        status_label.config(text=f"Error: {str(e)[:50]}...", fg="#FF0000")
                else:
                    log_event("No audio data recorded")
                    status_label.config(text="No audio detected. Please try again.", fg="#FF0000")
        except Exception as e:
            log_event(f"Voice recording error: {e}")
            status_label.config(text=f"Recording error: {str(e)[:50]}...", fg="#FF0000")
        finally:
            is_recording = False
            # Process result to update UI
            voice_window.after(500, process_result)
    
    # Add a Test Microphone button
    def test_microphone():
        """Test if microphone is working"""
        try:
            status_label.config(text="Testing microphone... Say something.", fg="#3498DB")
            r = sr.Recognizer()
            mic = sr.Microphone()
            
            # Brief countdown
            for i in range(3, 0, -1):
                countdown_label.config(text=f"Testing in {i}...")
                voice_window.update()
                time.sleep(0.5)
            countdown_label.config(text="Listening...")
            
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.record(source, duration=2)
                
                # Analyze audio volume
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                if len(audio_data) > 0:
                    rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                    peak = np.max(np.abs(audio_data))
                    
                    log_event(f"Mic test - RMS: {rms:.2f}, Peak: {peak}")
                    
                    if rms < 100:  # Very quiet
                        status_label.config(text="Microphone volume is too low. Please speak louder or check your mic.", fg="#FF0000")
                    elif peak > 30000:  # Very loud/clipping
                        status_label.config(text="Microphone volume is too high. Please speak softer or adjust your mic.", fg="#FF0000")
                    else:
                        try:
                            text = r.recognize_google(audio)
                            status_label.config(text=f"Microphone working! Heard: '{text}'", fg="#00FF00")
                        except sr.UnknownValueError:
                            status_label.config(text="Microphone working, but couldn't recognize speech.", fg="#FFA500")
                        except Exception as e:
                            log_event(f"Speech recognition error: {e}")
                            status_label.config(text="Microphone working, but speech recognition error.", fg="#FFA500")
                else:
                    status_label.config(text="No audio detected. Please check your microphone connection.", fg="#FF0000")
        except Exception as e:
            log_event(f"Microphone test error: {e}")
            status_label.config(text=f"Microphone test error: {str(e)[:50]}...", fg="#FF0000")
        finally:
            countdown_label.config(text="")
            # Reset status after a delay
            voice_window.after(3000, lambda: status_label.config(text="Ready to record", fg=MAIN_FG))
    
    # Add Record and Cancel buttons
    record_button = tk.Button(
        buttons_frame,
        text="Start Recording",
        command=lambda: update_countdown(3),
        bg=BUTTON_BG, fg=BUTTON_FG,
        font=(FONT_FAMILY, 12),
        width=15
    )
    record_button.pack(side=tk.LEFT, padx=10)
    
    # Add a Test Mic button
    test_mic_button = tk.Button(
        buttons_frame,
        text="Test Microphone",
        command=test_microphone,
        bg="#17a2b8", fg=BUTTON_FG,
        font=(FONT_FAMILY, 12),
        width=15
    )
    test_mic_button.pack(side=tk.LEFT, padx=10)
    
    # Add a "Done" button that appears when recording is successful
    done_button = tk.Button(
        buttons_frame,
        text="Done",
        command=voice_window.destroy,
        bg="#28A745", fg=BUTTON_FG,  # Green color for completion
        font=(FONT_FAMILY, 12),
        width=15,
        state=tk.DISABLED  # Initially disabled
    )
    done_button.pack(side=tk.LEFT, padx=10)
    done_button.pack_forget()  # Hide it initially
    
    cancel_button = tk.Button(
        buttons_frame,
        text="Cancel",
        command=cancel_recording,
        bg="#C72C41", fg=BUTTON_FG,
        font=(FONT_FAMILY, 12),
        width=15
    )
    cancel_button.pack(side=tk.LEFT, padx=10)
    
    # Override the process_result function to update the buttons
    original_process_result = process_result
    def updated_process_result():
        original_process_result()
        if voice_reference_recorded and reference_voice_embedding is not None:
            # Show the Done button and hide the Record button
            record_button.pack_forget()
            test_mic_button.pack_forget()
            done_button.pack(side=tk.LEFT, before=cancel_button, padx=10)
            done_button.config(state=tk.NORMAL)
    
    process_result = updated_process_result
    
    # Add debug mode and log display
    debug_frame = tk.Frame(voice_window, bg=MAIN_BG)
    debug_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    
    debug_var = tk.IntVar(value=0)
    debug_check = tk.Checkbutton(
        debug_frame, 
        text="Show Debug Log", 
        variable=debug_var,
        bg=MAIN_BG, fg=MAIN_FG,
        command=lambda: toggle_debug_log()
    )
    debug_check.pack(side=tk.LEFT, padx=10)
    
    debug_log = scrolledtext.ScrolledText(
        voice_window,
        height=6,
        width=60,
        font=(FONT_FAMILY, 8),
        bg="#000000",
        fg="#00FF00"
    )
    debug_log.insert(tk.END, "=== DEBUG LOG ===\n")
    debug_log.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
    debug_log.pack_forget()  # Initially hidden
    
    def toggle_debug_log():
        if debug_var.get():
            debug_log.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
            # Add current status information
            debug_log.delete(1.0, tk.END)
            debug_log.insert(tk.END, "=== DEBUG LOG ===\n")
            debug_log.insert(tk.END, f"Camera status: {'Connected' if cap and cap.isOpened() else 'Not connected'}\n")
            debug_log.insert(tk.END, f"Voice reference status: {'Recorded' if voice_reference_recorded else 'Not recorded'}\n")
            debug_log.insert(tk.END, f"Recording state: {'Active' if is_recording else 'Inactive'}\n")
            debug_log.insert(tk.END, f"Thresholds: Lip ratio={MIN_LIP_MOVEMENT_FRAMES_RATIO}, Variance={MIN_LIP_MOVEMENT_VARIANCE}, Similarity={MIN_PROMPT_SIMILARITY}\n")
        else:
            debug_log.pack_forget()
    
    # Modify log_event to also update debug log if visible
    original_log_event_global = globals()['log_event']
    def debug_log_event(message):
        original_log_event_global(message)
        # Check if voice_window and debug_var exist and are accessible before using them
        try:
            if 'voice_window' in globals() and voice_window is not None and voice_window.winfo_exists() and debug_var.get():
                debug_log.insert(tk.END, f"{message}\n")
                debug_log.see(tk.END)
        except (tk.TclError, NameError, AttributeError):
            # Safely handle errors when window is destroyed or variables are not accessible
            pass
    
    # Replace log_event temporarily within this function
    globals()['log_event'] = debug_log_event
    
    # Start preview thread
    preview_thread = threading.Thread(target=update_preview)
    preview_thread.daemon = True
    preview_thread.start()
    
    # Wait for window to be destroyed
    voice_window.wait_window()
    
    # Clean up
    face_mesh.close()
    preview_running = False
    process_running = False
    
    # Return status
    return voice_reference_recorded

def update_preview():
    """Thread to update webcam preview for lip movement detection"""
    global cap
    
    preview_width = 320
    preview_height = 240
    
    # Define the function body here
    pass

def update_progress():
    """Update the progress bar during recording"""
    pass

def update_countdown(count):
    """Update the countdown timer"""
    pass

def start_recording_process():
    """Start the audio recording thread"""
    pass

def process_recording(force_complete=False):
    """Process the recording after completion"""
    pass

def process_result():
    """Process recording results and update UI"""
    pass

# ---------------------------------------------------------
# 15) Interview Loop
# ---------------------------------------------------------
def interview_loop(chat_display):
    """
    Advanced interview loop with dynamic question generation, adaptive difficulty,
    and intelligent response handling.
    
    This function manages the entire interview process, including:
    - Analyzing the resume and role to generate appropriate questions
    - Progressing from basic to advanced questions based on candidate responses
    - Providing constructive feedback and follow-up questions
    - Handling both voice responses and coding challenges
    - Ensuring proper waiting for candidate responses
    - Generating a comprehensive evaluation at the end
    """
    global interview_running, user_submitted_answer
    global candidate_name, job_role, interview_context
    global challenge_submitted, compiler_active
    global code_editor_last_activity_time
    global is_recording_voice

    transcript = []
    candidate_responses_count = 0

    try:
        if not interview_running:
            return

        greeting = f"Hello, {candidate_name}! I'm your Virtual Interviewer for the {job_role} position. " \
                   "Please respond by clicking 'Record' for voice answers or use the code editor for technical challenges."
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
            text_to_speech(question_text)

            safe_update(record_btn, record_btn.config, state=tk.NORMAL)
            safe_update(code_editor, code_editor.config, state=tk.NORMAL)
            safe_update(run_code_btn, run_code_btn.config, state=tk.NORMAL)
            safe_update(submit_btn, submit_btn.config, state=tk.NORMAL)
            compiler_instructions_label.config(
                text="You may type your answer in the code editor (Run/Submit) or click 'Record' for voice. "
                     "You have 15s to start. Inactivity >15s will skip to the next question."
            )

            challenge_submitted = False
            compiler_active = False
            code_editor_last_activity_time = None
            user_submitted_answer = None

            start_wait_time = time.time()
            response_mode = None
            while interview_running and not response_mode:
                if is_recording_voice:
                    response_mode = "voice"
                    break
                if code_editor_last_activity_time is not None:
                    response_mode = "code"
                    break
                if time.time() - start_wait_time > 15:
                    break
                time.sleep(0.2)

            if not response_mode:
                skip_msg = "No answer initiated within 15 seconds. Moving to the next question."
                transcript.append(f"Interviewer: {skip_msg}")
                append_transcript(chat_display, f"Interviewer: {skip_msg}")
                text_to_speech(skip_msg)

                safe_update(record_btn, record_btn.config, state=tk.DISABLED)
                safe_update(code_editor, code_editor.config, state=tk.DISABLED)
                safe_update(run_code_btn, run_code_btn.config, state=tk.DISABLED)
                safe_update(submit_btn, submit_btn.config, state=tk.DISABLED)
                continue

            if response_mode == "voice":
                # Wait for recording to complete with a reasonable timeout
                recording_wait_start = time.time()
                # Wait up to 2 minutes for the recording to complete
                while interview_running and is_recording_voice and time.time() - recording_wait_start < 120:
                    time.sleep(0.5)
                
                # Get the user's answer - might be None if nothing was recognized
                candidate_response = user_submitted_answer if user_submitted_answer else ""

                safe_update(record_btn, record_btn.config, state=tk.DISABLED)
                safe_update(code_editor, code_editor.config, state=tk.DISABLED)
                safe_update(run_code_btn, run_code_btn.config, state=tk.DISABLED)
                safe_update(submit_btn, submit_btn.config, state=tk.DISABLED)

                if "stop" in candidate_response.lower() or "exit" in candidate_response.lower():
                    end_msg = "Understood. Ending interview now."
                    transcript.append(f"Interviewer: {end_msg}")
                    append_transcript(chat_display, f"Interviewer: {end_msg}")
                    text_to_speech(end_msg)
                    break

                if not candidate_response:
                    no_resp = "No voice recognized. Moving to the next question."
                    transcript.append(f"Interviewer: {no_resp}")
                    append_transcript(chat_display, f"Interviewer: {no_resp}")
                    text_to_speech(no_resp)
                    continue

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
                        text_to_speech(feed_line)
                except Exception as e:
                    log_event(f"Minor error in evaluate_response: {e}")

                if len(candidate_response.split()) < 10:
                    followup_q = generate_followup_question(current_question, candidate_response, interview_context, job_role)
                    transcript.append(followup_q)
                    append_transcript(chat_display, followup_q)
                    text_to_speech(followup_q.replace("Interviewer:", "").strip())

                    safe_update(record_btn, record_btn.config, state=tk.NORMAL)
                    user_submitted_answer = None
                    followup_start = time.time()
                    while interview_running and not is_recording_voice and (time.time() - followup_start < 15):
                        time.sleep(0.2)
                    safe_update(record_btn, record_btn.config, state=tk.DISABLED)

                    while interview_running and is_recording_voice:
                        time.sleep(0.5)
                    followup_response = user_submitted_answer if user_submitted_answer else ""

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

            elif response_mode == "code":
                wait_start = time.time()
                timeout = 60
                while interview_running and not challenge_submitted and (time.time() - wait_start < timeout):
                    time.sleep(0.5)

                safe_update(code_editor, code_editor.config, state=tk.DISABLED)
                safe_update(run_code_btn, run_code_btn.config, state=tk.DISABLED)
                safe_update(submit_btn, submit_btn.config, state=tk.DISABLED)
                safe_update(record_btn, record_btn.config, state=tk.DISABLED)

                if not challenge_submitted:
                    no_code_msg = "No code submission received. Moving to the next question."
                    transcript.append(f"Interviewer: {no_code_msg}")
                    append_transcript(chat_display, f"Interviewer: {no_code_msg}")
                    text_to_speech(no_code_msg)

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
                    text_to_speech(ov_line)

                sc, breakdown, _ = grade_interview_with_breakdown(t_text, interview_context)
                if sc > 0:
                    final_msg = f"Your final interview score is {sc}/100. Thank you for your time!"
                else:
                    final_msg = "Could not determine a numeric score at this time. Thank you for your time!"
                transcript.append(f"Interviewer: {final_msg}")
                append_transcript(chat_display, f"Interviewer: {final_msg}")
                text_to_speech(final_msg)

                generate_pdf_report(t_text, sc, breakdown)
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
# 16) Model Loading and Splash Screen
# ---------------------------------------------------------
def apply_theme():
    """Apply the UI theme to the application"""
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

def load_lip_sync_model():
    log_event("Lip sync stub loaded.")
    return True

def load_model_splash():
    """
    Show a loading splash screen while initializing models.
    """
    global tokenizer, model, summarizer, classifier, face_app
    global mp_face_mesh, mp_drawing, mp_face_detection, mp_drawing_styles
    global voice_encoder, vad
    
    # Create splash window
    splash = tk.Toplevel()
    splash.title("Loading Models")
    splash.geometry("600x400")
    splash.configure(bg=MAIN_BG)
    splash.overrideredirect(True)
    
    # Center the splash window
    splash.update_idletasks()
    width = splash.winfo_width()
    height = splash.winfo_height()
    x = (splash.winfo_screenwidth() // 2) - (width // 2)
    y = (splash.winfo_screenheight() // 2) - (height // 2)
    splash.geometry(f'{width}x{height}+{x}+{y}')
    
    # Create a canvas for the gradient background
    canvas = tk.Canvas(splash, width=600, height=400, bg=MAIN_BG, highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    
    def draw_gradient(canv, ww, hh, c1=GRADIENT_START, c2=GRADIENT_END):
        """Draw a vertical gradient."""
        for i in range(hh):
            # Calculate the color for this line
            r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
            r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
            
            # Linear interpolation
            r = r1 + (r2 - r1) * i // hh
            g = g1 + (g2 - g1) * i // hh
            b = b1 + (b2 - b1) * i // hh
            
            color = f'#{r:02x}{g:02x}{b:02x}'
            canv.create_line(0, i, ww, i, fill=color)
    
    draw_gradient(canvas, 600, 400)
    
    # Add loading text
    loading_label = tk.Label(
        splash, 
        text="Loading AI Models...", 
        font=(FONT_FAMILY, 24, "bold"),
        fg=MAIN_FG, 
        bg=GLASS_BG
    )
    loading_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
    
    # Add progress text
    progress_label = tk.Label(
        splash, 
        text="Initializing...", 
        font=(FONT_FAMILY, 12),
        fg=MAIN_FG, 
        bg=GLASS_BG
    )
    progress_label.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
    
    # Add a progress bar
    progress_bar = ttk.Progressbar(
        splash, 
        orient="horizontal",
        length=400, 
        mode="indeterminate"
    )
    progress_bar.place(relx=0.5, rely=0.8, anchor=tk.CENTER)
    progress_bar.start(10)
    
    def finish_loading():
        """Initialize models in a separate thread."""
        try:
            # Update progress
            safe_update(progress_label, progress_label.config, text="Loading MediaPipe models...")
            
            # Initialize MediaPipe
            mp_face_mesh = mp.solutions.face_mesh
            mp_drawing = mp.solutions.drawing_utils
            mp_face_detection = mp.solutions.face_detection
            mp_drawing_styles = mp.solutions.drawing_styles
            
            # Update progress
            safe_update(progress_label, progress_label.config, text="Loading InsightFace model...")
            
            # Initialize InsightFace
            initialize_insightface()
            
            # Update progress
            safe_update(progress_label, progress_label.config, text="Loading voice recognition models...")
            
            # Initialize Resemblyzer
            try:
                global voice_encoder
                voice_encoder = VoiceEncoder()
                log_event("Resemblyzer voice encoder initialized successfully")
            except Exception as e:
                log_event(f"Failed to initialize Resemblyzer: {e}")
            
            # Initialize WebRTC VAD
            try:
                global vad
                vad = webrtcvad.Vad(3)  # Aggressiveness mode 3 (highest)
                log_event("WebRTC VAD initialized successfully")
            except Exception as e:
                log_event(f"Failed to initialize WebRTC VAD: {e}")
            
            # Update progress
            safe_update(progress_label, progress_label.config, text="Loading YOLO model...")
            
            # Initialize YOLO
            load_yolo_model()
            
            # Update progress
            safe_update(progress_label, progress_label.config, text="All models loaded successfully!")
            
            # Close splash after a delay
            splash.after(1000, splash.destroy)
            
        except Exception as e:
            log_event(f"Error loading models: {e}")
            safe_update(progress_label, progress_label.config, text=f"Error: {str(e)}")
            # Close splash after a longer delay to show error
            splash.after(3000, splash.destroy)
    
    # Check if the loading thread is done
    def check_thread():
        if loading_thread.is_alive():
            splash.after(100, check_thread)
        else:
            # Close splash and start main app
            if splash.winfo_exists():
                splash.destroy()
            main_app()
    
    # Start loading in a separate thread
    loading_thread = threading.Thread(target=finish_loading)
    loading_thread.daemon = True
    loading_thread.start()
    
    # Start checking if thread is done
    splash.after(100, check_thread)

# ---------------------------------------------------------
# 17) Compiler / Code Execution (Updated with Output Window)
# ---------------------------------------------------------
def execute_sql_query(query):
    """
    Updated: Removed all predefined statements/data, so the in-memory
    DB is empty unless the user creates tables/data. The code typed
    in the editor is executed exactly as written.
    """
    import sqlite3
    output = ""
    try:
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        try:
            cur.execute(query)
            if query.strip().lower().startswith("select"):
                rows = cur.fetchall()
                if rows:
                    output = "Query Results:\n" + "\n".join(str(row) for row in rows)
                else:
                    output = "Query executed successfully, but returned no results."
            else:
                conn.commit()
                output = f"Query executed successfully. {cur.rowcount} row(s) affected."
        except Exception as e:
            output = f"SQL Error: {e}"
    except Exception as e:
        output = f"Database Error: {e}"
    finally:
        conn.close()
    return output

# Judge0 Compiler Integration
JUDGE0_API_URL = "https://judge0-extra-ce.p.rapidapi.com/submissions"
JUDGE0_HEADERS = {
    "x-rapidapi-host": "judge0-extra-ce.p.rapidapi.com",
    "x-rapidapi-key": "09cb6663e3msh50cfbb5473450fbp164d40jsn09a180c9e327",  # Replace with a valid RapidAPI key
    "content-type": "application/json"
}

JUDGE0_LANGUAGE_IDS = {
    "Python": 71,       # Python 3
    "Java": 62,         # Java (OpenJDK 17)
    "C++": 54,          # C++ (GCC 9.2)
    "JavaScript": 63,   # Node.js
    "SQL": 82           # MySQL on Judge0, if you prefer
}

def create_submission_judge0(source_code, language_id, stdin=""):
    payload = {
        "source_code": source_code,
        "language_id": language_id,
        "stdin": stdin
    }
    try:
        response = requests.post(JUDGE0_API_URL, json=payload, headers=JUDGE0_HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data["token"]
    except requests.RequestException as e:
        log_event(f"Judge0 create_submission error: {e}")
        return None

def get_submission_result_judge0(token):
    result_url = f"{JUDGE0_API_URL}/{token}"
    while True:
        try:
            response = requests.get(result_url, headers=JUDGE0_HEADERS, timeout=15)
            response.raise_for_status()
            result = response.json()
            if result["status"]["id"] in [1, 2]:
                time.sleep(1)
            else:
                return result
        except requests.RequestException as e:
            log_event(f"Judge0 get_submission error: {e}")
            break
    return None

def show_output_window(result_text):
    """
    Creates a small pop-up window to display the result of code or SQL execution.
    """
    output_window = tk.Toplevel(root)
    output_window.title("Code Execution Output")
    output_window.geometry("600x400")
    output_window.configure(bg=MAIN_BG)

    header = tk.Label(output_window, text="Program Output", bg=MAIN_BG, fg=ACCENT_COLOR,
                      font=(FONT_FAMILY, 12, "bold"))
    header.pack(pady=5)

    st = scrolledtext.ScrolledText(output_window, wrap=tk.WORD, bg="#222222", fg=ACCENT_COLOR,
                                   font=("Consolas", 10))
    st.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    st.insert(tk.END, result_text)
    st.config(state=tk.DISABLED)

def run_code():
    global code_editor, language_var, code_output
    code = code_editor.get("1.0", tk.END).strip()
    language = language_var.get()

    if language == "SQL":
        # Locally execute the SQL query (unless you prefer Judge0 for SQL).
        output = execute_sql_query(code)
    else:
        # Use Judge0 for non-SQL languages
        language_id = JUDGE0_LANGUAGE_IDS.get(language, 71)  # default to Python=71 if unknown
        submission_token = create_submission_judge0(code, language_id)
        if not submission_token:
            output = "Error: Could not create submission on Judge0."
        else:
            result = get_submission_result_judge0(submission_token)
            if not result:
                output = "Error: Could not retrieve submission result from Judge0."
            else:
                status_desc = result["status"].get("description", "Unknown")
                stdout = result.get("stdout", "")
                stderr = result.get("stderr", "")
                compile_out = result.get("compile_output", "")
                message = result.get("message", "")

                parts = [f"Status: {status_desc}"]
                if compile_out:
                    parts.append(f"\nCompiler Output:\n{compile_out}")
                if stderr:
                    parts.append(f"\nRuntime Error(s):\n{stderr}")
                if stdout:
                    parts.append(f"\nProgram Output:\n{stdout}")
                if message:
                    parts.append(f"\nMessage:\n{message}")

                joined = "\n".join(part for part in parts if part).strip()
                output = joined if joined else "No output produced."

    # Update the code_output text widget
    code_output.config(state=tk.NORMAL)
    code_output.delete("1.0", tk.END)
    code_output.insert(tk.END, output)
    code_output.config(state=tk.DISABLED)

    # Also show a pop-up window with the result
    show_output_window(output)

def submit_challenge():
    global challenge_submitted
    challenge_submitted = True
    append_transcript(chat_display, "(Candidate has submitted the challenge solution.)")
    text_to_speech("Challenge solution submitted.")

    candidate_code = code_editor.get("1.0", tk.END).strip()
    if candidate_code:
        question_text = "Candidate's Code Submission"
        try:
            feedback = evaluate_response(question_text, candidate_code, interview_context)
            if feedback.strip():
                feed_line = f"Interviewer: {feedback}"
                append_transcript(chat_display, feed_line)
                text_to_speech(feedback)
        except Exception as e:
            log_event(f"Minor error in code evaluation: {e}")

    run_code_btn.config(state=tk.DISABLED)
    submit_btn.config(state=tk.DISABLED)
    code_editor.config(state=tk.DISABLED)

def on_code_editor_activity(event=None):
    global code_editor_last_activity_time, compiler_active
    code_editor_last_activity_time = time.time()
    compiler_active = True

# ---------------------------------------------------------
# 18) UI and Main Functions
# ---------------------------------------------------------
def on_close():
    """Clean application shutdown"""
    global interview_running, cap, voice_encoder, vad
    try:
        # Stop the interview if running
        interview_running = False
        
        # Release camera
        if cap and cap.isOpened():
            cap.release()
            
        # Cleanup face recognition
        cleanup_face_recognition()
        
        # Cleanup voice recognition resources
        try:
            voice_encoder = None
            vad = None
            log_event("Voice recognition resources cleaned up")
        except Exception as e:
            log_event(f"Error cleaning up voice recognition: {e}")
        
        # Cleanup MediaPipe resources
        if mp_face_mesh is not None:
            try:
                mp_face_mesh.close()
                log_event("MediaPipe resources cleaned up")
            except Exception as e:
                log_event(f"Error cleaning up MediaPipe: {e}")
            
        # Destroy all windows and quit
        root.quit()
        root.destroy()
        
    except Exception as e:
        log_event(f"Error during application shutdown: {e}")
        try:
            # Force quit in case of error
            root.destroy()
        except:
            pass

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
    """Start the interview process with enhanced error checking."""
    global interview_running, cap, multi_face_counter
    
    # Check if candidate registration is done
    if not registered_face_id:
        safe_showerror("Registration Required", "Please register your face before starting the interview.")
        return
    
    if not voice_reference_recorded:
        safe_showerror("Registration Required", "Please register your voice before starting the interview.")
        return
        
    # Release any existing camera
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass

    # Initialize camera with enhanced error handling
    cap = open_camera_for_windows()
    if cap is None or not cap.isOpened():
        # Retry with different camera index
        for idx in range(3):
            log_event(f"Retrying camera initialization with index {idx}")
            cap = open_camera_for_windows(idx)
            if cap is not None and cap.isOpened():
                break
        
        # If still not working, show error
        if cap is None or not cap.isOpened():
            safe_showerror("Camera Error", "Cannot open webcam for interview. Please check your camera connection and permissions.")
        return
    
    # Set camera properties for better quality
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception as e:
        log_event(f"Warning: Could not set camera properties: {e}")

    multi_face_counter = 0
    interview_running = True
    update_camera_view()

    threading.Thread(target=monitor_webcam, daemon=True).start()
    threading.Thread(target=monitor_background_audio, daemon=True).start()
    threading.Thread(target=interview_loop, args=(chat_display,), daemon=True).start()

    safe_update(start_button, start_button.config, state=tk.DISABLED)
    safe_update(stop_button, stop_button.config, state=tk.NORMAL)

def main_app():
    """
    Main application UI setup with enhanced design and user experience.
    Features modern layout, improved visual feedback, and better organization.
    """
    global root
    global bot_label, camera_label, chat_display
    global resume_entry, role_entry
    global start_button, stop_button
    global record_btn, stop_record_btn
    global warning_count_label, metrics_label
    global code_editor, run_code_btn, code_output, language_var
    global compiler_instructions_label, submit_btn
    global voice_encoder, vad, warning_labels
    
    # Initialize voice recognition if not already done
    if voice_encoder is None:
        try:
            voice_encoder = VoiceEncoder()
            log_event("Resemblyzer voice encoder initialized successfully")
        except Exception as e:
            log_event(f"Failed to initialize Resemblyzer: {e}")
    
    if vad is None:
        try:
            vad = webrtcvad.Vad(3)  # Aggressiveness mode 3 (highest)
            log_event("WebRTC VAD initialized successfully")
        except Exception as e:
            log_event(f"Failed to initialize WebRTC VAD: {e}")

    # Initialize warning labels list
    warning_labels = []

    # Setup root window
    root.title(APP_TITLE)
    root.geometry("1280x820")
    root.configure(bg=MAIN_BG)
    apply_theme()
    root.protocol("WM_DELETE_WINDOW", on_close)

    # Create top banner with gradient
    banner_height = 70
    gradient_canvas = tk.Canvas(root, height=banner_height, bd=0, highlightthickness=0)
    gradient_canvas.pack(fill=tk.X)

    def draw_grad(canv, ww, hh, c1=GRADIENT_START, c2=GRADIENT_END):
        """Draw a horizontal gradient"""
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
        """Handle window resize events"""
        gradient_canvas.delete("all")
        draw_grad(gradient_canvas, e.width, banner_height)
    
    # Draw initial gradient
    draw_grad(gradient_canvas, root.winfo_screenwidth(), banner_height)
    
    # Add app title to banner
    title_label = tk.Label(
        gradient_canvas, 
        text=APP_TITLE,
        font=(FONT_FAMILY, 24, "bold"),
        fg=MAIN_FG,
        bg=GLASS_BG  # Using a defined color instead of transparent
    )
    title_label.place(relx=0.5, y=banner_height//2, anchor=tk.CENTER)
    
    # Add version info
    version_label = tk.Label(
        gradient_canvas,
        text="Version 1.0",
        font=(FONT_FAMILY, 10),
        fg=MAIN_FG,
        bg=GLASS_BG  # Using a defined color instead of transparent
    )
    version_label.place(relx=0.95, y=banner_height-10, anchor=tk.SE)
    
    # Main content frame with three columns
    main_frame = tk.Frame(root, bg=MAIN_BG)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    # Left column (30% width) - Registration & Controls
    left_column = tk.Frame(main_frame, bg=GLASS_BG, bd=1, relief=tk.RAISED)
    left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5, ipadx=5, ipady=5)
    left_column.configure(width=int(root.winfo_screenwidth() * 0.25))
    
    # Add title for registration section
    registration_title = tk.Label(
        left_column,
        text="CANDIDATE REGISTRATION",
        font=(FONT_FAMILY, 14, "bold"),
        fg=ACCENT_COLOR,
        bg=GLASS_BG
    )
    registration_title.pack(pady=(10, 5), fill=tk.X)
    
    # Resume upload
    resume_frame = tk.Frame(left_column, bg=GLASS_BG)
    resume_frame.pack(fill=tk.X, padx=10, pady=5)
    
    resume_label = tk.Label(
        resume_frame,
        text="Resume (PDF):",
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        fg=MAIN_FG,
        bg=GLASS_BG,
        anchor=tk.W
    )
    resume_label.pack(side=tk.LEFT, padx=5)
    
    resume_entry = tk.Entry(resume_frame, width=20)
    resume_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    resume_button = tk.Button(
        resume_frame,
        text="Browse",
        command=browse_resume,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL)
    )
    resume_button.pack(side=tk.RIGHT, padx=5)
    
    # Role selection
    role_frame = tk.Frame(left_column, bg=GLASS_BG)
    role_frame.pack(fill=tk.X, padx=10, pady=5)
    
    role_label = tk.Label(
        role_frame,
        text="Job Role:",
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        fg=MAIN_FG,
        bg=GLASS_BG,
        anchor=tk.W
    )
    role_label.pack(side=tk.LEFT, padx=5)
    
    # Predefined roles for dropdown
    roles = [
        "Software Engineer",
        "Data Scientist",
        "Web Developer",
        "UX Designer",
        "Product Manager",
        "DevOps Engineer",
        "Data Analyst",
        "Project Manager",
        "Cybersecurity Analyst",
        "Business Analyst"
    ]
    
    role_var = tk.StringVar()
    role_dropdown = ttk.Combobox(
        role_frame,
        textvariable=role_var,
        values=roles,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL)
    )
    role_dropdown.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    role_dropdown.current(0)  # Set default to first role
    
    # Face registration section
    face_section = tk.Frame(left_column, bg=GLASS_BG)
    face_section.pack(fill=tk.X, padx=10, pady=5)
    
    face_label = tk.Label(
        face_section,
        text="Face Registration:",
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        fg=MAIN_FG,
        bg=GLASS_BG
    )
    face_label.pack(side=tk.LEFT, padx=5)
    
    face_button = tk.Button(
        face_section,
        text="Register Face",
        command=register_candidate_face,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL)
    )
    face_button.pack(side=tk.RIGHT, padx=5)
    
    # Voice registration section
    voice_section = tk.Frame(left_column, bg=GLASS_BG)
    voice_section.pack(fill=tk.X, padx=10, pady=5)
    
    voice_label = tk.Label(
        voice_section,
        text="Voice Registration:",
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        fg=MAIN_FG,
        bg=GLASS_BG
    )
    voice_label.pack(side=tk.LEFT, padx=5)
    
    voice_button = tk.Button(
        voice_section,
        text="Register Voice",
        command=record_voice_reference,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL)
    )
    voice_button.pack(side=tk.RIGHT, padx=5)
    
    # Separator
    ttk.Separator(left_column, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=15)
    
    # Interview controls section
    control_title = tk.Label(
        left_column,
        text="INTERVIEW CONTROLS",
        font=(FONT_FAMILY, 14, "bold"),
        fg=ACCENT_COLOR,
        bg=GLASS_BG
    )
    control_title.pack(pady=(5, 10), fill=tk.X)
    
    # Start/Stop buttons
    buttons_frame = tk.Frame(left_column, bg=GLASS_BG)
    buttons_frame.pack(fill=tk.X, padx=10, pady=5)
    
    start_button = tk.Button(
        buttons_frame,
        text="Start Interview",
        command=start_interview,
        bg="#28a745",  # Green for start
        fg=MAIN_FG,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
        height=2
    )
    start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    
    stop_button = tk.Button(
        buttons_frame,
        text="Stop Interview",
        command=stop_interview,
        bg="#dc3545",  # Red for stop
        fg=MAIN_FG,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
        state=tk.DISABLED,
        height=2
    )
    stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
    
    # Voice recording controls
    recording_frame = tk.Frame(left_column, bg=GLASS_BG)
    recording_frame.pack(fill=tk.X, padx=10, pady=10)
    
    record_btn = tk.Button(
        recording_frame,
        text="Record Response",
        command=start_recording_voice,
        bg="#17a2b8",  # Blue for record
        fg=MAIN_FG,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        state=tk.DISABLED,
        height=2
    )
    record_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    
    stop_record_btn = tk.Button(
        recording_frame,
        text="Stop Recording",
        command=stop_recording_voice,
        bg="#6c757d",  # Gray for stop recording
        fg=MAIN_FG,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        state=tk.DISABLED,
        height=2
    )
    stop_record_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
    
    # Warning and metrics section
    metrics_frame = tk.Frame(left_column, bg=GLASS_BG)
    metrics_frame.pack(fill=tk.X, padx=10, pady=10)
    
    warning_count_label = tk.Label(
        metrics_frame,
        text="Warnings: 0",
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        fg=MAIN_FG,
        bg=GLASS_BG
    )
    warning_count_label.pack(fill=tk.X, pady=5)
    
    metrics_label = tk.Label(
        metrics_frame,
        text="Identity Match: 0.00\nFace Quality: 0.00\nStatus: Not verified",
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        fg=MAIN_FG,
        bg=GLASS_BG,
        justify=tk.LEFT
    )
    metrics_label.pack(fill=tk.X, pady=5)
    
    # Center column (40% width) - Webcam and AI Bot
    center_column = tk.Frame(main_frame, bg=MAIN_BG)
    center_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Camera view (top of center column)
    camera_frame = tk.Frame(center_column, bg=GLASS_BG, bd=1, relief=tk.RAISED)
    camera_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
    
    camera_label = tk.Label(camera_frame, bg="#000000")
    camera_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    # AI Bot view (bottom of center column)
    bot_frame = tk.Frame(center_column, bg=GLASS_BG, bd=1, relief=tk.RAISED)
    bot_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
    
    bot_label = tk.Label(bot_frame, bg="#000000")
    bot_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    # Right column (30% width) - Chat & Code editor
    right_column = tk.Frame(main_frame, bg=MAIN_BG)
    right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    right_column.configure(width=int(root.winfo_screenwidth() * 0.35))
    
    # Chat transcript (top of right column)
    chat_frame = tk.Frame(right_column, bg=GLASS_BG, bd=1, relief=tk.RAISED)
    chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
    
    chat_title = tk.Label(
        chat_frame,
        text="INTERVIEW TRANSCRIPT",
        font=(FONT_FAMILY, 14, "bold"),
        fg=ACCENT_COLOR,
        bg=GLASS_BG
    )
    chat_title.pack(pady=(5, 0))
    
    chat_display = scrolledtext.ScrolledText(
        chat_frame,
        wrap=tk.WORD,
        font=("Consolas", 10),
        bg="#f8f9fa",
        fg="#212529"
    )
    chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    chat_display.config(state=tk.DISABLED)
    
    # Code editor (bottom of right column)
    code_frame = tk.Frame(right_column, bg=GLASS_BG, bd=1, relief=tk.RAISED)
    code_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
    
    code_title = tk.Label(
        code_frame,
        text="CODE EDITOR",
        font=(FONT_FAMILY, 14, "bold"),
        fg=ACCENT_COLOR,
        bg=GLASS_BG
    )
    code_title.pack(pady=(5, 0))
    
    # Language selection
    lang_frame = tk.Frame(code_frame, bg=GLASS_BG)
    lang_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
    
    lang_label = tk.Label(
        lang_frame,
        text="Language:",
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        fg=MAIN_FG,
        bg=GLASS_BG
    )
    lang_label.pack(side=tk.LEFT, padx=5)
    
    languages = [
        "Python (3.8.1)",
        "JavaScript (Node.js 12.14.0)",
        "Java (OpenJDK 13.0.1)",
        "C++ (GCC 9.2.0)",
        "C# (Mono 6.6.0.161)",
        "SQL (SQLite 3.30.1)"
    ]
    
    language_var = tk.StringVar()
    language_dropdown = ttk.Combobox(
        lang_frame,
        textvariable=language_var,
        values=languages,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        width=25
    )
    language_dropdown.pack(side=tk.LEFT, padx=5)
    language_dropdown.current(0)  # Set default to Python
    
    # Code editor buttons
    code_btn_frame = tk.Frame(lang_frame, bg=GLASS_BG)
    code_btn_frame.pack(side=tk.RIGHT)
    
    run_code_btn = tk.Button(
        code_btn_frame,
        text="Run Code",
        command=run_code,
        bg="#28a745",  # Green
        fg=BUTTON_FG,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        state=tk.DISABLED
    )
    run_code_btn.pack(side=tk.LEFT, padx=5)
    
    submit_btn = tk.Button(
        code_btn_frame,
        text="Submit",
        command=submit_challenge,
        bg="#007bff",  # Blue
        fg=BUTTON_FG,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        state=tk.DISABLED
    )
    submit_btn.pack(side=tk.LEFT, padx=5)
    
    # Code editor
    code_editor = scrolledtext.ScrolledText(
        code_frame,
        wrap=tk.WORD,
        font=("Consolas", 11),
        bg="#282c34",
        fg="#abb2bf",
        insertbackground="#abb2bf"
    )
    code_editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    code_editor.config(state=tk.DISABLED)
    code_editor.bind("<KeyRelease>", on_code_editor_activity)
    
    # Code output
    global code_output
    code_output = scrolledtext.ScrolledText(
        code_frame,
        height=5,
        wrap=tk.WORD,
        font=("Consolas", 10),
        bg="#f8f9fa",
        fg="#212529"
    )
    code_output.pack(fill=tk.X, expand=False, padx=5, pady=5)
    code_output.config(state=tk.DISABLED)
    
    global compiler_instructions_label
    compiler_instructions_label = tk.Label(
        code_frame, 
        text="Compiler Instructions: Click or type in the code editor to activate it. "
             "Use 'Run Code' to test and 'Submit Challenge' to finalize.",
        fg=MAIN_FG,
        bg=GLASS_BG,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL)
    )
    compiler_instructions_label.pack(padx=5, pady=5)
    
    # Bind resize event
    root.bind("<Configure>", on_resize)
    
    # Start webcam monitoring and animation
    root.after(100, animate_bot)
    
    # Check for registered voice
    def check_voice_reference_status():
        global voice_reference_recorded
        if voice_reference_recorded:
            voice_button.config(text="Voice Registered ✓", bg="#28a745")
        else:
            voice_button.config(text="Register Voice", bg=BUTTON_BG)
        root.after(1000, check_voice_reference_status)
    
    # Start checking voice reference status
    check_voice_reference_status()

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-6)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        # Set up logging
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        log_event("Application starting")
        
        # Initialize environment
        os.environ["PYTHONIOENCODING"] = "utf-8"
        
        # Create and configure root window
        root = tk.Tk()
        root.title(APP_TITLE)
        root.geometry("1280x820")
        root.configure(bg=MAIN_BG)
        
        # Apply custom theme
        apply_theme()
        
        # Set window close handler
        root.protocol("WM_DELETE_WINDOW", on_close)
        
        # Show loading splash screen and load models
        load_model_splash()
        
        # Start main event loop
        log_event("Main application loop started")
        root.mainloop()
        
    except Exception as e:
        error_msg = f"Critical error during application startup: {e}"
        print(error_msg)
        traceback.print_exc()
        
        try:
            # Log the error
            log_event(error_msg)
            log_event(traceback.format_exc())
            
            # Clean up
            if 'root' in locals() and root is not None:
                root.destroy()
                
            # Show error dialog
            messagebox.showerror("Application Error", 
                               f"The application encountered a critical error and needs to close.\n\n"
                               f"Error: {str(e)}\n\n"
                               f"Please check the logs in the '{LOG_DIR}' directory for details.")
        except:
            pass
    finally:
        log_event("Application terminated")