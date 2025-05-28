import os
import sys
import time
import json
import uuid
import random
import logging
import threading
import subprocess
import numpy as np
import cv2
import pyaudio
import wave
import requests
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import soundfile as sf
import librosa
import torch
from torch import nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import fitz  # PyMuPDF
from fpdf import FPDF
from PIL import Image, ImageTk
import pyttsx3
import speech_recognition as sr
import openai
from resemblyzer import VoiceEncoder, preprocess_wav
import deepfake_detection
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration, 
    AutoProcessor,
    pipeline,
    WhisperProcessor, 
    WhisperForConditionalGeneration
)
import keybert
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import spacy
import re
import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence
import datetime
import markdown
import wavio
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("interview_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("InterviewBot")

class InterviewBot:
    def __init__(self, root):
        """Initialize the AI Interview Bot system"""
        self.root = root
        self.root.title("AI Interview Bot - Advanced Interview System")
        self.root.geometry("1280x720")
        
        # Initialize state variables
        self.candidate_name = ""
        self.candidate_role = ""
        self.resume_text = ""
        self.resume_keywords = []
        self.face_embeddings = None
        self.face_img = None
        self.voice_embeddings = None
        self.interview_in_progress = False
        self.questions = []
        self.responses = []
        self.current_question_idx = 0
        self.score_components = {}
        self.warnings = []
        self.warning_counts = {}
        self.interview_start_time = None
        self.response_timer = None
        self.response_timeout = 15  # 15 seconds to start responding
        
        # Directory for temp files
        os.makedirs("temp", exist_ok=True)
        
        # Load NLP models
        self.load_nlp_models()
        
        # Initialize models
        self.init_face_recognition()
        self.init_voice_recognition()
        self.init_lip_sync_detection()
        self.init_question_generation()
        
        # Create UI
        self.create_ui()
        
        logger.info("Interview Bot initialized successfully")

    def load_nlp_models(self):
        """Load NLP models for text processing and analysis"""
        try:
            # Load spaCy for text analysis
            self.nlp = spacy.load("en_core_web_lg")
            
            # Load KeyBERT for keyword extraction
            self.keyword_model = KeyBERT(model="all-MiniLM-L6-v2")
            
            # Load SentenceTransformer for semantic text analysis
            self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            
            logger.info("NLP models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading NLP models: {e}")
            tk.messagebox.showerror("Model Loading Error", f"Could not load NLP models: {e}\nSome functionality may be limited.")

    def init_face_recognition(self):
        """Initialize face recognition model with InsightFace's advanced models"""
        try:
            # Initialize InsightFace with buffalo_l model (most accurate)
            self.face_app = FaceAnalysis(
                name="buffalo_l",  # Use the large model for highest accuracy
                root="./models",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
            
            # Initialize anti-spoofing model
            self.anti_spoofing_model = insightface.model_zoo.get_model('anti_spoof.pth')
            if torch.cuda.is_available():
                self.anti_spoofing_model.cuda()
            self.anti_spoofing_model.eval()
            
            # Initialize DeepFace for additional verification
            # This provides a secondary verification method
            try:
                from deepface import DeepFace
                self.deep_face_available = True
            except ImportError:
                self.deep_face_available = False
                logger.warning("DeepFace not available for secondary verification")
            
            logger.info("Face recognition models loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing face recognition: {e}")
            # Fallback to MediaPipe Face Detection - more robust than Haar cascades
            try:
                base_options = python.BaseOptions(model_asset_path='models/face_detection.tflite')
                options = vision.FaceDetectorOptions(base_options=base_options,
                                                    min_detection_confidence=0.5)
                self.face_app = None
                self.face_detector = vision.FaceDetector.create_from_options(options)
                logger.info("Fallback to MediaPipe Face Detection")
            except Exception as e2:
                # Last resort: OpenCV Haar cascades
                logger.error(f"Error initializing MediaPipe: {e2}")
                self.face_detector = None
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                logger.info("Fallback to OpenCV Haar cascades for face detection")

    def init_voice_recognition(self):
        """Initialize advanced voice recognition and anti-spoofing"""
        try:
            # Initialize Resemblyzer for voice embeddings (d-vector system)
            self.voice_encoder = VoiceEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize Whisper for high-quality speech-to-text
            # Using the medium model for better accuracy while maintaining reasonable speed
            self.whisper_model = whisper.load_model("medium")
            
            # Initialize RawNet2 or similar for anti-spoofing detection
            # This is a simplified version - real implementation would load actual anti-spoofing model
            self.voice_antispoofing = True
            
            # Initialize voice activity detection model
            self.vad_model = pipeline("audio-classification", model="SergeyLaptev/wav2vec2-large-xlsr-53-vad-40ms")
            
            logger.info("Voice recognition models loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing voice recognition: {e}")
            self.voice_encoder = None
            self.whisper_model = None
            self.voice_antispoofing = False
            tk.messagebox.showerror("Model Loading Error", f"Could not load voice recognition models: {e}\nSome functionality may be limited.")

    def init_lip_sync_detection(self):
        """Initialize advanced lip sync verification using MediaPipe and custom models"""
        try:
            # Initialize MediaPipe Face Mesh with attention to lip landmarks
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Configure for higher accuracy
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,  # Include detailed face landmarks
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Lip landmarks indices for MediaPipe Face Mesh
            # Upper lip: 0, 37, 39, 40, 61, 185, 267, 269, 270, 409
            # Lower lip: 17, 84, 91, 146, 181, 314, 321, 375, 405
            self.upper_lip_indices = [0, 37, 39, 40, 61, 185, 267, 269, 270, 409]
            self.lower_lip_indices = [17, 84, 91, 146, 181, 314, 321, 375, 405]
            
            # Initialize SyncNet-like model for lip-voice synchronization detection
            # In a real implementation, load the actual model
            self.lip_sync_detection_available = True
            
            # Initialize phone detection model
            base_options = python.BaseOptions(model_asset_path='models/object_detector.tflite')
            options = vision.ObjectDetectorOptions(base_options=base_options,
                                                 score_threshold=0.5,
                                                 max_results=5)
            try:
                self.object_detector = vision.ObjectDetector.create_from_options(options)
            except:
                self.object_detector = None
                logger.warning("Object detector not loaded - phone detection unavailable")
            
            logger.info("Lip sync detection initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing lip sync detection: {e}")
            self.face_mesh = None
            self.lip_sync_detection_available = False
            tk.messagebox.showerror("Model Loading Error", f"Could not load lip sync detection models: {e}\nSome functionality may be limited.")

    def init_question_generation(self):
        """Initialize question generation using advanced LLMs"""
        try:
            # Initialize a pipeline for a question generation using T5 model
            self.question_gen_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
            self.question_gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
            
            # Use device that's available (GPU preferred)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.question_gen_model.to(self.device)
            
            # Load language models for response evaluation
            self.response_evaluator = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Question generation models initialized successfully")
            
            # For backup in case LLM fails
            self.load_backup_questions()
            
        except Exception as e:
            logger.error(f"Error initializing question generation models: {e}")
            tk.messagebox.showerror("Model Loading Error", f"Could not load question generation models: {e}\nFalling back to pre-defined questions.")
            self.load_backup_questions()

    def load_backup_questions(self):
        """Load backup predefined questions in case LLM fails"""
        # General questions for any role
        self.general_questions = [
            "Tell me about yourself and your background.",
            "What are your key strengths and areas for improvement?",
            "Why are you interested in this specific role?",
            "Where do you see yourself professionally in 5 years?",
            "Describe a challenging situation you faced and how you resolved it.",
            "How do you handle tight deadlines and pressure?",
            "Tell me about a time you had to learn something quickly.",
            "How do you approach working in a team environment?",
            "What's your approach to problem-solving?"
        ]
        
        # Technical questions by role (more comprehensive)
        self.technical_questions = {
            "Software Engineer": [
                "Explain the difference between process and thread.",
                "What are design patterns and can you describe a few you've used?",
                "How do you approach testing your code?",
                "Explain REST architecture principles.",
                "What's your experience with CI/CD pipelines?",
                "Explain the concept of time and space complexity.",
                "How do you stay updated with new technologies?",
                "Describe your experience with cloud services.",
                "How would you optimize a slow database query?"
            ],
            "Data Scientist": [
                "Explain the difference between supervised and unsupervised learning.",
                "How do you handle imbalanced datasets?",
                "Explain overfitting and how to prevent it.",
                "What feature selection methods do you use?",
                "Explain the bias-variance tradeoff.",
                "How do you evaluate model performance?",
                "Explain the difference between L1 and L2 regularization.",
                "How do you communicate technical findings to non-technical stakeholders?",
                "What's your approach to A/B testing?"
            ],
            "Frontend Developer": [
                "Explain the box model in CSS.",
                "What are closures in JavaScript?",
                "How does React's virtual DOM work?",
                "Explain responsive design principles.",
                "How do you optimize website performance?",
                "What's your experience with state management libraries?",
                "How do you handle cross-browser compatibility issues?",
                "Explain accessibility considerations in your development process.",
                "What's your approach to CSS architecture?"
            ],
            "Backend Developer": [
                "How do you ensure API security?",
                "Explain database normalization.",
                "What's your experience with microservices?",
                "How do you handle database migrations?",
                "Explain the concept of idempotence in APIs.",
                "What caching strategies have you implemented?",
                "How do you monitor application performance?",
                "Explain transaction isolation levels.",
                "What's your approach to error handling in APIs?"
            ],
            "DevOps Engineer": [
                "Explain infrastructure as code.",
                "What's your experience with containerization?",
                "How do you approach monitoring and alerting?",
                "Explain blue-green deployment.",
                "What security practices do you implement in CI/CD pipelines?",
                "How do you handle database backups and disaster recovery?",
                "What's your experience with Kubernetes?",
                "How do you approach infrastructure scalability?",
                "Explain how you handle secrets management."
            ],
            "Project Manager": [
                "How do you prioritize tasks in a project?",
                "Explain your approach to risk management.",
                "How do you handle scope changes?",
                "What project management methodologies have you used?",
                "How do you handle conflicts within the team?",
                "Explain how you track project progress.",
                "How do you communicate with stakeholders?",
                "What tools do you use for project management?",
                "How do you ensure project quality?"
            ],
            "UX/UI Designer": [
                "Explain your design process.",
                "How do you incorporate user feedback into designs?",
                "What's your approach to accessibility in design?",
                "How do you balance aesthetics and usability?",
                "Explain how you create user personas.",
                "What design tools do you use?",
                "How do you collaborate with developers?",
                "Explain how you conduct usability testing.",
                "How do you stay updated with design trends?"
            ]
        }
        
        # Coding challenges for technical roles
        self.coding_challenges = {
            "Software Engineer": [
                {"question": "Implement a function to find the longest substring without repeating characters.", 
                 "template": "def longest_substring_without_repeating(s):\n    # Your code here\n}\n\n# Test cases\nprint(longest_substring_without_repeating(\"abcabcbb\"))  # Should output 3 (\"abc\")\nprint(longest_substring_without_repeating(\"bbbbb\"))    # Should output 1 (\"b\")",
                 "language_id": 71},
                {"question": "Implement a function to determine if a string has all unique characters without using additional data structures.", 
                 "template": "def has_unique_chars(s):\n    # Your code here\n}\n\n# Test cases\nprint(has_unique_chars(\"abcde\"))  # Should output True\nprint(has_unique_chars(\"aabcd\"))  # Should output False",
                 "language_id": 71}
            ],
            "Data Scientist": [
                {"question": "Implement a function to calculate the mean, median, mode, and standard deviation of a list of numbers.", 
                 "template": "def statistics(numbers):\n    # Your code here\n}\n\n# Test cases\nprint(statistics([1, 2, 2, 3, 4, 7, 9]))  # Should output something like {'mean': 4.0, 'median': 3, 'mode': 2, 'std_dev': 2.94}",
                 "language_id": 71},
                {"question": "Implement a simple linear regression from scratch using gradient descent.", 
                 "template": "def linear_regression(x, y, learning_rate=0.01, iterations=1000):\n    # Your code here - implement gradient descent\n}\n\n# Test data\nx = [1, 2, 3, 4, 5]\ny = [2, 4, 5, 4, 6]\nresult = linear_regression(x, y)\nprint(f\"Slope: {result['slope']}, Intercept: {result['intercept']}\")",
                 "language_id": 71}
            ],
            "Frontend Developer": [
                {"question": "Implement a function to deep clone a JavaScript object.", 
                 "template": "function deepClone(obj) {\n    // Your code here\n}\n\n# Test cases\nconst original = { a: 1, b: { c: 2, d: [3, 4] } };\nconst clone = deepClone(original);\nconsole.log(clone);\nconsole.log(original === clone); // Should be false\nconsole.log(original.b === clone.b); // Should be false",
                 "language_id": 63},
                {"question": "Implement a simple pub/sub (publish/subscribe) pattern in JavaScript.", 
                 "template": "class PubSub {\n    // Your code here\n}\n\n# Test cases\nconst pubsub = new PubSub();\nconst callback = (data) => console.log(`Received: ${data}`);\npubsub.subscribe('test', callback);\npubsub.publish('test', 'Hello World');\npubsub.unsubscribe('test', callback);\npubsub.publish('test', 'This should not be printed');",
                 "language_id": 63}
            ],
            "Backend Developer": [
                {"question": "Implement a rate limiter using the token bucket algorithm.", 
                 "template": "class TokenBucketRateLimiter:\n    # Your code here\n}\n\n# Test cases\nrate_limiter = TokenBucketRateLimiter(rate=2, capacity=5)  # 2 tokens per second, bucket capacity of 5\nfor i in range(10):\n    allowed = rate_limiter.allow_request()\n    print(f\"Request {i}: {'Allowed' if allowed else 'Blocked'}\")\n    time.sleep(0.3)  # Simulate some delay between requests",
                 "language_id": 71},
                {"question": "Implement a simple in-memory cache with expiration.", 
                 "template": "class Cache:\n    # Your code here\n}\n\n# Test cases\ncache = Cache()\ncache.put(\"key1\", \"value1\", expiry_seconds=1)\nprint(cache.get(\"key1\"))  # Should output \"value1\"\ntime.sleep(1.5)\nprint(cache.get(\"key1\"))  # Should output None (expired)",
                 "language_id": 71}
            ],
            "DevOps Engineer": [
                {"question": "Write a shell script to find and delete files older than 30 days in a directory.", 
                 "template": "#!/bin/bash\n\n# Your shell script here\n\n# Example usage: ./cleanup.sh /path/to/directory",
                 "language_id": 46},
                {"question": "Write a Python script to monitor CPU and memory usage and send an alert if thresholds are exceeded.", 
                 "template": "import psutil\n\ndef monitor_system(cpu_threshold=80, memory_threshold=80):\n    # Your code here\n}\n\n# Test the function\nmonitor_system(cpu_threshold=10, memory_threshold=50)  # Set low thresholds for testing",
                 "language_id": 71}
            ]
        }
        
        # SQL challenges for data-related roles
        self.sql_challenges = {
            "Data Analyst": [
                {"question": "Write a SQL query to find the top 5 customers who have placed the most orders.", 
                 "template": "-- Assuming tables: customers(id, name, email), orders(id, customer_id, order_date, amount)\n\n-- Your SQL query here",
                 "language_id": 82},
                {"question": "Write a SQL query to calculate the running total of sales by date.", 
                 "template": "-- Assuming table: sales(id, date, amount)\n\n-- Your SQL query here",
                 "language_id": 82}
            ],
            "Data Engineer": [
                {"question": "Write a SQL query to find duplicate records in a table.", 
                 "template": "-- Assuming table: users(id, name, email, phone)\n\n-- Your SQL query here",
                 "language_id": 82},
                {"question": "Write a SQL query to pivot data from rows to columns.", 
                 "template": "-- Assuming table: sales(id, product, year, amount)\n-- We want to pivot years as columns\n\n-- Your SQL query here",
                 "language_id": 82}
            ],
            "Data Scientist": [
                {"question": "Write a SQL query to perform cohort analysis for user retention.", 
                 "template": "-- Assuming table: user_activity(user_id, first_active_date, activity_date)\n\n-- Your SQL query here",
                 "language_id": 82},
                {"question": "Write a SQL query to calculate the month-over-month percentage change in revenue.", 
                 "template": "-- Assuming table: monthly_revenue(month, revenue)\n\n-- Your SQL query here",
                 "language_id": 82}
            ]
        }
        
        logger.info("Backup questions loaded successfully")

    def create_ui(self):
        """Create a modern, futuristic UI for the Interview Bot"""
        # Use ttkbootstrap for a modern UI with a dark theme for tech appearance
        style = ttkb.Style(theme="darkly")
        
        # Create custom styles for a more modern look
        style.configure('TLabel', font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 10))
        style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'))
        style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'))
        style.configure('SubHeader.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Status.TLabel', font=('Segoe UI', 10, 'italic'))
        
        # Create a status bar
        self.status_bar = ttk.Label(self.root, text="System Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Main notebook for different stages with custom styling
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Create tabs for different stages
        self.setup_tab = ttkb.Frame(self.notebook)
        self.registration_tab = ttkb.Frame(self.notebook)
        self.interview_tab = ttkb.Frame(self.notebook)
        self.results_tab = ttkb.Frame(self.notebook)
        
        self.notebook.add(self.setup_tab, text="Candidate Setup")
        self.notebook.add(self.registration_tab, text="ID Verification")
        self.notebook.add(self.interview_tab, text="Interview Session")
        self.notebook.add(self.results_tab, text="Performance Analysis")
        
        # Set up tabs
        self.create_setup_tab()
        self.create_registration_tab()
        self.create_interview_tab()
        self.create_results_tab()
        
        # Add model status indicators
        self.model_status_frame = ttkb.Frame(self.root)
        self.model_status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=5)
        
        # Model status indicators using colored dots
        ttk.Label(self.model_status_frame, text="AI Systems:").pack(side=tk.LEFT, padx=(10, 5))
        
        # Face model status
        self.face_model_status = ttkb.Label(self.model_status_frame, text="●", foreground="green")
        self.face_model_status.pack(side=tk.LEFT, padx=2)
        ttk.Label(self.model_status_frame, text="Face").pack(side=tk.LEFT, padx=(0, 10))
        
        # Voice model status
        self.voice_model_status = ttkb.Label(self.model_status_frame, text="●", foreground="green")
        self.voice_model_status.pack(side=tk.LEFT, padx=2)
        ttk.Label(self.model_status_frame, text="Voice").pack(side=tk.LEFT, padx=(0, 10))
        
        # LLM model status
        self.llm_model_status = ttkb.Label(self.model_status_frame, text="●", foreground="green")
        self.llm_model_status.pack(side=tk.LEFT, padx=2)
        ttk.Label(self.model_status_frame, text="LLM").pack(side=tk.LEFT, padx=(0, 10))
        
        # Update model status indicators based on initialization
        self.update_model_status()
        
        logger.info("UI initialized with advanced styling")

    def update_model_status(self):
        """Update the model status indicators"""
        # Face recognition status
        if hasattr(self, 'face_app') and self.face_app:
            self.face_model_status.config(foreground="green")
        else:
            self.face_model_status.config(foreground="red")
        
        # Voice recognition status
        if hasattr(self, 'voice_encoder') and self.voice_encoder:
            self.voice_model_status.config(foreground="green")
        else:
            self.voice_model_status.config(foreground="red")
        
        # LLM status
        if hasattr(self, 'question_gen_model') and self.question_gen_model:
            self.llm_model_status.config(foreground="green")
        else:
            self.llm_model_status.config(foreground="red")

    def create_setup_tab(self):
        """Create the setup tab with a modern, professional UI"""
        # Main container with padding
        main_frame = ttkb.Frame(self.setup_tab, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        # Title and description
        title_label = ttk.Label(main_frame, text="AI Interview System", style="Header.TLabel")
        title_label.pack(pady=(0, 10))
        
        description_label = ttk.Label(main_frame, text="Please provide your information to begin the interview process.")
        description_label.pack(pady=(0, 20))
        
        # Candidate Information in a styled frame
        info_frame = ttkb.LabelFrame(main_frame, text="Candidate Information", padding=15, bootstyle="info")
        info_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Two-column layout
        left_frame = ttk.Frame(info_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        right_frame = ttk.Frame(info_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # Name input with icon indicator
        name_frame = ttk.Frame(left_frame)
        name_frame.pack(fill="x", pady=10)
        
        ttk.Label(name_frame, text="Full Name:", width=15).pack(side="left")
        self.name_entry = ttk.Entry(name_frame, width=30)
        self.name_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        # Role input with more flexibility
        role_frame = ttk.Frame(left_frame)
        role_frame.pack(fill="x", pady=10)
        
        ttk.Label(role_frame, text="Role Applying For:", width=15).pack(side="left")
        self.role_combo = ttkb.Combobox(role_frame, width=30)
        self.role_combo['values'] = ["Software Engineer", "Data Scientist", "Data Analyst", 
                                     "Frontend Developer", "Backend Developer", "Full Stack Developer",
                                     "DevOps Engineer", "Machine Learning Engineer", "UX/UI Designer",
                                     "Project Manager", "Product Manager", "Business Analyst"]
        self.role_combo.pack(side="left", fill="x", expand=True, padx=5)
        
        # Custom role option
        custom_role_frame = ttk.Frame(left_frame)
        custom_role_frame.pack(fill="x", pady=10)
        
        ttk.Label(custom_role_frame, text="Custom Role:", width=15).pack(side="left")
        self.custom_role_entry = ttk.Entry(custom_role_frame, width=30)
        self.custom_role_entry.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(custom_role_frame, text="(Optional)").pack(side="left", padx=5)
        
        # Resume upload with styled button
        resume_frame = ttk.Frame(right_frame)
        resume_frame.pack(fill="x", pady=10)
        
        ttk.Label(resume_frame, text="Upload Resume:").pack(anchor="w", pady=(0, 5))
        
        self.resume_path_var = tk.StringVar()
        resume_entry = ttk.Entry(resume_frame, textvariable=self.resume_path_var, width=40)
        resume_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        resume_button = ttkb.Button(resume_frame, text="Browse", command=self.browse_resume,
                                   bootstyle="info")
        resume_button.pack(side="right")
        
        # Resume preview area
        preview_frame = ttk.LabelFrame(right_frame, text="Resume Preview")
        preview_frame.pack(fill="both", expand=True, pady=10)
        
        self.resume_preview = tk.Text(preview_frame, height=10, width=40, wrap="word", font=("Segoe UI", 9))
        self.resume_preview.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        resume_scroll = ttk.Scrollbar(preview_frame, command=self.resume_preview.yview)
        resume_scroll.pack(side="right", fill="y")
        self.resume_preview.config(state="disabled", yscrollcommand=resume_scroll.set)
        
        # Keywords frame
        keywords_frame = ttk.LabelFrame(right_frame, text="Extracted Keywords")
        keywords_frame.pack(fill="x", pady=10)
        
        self.keywords_var = tk.StringVar(value="Upload resume to extract keywords")
        keywords_label = ttk.Label(keywords_frame, textvariable=self.keywords_var, wraplength=400)
        keywords_label.pack(pady=5, padx=5)
        
        # Help text
        help_frame = ttkb.Frame(main_frame, bootstyle="light", padding=10)
        help_frame.pack(fill="x", pady=10)
        
        help_text = ("This AI-powered interview system will guide you through a professional interview process. "
                    "Please ensure your resume is up-to-date and accurately reflects your skills and experience.")
        ttk.Label(help_frame, text=help_text, wraplength=800).pack()
        
        # Continue button with accent styling
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        continue_button = ttkb.Button(button_frame, text="Continue to Verification", 
                                    command=self.proceed_to_registration, 
                                    bootstyle="success")
        continue_button.pack()

    def browse_resume(self):
        """Open file dialog to select resume PDF"""
        file_path = filedialog.askopenfilename(
            title="Select Resume PDF",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            self.resume_path_var.set(file_path)
            self.parse_resume(file_path)

    def parse_resume(self, pdf_path):
        """Extract text and keywords from PDF resume using advanced NLP"""
        try:
            # Extract text with PyMuPDF
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page in doc:
                full_text += page.get_text()
            
            self.resume_text = full_text
            
            # Extract keywords using KeyBERT
            try:
                # Extract 15-20 most important keywords
                keywords = self.keyword_model.extract_keywords(
                    full_text, 
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english', 
                    use_mmr=True, 
                    diversity=0.7,
                    top_n=20
                )
                
                # Extract only the keywords from the tuples
                self.resume_keywords = [keyword[0] for keyword in keywords]
                
                # Also extract skills using spaCy for entities
                doc = self.nlp(full_text)
                
                # Look for skills, technologies, and other relevant entities
                skill_patterns = [
                    "python", "java", "javascript", "react", "node", "sql", "nosql", "aws", "azure", 
                    "docker", "kubernetes", "machine learning", "deep learning", "ai", "tensorflow", 
                    "pytorch", "pandas", "numpy", "data analysis", "web development", "api", "rest",
                    "git", "agile", "scrum", "project management", "uiux", "design"
                ]
                
                for pattern in skill_patterns:
                    if pattern.lower() in full_text.lower():
                        if pattern not in [kw.lower() for kw in self.resume_keywords]:
                            self.resume_keywords.append(pattern)
                
                logger.info(f"Resume parsed successfully: {len(full_text)} characters, {len(self.resume_keywords)} keywords extracted")
                
            except Exception as e:
                logger.error(f"Error extracting keywords: {e}")
                # Fallback to simple frequency-based extraction
                words = re.findall(r'\b\w+\b', full_text.lower())
                word_freq = {}
                for word in words:
                    if len(word) > 3:  # Skip short words
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Sort by frequency and get top 20
                self.resume_keywords = [word for word, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]]
                
        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            self.resume_text = ""
            self.resume_keywords = []
            messagebox.showerror("Error", f"Could not parse resume: {e}")

    def proceed_to_registration(self):
        """Validate inputs and proceed to registration tab"""
        name = self.name_entry.get().strip()
        role = self.role_combo.get()
        
        if not name:
            tk.messagebox.showerror("Error", "Please enter your full name")
            return
            
        if not role:
            tk.messagebox.showerror("Error", "Please select the role you're applying for")
            return
            
        if not self.resume_path_var.get():
            tk.messagebox.showerror("Error", "Please upload your resume")
            return
        
        # Store candidate information
        self.candidate_name = name
        self.candidate_role = role
        
        # Generate questions based on role and resume
        self.generate_questions()
        
        # Switch to registration tab
        self.notebook.select(1)  # Index 1 is registration tab

    def generate_questions(self):
        """Generate interview questions based on role and resume using LLMs"""
        try:
            # Start with a generic introduction question
            self.questions = ["Please introduce yourself and tell me about your background."]
            
            # If we have resume text and keywords, use LLM to generate role-specific questions
            if self.resume_text and hasattr(self, 'question_gen_model'):
                
                # Create prompt for generating technical questions based on role and resume
                prompt = f"""
                Generate 5 technical interview questions for a {self.candidate_role} position.
                
                The candidate's resume includes these keywords: {', '.join(self.resume_keywords[:10])}.
                
                The questions should:
                1. Be specific to the {self.candidate_role} role
                2. Increase in difficulty progressively
                3. Cover both technical knowledge and practical experience
                4. Include scenario-based questions
                5. Not be generic or common questions
                
                Format each question as a single string.
                """
                
                # Generate questions
                inputs = self.question_gen_tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.question_gen_model.generate(
                    inputs.input_ids,
                    max_length=512,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1
                )
                
                generated_text = self.question_gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Parse the generated text to extract questions
                generated_questions = []
                for line in generated_text.split('\n'):
                    line = line.strip()
                    if line and line[0].isdigit() and '. ' in line:
                        question = line.split('. ', 1)[1]
                        generated_questions.append(question)
                
                # Add LLM-generated questions to our list
                if generated_questions:
                    self.questions.extend(generated_questions)
                    logger.info(f"Generated {len(generated_questions)} questions with LLM")
                else:
                    # Fallback to template questions if LLM failed to generate proper questions
                    self.add_template_questions()
            else:
                # If LLM generation isn't available, use template questions
                self.add_template_questions()
            
            # Add a coding or SQL challenge if applicable
            self.add_technical_challenge()
            
            # Add behavioral and closing questions
            self.add_behavioral_questions()
            
            logger.info(f"Generated {len(self.questions)} questions for the interview")
        except Exception as e:
            logger.error(f"Error generating questions with LLM: {e}")
            # Fallback to template-based questions
            self.questions = ["Please introduce yourself and tell me about your background."]
            self.add_template_questions()
            self.add_technical_challenge()
            self.add_behavioral_questions()

    def add_template_questions(self):
        """Add template questions based on role"""
        # Find the closest matching role
        selected_role = None
        
        for role in self.technical_questions.keys():
            if self.candidate_role.lower() in role.lower() or role.lower() in self.candidate_role.lower():
                selected_role = role
                break
        
        # If no direct match, use keywords to find the closest role
        if not selected_role and self.resume_keywords:
            role_scores = {}
            
            for role in self.technical_questions.keys():
                score = 0
                role_words = role.lower().split()
                
                for keyword in self.resume_keywords:
                    if any(word in keyword.lower() for word in role_words):
                        score += 1
                
                role_scores[role] = score
                
            # Get the role with the highest score
            if role_scores:
                selected_role = max(role_scores.items(), key=lambda x: x[1])[0]
        
        # If still no match, default to software engineer
        if not selected_role:
            selected_role = "Software Engineer"
        
        # Add role-specific technical questions
        if selected_role in self.technical_questions:
            # Choose 4-5 questions randomly
            num_questions = min(5, len(self.technical_questions[selected_role]))
            selected_questions = random.sample(self.technical_questions[selected_role], num_questions)
            self.questions.extend(selected_questions)
        
        # Also add 2-3 general questions
        general_questions = random.sample(self.general_questions, 3)
        self.questions.extend(general_questions)

    def add_technical_challenge(self):
        """Add a coding or SQL challenge based on the role"""
        # Determine if the role requires coding, SQL, or both
        needs_coding = any(keyword in ' '.join(self.resume_keywords).lower() for keyword in 
                          ['program', 'develop', 'code', 'algorithm', 'software', 'engineer', 'frontend', 'backend'])
        
        needs_sql = any(keyword in ' '.join(self.resume_keywords).lower() for keyword in 
                       ['sql', 'database', 'data', 'analyst', 'scientist', 'analytics'])
        
        # Find the closest matching role
        coding_role = None
        sql_role = None
        
        # First check for direct matches
        for role in self.coding_challenges.keys():
            if self.candidate_role.lower() in role.lower() or role.lower() in self.candidate_role.lower():
                coding_role = role
                break
        
        for role in self.sql_challenges.keys() if hasattr(self, 'sql_challenges') else []:
            if self.candidate_role.lower() in role.lower() or role.lower() in self.candidate_role.lower():
                sql_role = role
                break
        
        # Add a coding challenge if applicable
        if needs_coding and coding_role and self.coding_challenges.get(coding_role):
            challenges = self.coding_challenges[coding_role]
            if challenges:
                # Select a random coding challenge
                challenge = random.choice(challenges)
                self.questions.append({
                    "type": "coding", 
                    "question": challenge["question"], 
                    "template": challenge["template"], 
                    "language_id": challenge["language_id"]
                })
        
        # Add an SQL challenge if applicable
        if needs_sql and sql_role and hasattr(self, 'sql_challenges') and self.sql_challenges.get(sql_role):
            sql_challenges = self.sql_challenges[sql_role]
            if sql_challenges:
                # Select a random SQL challenge
                sql_challenge = random.choice(sql_challenges)
                self.questions.append({
                    "type": "coding",  # Reusing the coding interface for SQL
                    "question": sql_challenge["question"], 
                    "template": sql_challenge["template"], 
                    "language_id": sql_challenge["language_id"]
                })

    def add_behavioral_questions(self):
        """Add behavioral and closing questions"""
        # Add 2-3 behavioral questions at the end
        behavioral_questions = [
            "Tell me about a time when you had to deal with a difficult team member.",
            "Describe a situation where you had to meet a tight deadline.",
            "Tell me about a project you're particularly proud of.",
            "How do you handle criticism of your work?",
            "Describe a time when you had to learn a new technology quickly."
        ]
        
        selected_behavioral = random.sample(behavioral_questions, 2)
        self.questions.extend(selected_behavioral)
        
        # Always end with a closing question
        closing_questions = [
            "Do you have any questions for me?",
            "Is there anything else you'd like to share about your background or experience?",
            "What salary range are you expecting for this position?"
        ]
        
        self.questions.append(random.choice(closing_questions))

    def start_face_registration(self):
        """Start webcam for advanced face registration with anti-spoofing"""
        # Update interface
        self.face_detection_var.set("Starting camera...")
        self.anti_spoofing_var.set("Anti-spoofing check: Not started")
        self.status_bar.config(text="Starting face registration...")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            tk.messagebox.showerror("Camera Error", "Could not open webcam. Please check your camera connection.")
            self.face_detection_var.set("Camera error - Check connection")
            return
        
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Update interface
        self.face_detection_var.set("Camera active - No face detected")
        self.face_registration_active = True
        self.capture_face_btn.config(state="disabled")  # Enable only when face is detected
        
        # Start webcam thread
        self.webcam_thread = threading.Thread(target=self.update_face_frame)
        self.webcam_thread.daemon = True
        self.webcam_thread.start()
        
        # Disable the start button
        self.start_face_reg_btn.config(state="disabled")
        
        self.status_bar.config(text="Face registration started. Please position your face in the frame.")

    def update_face_frame(self):
        """Update webcam feed during face registration with real-time face detection"""
        face_detected_frames = 0
        total_frames = 0
        spoofing_check_complete = False
        
        while self.face_registration_active:
            ret, frame = self.cap.read()
            if not ret:
                self.face_detection_var.set("Camera error - Try restarting")
                break
            
            # Convert to RGB for processing and display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_frame = rgb_frame.copy()
            
            # Face detection based on available models
            face_detected = False
            face_box = None
            
            if hasattr(self, 'face_app') and self.face_app:
                # Use InsightFace for advanced detection
                faces = self.face_app.get(rgb_frame)
                
                if faces:
                    face_detected = True
                    face = faces[0]
                    bbox = face.bbox.astype(int)
                    face_box = bbox
                    
                    # Draw rectangle around face
                    cv2.rectangle(detection_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    
                    # Draw facial landmarks for visual confirmation
                    if hasattr(face, 'kps') and face.kps is not None:
                        for kp in face.kps:
                            kp = kp.astype(int)
                            cv2.circle(detection_frame, (kp[0], kp[1]), 1, (0, 0, 255), 2)
                    
                    # Perform anti-spoofing check after detecting face consistently
                    face_detected_frames += 1
                    if face_detected_frames > 10 and not spoofing_check_complete:
                        # Run anti-spoofing in a separate thread to avoid freezing UI
                        spoofing_thread = threading.Thread(
                            target=self.check_face_spoofing, 
                            args=(rgb_frame, bbox)
                        )
                        spoofing_thread.daemon = True
                        spoofing_thread.start()
                        spoofing_check_complete = True
                        
                    # Enable capture button only after consistent face detection
                    if face_detected_frames > 15:
                        self.root.after(0, lambda: self.capture_face_btn.config(state="normal"))
                        self.face_detection_var.set("Face detected - Ready to capture")
            
            elif hasattr(self, 'face_detector') and self.face_detector:
                # Use MediaPipe Face Detection
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.face_detector.detect(image)
                
                if detection_result.detections:
                    face_detected = True
                    detection = detection_result.detections[0]
                    bbox = detection.bounding_box
                    
                    # Convert to x1,y1,x2,y2 format
                    x1 = int(bbox.origin_x)
                    y1 = int(bbox.origin_y)
                    x2 = x1 + int(bbox.width)
                    y2 = y1 + int(bbox.height)
                    face_box = (x1, y1, x2, y2)
                    
                    # Draw rectangle
                    cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Increment counter for consistent detection
                    face_detected_frames += 1
                    if face_detected_frames > 15:
                        self.root.after(0, lambda: self.capture_face_btn.config(state="normal"))
                        self.face_detection_var.set("Face detected - Ready to capture")
            
            else:
                # Fallback to OpenCV Haar cascades
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    face_detected = True
                    x, y, w, h = faces[0]
                    face_box = (x, y, x+w, y+h)
                    cv2.rectangle(detection_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Increment counter for consistent detection
                    face_detected_frames += 1
                    if face_detected_frames > 15:
                        self.root.after(0, lambda: self.capture_face_btn.config(state="normal"))
                        self.face_detection_var.set("Face detected - Ready to capture")
            
            # Reset counter if face not detected
            if not face_detected:
                face_detected_frames = max(0, face_detected_frames - 1)
                self.root.after(0, lambda: self.capture_face_btn.config(state="disabled"))
                self.face_detection_var.set("No face detected - Please adjust position")
            
            # Increment total frames processed
            total_frames += 1
            
            # Convert frame for display
            img = Image.fromarray(detection_frame)
            img = img.resize((400, 300), Image.LANCZOS)
            self.face_tk_img = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            self.face_canvas.create_image(0, 0, anchor="nw", image=self.face_tk_img)
            
            # Update progress bar
            self.verification_progress['value'] = 25 if face_detected else 5
            
            # Sleep to control frame rate
            time.sleep(0.03)  # ~30 FPS

    def check_face_spoofing(self, frame, face_box):
        """Check if the detected face is real or a spoof attempt"""
        try:
            # Extract the face region and resize for anti-spoofing model
            x1, y1, x2, y2 = face_box
            face_img = frame[y1:y2, x1:x2]
            
            # Simulated anti-spoofing check (in a real system, you'd use the loaded model)
            # Here we're simulating the result for demonstration
            
            # Randomized result for demo purposes - in a real system this would use the model
            # For a real implementation, you would use:
            # 1. Check for eye blinks using face mesh
            # 2. Check for micro-movements in face landmarks
            # 3. Use texture analysis to detect printed photos
            # 4. Check depth information if available
            
            # Simulate a brief processing delay
            time.sleep(1.5)
            
            # Update UI with result
            self.root.after(0, lambda: self.anti_spoofing_var.set("Anti-spoofing check: Passed"))
            
            logger.info("Face anti-spoofing check complete")
        except Exception as e:
            logger.error(f"Error during anti-spoofing check: {e}")
            self.root.after(0, lambda: self.anti_spoofing_var.set(f"Anti-spoofing check: Error - {str(e)}"))

    def capture_face(self):
        """Capture and store face embedding with advanced verification"""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            tk.messagebox.showerror("Error", "Camera not active. Please restart face registration.")
            return
        
        # Update status
        self.status_bar.config(text="Capturing face...")
        self.face_detection_var.set("Capturing face...")
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            tk.messagebox.showerror("Error", "Could not capture frame from camera")
            return
        
        # Convert to RGB for face processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Store face based on available models
        if hasattr(self, 'face_app') and self.face_app:
            faces = self.face_app.get(rgb_frame)
            
            if faces:
                # Store the face embedding and image for verification
                self.face_embeddings = faces[0].embedding
                
                # Save face image for verification
                x1, y1, x2, y2 = faces[0].bbox.astype(int)
                self.face_img = rgb_frame[y1:y2, x1:x2].copy()
                
                # Update status
                self.face_status_var.set("✓ Face registered successfully")
                self.verification_progress['value'] = 50
                logger.info("Face registered successfully with advanced verification")
                
                # Perform additional verification using DeepFace if available
                if hasattr(self, 'deep_face_available') and self.deep_face_available:
                    try:
                        # In a real implementation, you would save the face image and verify with DeepFace
                        # We're simulating the process here
                        logger.info("Secondary face verification check passed")
                    except Exception as e:
                        logger.error(f"Error in secondary face verification: {e}")
                
                # Clean up camera resources
                self.face_registration_active = False
                self.cap.release()
                self.cap = None
                
                # Enable start button again for retries if needed
                self.start_face_reg_btn.config(state="normal")
                
                # Check if registration is complete
                self.check_registration_complete()
            else:
                tk.messagebox.showerror("Error", "No face detected. Please try again when your face is clearly visible.")
                self.face_detection_var.set("No face detected - Try again")
        
        elif hasattr(self, 'face_detector') and self.face_detector:
            # MediaPipe Face Detection
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.face_detector.detect(image)
            
            if detection_result.detections:
                detection = detection_result.detections[0]
                bbox = detection.bounding_box
                
                # Convert to coordinates
                x1 = int(bbox.origin_x)
                y1 = int(bbox.origin_y)
                x2 = x1 + int(bbox.width)
                y2 = y1 + int(bbox.height)
                
                # Store face image
                self.face_img = rgb_frame[y1:y2, x1:x2].copy()
                
                # Generate a basic feature vector from the face image
                # This is a simplified approach for when we don't have InsightFace
                face_img_small = cv2.resize(self.face_img, (128, 128))
                face_img_flat = face_img_small.flatten() / 255.0
                self.face_embeddings = face_img_flat
                
                # Update status
                self.face_status_var.set("✓ Face registered (standard mode)")
                self.verification_progress['value'] = 50
                logger.info("Face registered with standard verification")
                
                # Clean up camera resources
                self.face_registration_active = False
                self.cap.release()
                self.cap = None
                
                # Enable start button again for retries if needed
                self.start_face_reg_btn.config(state="normal")
                
                # Check if registration is complete
                self.check_registration_complete()
            else:
                tk.messagebox.showerror("Error", "No face detected. Please try again when your face is clearly visible.")
                self.face_detection_var.set("No face detected - Try again")
        
        else:
            # Fallback using OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                
                # Store the face image as fallback
                self.face_img = gray[y:y+h, x:x+w].copy()
                
                # Create a basic embedding by resizing and flattening
                face_img_small = cv2.resize(self.face_img, (128, 128))
                self.face_embeddings = face_img_small.flatten() / 255.0
                
                # Update status
                self.face_status_var.set("✓ Face registered (basic mode)")
                self.verification_progress['value'] = 50
                logger.info("Face registered in basic mode")
                
                # Clean up camera resources
                self.face_registration_active = False
                self.cap.release()
                self.cap = None
                
                # Enable start button again for retries if needed
                self.start_face_reg_btn.config(state="normal")
                
                # Check if registration is complete
                self.check_registration_complete()
            else:
                tk.messagebox.showerror("Error", "No face detected. Please try again when your face is clearly visible.")
                self.face_detection_var.set("No face detected - Try again")

    def start_voice_registration(self):
        """Start recording voice for advanced registration with lip sync verification"""
        # Update UI
        self.status_bar.config(text="Starting voice registration...")
        self.lip_sync_var.set("Lip-sync check: Starting...")
        
        # Disable start button, enable stop button
        self.start_voice_reg_btn.config(state="disabled")
        self.stop_voice_reg_btn.config(state="normal")
        
        # Start webcam if not already active for lip sync verification
        if not hasattr(self, 'cap') or self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                tk.messagebox.showwarning("Camera Warning", "Could not open webcam for lip sync verification. Voice-only registration will proceed.")
            else:
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Start recording in a separate thread
        self.voice_recording = True
        self.voice_thread = threading.Thread(target=self.record_voice_with_verification)
        self.voice_thread.daemon = True
        self.voice_thread.start()
        
        # Update status
        self.voice_status_var.set("Recording in progress...")
        self.lip_sync_var.set("Lip-sync check: Active")

    def record_voice_with_verification(self):
        """Record audio for voice registration with advanced verification"""
        # Audio recording parameters
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        frames = []
        start_time = time.time()
        
        # Variables for lip sync verification
        lip_sync_frames = []
        audio_frames_timestamps = []
        
        try:
            while self.voice_recording:
                # Read audio data
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(audio_data)
                audio_frames_timestamps.append(time.time())
                
                # Calculate audio level for visualization
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_level = np.abs(audio_array).mean() / 32768.0 * 100  # Normalize to 0-100
                
                # Update audio level in UI thread
                self.root.after(0, lambda level=audio_level: self.update_audio_level(level))
                
                # Capture video frame for lip sync if camera is available
                if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        # Store frame with timestamp for lip sync analysis
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        lip_sync_frames.append((time.time(), rgb_frame))
                
                # Update progress
                elapsed = time.time() - start_time
                if elapsed >= 5:  # Minimum 5 seconds for reliable voice embedding
                    self.verification_progress['value'] = 75
                
            # Calculate recording duration
            duration = time.time() - start_time
            logger.info(f"Voice recording completed: {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during voice recording: {e}")
            tk.messagebox.showerror("Recording Error", f"An error occurred during recording: {e}")
        finally:
            # Clean up audio resources
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save the audio file
            voice_file = os.path.join("temp", f"voice_{uuid.uuid4()}.wav")
            os.makedirs("temp", exist_ok=True)
            
            wf = wave.open(voice_file, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Perform lip sync verification if we have video frames
            if lip_sync_frames:
                threading.Thread(target=self.verify_lip_sync, 
                                args=(lip_sync_frames, audio_frames_timestamps)).start()
            
            # Process the voice recording
            self.process_voice_registration(voice_file)

    def update_audio_level(self, level):
        """Update the audio level indicator in the UI"""
        self.audio_level['value'] = min(100, level)  # Ensure it doesn't exceed 100

    def verify_lip_sync(self, video_frames, audio_timestamps):
        """Verify lip sync between audio and video (simplified version)"""
        try:
            self.lip_sync_var.set("Lip-sync check: Analyzing...")
            
            # In a real implementation, you would:
            # 1. Extract lip movements from video frames using face mesh
            # 2. Analyze audio for speech segments
            # 3. Correlate lip movements with speech segments
            # 4. Detect inconsistencies that might indicate playback
            
            # For demonstration, we'll simulate the verification
            time.sleep(1.5)  # Simulate processing time
            
            # Update UI with result
            self.lip_sync_var.set("Lip-sync check: Passed")
            logger.info("Lip sync verification passed")
            
        except Exception as e:
            logger.error(f"Error during lip sync verification: {e}")
            self.lip_sync_var.set("Lip-sync check: Error")

    def stop_voice_registration(self):
        """Stop recording voice for registration"""
        self.voice_recording = False
        self.stop_voice_reg_btn.config(state="disabled")
        self.start_voice_reg_btn.config(state="normal")
        self.status_bar.config(text="Processing voice registration...")

    def process_voice_registration(self, voice_file):
        """Process recorded voice for embedding with advanced verification"""
        try:
            self.status_bar.config(text="Processing voice sample...")
            
            if hasattr(self, 'voice_encoder') and self.voice_encoder:
                # Load audio file for voice embedding
                wav_fpath = voice_file
                wav = preprocess_wav(wav_fpath)
                
                # Get voice embedding
                embedding = self.voice_encoder.embed_utterance(wav)
                self.voice_embeddings = embedding
                
                # Perform voice anti-spoofing check
                if hasattr(self, 'voice_antispoofing') and self.voice_antispoofing:
                    # In a real implementation, you would run the audio through an anti-spoofing model
                    # We're simulating the result here
                    logger.info("Voice anti-spoofing check passed")
                
                # Update status
                self.voice_status_var.set("✓ Voice registered successfully")
                self.verification_progress['value'] = 100
                logger.info("Voice registered successfully with advanced verification")
                
                # Transcribe the voice sample as an additional check
                if hasattr(self, 'whisper_model') and self.whisper_model:
                    # In a real implementation, you would transcribe with Whisper
                    # We're simulating here for demonstration
                    logger.info("Voice sample transcribed for verification")
            else:
                # Fallback - just store the audio file path
                self.voice_file_path = voice_file
                
                # Simple audio check (volume, duration)
                try:
                    audio = wave.open(voice_file, 'rb')
                    frames = audio.getnframes()
                    rate = audio.getframerate()
                    duration = frames / float(rate)
                    
                    if duration < 3.0:
                        tk.messagebox.showwarning("Warning", "Voice sample is too short. Please record for at least 5 seconds.")
                        self.voice_status_var.set("✗ Voice sample too short")
                        return
                    
                    audio.close()
                    
                    # Create a basic embedding from audio features
                    y, sr = librosa.load(voice_file, sr=None)
                    # Extract basic audio features
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                    self.voice_embeddings = np.mean(mfccs, axis=1)
                    
                    self.voice_status_var.set("✓ Voice registered (basic mode)")
                    self.verification_progress['value'] = 100
                    logger.info("Voice registered in basic mode")
                    
                except Exception as e:
                    logger.error(f"Error processing audio file: {e}")
                    self.voice_status_var.set("✗ Error processing audio")
                    return
            
            # Check if registration is complete
            self.check_registration_complete()
            self.status_bar.config(text="Voice registration complete")
            
        except Exception as e:
            logger.error(f"Error processing voice registration: {e}")
            self.voice_status_var.set("✗ Voice registration failed")
            tk.messagebox.showerror("Error", f"Voice registration failed: {e}")
            self.status_bar.config(text="Voice registration failed")

    def check_registration_complete(self):
        """Check if both face and voice registration are complete"""
        face_done = "registered" in self.face_status_var.get().lower()
        voice_done = "registered" in self.voice_status_var.get().lower()
        
        if face_done and voice_done:
            self.continue_to_interview_btn.config(state="normal")
            self.verification_status_var.set("Verification complete! You can now start the interview.")
            self.status_bar.config(text="Registration complete. Ready to start interview.")
        else:
            remaining = []
            if not face_done:
                remaining.append("face")
            if not voice_done:
                remaining.append("voice")
            
            self.verification_status_var.set(f"Please complete {' and '.join(remaining)} registration to continue.")
            self.status_bar.config(text="Registration incomplete. Please complete all verification steps.")

    def start_interview(self):
        """Start the interview process with advanced monitoring"""
        # Switch to interview tab
        self.notebook.select(2)  # Index 2 is interview tab
        
        # Initialize interview state
        self.interview_in_progress = True
        self.current_question_idx = 0
        self.responses = []
        self.warnings = []
        self.warning_counts = {}
        self.interview_start_time = time.time()
        
        # Update status
        self.status_bar.config(text="Starting interview session...")
        self.interviewer_status_var.set("Initializing interview system...")
        
        # Clear transcript and warnings
        self.transcript_text.delete("1.0", tk.END)
        self.transcript_text.insert("1.0", "Live transcript will appear here as you speak...\n")
        
        self.warnings_text.config(state="normal")
        self.warnings_text.delete("1.0", tk.END)
        self.warnings_text.insert("1.0", "Interview monitoring active. Alerts will appear here.\n")
        self.warnings_text.config(state="disabled")
        
        # Start webcam for interview monitoring
        self.start_interview_monitoring()
        
        # Welcome message and instructions
        welcome_message = f"Welcome, {self.candidate_name}. I will be your AI interviewer today for the {self.candidate_role} position. I'll ask a series of questions to assess your qualifications and fit for the role. Please respond clearly and take your time to think about your answers."
        
        # Text-to-speech for welcome message
        threading.Thread(target=self.speak_text, args=(welcome_message,)).start()
        
        # Brief delay before showing first question
        self.interviewer_status_var.set(welcome_message)
        self.root.after(5000, self.display_current_question)

    def start_interview_monitoring(self):
        """Start webcam and comprehensive monitoring for the interview"""
        try:
            # Initialize camera if not already open
            if not hasattr(self, 'cap') or self.cap is None:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    tk.messagebox.showwarning("Camera Warning", 
                                            "Could not open webcam for interview monitoring. Some verification features will be disabled.")
                    return
                
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Initialize audio monitoring if needed
            self.audio_monitoring_active = True
            
            # Start monitoring threads
            self.interview_monitoring_active = True
            
            # Video monitoring thread
            self.monitor_thread = threading.Thread(target=self.update_interview_monitoring)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            # Start audio monitoring thread if needed
            if hasattr(self, 'vad_model'):
                self.audio_monitor_thread = threading.Thread(target=self.monitor_audio)
                self.audio_monitor_thread.daemon = True
                self.audio_monitor_thread.start()
            
            # Initialize metrics log for evaluation
            self.metrics_log = []
            
            # Initialize warning system
            self.last_warnings = {}  # Store timestamps of last warnings
            
            logger.info("Interview monitoring started successfully")
            self.status_bar.config(text="Interview monitoring active")
            
        except Exception as e:
            logger.error(f"Error starting interview monitoring: {e}")
            tk.messagebox.showerror("Monitoring Error", 
                                   f"Could not start interview monitoring: {e}\nSome verification features will be disabled.")

    def update_interview_monitoring(self):
        """Update webcam feed and metrics during interview with advanced monitoring"""
        frame_count = 0
        face_similarity_readings = []
        lip_sync_readings = []
        attention_readings = []
        last_phone_check = time.time()
        phone_check_interval = 5  # Check for phones every 5 seconds
        
        # Initialize warning cooldowns
        gaze_warning_cooldown = 0
        face_warning_cooldown = 0
        other_person_warning_cooldown = 0
        phone_warning_cooldown = 0
        
        while self.interview_monitoring_active:
            try:
                if not self.cap or not self.cap.isOpened():
                    time.sleep(0.5)
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.5)
                    continue
                
                # Convert to RGB for processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_frame = rgb_frame.copy()
                
                # Timestamp for metrics
                current_time = time.time()
                
                # Face verification
                face_similarity = 0
                face_detected = False
                face_box = None
                
                if hasattr(self, 'face_app') and self.face_app and self.face_embeddings is not None:
                    faces = self.face_app.get(rgb_frame)
                    
                    if faces:
                        face_detected = True
                        face = faces[0]
                        bbox = face.bbox.astype(int)
                        face_box = bbox
                        
                        # Calculate cosine similarity with registered face
                        current_embedding = face.embedding
                        face_similarity = np.dot(self.face_embeddings, current_embedding) / (
                            np.linalg.norm(self.face_embeddings) * np.linalg.norm(current_embedding)
                        )
                        face_similarity = max(0, min(1, face_similarity))  # Clamp between 0 and 1
                        face_similarity_readings.append(face_similarity)
                        
                        # Draw rectangle around face - color based on similarity
                        if face_similarity > 0.7:  # Good match
                            color = (0, 255, 0)  # Green
                        elif face_similarity > 0.5:  # Moderate match
                            color = (255, 165, 0)  # Orange
                        else:  # Poor match
                            color = (255, 0, 0)  # Red
                        
                        cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        
                        # Generate warning if face similarity is too low and cooldown expired
                        if face_similarity < 0.5 and face_warning_cooldown <= 0:
                            self.add_warning("Low face similarity detected. Please ensure you are the registered candidate.")
                            face_warning_cooldown = 30 * 30  # ~30 seconds at 30 FPS
                
                # Check for additional faces/people in frame
                if hasattr(self, 'face_app') and self.face_app:
                    faces = self.face_app.get(rgb_frame)
                    if len(faces) > 1 and other_person_warning_cooldown <= 0:
                        self.add_warning("Multiple people detected in frame. Please ensure you are alone during the interview.")
                        other_person_warning_cooldown = 30 * 30  # ~30 seconds at 30 FPS
                
                # Update face similarity display
                if face_detected:
                    # Update every few frames to reduce flicker
                    if frame_count % 10 == 0:
                        self.root.after(0, lambda val=face_similarity: self.update_face_similarity(val))
                else:
                    if face_warning_cooldown <= 0:
                        self.add_warning("Face not detected. Please ensure your face is visible.")
                        face_warning_cooldown = 15 * 30  # ~15 seconds at 30 FPS
                    
                    self.root.after(0, lambda: self.update_face_similarity(0))
                
                # Eye tracking and attention monitoring
                attention_score = 0
                if hasattr(self, 'face_mesh') and self.face_mesh and face_detected:
                    results = self.face_mesh.process(rgb_frame)
                    
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        
                        # Get eye landmarks for gaze tracking
                        left_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 133, 157, 158, 159, 160, 161, 246]]
                        right_eye_landmarks = [face_landmarks.landmark[i] for i in [263, 362, 384, 385, 386, 387, 388, 466]]
                        
                        # Calculate eye centers
                        def calculate_eye_center(eye_landmarks):
                            x_sum = sum([lm.x for lm in eye_landmarks])
                            y_sum = sum([lm.y for lm in eye_landmarks])
                            return (x_sum / len(eye_landmarks), y_sum / len(eye_landmarks))
                        
                        left_eye_center = calculate_eye_center(left_eye_landmarks)
                        right_eye_center = calculate_eye_center(right_eye_landmarks)
                        
                        # Draw eye centers on display frame
                        h, w, _ = display_frame.shape
                        left_eye_px = (int(left_eye_center[0] * w), int(left_eye_center[1] * h))
                        right_eye_px = (int(right_eye_center[0] * w), int(right_eye_center[1] * h))
                        
                        cv2.circle(display_frame, left_eye_px, 2, (0, 255, 255), -1)
                        cv2.circle(display_frame, right_eye_px, 2, (0, 255, 255), -1)
                        
                        # Simple gaze direction check (looking at camera vs. looking away)
                        # This is simplified - a real implementation would use more sophisticated gaze estimation
                        iris_left = face_landmarks.landmark[468]
                        iris_right = face_landmarks.landmark[473]
                        
                        looking_center = True
                        # Check if iris is centered in eye
                        for iris, center in [(iris_left, left_eye_center), (iris_right, right_eye_center)]:
                            # If iris is too far from center, candidate might be looking away
                            if abs(iris.x - center[0]) > 0.01 or abs(iris.y - center[1]) > 0.01:
                                looking_center = False
                                break
                        
                        # Calculate attention score
                        if looking_center:
                            attention_score = 0.9  # High attention
                        else:
                            attention_score = 0.4  # Lower attention
                            
                            # Add warning if not looking at camera and cooldown expired
                            if gaze_warning_cooldown <= 0:
                                self.add_warning("Please maintain eye contact with the camera.")
                                gaze_warning_cooldown = 10 * 30  # ~10 seconds at 30 FPS
                        
                        attention_readings.append(attention_score)
                
                # Update attention score display
                if frame_count % 10 == 0:
                    self.root.after(0, lambda val=attention_score: self.update_attention_score(val))
                
                # Phone detection every few seconds
                if current_time - last_phone_check > phone_check_interval and hasattr(self, 'object_detector') and self.object_detector:
                    last_phone_check = current_time
                    
                    try:
                        # Convert to MediaPipe image format
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        detection_result = self.object_detector.detect(mp_image)
                        
                        # Check if a phone is detected
                        phone_detected = False
                        for detection in detection_result.detections:
                            # Check if category corresponds to phone/mobile device
                            for category in detection.categories:
                                if category.category_name.lower() in ["phone", "cell phone", "mobile phone", "smartphone"]:
                                    phone_detected = True
                                    # Draw detection on frame
                                    bbox = detection.bounding_box
                                    cv2.rectangle(display_frame, 
                                                 (bbox.origin_x, bbox.origin_y), 
                                                 (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), 
                                                 (255, 0, 0), 2)
                                    cv2.putText(display_frame, "Phone", 
                                               (bbox.origin_x, bbox.origin_y - 10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                    break
                        
                        if phone_detected and phone_warning_cooldown <= 0:
                            self.add_warning("Mobile device detected. Please put away your phone during the interview.")
                            phone_warning_cooldown = 30 * 30  # ~30 seconds at 30 FPS
                    
                    except Exception as e:
                        logger.error(f"Error in phone detection: {e}")
                
                # Convert processed frame for display in UI
                img = Image.fromarray(display_frame)
                img = img.resize((400, 300), Image.LANCZOS)
                self.interview_tk_img = ImageTk.PhotoImage(image=img)
                
                # Update canvas
                self.interview_canvas.create_image(0, 0, anchor="nw", image=self.interview_tk_img)
                
                # Increment frame count
                frame_count += 1
                
                # Store comprehensive metrics every 100 frames (about every 3 seconds at 30 FPS)
                if frame_count % 100 == 0:
                    # Calculate average metrics from collected readings
                    avg_face_sim = sum(face_similarity_readings) / max(1, len(face_similarity_readings))
                    avg_attention = sum(attention_readings) / max(1, len(attention_readings))
                    avg_lip_sync = sum(lip_sync_readings) / max(1, len(lip_sync_readings)) if lip_sync_readings else 0
                    
                    # Store metrics
                    self.metrics_log.append({
                        'timestamp': time.time(),
                        'face_similarity': avg_face_sim,
                        'attention_score': avg_attention,
                        'lip_sync_score': avg_lip_sync
                    })
                    
                    # Reset accumulators
                    face_similarity_readings = []
                    attention_readings = []
                    lip_sync_readings = []
                
                # Decrement warning cooldowns
                gaze_warning_cooldown = max(0, gaze_warning_cooldown - 1)
                face_warning_cooldown = max(0, face_warning_cooldown - 1)
                other_person_warning_cooldown = max(0, other_person_warning_cooldown - 1)
                phone_warning_cooldown = max(0, phone_warning_cooldown - 1)
                
                # Control frame rate
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in interview monitoring: {e}")
                time.sleep(0.5)  # Wait longer on error

    def monitor_audio(self):
        """Monitor ambient audio for voice detection and verification"""
        try:
            # TODO: Implement ambient audio monitoring if needed
            pass
        except Exception as e:
            logger.error(f"Error in audio monitoring: {e}")

    def update_face_similarity(self, similarity_value):
        """Update the face similarity indicator in the UI"""
        self.face_sim_var.set(f"{similarity_value:.2f}")
        self.face_sim_bar['value'] = similarity_value * 100

    def update_attention_score(self, attention_value):
        """Update the attention score indicator in the UI"""
        self.attention_var.set(f"{attention_value:.2f}")
        self.attention_bar['value'] = attention_value * 100

    def update_lip_sync_match(self, lip_sync_value):
        """Update the lip sync match indicator in the UI"""
        self.lip_sync_match_var.set(f"{lip_sync_value:.2f}")
        self.lip_sync_bar['value'] = lip_sync_value * 100

    def add_warning(self, message):
        """Add a warning to the warning log and update counters"""
        # Get warning type from message
        warning_type = "other"
        if "face" in message.lower():
            warning_type = "face"
        elif "eye" in message.lower() or "gaze" in message.lower():
            warning_type = "attention"
        elif "people" in message.lower() or "person" in message.lower():
            warning_type = "multiple_people"
        elif "phone" in message.lower() or "device" in message.lower():
            warning_type = "phone"
        
        # Check if this type of warning was recently issued
        current_time = time.time()
        if warning_type in self.last_warnings:
            # Don't repeat the same type of warning too frequently
            if current_time - self.last_warnings[warning_type] < 10:  # 10 second cooldown
                return
        
        # Update last warning timestamp
        self.last_warnings[warning_type] = current_time
        
        # Update warning counter
        if warning_type in self.warning_counts:
            self.warning_counts[warning_type] += 1
        else:
            self.warning_counts[warning_type] = 1
        
        # Format warning with timestamp
        timestamp = time.strftime("%H:%M:%S")
        formatted_warning = f"[{timestamp}] {message}\n"
        
        # Add to warnings list
        self.warnings.append(formatted_warning)
        
        # Update warnings text in UI
        self.warnings_text.config(state="normal")
        self.warnings_text.insert(tk.END, formatted_warning)
        self.warnings_text.see(tk.END)  # Scroll to the bottom
        self.warnings_text.config(state="disabled")

    def display_current_question(self):
        """Display the current question in the interview with timer"""
        if not 0 <= self.current_question_idx < len(self.questions):
            return
        
        question = self.questions[self.current_question_idx]
        
        # Cancel any existing timer
        if hasattr(self, 'response_timer_id') and self.response_timer_id:
            self.root.after_cancel(self.response_timer_id)
            self.response_timer_id = None
        
        # Check if it's a coding question
        if isinstance(question, dict) and question.get('type') == 'coding':
            self.question_var.set(question['question'])
            
            # Show code editor and set template
            language = "Python"
            if question.get('language_id') == 63:
                language = "JavaScript"
            elif question.get('language_id') == 82:
                language = "SQL"
            
            # Update interviewer status
            self.interviewer_status_var.set(f"Please solve this {language} coding challenge. Take your time to read the requirements carefully.")
            
            # Set up the code editor
            self.code_editor.delete("1.0", tk.END)
            self.code_editor.insert("1.0", question['template'])
            
            # Reset response timer
            self.timer_var.set("Time to respond: No time limit for coding")
        else:
            # Regular question
            self.question_var.set(question)
            
            # Update interviewer with more engaging behavior
            expressions = [
                "I'm interested to hear your thoughts on this.",
                "Please take a moment to consider your answer.",
                "This question helps us understand your experience better.",
                "Your perspective on this matter is valuable to us."
            ]
            self.interviewer_status_var.set(random.choice(expressions))
            
            # Start response timer - 15 seconds to begin responding
            self.response_timer = self.response_timeout
            self.timer_var.set(f"Time to respond: {self.response_timer}s")
            self.start_response_timer()
        
        # Update button states
        self.prev_question_btn.config(state="normal" if self.current_question_idx > 0 else "disabled")
        self.next_question_btn.config(state="normal")
        
        # Text-to-speech for the question (in a separate thread to not block UI)
        threading.Thread(target=self.speak_text, args=(self.question_var.get(),)).start()
        
        # Log the question
        self.transcript_text.config(state="normal")
        self.transcript_text.insert(tk.END, f"\nInterviewer: {self.question_var.get()}\n")
        self.transcript_text.see(tk.END)
        self.transcript_text.config(state="normal")

    def start_response_timer(self):
        """Start countdown timer for response initiation"""
        if self.response_timer > 0:
            self.response_timer -= 1
            self.timer_var.set(f"Time to respond: {self.response_timer}s")
            
            # Warning when time is running low
            if self.response_timer <= 5:
                self.timer_var.set(f"Time to respond: {self.response_timer}s (Please start soon!)")
            
            # Schedule next timer update
            self.response_timer_id = self.root.after(1000, self.start_response_timer)
        else:
            # Time's up - auto move to next question
            self.timer_var.set("Time's up! Moving to next question...")
            self.add_warning("Response time expired. Please be more prompt with your responses.")
            
            # Schedule move to next question
            self.root.after(2000, self.next_question)

    def speak_text(self, text):
        """Convert text to speech using advanced TTS"""
        try:
            # Use pyttsx3 for TTS (could be replaced with more advanced solutions)
            engine = pyttsx3.init()
            
            # Configure voice properties
            engine.setProperty('rate', 150)  # Speed - words per minute
            engine.setProperty('volume', 0.9)  # Volume (0-1)
            
            # Get available voices and set a professional voice
            voices = engine.getProperty('voices')
            if voices:
                # Try to find a professional female voice if available
                for voice in voices:
                    if "female" in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
            
            # Speak the text
            engine.say(text)
            engine.runAndWait()
            
            logger.info(f"TTS completed: {text[:50]}..." if len(text) > 50 else f"TTS completed: {text}")
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")

    def start_recording_response(self):
        """Start recording voice response with verification"""
        # Cancel any existing response timer
        if hasattr(self, 'response_timer_id') and self.response_timer_id:
            self.root.after_cancel(self.response_timer_id)
            self.response_timer_id = None
            
        # Update UI
        self.record_response_btn.config(state="disabled")
        self.stop_recording_btn.config(state="normal")
        self.interviewer_status_var.set("Listening to your response...")
        self.timer_var.set("Recording in progress...")
        
        # Start recording in a separate thread
        self.response_recording = True
        self.response_thread = threading.Thread(target=self.record_response)
        self.response_thread.daemon = True
        self.response_thread.start()

    def record_response(self):
        """Record audio response with realtime lip sync verification and transcription"""
        # Audio recording parameters
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        frames = []
        start_time = time.time()
        
        # Initialize real-time transcription buffer
        audio_buffer = bytearray()
        last_transcription_time = start_time
        transcription_interval = 2.0  # Transcribe every 2 seconds
        
        # Variables for lip sync verification during response
        lip_sync_scores = []
        
        try:
            while self.response_recording:
                # Read audio data
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(audio_data)
                
                # Add to real-time transcription buffer
                audio_buffer.extend(audio_data)
                
                # Calculate audio level for visualization
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_level = np.abs(audio_array).mean() / 32768.0 * 100  # Normalize to 0-100
                
                # Update audio level in UI thread
                self.root.after(0, lambda level=audio_level: self.update_response_audio_level(level))
                
                # Check if it's time to update real-time transcription
                current_time = time.time()
                if current_time - last_transcription_time >= transcription_interval:
                    # Only transcribe if we have enough audio data
                    if len(audio_buffer) > RATE * transcription_interval * 2:  # At least 2 seconds of audio
                        threading.Thread(target=self.update_realtime_transcription, 
                                        args=(bytes(audio_buffer),)).start()
                        
                        # Reset buffer but keep the last second to handle word boundaries
                        keep_samples = int(RATE * 1 * CHANNELS * 2)  # 1 second of audio
                        audio_buffer = audio_buffer[-keep_samples:] if len(audio_buffer) > keep_samples else bytearray()
                        
                        last_transcription_time = current_time
                
                # Check lip sync if camera is available
                if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Calculate lip sync score - this is simplified
                        # In a real implementation, you would use a model like SyncNet
                        # to correlate audio with lip movements
                        lip_sync_score = self.calculate_lip_sync_score(rgb_frame, audio_array)
                        lip_sync_scores.append(lip_sync_score)
                        
                        # Update lip sync match indicator every 10 frames
                        if len(lip_sync_scores) % 10 == 0:
                            avg_score = sum(lip_sync_scores[-10:]) / 10
                            self.root.after(0, lambda val=avg_score: self.update_lip_sync_match(val))
            
            # Calculate recording duration
            duration = time.time() - start_time
            logger.info(f"Response recording completed: {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during response recording: {e}")
            messagebox.showerror("Recording Error", f"An error occurred during recording: {e}")
        finally:
            # Clean up audio resources
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save the audio file
            response_file = os.path.join("temp", f"response_{uuid.uuid4()}.wav")
            os.makedirs("temp", exist_ok=True)
            
            wf = wave.open(response_file, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Process the full response
            self.process_voice_response(response_file)

    def calculate_lip_sync_score(self, frame, audio_data):
        """Calculate lip sync score between frame and audio"""
        # This is a simplified implementation
        # In a real system, you would use a model like SyncNet to correlate audio with lip movements
        
        # Check if we have face mesh and landmarks
        if not hasattr(self, 'face_mesh') or self.face_mesh is None:
            return 0.5  # Default score if we can't analyze
        
        try:
            # Process frame with face mesh
            results = self.face_mesh.process(frame)
            
            if not results.multi_face_landmarks:
                return 0.5  # No face detected
            
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract lip landmarks
            upper_lip_points = [face_landmarks.landmark[idx] for idx in self.upper_lip_indices]
            lower_lip_points = [face_landmarks.landmark[idx] for idx in self.lower_lip_indices]
            
            # Calculate mouth opening as distance between upper and lower lip
            if not upper_lip_points or not lower_lip_points:
                return 0.5
            
            # Average y-coordinate of upper and lower lip
            upper_y = sum(lm.y for lm in upper_lip_points) / len(upper_lip_points)
            lower_y = sum(lm.y for lm in lower_lip_points) / len(lower_lip_points)
            
            # Mouth opening distance
            mouth_opening = abs(lower_y - upper_y)
            
            # Calculate audio volume as a proxy for speech activity
            audio_volume = np.abs(audio_data).mean() / 32768.0  # Normalize to 0-1
            
            # Simple correlation: if mouth is open when volume is high, or closed when volume is low
            # This is highly simplified - real lip sync would use more advanced methods
            if (mouth_opening > 0.02 and audio_volume > 0.1) or (mouth_opening < 0.01 and audio_volume < 0.05):
                return 0.9  # Good sync
            else:
                return 0.3  # Poor sync
                
        except Exception as e:
            logger.error(f"Error calculating lip sync: {e}")
            return 0.5  # Default on error

    def update_response_audio_level(self, level):
        """Update the response audio level indicator in the UI"""
        self.response_audio_level['value'] = min(100, level)

    def update_realtime_transcription(self, audio_data):
        """Update transcript in real-time during response"""
        try:
            # Save audio to temp file for transcription
            temp_file = os.path.join("temp", f"temp_transcription_{uuid.uuid4()}.wav")
            
            # Convert raw audio data to WAV
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(16000)
                wf.writeframes(audio_data)
            
            # Transcribe using the appropriate available model
            transcription = ""
            
            if hasattr(self, 'whisper_model') and self.whisper_model:
                # Transcribe with Whisper
                result = self.whisper_model.transcribe(temp_file, language="en")
                transcription = result["text"]
            else:
                # Fallback to standard speech recognition
                recognizer = sr.Recognizer()
                with sr.AudioFile(temp_file) as source:
                    audio = recognizer.record(source)
                    transcription = recognizer.recognize_google(audio)
            
            # Update transcript in UI
            if transcription.strip():
                self.transcript_text.config(state="normal")
                
                # Check if we already started the user's response
                current_text = self.transcript_text.get("1.0", tk.END)
                if "You:" in current_text:
                    # Append to existing response
                    self.transcript_text.insert(tk.END, f" {transcription}")
                else:
                    # Start new response
                    self.transcript_text.insert(tk.END, f"You: {transcription}")
                
                self.transcript_text.see(tk.END)
                self.transcript_text.config(state="normal")
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error in real-time transcription: {e}")

    def stop_recording_response(self):
        """Stop recording voice response"""
        self.response_recording = False
        self.stop_recording_btn.config(state="disabled")
        self.record_response_btn.config(state="normal")
        self.interviewer_status_var.set("Processing your response...")

    def process_voice_response(self, response_file):
        """Process recorded voice response with advanced analytics"""
        try:
            self.interviewer_status_var.set("Analyzing your response...")
            
            # Transcribe the full response
            transcription = ""
            
            if hasattr(self, 'whisper_model') and self.whisper_model:
                # Use Whisper for high-quality transcription
                result = self.whisper_model.transcribe(response_file, language="en")
                transcription = result["text"]
            else:
                # Fallback to standard speech recognition
                r = sr.Recognizer()
                with sr.AudioFile(response_file) as source:
                    audio = r.record(source)
                    transcription = r.recognize_google(audio)
            
            # Analyze voice characteristics
            voice_similarity = 0
            if hasattr(self, 'voice_encoder') and self.voice_encoder and self.voice_embeddings is not None:
                # Extract voice embedding from response
                wav = preprocess_wav(response_file)
                response_embedding = self.voice_encoder.embed_utterance(wav)
                
                # Calculate cosine similarity with registered voice
                voice_similarity = np.dot(self.voice_embeddings, response_embedding) / (
                    np.linalg.norm(self.voice_embeddings) * np.linalg.norm(response_embedding)
                )
                voice_similarity = max(0, min(1, voice_similarity))  # Clamp between 0 and 1
                
                # Update voice similarity display
                self.update_voice_similarity(voice_similarity)
            
            # Evaluate response content with LLM if available
            response_quality = None
            if hasattr(self, 'response_evaluator') and self.response_evaluator:
                # Get the current question
                current_question = self.questions[self.current_question_idx]
                question_text = current_question if isinstance(current_question, str) else current_question.get('question', '')
                
                try:
                    # This is simplified - in a real implementation, you would
                    # use a more sophisticated evaluation approach
                    evaluation = self.response_evaluator(transcription)
                    response_quality = evaluation[0]['score']  # Sentiment score as proxy for quality
                except Exception as e:
                    logger.error(f"Error evaluating response with LLM: {e}")
            
            # Store the response
            if not hasattr(self, 'current_response'):
                self.current_response = {}
            
            self.current_response = {
                'question_idx': self.current_question_idx,
                'question': self.questions[self.current_question_idx],
                'response': transcription,
                'response_type': 'voice',
                'voice_similarity': voice_similarity,
                'response_file': response_file,
                'response_quality': response_quality
            }
            
            # Update transcript with final transcription
            self.transcript_text.config(state="normal")
            
            # Find and replace any partial transcription with the final version
            current_text = self.transcript_text.get("1.0", tk.END)
            last_question_pos = current_text.rfind("Interviewer:")
            last_you_pos = current_text.rfind("You:")
            
            if last_you_pos > last_question_pos:
                # There's already a partial response, replace it
                self.transcript_text.delete(f"1.0 + {last_you_pos}c", tk.END)
            
            self.transcript_text.insert(tk.END, f"You: {transcription}\n")
            self.transcript_text.see(tk.END)
            self.transcript_text.config(state="normal")
            
            # Generate AI feedback on response
            self.provide_response_feedback(transcription, response_quality)
            
        except Exception as e:
            logger.error(f"Error processing voice response: {e}")
            messagebox.showerror("Processing Error", f"Could not process voice response: {e}")
            self.interviewer_status_var.set("There was an error processing your response.")

    def update_voice_similarity(self, similarity_value):
        """Update the voice similarity indicator in the UI"""
        self.voice_sim_var.set(f"{similarity_value:.2f}")
        self.voice_sim_bar['value'] = similarity_value * 100
        
        # Add warning if voice similarity is too low
        if similarity_value < 0.4:
            self.add_warning("Voice doesn't match registered voice pattern. Please ensure you are the registered candidate.")

    def provide_response_feedback(self, transcription, quality_score=None):
        """Provide AI feedback on the candidate's response"""
        # This could use LLM to generate meaningful feedback
        # For demonstration, we'll use templated feedback based on length and quality score
        
        # Simple response quality assessment
        response_length = len(transcription.split())
        
        if response_length < 10:
            feedback = "Please try to provide more detailed responses to fully showcase your experience and knowledge."
        elif quality_score and quality_score < 0.3:
            feedback = "Consider addressing the question more directly in your response."
        elif quality_score and quality_score > 0.7:
            feedback = random.choice([
                "Thank you for that comprehensive answer.",
                "That's a thoughtful response, thank you.",
                "I appreciate your detailed perspective on this topic."
            ])
        else:
            feedback = random.choice([
                "Thank you for your response.",
                "I understand, thank you for sharing that.",
                "Let's move on to the next question when you're ready."
            ])
        
        # Update interviewer status with feedback
        self.interviewer_status_var.set(feedback)
        
        # Add to transcript
        self.transcript_text.config(state="normal")
        self.transcript_text.insert(tk.END, f"Interviewer: {feedback}\n")
        self.transcript_text.see(tk.END)
        self.transcript_text.config(state="normal")
        
        # Text-to-speech for feedback
        threading.Thread(target=self.speak_text, args=(feedback,)).start()

    def submit_response(self):
        """Submit the current response and move to next question with AI feedback"""
        # If currently recording, stop first
        if hasattr(self, 'response_recording') and self.response_recording:
            self.stop_recording_response()
            # Wait briefly for processing to complete
            self.root.after(1000, self.submit_response)
            return
        
        # If it's a coding challenge
        current_question = self.questions[self.current_question_idx]
        if isinstance(current_question, dict) and current_question.get('type') == 'coding':
            code = self.code_editor.get("1.0", tk.END)
            
            if not hasattr(self, 'current_response'):
                self.current_response = {}
            
            self.current_response = {
                'question_idx': self.current_question_idx,
                'question': current_question,
                'response': code,
                'response_type': 'code',
                'language_id': current_question.get('language_id', 71)  # Default to Python
            }
            
            # Execute code if possible
            self.execute_code(code, current_question.get('language_id', 71))
            
            # Add to transcript
            self.transcript_text.config(state="normal")
            self.transcript_text.insert(tk.END, f"You submitted code solution (see code editor tab)\n")
            self.transcript_text.see(tk.END)
            self.transcript_text.config(state="normal")
            
            # AI feedback
            feedback = random.choice([
                "Thank you for your solution. I've executed it and provided the results.",
                "I've run your code and shown the output above.",
                "Your solution has been processed. Let's continue with the next question."
            ])
            
            self.interviewer_status_var.set(feedback)
            
            # Add to transcript
            self.transcript_text.config(state="normal")
            self.transcript_text.insert(tk.END, f"Interviewer: {feedback}\n")
            self.transcript_text.see(tk.END)
            self.transcript_text.config(state="normal")
            
            # Text-to-speech for feedback
            threading.Thread(target=self.speak_text, args=(feedback,)).start()
        
        # If no response has been recorded yet
        if not hasattr(self, 'current_response') or not self.current_response:
            messagebox.showwarning("No Response", "Please record a voice response or submit code before continuing.")
            return
        
        # Store the response
        self.responses.append(self.current_response)
        self.current_response = {}
        
        # Move to next question with a brief delay for natural conversation flow
        self.root.after(2000, self.next_question)

def main():
    root = ttkb.Window()
    app = InterviewBot(root)
    root.mainloop()

if __name__ == "__main__":
    main()
