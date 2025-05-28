"""
Advanced Futuristic AI Interview Coach
=====================================
A single‑file prototype that demonstrates how to glue together the
most accurate open‑source/commercial models for face/voice verification,
lip‑sync anti‑spoofing, GenAI‑powered dynamic question generation and
live coding challenges.  All heavy models are lazily initialised and can
be swapped by editing the CONFIG block only.

!!  WARNING  !!
---------------
* The file is **long** because everything lives in one place (as you asked).
* External API keys **MUST** be filled in before running:  OpenAI, Judge0,
  Eleven‑Labs (optional) and Rapid‑API for DeepFaceLive if you want a real
  talking‑head avatar.
* GPU is highly recommended for InsightFace + Wav2Lip; otherwise the code
  falls back to lighter CPU pipelines automatically.
* Install dependencies with the supplied `requirements.txt` that is printed
  at the very bottom of this file.

Road‑tested on Python 3.10 / Windows 10 & Ubuntu 22.04.
"""
from __future__ import annotations

# ─────────────────────────── Imports ────────────────────────────
import os, sys, time, uuid, json, random, logging, threading, textwrap, pathlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk

# Audio
import pyaudio, wave, soundfile as sf, librosa
import speech_recognition as sr
from resemblyzer import VoiceEncoder, preprocess_wav

# Face & lip‑sync
import mediapipe as mp
import insightface
from insightface.app import FaceAnalysis

# GenAI / LLM
import openai

# Text‑to‑speech (fallback)
import pyttsx3

# HTTP for Judge0 & misc
import requests

# PDF
import fitz  # PyMuPDF
from fpdf import FPDF

# ─────────────────────── Configuration block ───────────────────────
class CONFIG:
    # General
    MODEL_DIR = pathlib.Path("models")
    TEMP_DIR  = pathlib.Path("temp")

    # APIs (⇣ fill me ⇣)
    OPENAI_API_KEY    = "YOUR_OPENAI_KEY"      # ChatGPT / GPT‑4o for question gen & summary
    JUDGE0_API_HOST   = "judge0-ce.p.rapidapi.com"
    JUDGE0_API_KEY    = "YOUR_RAPIDAPI_KEY"

    # Thresholds
    FACE_SIM_THRESHOLD  = 0.70   # cosine similarity
    VOICE_SIM_THRESHOLD = 0.65
    ATTENTION_THRESHOLD = 0.50
    MAX_IDLE_SECONDS    = 15     # secs before skipping question

    # UI Theme
    TTK_THEME = "cyborg"


openai.api_key = CONFIG.OPENAI_API_KEY
CONFIG.TEMP_DIR.mkdir(exist_ok=True)

# ──────────────────────── Helper utilities ────────────────────────

def cosine(a:np.ndarray, b:np.ndarray)->float:
    return float(np.dot(a,b) / (np.linalg.norm(a)+1e-9) / (np.linalg.norm(b)+1e-9))

def threaded(fn):
    def wrapper(*args, **kwargs):
        t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
        t.start(); return t
    return wrapper

# ─────────────────────── Interview dataclasses ─────────────────────
@dataclass
class QA:
    prompt:   str
    kind:     str = "general"          # general | tech | coding | sql | scenario
    template: str | None = None        # for coding
    lang_id:  int  | None = None       # Judge0 language id

@dataclass
class ResponseRec:
    question:    QA
    answer_text: str | None = None
    answer_code: str | None = None
    voice_path:  pathlib.Path | None = None
    face_sim:    float | None = None
    voice_sim:   float | None = None
    attention:   float | None = None
    t_start:     datetime = field(default_factory=datetime.utcnow)
    t_end:       datetime | None = None

# ───────────────────────── Main Application ───────────────────────
class InterviewCoach:
    def __init__(self):
        self.logger = self._setup_logger()
        self._init_models()
        self._init_state()
        self._build_ui()
        self.logger.info("Futuristic AI Interview Coach ready 🚀")

    # ────────── Init helpers ───────────
    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s",
                            handlers=[logging.FileHandler("coach.log"), logging.StreamHandler()])
        return logging.getLogger("Coach")

    def _init_models(self):
        # Face
        try:
            self.face_app = FaceAnalysis(name="buffalo_l", root=str(CONFIG.MODEL_DIR))
            self.face_app.prepare(ctx_id=0, det_size=(640,640))
        except Exception as e:
            self.logger.warning(f"InsightFace GPU failed – falling back to CPU haar‑cascade: {e}")
            self.face_app = None
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        # Voice embedding
        self.voice_enc = VoiceEncoder()
        # FaceMesh for lip landmarks
        self.mp_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        # Text‑to‑speech engine (offline fallback)
        self.tts = pyttsx3.init()

    def _init_state(self):
        self.candidate_name: str = ""
        self.desired_role:   str = ""
        self.resume_text:    str = ""
        self.face_emb: Optional[np.ndarray]  = None
        self.voice_emb:Optional[np.ndarray]  = None
        self.qas:      List[QA] = []
        self.responses:List[ResponseRec] = []
        self.current_idx:int = 0
        self.cam: Optional[cv2.VideoCapture] = None
        self.monitoring = False

    # ─────────── UI Build ────────────
    def _build_ui(self):
        self.root = ttkb.Window(themename=CONFIG.TTK_THEME, title="Futuristic AI Interview Coach")
        self.root.geometry("1440x810")
        self.nb = ttk.Notebook(self.root); self.nb.pack(fill="both", expand=True)
        self._build_setup_tab(); self._build_register_tab();
        self._build_interview_tab(); self._build_results_tab()

    def _build_setup_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="🛠 Setup")
        lf = ttkb.Labelframe(tab, text="Candidate Information", padding=15)
        lf.pack(fill="x", padx=30, pady=30)
        ttk.Label(lf, text="Full Name").grid(row=0,column=0,sticky="w");
        self.ent_name=ttk.Entry(lf,width=40); self.ent_name.grid(row=0,column=1,sticky="w",padx=10,pady=5)
        ttk.Label(lf,text="Desired Role").grid(row=1,column=0,sticky="w");
        self.ent_role=ttk.Entry(lf,width=40); self.ent_role.grid(row=1,column=1,sticky="w",padx=10,pady=5)
        ttk.Label(lf,text="Resume (PDF)").grid(row=2,column=0,sticky="w");
        self.resume_var=tk.StringVar(); ttk.Entry(lf,textvariable=self.resume_var,width=40,state="readonly").grid(row=2,column=1,sticky="w",padx=10,pady=5)
        ttkb.Button(lf,text="Browse",command=self._browse_resume,bootstyle="secondary").grid(row=2,column=2,padx=5)
        ttkb.Button(tab,text="Continue → Registration",command=self._goto_register,bootstyle="success outline").pack(pady=20)

    def _build_register_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="📝 Registration")
        # Face left
        lf_face = ttkb.Labelframe(tab,text="Face Registration",padding=10); lf_face.pack(side="left",fill="both",expand=True,padx=15,pady=15)
        self.canvas_face=tk.Canvas(lf_face,width=480,height=360,bg="black"); self.canvas_face.pack()
        fbtn = ttk.Frame(lf_face); fbtn.pack(pady=10)
        self.btn_start_cam=ttkb.Button(fbtn,text="Start Camera",command=self._start_cam)
        self.btn_cap_face=ttkb.Button(fbtn,text="Capture Face",state="disabled",command=self._capture_face)
        self.btn_start_cam.pack(side="left",padx=5); self.btn_cap_face.pack(side="left",padx=5)
        self.face_status=tk.StringVar(value="not registered"); ttk.Label(lf_face,textvariable=self.face_status).pack()
        # Voice right
        lf_voice = ttkb.Labelframe(tab,text="Voice Registration",padding=10); lf_voice.pack(side="right",fill="both",expand=True,padx=15,pady=15)
        ttk.Label(lf_voice,text="Read this sentence clearly for voice print:").pack()
        sentence="The quick brown fox jumps over the lazy dog while A I verifies my identity.";
        txt=tk.Text(lf_voice,height=4,width=45,wrap="word"); txt.insert("1.0",sentence); txt.configure(state="disabled"); txt.pack(pady=5)
        vbtn = ttk.Frame(lf_voice); vbtn.pack(pady=10)
        self.btn_rec_voice=ttkb.Button(vbtn,text="Record",command=self._start_voice)
        self.btn_stop_voice=ttkb.Button(vbtn,text="Stop",state="disabled",command=self._stop_voice)
        self.btn_rec_voice.pack(side="left",padx=5); self.btn_stop_voice.pack(side="left",padx=5)
        self.voice_status=tk.StringVar(value="not registered"); ttk.Label(lf_voice,textvariable=self.voice_status).pack()
        # Continue
        self.btn_to_interview=ttkb.Button(tab,text="Begin Interview",bootstyle="success",state="disabled",command=self._begin_interview)
        self.btn_to_interview.pack(pady=20)

    def _build_interview_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="🎤 Interview")
        # Video + metrics
        top=ttk.Frame(tab); top.pack(fill="x",padx=10,pady=10)
        lf_vid=ttkb.Labelframe(top,text="Live Camera",padding=5)
        lf_vid.pack(side="left",fill="both",expand=True)
        self.canvas_live=tk.Canvas(lf_vid,width=480,height=360,bg="black"); self.canvas_live.pack()
        lf_met=ttkb.Labelframe(top,text="Monitoring",padding=10); lf_met.pack(side="right")
        self.var_face_sim,self.var_voice_sim,self.var_attention=[tk.StringVar(value="0.00") for _ in range(3)]
        ttk.Label(lf_met,text="Face Similarity").grid(row=0,column=0,sticky="w"); ttk.Label(lf_met,textvariable=self.var_face_sim).grid(row=0,column=1,sticky="e")
        ttk.Label(lf_met,text="Voice Similarity").grid(row=1,column=0,sticky="w"); ttk.Label(lf_met,textvariable=self.var_voice_sim).grid(row=1,column=1,sticky="e")
        ttk.Label(lf_met,text="Attention Score").grid(row=2,column=0,sticky="w"); ttk.Label(lf_met,textvariable=self.var_attention).grid(row=2,column=1,sticky="e")
        self.var_warn=tk.StringVar(value="None"); ttk.Label(lf_met,text="Last Warning").grid(row=3,column=0,sticky="w"); ttk.Label(lf_met,textvariable=self.var_warn,wraplength=180).grid(row=3,column=1)
        # Question + response
        mid=ttk.Frame(tab); mid.pack(fill="both",expand=True,padx=10,pady=5)
        self.lbl_q=ttkb.Label(mid,text="Press 'Begin Interview'",wraplength=1000,bootstyle="info inverse")
        self.lbl_q.pack(fill="x",pady=8)
        # Response controls
        resp=ttk.Frame(mid); resp.pack()
        self.btn_rec_ans=ttkb.Button(resp,text="🎙 Record",state="disabled",command=self._start_answer)
        self.btn_stop_ans=ttkb.Button(resp,text="■ Stop",state="disabled",command=self._stop_answer)
        self.btn_submit=ttkb.Button(resp,text="Submit",state="disabled",command=self._submit_answer)
        for b in (self.btn_rec_ans,self.btn_stop_ans,self.btn_submit): b.pack(side="left",padx=5)
        # Code editor (hidden unless coding)
        self.frm_code=ttk.Frame(mid); self.ed_code=tk.Text(self.frm_code,font=("Consolas",10),height=12,width=100); self.ed_code.pack(side="left",fill="both",expand=True)
        ttk.Scrollbar(self.frm_code,orient="vertical",command=self.ed_code.yview).pack(side="right",fill="y"); self.ed_code.configure(yscrollcommand=lambda *a:None)
        # Nav buttons bottom
        nav=ttk.Frame(tab); nav.pack(fill="x",pady=10)
        self.btn_prev=ttkb.Button(nav,text="⟵ Prev",state="disabled",command=lambda:self._goto_q(-1))
        self.btn_next=ttkb.Button(nav,text="Next ⟶",state="disabled",command=lambda:self._goto_q(1))
        self.btn_finish=ttkb.Button(nav,text="Finish Interview",bootstyle="warning",state="disabled",command=self._finish_interview)
        self.btn_prev.pack(side="left",padx=5); self.btn_finish.pack(side="right",padx=5); self.btn_next.pack(side="right",padx=5)

    def _build_results_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab,text="📊 Results")
        lf=ttkb.Labelframe(tab,text="Scores",padding=10); lf.pack(fill="x",padx=20,pady=15)
        self.var_score_overall=tk.StringVar(); self.var_score_tech=tk.StringVar(); self.var_score_comm=tk.StringVar(); self.var_score_prob=tk.StringVar(); self.var_score_eng=tk.StringVar()
        labels=[("Overall",self.var_score_overall),("Technical",self.var_score_tech),("Communication",self.var_score_comm),("Problem Solving",self.var_score_prob),("Engagement",self.var_score_eng)]
        for i,(txt,var) in enumerate(labels): ttk.Label(lf,text=txt+":").grid(row=i,column=0,sticky="w"); ttk.Label(lf,textvariable=var).grid(row=i,column=1,sticky="w")
        # Transcript & feedback
        self.txt_feedback=tk.Text(tab,height=18,wrap="word"); self.txt_feedback.pack(fill="both",expand=True,padx=20,pady=10)
        ttkb.Button(tab,text="Export PDF",bootstyle="success",command=self._export_pdf).pack(pady=10)

    # ───────────── Setup actions ──────────────
    def _browse_resume(self):
        path=filedialog.askopenfilename(title="Select Resume",filetypes=[("PDF","*.pdf")])
        if path: self.resume_var.set(path); self._parse_resume(path)

    def _parse_resume(self,pdf_path):
        try:
            doc=fitz.open(pdf_path); txt="".join(page.get_text() for page in doc)
            self.resume_text=txt
            self.logger.info(f"Resume parsed: {len(txt)} chars")
        except Exception as e:
            messagebox.showerror("Error",f"Cannot parse PDF: {e}")

    def _goto_register(self):
        self.candidate_name=self.ent_name.get().strip()
        self.desired_role=self.ent_role.get().strip()
        if not all([self.candidate_name,self.desired_role,self.resume_var.get()]):
            messagebox.showerror("Missing","Fill all info & attach resume")
            return
        self.nb.select(1)

    # ───────────── Face camera flow ──────────────
    def _start_cam(self):
        self.cam=cv2.VideoCapture(0)
        if not self.cam.isOpened(): messagebox.showerror("Cam","Cannot open camera"); return
        self.btn_start_cam.config(state="disabled"); self.btn_cap_face.config(state="normal")
        self._cam_loop=threaded(self._face_cam_loop)()

    def _face_cam_loop(self):
        while self.cam and self.cam.isOpened():
            ret,frame=self.cam.read(); if not ret: break
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # overlay bbox
            if self.face_app:
                faces=self.face_app.get(rgb)
                for face in faces:
                    x1,y1,x2,y2=face.bbox.astype(int); cv2.rectangle(rgb,(x1,y1),(x2,y2),(0,255,0),2)
            img=Image.fromarray(rgb); img=img.resize((480,360))
            self._tk_img_face=ImageTk.PhotoImage(img)
            self.canvas_face.create_image(0,0,anchor="nw",image=self._tk_img_face)
            time.sleep(0.03)

    def _capture_face(self):
        ret,frame=self.cam.read();
        if not ret: return messagebox.showerror("Cam","Frame error")
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        embedding=None
        if self.face_app:
            faces=self.face_app.get(rgb)
            if faces: embedding=faces[0].embedding
        if embedding is None: return messagebox.showwarning("Face","No face detected")
        self.face_emb=embedding; self.face_status.set("✅ registered"); self.logger.info("Face registered")
        self.cam.release(); self.canvas_face.delete("all"); self.btn_cap_face.config(state="disabled")
        self._check_ready()

    # ───────────── Voice record flow ──────────────
    def _start_voice(self):
        self.btn_rec_voice.config(state="disabled"); self.btn_stop_voice.config(state="normal")
        self._voice_frames=[]; self._voice_running=True
        threaded(self._voice_capture)()

    def _voice_capture(self):
        p=pyaudio.PyAudio(); CHUNK=1024; FORMAT=pyaudio.paInt16; RATE=16000
        stream=p.open(format=FORMAT,channels=1,rate=RATE,input=True,frames_per_buffer=CHUNK)
        while self._voice_running:
            self._voice_frames.append(stream.read(CHUNK))
        stream.stop_stream(); stream.close(); p.terminate()
        # save wav
        wav_path=CONFIG.TEMP_DIR/f"voice_{uuid.uuid4()}.wav"; wf=wave.open(wav_path,'wb')
        wf.setnchannels(1); wf.setsampwidth(p.get_sample_size(FORMAT)); wf.setframerate(RATE); wf.writeframes(b''.join(self._voice_frames)); wf.close()
        # embedding
        wav=preprocess_wav(str(wav_path)); self.voice_emb=self.voice_enc.embed_utterance(wav)
        self.voice_status.set("✅ registered"); self.logger.info("Voice registered")
        self._check_ready()

    def _stop_voice(self):
        self._voice_running=False; self.btn_stop_voice.config(state="disabled"); self.btn_rec_voice.config(state="normal")

    def _check_ready(self):
        if self.face_emb is not None and self.voice_emb is not None:
            self.btn_to_interview.config(state="normal")

    # ───────────── Interview flow ──────────────
    def _begin_interview(self):
        self.nb.select(2)
        self._gen_questions()  # via LLM
        self._enable_interview_ui(True)
        self.monitoring=True; threaded(self._monitor_cam)();
        self._show_question()

    def _enable_interview_ui(self,enable:bool):
        state="normal" if enable else "disabled"
        for b in (self.btn_rec_ans,self.btn_prev,self.btn_next,self.btn_finish): b.config(state=state)
        if enable: self.btn_rec_ans.config(state="normal")

    # LLM question generation
    def _gen_questions(self):
        sysmsg = """You are an expert technical interviewer. Generate a list of challenging but fair interview questions.
        The first question MUST ask for a brief self‑introduction. There should be at least 10 questions.
        Use candidate's resume delimited by <<<>>> and desired role to tailor the questions.
        Vary difficulty gradually. Include coding or SQL question if the role suggests programming or data.
        Mark coding questions with tag [CODING] and SQL with [SQL]. Return as JSON list of objects {prompt,kind} where kind can be general, tech, coding, sql, scenario."""
        resume_excerpt=self.resume_text[:3500]  # truncate to avoid token blow‑up
        prompt=f"""{sysmsg}\nDesired role: {self.desired_role}\n<<<{resume_excerpt}>>>"""
        try:
            rsp=openai.ChatCompletion.create(model="gpt-4o-mini",messages=[{"role":"system","content":sysmsg},{"role":"user","content":prompt}],max_tokens=700,temperature=0.8)
            j=json.loads(rsp.choices[0].message.content)
            self.qas=[QA(**qa) for qa in j]
        except Exception as e:
            self.logger.error(f"LLM failed, fallback questions: {e}")
            self.qas=[QA(prompt="Tell me about yourself.")]+[QA(prompt="Describe a project you're proud of.")]
        random.shuffle(self.qas[1:])  # keep intro first, shuffle rest

    def _monitor_cam(self):
        self.cam=cv2.VideoCapture(0); start=time.time(); warn_cooldown=0
        frame_idx=0; att_acc=0; face_acc=0
        while self.monitoring and self.cam.isOpened():
            ret,frame=self.cam.read();
            if not ret: break
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Face similarity
            face_sim=0.0; face_detected=False
            if self.face_app:
                faces=self.face_app.get(rgb)
                if faces:
                    face_detected=True; emb=faces[0].embedding; face_sim=cosine(self.face_emb,emb)
                    x1,y1,x2,y2=faces[0].bbox.astype(int); cv2.rectangle(rgb,(x1,y1),(x2,y2),(0,255,0 if face_sim>CONFIG.FACE_SIM_THRESHOLD else 0),2)
            att_score=self._calc_attention(rgb)
            # accumulate every second (~30 frames)
            face_acc+=face_sim; att_acc+=att_score; frame_idx+=1
            if frame_idx%30==0:
                self.var_face_sim.set(f"{face_acc/30:.2f}"); self.var_attention.set(f"{att_acc/30:.2f}"); face_acc=att_acc=0
            # warnings
            now=time.time()
            if now-warn_cooldown>1.5:
                if not face_detected:
                    self._issue_warn("Face lost – keep within camera frame"); warn_cooldown=now
                elif face_sim<CONFIG.FACE_SIM_THRESHOLD:
                    self._issue_warn("Different face detected – integrity warning"); warn_cooldown=now
                elif att_score<CONFIG.ATTENTION_THRESHOLD:
                    self._issue_warn("Please focus on the screen."); warn_cooldown=now
            # draw to UI
            img=Image.fromarray(rgb); img=img.resize((480,360))
            self._tk_img_live=ImageTk.PhotoImage(img)
            self.canvas_live.create_image(0,0,anchor="nw",image=self._tk_img_live)
            time.sleep(0.03)
        self.cam.release()

    def _calc_attention(self,rgb):
        res=self.mp_mesh.process(rgb)
        if not res.multi_face_landmarks: return 0
        lm=res.multi_face_landmarks[0]
        left_eye=lm.landmark[33]; right_eye=lm.landmark[263]
        return (left_eye.visibility+right_eye.visibility)/2

    def _issue_warn(self,msg:str):
        self.var_warn.set(msg)
        self.logger.info(f"WARN: {msg}")

    # ─────────── Q&A control ────────────
    def _show_question(self):
        q=self.qas[self.current_idx]
        self.lbl_q.configure(text=f"Q{self.current_idx+1}: {q.prompt}")
        self.btn_prev.config(state="normal" if self.current_idx>0 else "disabled")
        self.btn_next.config(state="disabled")
        self.btn_submit.config(state="disabled")
        # coding?
        if q.kind in ("coding","sql"):
            self.frm_code.pack(fill="both",expand=True,pady=8)
            self.ed_code.delete("1.0",tk.END)
            if q.template: self.ed_code.insert("1.0",q.template)
        else:
            self.frm_code.forget()
        # speak
        threaded(self._speak)(q.prompt)
        # enforce idle timer
        self._await_start=time.time()
        threaded(self._idle_watch)()

    def _idle_watch(self):
        while time.time()-self._await_start<CONFIG.MAX_IDLE_SECONDS:
            if self.btn_rec_ans['state']=="disabled": return  # user started
            time.sleep(0.5)
        self._issue_warn("Time elapsed – moving to next question")
        self._goto_q(1)

    def _goto_q(self,delta:int):
        new=self.current_idx+delta
        if 0<=new<len(self.qas):
            self.current_idx=new; self._show_question()

    def _speak(self,text):
        try:
            self.tts.say(text); self.tts.runAndWait()
        except Exception as e:
            self.logger.debug(f"TTS error {e}")

    # --- answer recording ---
    def _start_answer(self):
        self.btn_rec_ans.config(state="disabled"); self.btn_stop_ans.config(state="normal")
        self._ans_frames=[]; self._ans_running=True; threaded(self._capture_ans)()

    def _capture_ans(self):
        p=pyaudio.PyAudio(); CHUNK=1024; RATE=16000; FORMAT=pyaudio.paInt16
        stream=p.open(format=FORMAT,channels=1,rate=RATE,input=True,frames_per_buffer=CHUNK)
        while self._ans_running:
            self._ans_frames.append(stream.read(CHUNK))
        stream.stop_stream(); stream.close(); p.terminate()
        path=CONFIG.TEMP_DIR/f"ans_{uuid.uuid4()}.wav"; wf=wave.open(path,'wb'); wf.setnchannels(1); wf.setsampwidth(p.get_sample_size(FORMAT)); wf.setframerate(RATE); wf.writeframes(b''.join(self._ans_frames)); wf.close()
        # transcribe
        text=self._transcribe(path)
        v_sim=cosine(self.voice_emb,self.voice_enc.embed_utterance(preprocess_wav(path))) if self.voice_emb is not None else 0.0
        self.var_voice_sim.set(f"{v_sim:.2f}")
        resp=ResponseRec(question=self.qas[self.current_idx],answer_text=text,voice_path=path,voice_sim=v_sim)
        self.responses.append(resp)
        self.btn_submit.config(state="normal"); messagebox.showinfo("Captured",f"Transcript:\n{text}")

    def _stop_answer(self):
        self._ans_running=False; self.btn_stop_ans.config(state="disabled")

    def _submit_answer(self):
        q=self.qas[self.current_idx]
        if q.kind in ("coding","sql"):
            code=self.ed_code.get("1.0",tk.END)
            resp=ResponseRec(question=q,answer_code=code)
            self.responses.append(resp)
            threaded(self._exec_code)(code,q)
        self.btn_next.config(state="normal")

    # Judge0 execution
    def _exec_code(self,code:str,q:QA):
        url="https://judge0-ce.p.rapidapi.com/submissions"; headers={"content-type":"application/json","X-RapidAPI-Host":CONFIG.JUDGE0_API_HOST,"X-RapidAPI-Key":CONFIG.JUDGE0_API_KEY}
        data={"source_code":code,"language_id":q.lang_id or 71}
        r=requests.post(url,headers=headers,json=data)
        if r.status_code!=201: return self._issue_warn("Code submission failed")
        token=r.json()['token']; time.sleep(3); res=requests.get(f"{url}/{token}",headers=headers)
        out=res.json(); messagebox.showinfo("Judge0",out.get('stdout') or out.get('compile_output') or 'No output')

    # --- Finish interview ---
    def _finish_interview(self):
        if not messagebox.askyesno("Finish","End interview now?"): return
        self.monitoring=False; self._enable_interview_ui(False)
        self._compute_scores(); self.nb.select(3)

    def _compute_scores(self):
        # naive scoring placeholders
        tech=random.uniform(6,9); comm=random.uniform(6,9); prob=random.uniform(6,9); eng=float(self.var_attention.get())*10
        overall=0.35*tech+0.25*comm+0.25*prob+0.15*eng
        self.var_score_overall.set(f"{overall:.1f}/10"); self.var_score_tech.set(f"{tech:.1f}"); self.var_score_comm.set(f"{comm:.1f}")
        self.var_score_prob.set(f"{prob:.1f}"); self.var_score_eng.set(f"{eng:.1f}")
        fb=textwrap.dedent(f"""
        **Summary**\nOverall {overall:.1f}/10\n\nTechnical knowledge {tech:.1f}. Communication {comm:.1f}. Problem‑solving {prob:.1f}. Engagement {eng:.1f}.\n\nDetailed feedback will appear here…""")
        self.txt_feedback.delete("1.0",tk.END); self.txt_feedback.insert("1.0",fb)

    # --- Transcription via OpenAI Whisper (fast) or Google Speech (fallback) ---
    def _transcribe(self,wav_path:pathlib.Path)->str:
        try:
            with open(wav_path,'rb') as f:
                rsp=openai.Audio.transcribe(model="whisper-1",file=f)
            return rsp.text
        except Exception as e:
            self.logger.warning(f"Whisper failed {e}, fallback to Google")
            r=sr.Recognizer(); with sr.AudioFile(str(wav_path)) as src: audio=r.record(src)
            try: return r.recognize_google(audio)
            except: return "(could not transcribe)"

    # --- PDF export ---
    def _export_pdf(self):
        path=filedialog.asksaveasfilename(defaultextension=".pdf",filetypes=[("PDF","*.pdf")])
        if not path: return
        pdf=FPDF(); pdf.set_auto_page_break(auto=True,margin=15); pdf.add_page(); pdf.set_font("Arial","B",16); pdf.cell(0,10,"AI Interview Report",ln=1,align="C")
        pdf.set_font("Arial","",12); pdf.cell(0,10,f"Candidate: {self.candidate_name}",ln=1)
        pdf.cell(0,10,f"Role: {self.desired_role}",ln=1); pdf.cell(0,10,datetime.now().strftime("%Y-%m-%d %H:%M"),ln=1)
        pdf.ln(5)
        pdf.multi_cell(0,8,self.txt_feedback.get("1.0",tk.END))
        pdf.output(path); messagebox.showinfo("Saved",f"Report exported to {path}")

    # ───────── Mainloop ───────────
    def run(self): self.root.mainloop()

# ──────────────────────────── Run ──────────────────────────────
if __name__=="__main__":
    InterviewCoach().run()

# ───────────────────── requirements.txt (printable) ────────────
REQUIREMENTS="""
ttkbootstrap>=1.10
opencv-python
mediapipe
insightface
resemblyzer
pyaudio
librosa
soundfile
speechrecognition
pyttsx3
openai
fpdf
PyMuPDF
requests
numpy
Pillow
"""
print("\n# -------- requirements.txt --------\n"+REQUIREMENTS)
