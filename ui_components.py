import os
import sys
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import threading
import logging

# Import local config
import config

# Setup logging
logger = logging.getLogger("UI")

# Set customtkinter appearance
ctk.set_appearance_mode(config.UI_THEME)
ctk.set_default_color_theme("blue")

class FuturisticFrame(ctk.CTkFrame):
    """Enhanced frame with futuristic appearance"""
    
    def __init__(self, master=None, **kwargs):
        border_width = kwargs.pop("border_width", 1)
        border_color = kwargs.pop("border_color", config.UI_ACCENT_COLOR)
        
        super().__init__(master, border_width=border_width, border_color=border_color, **kwargs)
        
        # Create pulsing border effect
        self.pulsing = False
        self.pulse_thread = None
        self.pulse_alpha = 0  # Current pulse opacity
        self.pulse_direction = 1  # 1 for increasing, -1 for decreasing
        
    def start_pulse(self):
        """Start pulsing border effect"""
        if self.pulsing:
            return
            
        self.pulsing = True
        self.pulse_thread = threading.Thread(target=self._pulse_animation)
        self.pulse_thread.daemon = True
        self.pulse_thread.start()
        
    def stop_pulse(self):
        """Stop pulsing border effect"""
        self.pulsing = False
        
    def _pulse_animation(self):
        """Animate the border pulsing effect"""
        while self.pulsing:
            # Update pulse alpha
            self.pulse_alpha += 0.05 * self.pulse_direction
            
            # Change direction when reaching limits
            if self.pulse_alpha >= 1:
                self.pulse_direction = -1
            elif self.pulse_alpha <= 0:
                self.pulse_direction = 1
                
            # Apply new border color with alpha
            alpha_hex = format(int(self.pulse_alpha * 255), '02x')
            color = config.UI_ACCENT_COLOR + alpha_hex
            
            try:
                self.configure(border_color=color)
            except:
                # May fail if widget is being destroyed
                pass
                
            time.sleep(0.05)

class CameraFrame(FuturisticFrame):
    """Enhanced frame for displaying the camera feed with overlays"""
    
    def __init__(self, master=None, **kwargs):
        width = kwargs.pop("width", 640)
        height = kwargs.pop("height", 480)
        
        super().__init__(master, **kwargs)
        
        self.width = width
        self.height = height
        
        # Create canvas for camera display
        self.canvas = ctk.CTkCanvas(self, width=width, height=height, 
                                   bg=self._apply_appearance_mode(config.UI_SECONDARY_COLOR),
                                   highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create placeholder image
        self.placeholder_img = self._create_placeholder()
        self.current_image = self.placeholder_img
        self.canvas_image = self.canvas.create_image(0, 0, anchor="nw", image=self.current_image)
        
        # Camera status display
        self.status_var = tk.StringVar(value="Camera: Standby")
        self.status_label = ctk.CTkLabel(self, textvariable=self.status_var, 
                                        text_color=config.UI_ACCENT_COLOR,
                                        font=(config.UI_FONT, config.UI_TEXT_FONT_SIZE))
        self.status_label.pack(pady=(0, 10))
        
        # Metrics display
        self.metrics_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Metrics variables
        self.face_sim_var = tk.StringVar(value="Face Match: --")
        self.attention_var = tk.StringVar(value="Attention: --")
        self.emotion_var = tk.StringVar(value="Emotion: --")
        self.warnings_var = tk.StringVar(value="Warnings: 0")
        
        # Metric labels
        metrics_grid = [
            ("Face", self.face_sim_var),
            ("Attention", self.attention_var),
            ("Emotion", self.emotion_var),
            ("Warnings", self.warnings_var)
        ]
        
        # Add metrics to the frame
        for i, (label, var) in enumerate(metrics_grid):
            metric_frame = ctk.CTkFrame(self.metrics_frame, fg_color="transparent")
            metric_frame.grid(row=0, column=i, padx=5, pady=5, sticky="ew")
            
            ctk.CTkLabel(metric_frame, text=label, 
                       font=(config.UI_FONT, config.UI_TEXT_FONT_SIZE - 2),
                       text_color="gray").pack(anchor="w")
                       
            ctk.CTkLabel(metric_frame, textvariable=var, 
                       font=(config.UI_FONT, config.UI_TEXT_FONT_SIZE),
                       text_color=config.UI_ACCENT_COLOR).pack(anchor="w")
                       
        # Configure grid for equal column widths
        for i in range(4):
            self.metrics_frame.grid_columnconfigure(i, weight=1)
            
        # Warning display
        self.warning_var = tk.StringVar(value="")
        self.warning_label = ctk.CTkLabel(self, textvariable=self.warning_var, 
                                        text_color=config.UI_WARNING_COLOR,
                                        font=(config.UI_FONT, config.UI_TEXT_FONT_SIZE))
        self.warning_label.pack(pady=5)
        self.warning_label.pack_forget()  # Hide initially
        
        # Active camera flag
        self.is_active = False
        
    def _create_placeholder(self):
        """Create a placeholder image for when no camera is active"""
        placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        placeholder.fill(50)  # Dark gray
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Camera Inactive"
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        
        # Position text at center
        text_x = (placeholder.shape[1] - textsize[0]) // 2
        text_y = (placeholder.shape[0] + textsize[1]) // 2
        
        cv2.putText(placeholder, text, (text_x, text_y), font, 1, (200, 200, 200), 2)
        
        # Convert to PhotoImage
        img = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return ImageTk.PhotoImage(image=img)
        
    def update_frame(self, frame):
        """Update the camera frame with a new image"""
        if frame is None:
            self.canvas.itemconfig(self.canvas_image, image=self.placeholder_img)
            self.current_image = self.placeholder_img
            return
            
        # Resize frame to fit canvas
        frame = cv2.resize(frame, (self.width, self.height))
        
        # Convert to PhotoImage
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        photo_img = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.canvas.itemconfig(self.canvas_image, image=photo_img)
        self.current_image = photo_img  # Keep reference to prevent garbage collection
        
    def set_active(self, active=True):
        """Set camera active status"""
        self.is_active = active
        
        if active:
            self.status_var.set("Camera: Active")
            self.start_pulse()
        else:
            self.status_var.set("Camera: Standby")
            self.stop_pulse()
            self.update_frame(None)
            
    def update_metrics(self, face_sim=None, attention=None, emotion=None, warnings=None):
        """Update the displayed metrics"""
        if face_sim is not None:
            # Format as percentage
            face_sim_pct = min(100, max(0, face_sim * 100))
            self.face_sim_var.set(f"Face Match: {face_sim_pct:.1f}%")
            
        if attention is not None:
            # Format as percentage
            attention_pct = min(100, max(0, attention * 100))
            self.attention_var.set(f"Attention: {attention_pct:.1f}%")
            
        if emotion is not None:
            self.emotion_var.set(f"Emotion: {emotion.capitalize()}")
            
        if warnings is not None:
            self.warnings_var.set(f"Warnings: {warnings}")
            
    def show_warning(self, message, duration=3000):
        """Show a warning message for the specified duration in milliseconds"""
        if not message:
            return
            
        self.warning_var.set(message)
        self.warning_label.pack(pady=5)
        
        # Hide warning after duration
        self.after(duration, self.hide_warning)
        
    def hide_warning(self):
        """Hide the warning message"""
        self.warning_label.pack_forget()

class VoiceVisualizer(ctk.CTkFrame):
    """Voice level visualizer with animated bars"""
    
    def __init__(self, master=None, **kwargs):
        width = kwargs.pop("width", 200)
        height = kwargs.pop("height", 60)
        
        super().__init__(master, **kwargs)
        
        self.width = width
        self.height = height
        
        # Create canvas for visualization
        self.canvas = ctk.CTkCanvas(self, width=width, height=height, 
                                  bg=self._apply_appearance_mode(config.UI_SECONDARY_COLOR),
                                  highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bar settings
        self.num_bars = 20
        self.bar_width = width / (self.num_bars * 1.5)
        self.bar_spacing = self.bar_width / 2
        self.max_bar_height = height - 10
        
        # Create bars
        self.bars = []
        for i in range(self.num_bars):
            x = i * (self.bar_width + self.bar_spacing) + 10
            bar = self.canvas.create_rectangle(
                x, height - 5, x + self.bar_width, height - 5,
                fill=config.UI_ACCENT_COLOR, outline=""
            )
            self.bars.append(bar)
            
        # Voice match indicator
        self.match_var = tk.StringVar(value="Voice Match: --")
        self.match_label = ctk.CTkLabel(self, textvariable=self.match_var, 
                                      text_color=config.UI_ACCENT_COLOR,
                                      font=(config.UI_FONT, config.UI_TEXT_FONT_SIZE))
        self.match_label.pack(pady=5)
        
        # Recording state
        self.is_recording = False
        self.animation_active = False
        
        # Start animation
        self._start_animation()
        
    def _start_animation(self):
        """Start the bar animation"""
        self.animation_active = True
        self._animate_bars()
        
    def _animate_bars(self):
        """Animate the bars based on recording state"""
        if not self.animation_active:
            return
            
        if self.is_recording:
            # Generate random heights when recording
            heights = []
            for i in range(self.num_bars):
                if i > 0:
                    # Ensure some continuity between adjacent bars
                    prev_height = heights[i-1]
                    max_diff = self.max_bar_height * 0.3
                    min_h = max(5, prev_height - max_diff)
                    max_h = min(self.max_bar_height, prev_height + max_diff)
                    height = np.random.uniform(min_h, max_h)
                else:
                    height = np.random.uniform(5, self.max_bar_height)
                    
                heights.append(height)
        else:
            # When not recording, show minimal movement
            heights = [np.random.uniform(3, 8) for _ in range(self.num_bars)]
            
        # Update bar heights
        for i, bar in enumerate(self.bars):
            height = heights[i]
            x1, _, x2, y2 = self.canvas.coords(bar)
            self.canvas.coords(bar, x1, self.height - height - 5, x2, y2)
            
            # Update bar color based on height
            if self.is_recording:
                # Calculate color based on height - red for high levels
                intensity = height / self.max_bar_height
                r = min(255, int(100 + intensity * 155))
                g = min(255, int(100 + 155 * (1 - intensity)))
                b = min(255, int(100 + intensity * 50))
                color = f"#{r:02x}{g:02x}{b:02x}"
            else:
                color = config.UI_ACCENT_COLOR
                
            self.canvas.itemconfig(bar, fill=color)
            
        # Schedule next animation frame
        delay = 50 if self.is_recording else 200
        self.after(delay, self._animate_bars)
        
    def set_recording(self, recording=True):
        """Set recording status"""
        self.is_recording = recording
        
    def set_voice_match(self, match_value=None):
        """Set voice match percentage"""
        if match_value is None:
            self.match_var.set("Voice Match: --")
        else:
            # Format as percentage
            match_pct = min(100, max(0, match_value * 100))
            self.match_var.set(f"Voice Match: {match_pct:.1f}%")
            
    def stop_animation(self):
        """Stop the animation when widget is destroyed"""
        self.animation_active = False

class TranscriptArea(FuturisticFrame):
    """Enhanced scrollable area for interview transcript"""
    
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        
        # Create transcript text widget
        self.transcript = ctk.CTkTextbox(self, wrap="word", height=300,
                                        font=(config.UI_FONT, config.UI_TEXT_FONT_SIZE))
        self.transcript.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Make read-only
        self.transcript.configure(state="disabled")
        
        # Automatic scrolling flag
        self.auto_scroll = True
        
    def add_interviewer_message(self, message):
        """Add an interviewer message to the transcript"""
        if not message:
            return
            
        # Enable editing
        self.transcript.configure(state="normal")
        
        # Add timestamp and speaker prefix
        timestamp = time.strftime("%H:%M:%S")
        self.transcript.insert("end", f"[{timestamp}] ", "timestamp")
        self.transcript.insert("end", "AI Interviewer: ", "interviewer")
        self.transcript.insert("end", f"{message}\n\n", "interviewer_text")
        
        # Configure tags
        self.transcript.tag_config("timestamp", foreground="gray")
        self.transcript.tag_config("interviewer", foreground=config.UI_ACCENT_COLOR, font=(config.UI_FONT, config.UI_TEXT_FONT_SIZE, "bold"))
        self.transcript.tag_config("interviewer_text", foreground="white")
        
        # Disable editing
        self.transcript.configure(state="disabled")
        
        # Auto-scroll to bottom
        if self.auto_scroll:
            self.transcript.see("end")
            
    def add_candidate_message(self, message):
        """Add a candidate message to the transcript"""
        if not message:
            return
            
        # Enable editing
        self.transcript.configure(state="normal")
        
        # Add timestamp and speaker prefix
        timestamp = time.strftime("%H:%M:%S")
        self.transcript.insert("end", f"[{timestamp}] ", "timestamp")
        self.transcript.insert("end", "You: ", "candidate")
        self.transcript.insert("end", f"{message}\n\n", "candidate_text")
        
        # Configure tags
        self.transcript.tag_config("timestamp", foreground="gray")
        self.transcript.tag_config("candidate", foreground="#2ECC71", font=(config.UI_FONT, config.UI_TEXT_FONT_SIZE, "bold"))
        self.transcript.tag_config("candidate_text", foreground="white")
        
        # Disable editing
        self.transcript.configure(state="disabled")
        
        # Auto-scroll to bottom
        if self.auto_scroll:
            self.transcript.see("end")
            
    def add_system_message(self, message):
        """Add a system message to the transcript"""
        if not message:
            return
            
        # Enable editing
        self.transcript.configure(state="normal")
        
        # Add timestamp and message
        timestamp = time.strftime("%H:%M:%S")
        self.transcript.insert("end", f"[{timestamp}] ", "timestamp")
        self.transcript.insert("end", f"{message}\n", "system")
        
        # Configure tags
        self.transcript.tag_config("timestamp", foreground="gray")
        self.transcript.tag_config("system", foreground="orange")
        
        # Disable editing
        self.transcript.configure(state="disabled")
        
        # Auto-scroll to bottom
        if self.auto_scroll:
            self.transcript.see("end")
            
    def clear(self):
        """Clear the transcript"""
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        
    def get_text(self):
        """Get the full transcript text"""
        return self.transcript.get("1.0", "end")

class CodeEditor(ctk.CTkFrame):
    """Code editor with syntax highlighting"""
    
    def __init__(self, master=None, **kwargs):
        language = kwargs.pop("language", "python")
        
        super().__init__(master, **kwargs)
        
        # Store language
        self.language = language
        
        # Create toolbar
        self.toolbar = ctk.CTkFrame(self)
        self.toolbar.pack(fill=tk.X, pady=(0, 5))
        
        # Language selector
        lang_values = ["python", "javascript", "java", "csharp", "sql"]
        self.lang_var = tk.StringVar(value=language)
        
        lang_label = ctk.CTkLabel(self.toolbar, text="Language:")
        lang_label.pack(side=tk.LEFT, padx=(10, 5))
        
        self.lang_menu = ctk.CTkOptionMenu(self.toolbar, values=lang_values, 
                                        variable=self.lang_var,
                                        command=self._on_language_change)
        self.lang_menu.pack(side=tk.LEFT, padx=5)
        
        # Run button
        self.run_btn = ctk.CTkButton(self.toolbar, text="Run Code", 
                                   command=self._on_run_code)
        self.run_btn.pack(side=tk.RIGHT, padx=10)
        
        # Create code editor
        self.editor = ctk.CTkTextbox(self, wrap="none", height=300,
                                   font=("Consolas", config.UI_TEXT_FONT_SIZE))
        self.editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Line numbers
        self.line_numbers = ctk.CTkTextbox(self, width=30, wrap="none",
                                         font=("Consolas", config.UI_TEXT_FONT_SIZE))
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0), pady=5)
        self.line_numbers.configure(state="disabled")
        
        # Reposition line numbers
        self.editor.pack_forget()
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0), pady=5)
        self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5), pady=5)
        
        # Code output area
        self.output_label = ctk.CTkLabel(self, text="Output:")
        self.output_label.pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        self.output = ctk.CTkTextbox(self, wrap="word", height=100,
                                   font=("Consolas", config.UI_TEXT_FONT_SIZE))
        self.output.pack(fill=tk.X, expand=False, padx=10, pady=(0, 10))
        self.output.configure(state="disabled")
        
        # Run callback
        self.run_callback = None
        
        # Bind events
        self.editor.bind("<KeyRelease>", self._on_text_change)
        
        # Initial line numbers update
        self._update_line_numbers()
        
    def _on_language_change(self, lang):
        """Handle language change"""
        self.language = lang
        
    def _on_text_change(self, event=None):
        """Handle text change in editor"""
        self._update_line_numbers()
        
    def _update_line_numbers(self):
        """Update line numbers display"""
        # Get text content
        text_content = self.editor.get("1.0", "end")
        
        # Count lines
        num_lines = text_content.count("\n") + 1
        
        # Generate line numbers text
        line_numbers_text = "\n".join(str(i) for i in range(1, num_lines))
        
        # Update line numbers
        self.line_numbers.configure(state="normal")
        self.line_numbers.delete("1.0", "end")
        self.line_numbers.insert("1.0", line_numbers_text)
        self.line_numbers.configure(state="disabled")
        
    def _on_run_code(self):
        """Handle run code button click"""
        if self.run_callback:
            code = self.editor.get("1.0", "end")
            self.run_callback(code, self.language)
            
    def set_code(self, code_text):
        """Set code in the editor"""
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", code_text)
        self._update_line_numbers()
        
    def get_code(self):
        """Get code from the editor"""
        return self.editor.get("1.0", "end")
        
    def set_output(self, output_text):
        """Set output in the output area"""
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", output_text)
        self.output.configure(state="disabled")
        
    def set_run_callback(self, callback):
        """Set callback for run button"""
        self.run_callback = callback 