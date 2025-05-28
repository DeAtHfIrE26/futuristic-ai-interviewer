import os
import cv2
import numpy as np
import threading
import time
import logging
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import mediapipe as mp
import dlib
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Import local config
import config

# Setup logging
logger = logging.getLogger("FaceProcessor")

class FaceProcessor:
    """Advanced face processing class for the interview system"""
    
    def __init__(self):
        """Initialize face processing models and detectors"""
        self.lock = threading.Lock()
        
        # Initialize primary face recognition (InsightFace)
        try:
            self.face_app = FaceAnalysis(name=config.FACE_ANALYSIS_MODEL, 
                                     root=str(config.MODELS_DIR))
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing InsightFace: {e}")
            self.face_app = None
        
        # Initialize backup/supplementary face recognition (dlib)
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            model_path = os.path.join(config.MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
            
            # Download if not exists
            if not os.path.exists(model_path):
                logger.warning(f"Dlib model not found at {model_path}, using basic detector only")
                self.shape_predictor = None
                self.face_recognition_model = None
            else:
                self.shape_predictor = dlib.shape_predictor(model_path)
                self.face_recognition_model = dlib.face_recognition_model_v1(
                    os.path.join(config.MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")
                )
            logger.info("Dlib models loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing Dlib models: {e}")
            self.face_detector = None
            self.shape_predictor = None
            self.face_recognition_model = None
            
        # Initialize MediaPipe for facial landmarks and attention tracking
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe face mesh initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MediaPipe: {e}")
            self.face_mesh = None
            
        # DeepFace settings
        try:
            DeepFace.build_model("VGG-Face")
            logger.info("DeepFace model initialized")
            self.deepface_available = True
        except Exception as e:
            logger.error(f"Error initializing DeepFace: {e}")
            self.deepface_available = False
            
        # Register reference embeddings
        self.reference_embedding = None
        self.reference_descriptor = None
        self.reference_image = None
        
        # Cache for face analysis
        self.last_attention_score = 0
        self.last_emotion = "neutral"
        self.last_face_count = 0
        self.last_analysis_time = 0
        self.analysis_interval = 0.1  # seconds
        
        # Eye aspect ratio thresholds for attention
        self.EAR_THRESHOLD = 0.2
        
        # Initialize warning system
        self.warning_history = {}
        
        logger.info("Face processor initialized successfully")
        
    def register_face(self, frame):
        """Register a face for future verification"""
        with self.lock:
            try:
                if frame is None:
                    return False, "Invalid frame"
                
                # Convert to RGB for processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                success = False
                error_message = "No face detected"
                
                # Primary method: InsightFace
                if self.face_app:
                    faces = self.face_app.get(rgb_frame)
                    if faces:
                        # Store the face embedding
                        self.reference_embedding = faces[0].embedding
                        # Save aligned face
                        aligned_face = face_align.norm_crop(rgb_frame, faces[0].kps)
                        self.reference_image = aligned_face
                        success = True
                        error_message = ""
                
                # Backup method: Dlib
                if not success and self.face_detector and self.face_recognition_model:
                    dlib_faces = self.face_detector(rgb_frame)
                    if dlib_faces:
                        shape = self.shape_predictor(rgb_frame, dlib_faces[0])
                        self.reference_descriptor = self.face_recognition_model.compute_face_descriptor(
                            rgb_frame, shape
                        )
                        # Extract face region
                        rect = dlib_faces[0]
                        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                        self.reference_image = rgb_frame[y:y+h, x:x+w]
                        success = True
                        error_message = ""
                
                # Save reference image if registration was successful
                if success and self.reference_image is not None:
                    # Ensure directory exists
                    os.makedirs(config.FACE_SAMPLES_DIR, exist_ok=True)
                    file_path = os.path.join(config.FACE_SAMPLES_DIR, f"reference_face_{int(time.time())}.jpg")
                    cv2.imwrite(file_path, cv2.cvtColor(self.reference_image, cv2.COLOR_RGB2BGR))
                    logger.info(f"Reference face saved to {file_path}")
                
                return success, error_message
                
            except Exception as e:
                logger.error(f"Error in face registration: {e}")
                return False, str(e)
    
    def verify_face(self, frame):
        """Verify if the face in frame matches the registered face"""
        with self.lock:
            try:
                if frame is None or (self.reference_embedding is None and 
                                    self.reference_descriptor is None and
                                    self.reference_image is None):
                    return False, 0.0, "Reference face or current frame invalid"
                
                # Convert to RGB for processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                similarity = 0.0
                verified = False
                
                # Primary method: InsightFace
                if self.face_app and self.reference_embedding is not None:
                    faces = self.face_app.get(rgb_frame)
                    if faces:
                        # Calculate cosine similarity
                        current_embedding = faces[0].embedding
                        # Convert to similarity score (1 - cosine distance)
                        similarity = 1 - cosine(self.reference_embedding, current_embedding)
                        similarity = max(0, min(1, similarity))  # Clamp between 0 and 1
                        verified = similarity >= config.FACE_SIMILARITY_THRESHOLD
                
                # Backup method: Dlib
                elif not verified and self.face_detector and self.face_recognition_model and self.reference_descriptor is not None:
                    dlib_faces = self.face_detector(rgb_frame)
                    if dlib_faces:
                        shape = self.shape_predictor(rgb_frame, dlib_faces[0])
                        face_descriptor = self.face_recognition_model.compute_face_descriptor(rgb_frame, shape)
                        # Calculate Euclidean distance and convert to similarity
                        distance = np.linalg.norm(np.array(face_descriptor) - np.array(self.reference_descriptor))
                        similarity = max(0, 1 - (distance / 0.6))  # Normalize, 0.6 is a threshold for dlib
                        verified = similarity >= config.FACE_SIMILARITY_THRESHOLD
                
                # Final fallback: DeepFace
                elif not verified and self.deepface_available and self.reference_image is not None:
                    try:
                        # Save current frame temporarily
                        temp_path = os.path.join(config.TEMP_DIR, "temp_verify.jpg")
                        cv2.imwrite(temp_path, frame)
                        
                        # Save reference image temporarily if it's not already saved
                        ref_path = os.path.join(config.TEMP_DIR, "temp_reference.jpg")
                        if isinstance(self.reference_image, np.ndarray):
                            cv2.imwrite(ref_path, cv2.cvtColor(self.reference_image, cv2.COLOR_RGB2BGR))
                        
                        # Verify with DeepFace
                        result = DeepFace.verify(temp_path, ref_path, model_name="VGG-Face", enforce_detection=False)
                        verified = result["verified"]
                        similarity = result.get("similarity", 0) / 100  # Normalize to 0-1
                        
                        # Clean up temp files
                        try:
                            os.remove(temp_path)
                            os.remove(ref_path)
                        except:
                            pass
                    except Exception as e:
                        logger.error(f"DeepFace verification error: {e}")
                
                return verified, similarity, ""
                
            except Exception as e:
                logger.error(f"Error in face verification: {e}")
                return False, 0.0, str(e)
    
    def analyze_face(self, frame, timestamp):
        """Analyze face for attention, emotion, and other metrics"""
        if frame is None:
            return {
                "attention_score": 0, 
                "emotion": "unknown",
                "face_count": 0,
                "eye_contact": False,
                "warnings": []
            }
        
        # Only run analysis at intervals to reduce CPU usage
        if time.time() - self.last_analysis_time < self.analysis_interval:
            return {
                "attention_score": self.last_attention_score,
                "emotion": self.last_emotion,
                "face_count": self.last_face_count,
                "eye_contact": self.last_attention_score > config.ATTENTION_THRESHOLD,
                "warnings": []
            }
        
        self.last_analysis_time = time.time()
        
        try:
            # Convert to RGB for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Initialize results
            attention_score = 0
            emotion = "neutral"
            face_count = 0
            warnings = []
            
            # Count faces
            if self.face_app:
                faces = self.face_app.get(rgb_frame)
                face_count = len(faces)
                self.last_face_count = face_count
                
                # Check for multiple faces warning
                if face_count > 1:
                    warning = self._create_warning("multiple_faces", timestamp)
                    if warning:
                        warnings.append(warning)
                
                # Check for no face warning
                elif face_count == 0:
                    warning = self._create_warning("no_face", timestamp)
                    if warning:
                        warnings.append(warning)
            
            # Calculate attention based on eye tracking
            if self.face_mesh:
                results = self.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    attention_score = self._calculate_attention(results.multi_face_landmarks[0], frame.shape)
                    self.last_attention_score = attention_score
                    
                    # Check for looking away warning
                    if attention_score < config.ATTENTION_THRESHOLD:
                        warning = self._create_warning("looking_away", timestamp)
                        if warning:
                            warnings.append(warning)
            
            # Analyze emotion if DeepFace is available
            if self.deepface_available and face_count > 0:
                try:
                    # Save frame temporarily
                    temp_path = os.path.join(config.TEMP_DIR, "temp_emotion.jpg")
                    cv2.imwrite(temp_path, frame)
                    
                    # Analyze emotion
                    result = DeepFace.analyze(temp_path, actions=['emotion'], enforce_detection=False, silent=True)
                    if result and isinstance(result, list) and len(result) > 0:
                        emotion = result[0]['dominant_emotion']
                        self.last_emotion = emotion
                    
                    # Clean up
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                except Exception as e:
                    logger.debug(f"Emotion analysis error: {e}")
            
            return {
                "attention_score": attention_score,
                "emotion": emotion,
                "face_count": face_count,
                "eye_contact": attention_score > config.ATTENTION_THRESHOLD,
                "warnings": warnings
            }
        
        except Exception as e:
            logger.error(f"Error in face analysis: {e}")
            return {
                "attention_score": 0,
                "emotion": "error",
                "face_count": 0,
                "eye_contact": False,
                "warnings": []
            }
    
    def detect_phone(self, frame, timestamp):
        """Detect if the candidate is using a phone or another device"""
        # This is a simplified implementation - a real system would use object detection
        # For now, we'll check for rectangular objects with specific aspect ratios
        
        # Skip if no frame
        if frame is None:
            return False, []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Skip small contours
                if cv2.contourArea(contour) < 5000:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                # Phone-like aspect ratio (between 0.4 and 0.7 for most phones)
                if 0.4 <= aspect_ratio <= 0.7:
                    # Check if it's not near a face (to avoid false positives)
                    if self.face_app:
                        faces = self.face_app.get(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        is_phone = True
                        
                        for face in faces:
                            bbox = face.bbox.astype(int)
                            face_center_x = (bbox[0] + bbox[2]) / 2
                            face_center_y = (bbox[1] + bbox[3]) / 2
                            contour_center_x = x + w/2
                            contour_center_y = y + h/2
                            
                            # Calculate distance between face center and contour center
                            distance = np.sqrt((face_center_x - contour_center_x)**2 + 
                                              (face_center_y - contour_center_y)**2)
                            
                            # If the contour is close to a face, it's probably not a phone
                            if distance < 200:
                                is_phone = False
                                break
                        
                        if is_phone:
                            warning = self._create_warning("phone_detected", timestamp)
                            if warning:
                                return True, [warning]
            
            return False, []
            
        except Exception as e:
            logger.error(f"Error in phone detection: {e}")
            return False, []
    
    def _calculate_attention(self, face_landmarks, frame_shape):
        """Calculate attention score based on eye position and head pose"""
        try:
            # Extract relevant landmarks for eye tracking
            # Using MediaPipe Face Mesh landmarks
            # Left eye: landmarks 33, 145, 159
            # Right eye: landmarks 263, 374, 386
            
            # Get frame dimensions
            h, w = frame_shape[0:2]
            
            # Get eye landmarks
            left_eye_landmarks = [face_landmarks.landmark[33], face_landmarks.landmark[145], face_landmarks.landmark[159]]
            right_eye_landmarks = [face_landmarks.landmark[263], face_landmarks.landmark[374], face_landmarks.landmark[386]]
            
            # Get nose tip and forehead for head pose
            nose_tip = face_landmarks.landmark[4]
            forehead = face_landmarks.landmark[10]
            
            # Convert normalized coordinates to pixel coordinates
            left_eye_points = [(lm.x * w, lm.y * h) for lm in left_eye_landmarks]
            right_eye_points = [(lm.x * w, lm.y * h) for lm in right_eye_landmarks]
            nose_tip = (nose_tip.x * w, nose_tip.y * h)
            forehead = (forehead.x * w, forehead.y * h)
            
            # Calculate eye aspect ratio (EAR) for both eyes
            left_ear = self._calculate_ear(left_eye_points)
            right_ear = self._calculate_ear(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2
            
            # Check if eyes are open
            eyes_open = avg_ear > self.EAR_THRESHOLD
            
            # Check if looking at camera based on eye position and head pose
            looking_forward = True
            
            # Simple head pose estimation based on nose-forehead line
            head_angle = np.arctan2(forehead[1] - nose_tip[1], forehead[0] - nose_tip[0]) * 180 / np.pi
            looking_forward = looking_forward and (-45 < head_angle < 45)
            
            # Calculate attention score (0-1)
            attention_score = 0.0
            
            if eyes_open and looking_forward:
                attention_score = 0.9  # High attention
            elif eyes_open and not looking_forward:
                attention_score = 0.5  # Medium attention
            elif not eyes_open:
                attention_score = 0.2  # Low attention (eyes closed)
            
            return attention_score
            
        except Exception as e:
            logger.debug(f"Error calculating attention: {e}")
            return self.last_attention_score  # Return last known score on error
    
    def _calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio (EAR)"""
        # This is a simplified EAR calculation with 3 points per eye
        # A full implementation would use all 6 points per eye
        try:
            # Compute distances
            p1, p2, p3 = eye_points
            height = np.linalg.norm(np.array(p2) - np.array(p3))
            width = np.linalg.norm(np.array(p1) - np.array(p3))
            
            # Avoid division by zero
            if width == 0:
                return 0
                
            ear = height / width
            return ear
            
        except Exception as e:
            logger.debug(f"Error calculating EAR: {e}")
            return 0
    
    def _create_warning(self, warning_type, timestamp):
        """Create a warning if enough time has elapsed since the last warning of this type"""
        with self.lock:
            if warning_type not in config.WARNING_TYPES:
                return None
                
            # Check if we've already warned about this recently
            last_warning_time = self.warning_history.get(warning_type, 0)
            if timestamp - last_warning_time < config.WARNING_TIMEOUT:
                return None
                
            # Update warning history
            self.warning_history[warning_type] = timestamp
            
            return {
                "type": warning_type,
                "message": config.WARNING_TYPES[warning_type]["message"],
                "impact": config.WARNING_TYPES[warning_type]["impact"],
                "timestamp": timestamp
            }
    
    def get_marked_frame(self, frame, analysis_result=None):
        """Return the frame with visual annotations for debugging/display"""
        if frame is None:
            return None
            
        try:
            # Create a copy of the frame
            marked_frame = frame.copy()
            
            # Draw face detection boxes
            if self.face_app:
                faces = self.face_app.get(cv2.cvtColor(marked_frame, cv2.COLOR_BGR2RGB))
                for face in faces:
                    bbox = face.bbox.astype(int)
                    
                    # Determine color based on verification result
                    color = (0, 255, 0)  # Default green
                    if analysis_result:
                        if "attention_score" in analysis_result:
                            # Red if attention is low, green if high
                            attention = analysis_result["attention_score"]
                            if attention < config.ATTENTION_THRESHOLD:
                                color = (0, 0, 255)  # Red
                    
                    # Draw rectangle around face
                    cv2.rectangle(marked_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    
            # Add metrics text overlay
            if analysis_result:
                metrics_text = []
                
                # Add attention score
                if "attention_score" in analysis_result:
                    metrics_text.append(f"Attention: {analysis_result['attention_score']:.2f}")
                
                # Add emotion
                if "emotion" in analysis_result:
                    metrics_text.append(f"Emotion: {analysis_result['emotion']}")
                
                # Add face count
                if "face_count" in analysis_result:
                    metrics_text.append(f"Faces: {analysis_result['face_count']}")
                
                # Add warnings count
                if "warnings" in analysis_result:
                    metrics_text.append(f"Warnings: {len(analysis_result['warnings'])}")
                
                # Draw metrics text
                y_pos = 30
                for text in metrics_text:
                    cv2.putText(marked_frame, text, (10, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y_pos += 30
            
            return marked_frame
            
        except Exception as e:
            logger.error(f"Error creating marked frame: {e}")
            return frame 