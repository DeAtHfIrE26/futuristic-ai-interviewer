import os
import time
import json
import logging
import threading
from pathlib import Path
import re
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import torch
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

# Import local config
import config

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Setup logging
logger = logging.getLogger("LanguageModel")

class LanguageModel:
    """Advanced language model processor for the interview system"""
    
    def __init__(self):
        """Initialize language model and processors"""
        self.lock = threading.Lock()
        
        # Initialize main language model for question generation and evaluation
        try:
            model_path = os.path.join(config.MODELS_DIR, config.LLM_MODEL_PATH)
            if not os.path.exists(model_path):
                model_path = config.LLM_MODEL_PATH  # Try as absolute path
                
            self.llm = Llama(
                model_path=model_path,
                n_ctx=config.CONTEXT_LENGTH,
                n_threads=os.cpu_count(),
                n_gpu_layers=-1,
                verbose=False
            )
            logger.info(f"Language model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error initializing primary language model: {e}")
            self.llm = None
            
        # Initialize embeddings model for resume analysis
        try:
            self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            logger.info("Embeddings model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing embeddings model: {e}")
            self.embeddings_model = None
            
        # Resume analysis storage
        self.resume_texts = []
        self.resume_skills = set()
        self.resume_experience = []
        self.resume_education = []
        self.resume_vector_store = None
        
        # Interview state
        self.asked_questions = []
        self.current_role = ""
        self.interview_responses = []
        self.generated_skills = set()
        self.response_evaluations = []
        
        logger.info("Language model processor initialized successfully")
        
    def analyze_resume(self, resume_text, desired_role):
        """Analyze resume to extract skills, experience, and relevant information"""
        with self.lock:
            try:
                self.current_role = desired_role.lower()
                self.resume_texts = []
                self.resume_skills = set()
                self.resume_experience = []
                self.resume_education = []
                
                # Store the full text
                self.resume_texts.append(resume_text)
                
                # Extract sections from resume
                sections = self._extract_resume_sections(resume_text)
                
                # Extract skills and other relevant information
                if "skills" in sections:
                    self.resume_skills = self._extract_skills(sections["skills"])
                    
                if "experience" in sections:
                    self.resume_experience = self._extract_experience(sections["experience"])
                    
                if "education" in sections:
                    self.resume_education = self._extract_education(sections["education"])
                
                # Create a vector store for semantic search
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                resume_chunks = text_splitter.split_text(resume_text)
                
                if self.embeddings:
                    self.resume_vector_store = FAISS.from_texts(resume_chunks, self.embeddings)
                
                # Generate skills for the specified role if none are found in resume
                if not self.resume_skills and self.current_role:
                    self.generated_skills = self._generate_skills_for_role(self.current_role)
                    
                logger.info(f"Resume analyzed successfully: {len(self.resume_skills)} skills, {len(self.resume_experience)} experience items")
                return True
                
            except Exception as e:
                logger.error(f"Error analyzing resume: {e}")
                return False
    
    def _extract_resume_sections(self, resume_text):
        """Extract common sections from resume text"""
        sections = {}
        
        # Common section headers
        section_patterns = {
            "skills": r"(?i)(skills|technical skills|core competencies|areas of expertise)",
            "experience": r"(?i)(experience|work experience|employment history|professional experience)",
            "education": r"(?i)(education|educational background|academic background|qualifications)",
            "projects": r"(?i)(projects|personal projects|key projects)",
            "summary": r"(?i)(summary|professional summary|profile|about)",
            "certifications": r"(?i)(certifications|licenses|professional certifications)"
        }
        
        # Find positions of all section headers
        section_positions = []
        for section_name, pattern in section_patterns.items():
            for match in re.finditer(pattern, resume_text):
                section_positions.append((match.start(), section_name))
        
        # Sort by position
        section_positions.sort()
        
        # Extract section content
        for i, (pos, section_name) in enumerate(section_positions):
            start = pos
            # Find the end of this section (start of next section or end of text)
            if i < len(section_positions) - 1:
                end = section_positions[i+1][0]
            else:
                end = len(resume_text)
                
            # Extract the section content
            section_text = resume_text[start:end].strip()
            # Remove the section header from the content
            section_header_match = re.search(section_patterns[section_name], section_text)
            if section_header_match:
                header_end = section_header_match.end()
                section_text = section_text[header_end:].strip()
                
            sections[section_name] = section_text
            
        return sections
    
    def _extract_skills(self, skills_text):
        """Extract individual skills from skills section"""
        skills = set()
        
        # Split by common delimiters
        for delimiter in [',', '•', '|', '/', '\n', ';']:
            if delimiter in skills_text:
                parts = [p.strip() for p in skills_text.split(delimiter)]
                for part in parts:
                    if part and len(part) < 50:  # Reasonable skill length
                        skills.add(part.lower())
                
                # If we found skills, break
                if skills:
                    break
        
        # If no skills found with delimiters, try to extract using text prompt
        if not skills and self.llm:
            prompt = f"""Extract a list of technical and professional skills from the following resume text. 
Format the output as a simple comma-separated list of skills.

Text:
{skills_text}

Skills:"""
            
            result = self.llm(prompt, max_tokens=200, temperature=0.0, stop=["Text:", "\n\n"])
            if result and "choices" in result and len(result["choices"]) > 0:
                extracted_text = result["choices"][0]["text"]
                # Process the skills
                skill_parts = [s.strip().lower() for s in extracted_text.split(',')]
                for skill in skill_parts:
                    if skill and len(skill) < 50:
                        skills.add(skill)
        
        return skills
    
    def _extract_experience(self, experience_text):
        """Extract experience items from experience section"""
        experiences = []
        
        # Split into paragraphs
        paragraphs = experience_text.split('\n\n')
        
        for paragraph in paragraphs:
            if len(paragraph.strip()) > 10:  # Minimum length for a meaningful entry
                experiences.append(paragraph.strip())
        
        return experiences
    
    def _extract_education(self, education_text):
        """Extract education items from education section"""
        education = []
        
        # Split into paragraphs
        paragraphs = education_text.split('\n\n')
        
        for paragraph in paragraphs:
            if len(paragraph.strip()) > 10:  # Minimum length for a meaningful entry
                education.append(paragraph.strip())
        
        return education
    
    def _generate_skills_for_role(self, role):
        """Generate typical skills for a role if none found in resume"""
        if not self.llm:
            return set()
            
        role_skills = set()
        
        # Check if we have predefined skills for this role
        role_lower = role.lower()
        for role_key, skills_list in config.ROLE_SKILLS_MAPPING.items():
            if role_key in role_lower:
                return set(skills_list)
        
        # If not in predefined mappings, generate with LLM
        prompt = f"""What are the top 10 most important technical and professional skills for a {role} position? 
Format the output as a simple comma-separated list of skills.

Skills:"""
        
        try:
            result = self.llm(prompt, max_tokens=200, temperature=0.0, stop=["\n\n"])
            if result and "choices" in result and len(result["choices"]) > 0:
                extracted_text = result["choices"][0]["text"]
                # Process the skills
                skill_parts = [s.strip().lower() for s in extracted_text.split(',')]
                for skill in skill_parts:
                    if skill and len(skill) < 50:
                        role_skills.add(skill)
        except Exception as e:
            logger.error(f"Error generating skills for role: {e}")
            
        return role_skills
    
    def generate_question(self, question_number, previous_questions=None, previous_responses=None):
        """Generate the next interview question based on resume, role, and previous interactions"""
        with self.lock:
            try:
                if not self.llm:
                    return "Tell me about your experience and how it relates to this role.", False
                
                # Determine question type based on the question number
                # First question is always an introduction
                if question_number == 1:
                    return "Tell me about yourself and why you're interested in this role.", False
                
                # Last 1-2 questions should be coding/technical challenges if appropriate for the role
                if (question_number >= config.MIN_QUESTIONS and 
                    any(tech_role in self.current_role.lower() for tech_role in 
                        ["developer", "engineer", "programmer", "coder", "data scientist"])):
                    
                    # Check if we've already asked a coding question
                    if previous_questions and any("coding" in q.lower() for q in previous_questions):
                        # If yes, generate a normal question
                        pass
                    else:
                        # If no, generate a coding question
                        return self._generate_coding_question(), True
                
                # Build context from resume and previous interactions
                context = self._build_question_context(previous_questions, previous_responses)
                
                # Create the prompt for question generation
                prompt = self._create_question_generation_prompt(context, question_number)
                
                # Generate the question
                result = self.llm(prompt, max_tokens=300, temperature=config.TEMPERATURE, 
                                 top_p=config.TOP_P, repeat_penalty=config.REPETITION_PENALTY)
                
                if result and "choices" in result and len(result["choices"]) > 0:
                    generated_text = result["choices"][0]["text"].strip()
                    
                    # Extract just the question (remove any preamble or explanations)
                    questions = re.findall(r"(?:^|\n)(.+\?)", generated_text)
                    if questions:
                        question = questions[0].strip()
                    else:
                        # If no question mark, take the first sentence or the whole text if it's short
                        sentences = sent_tokenize(generated_text)
                        question = sentences[0] if sentences else generated_text
                        
                        # Make sure it ends with a question mark
                        if not question.endswith("?"):
                            question += "?"
                    
                    # Store in asked questions
                    self.asked_questions.append(question)
                    
                    return question, False
                    
                # Fallback questions if generation fails
                fallback_questions = [
                    "What are your greatest strengths as they relate to this position?",
                    "How do you handle challenging situations at work?",
                    "Describe a project you're particularly proud of and your role in it.",
                    "How do you stay current with industry trends and new technologies?",
                    "What are you looking for in your next role?"
                ]
                
                fallback = fallback_questions[question_number % len(fallback_questions)]
                self.asked_questions.append(fallback)
                return fallback, False
                
            except Exception as e:
                logger.error(f"Error generating question: {e}")
                fallback = "Tell me about a challenging project you worked on recently."
                self.asked_questions.append(fallback)
                return fallback, False
    
    def _generate_coding_question(self):
        """Generate a coding challenge appropriate for the role"""
        if not self.llm:
            return "Write a function to find the maximum value in an array."
        
        role_lower = self.current_role.lower()
        
        # Determine the appropriate difficulty and domain for the coding question
        difficulty = "medium"
        domain = "algorithms"
        language = "python"
        
        if "junior" in role_lower or "entry" in role_lower:
            difficulty = "easy"
        elif "senior" in role_lower or "lead" in role_lower:
            difficulty = "hard"
            
        if "front" in role_lower:
            domain = "frontend"
            language = "javascript"
        elif "back" in role_lower:
            domain = "backend"
        elif "data" in role_lower:
            domain = "data processing"
        elif "machine learning" in role_lower or "ml" in role_lower:
            domain = "machine learning"
            
        # Create prompt for coding question
        prompt = f"""Generate a {difficulty} coding challenge for a {self.current_role} interview.
The challenge should focus on {domain} and be solvable in {language}.
Give a clear problem statement, input format, expected output, and an example input/output.
Do not provide the solution.

Coding challenge:"""
        
        try:
            result = self.llm(prompt, max_tokens=500, temperature=0.7)
            if result and "choices" in result and len(result["choices"]) > 0:
                challenge_text = result["choices"][0]["text"].strip()
                return challenge_text
        except Exception as e:
            logger.error(f"Error generating coding question: {e}")
            
        # Fallback coding questions
        fallbacks = {
            "python": "Write a Python function to find the longest substring without repeating characters in a given string.",
            "javascript": "Write a JavaScript function to implement a debounce function which will postpone a function's execution until after a given time has elapsed.",
            "java": "Write a Java function to find all pairs of elements in an array whose sum is equal to a given target.",
            "c#": "Write a C# function that checks if a string is a palindrome, considering only alphanumeric characters and ignoring case.",
            "data": "Write a function to clean a dataset by removing outliers based on the IQR method.",
        }
        
        # Choose appropriate fallback based on role
        if "javascript" in role_lower or "frontend" in role_lower:
            return fallbacks["javascript"]
        elif "java" in role_lower:
            return fallbacks["java"]
        elif "c#" in role_lower:
            return fallbacks["c#"]
        elif "data" in role_lower:
            return fallbacks["data"]
        else:
            return fallbacks["python"]
    
    def _build_question_context(self, previous_questions, previous_responses):
        """Build context for question generation from resume and previous interactions"""
        context = []
        
        # Add role information
        context.append(f"Role: {self.current_role}")
        
        # Add resume highlights
        if self.resume_skills:
            context.append(f"Skills: {', '.join(list(self.resume_skills)[:10])}")
            
        if self.resume_experience:
            context.append("Experience Highlights:")
            for exp in self.resume_experience[:2]:  # Just first 2 experiences for brevity
                summary = exp[:200] + "..." if len(exp) > 200 else exp
                context.append(f"- {summary}")
                
        # Add previous Q&A to maintain coherent conversation
        if previous_questions and previous_responses:
            context.append("Previous Questions and Answers:")
            for i, (q, a) in enumerate(zip(previous_questions, previous_responses)):
                if i >= 3:  # Only include the last 3 Q&As to keep context manageable
                    break
                a_summary = a[:150] + "..." if len(a) > 150 else a
                context.append(f"Q: {q}")
                context.append(f"A: {a_summary}")
        
        return "\n".join(context)
    
    def _create_question_generation_prompt(self, context, question_number):
        """Create the prompt for question generation"""
        # Determine appropriate question type/category based on question progression
        if question_number <= 2:
            question_type = "general background and experience"
        elif question_number <= 4:
            question_type = "technical skills and expertise"
        elif question_number <= 6:
            question_type = "problem solving and approach"
        elif question_number <= 8:
            question_type = "behavioral and situational"
        else:
            question_type = "career goals and cultural fit"
            
        # Format the system prompt template
        system_prompt = config.SYSTEM_PROMPTS["question_generation"].format(role=self.current_role)
        
        # Create the full prompt
        prompt = f"""{system_prompt}

Context for the interview:
{context}

The candidate is interviewing for a {self.current_role} position.
This is question #{question_number} in the interview.
This question should focus on {question_type}.
Questions already asked: {'; '.join(self.asked_questions)}

Generate one clear, concise, and specific interview question:"""

        return prompt
    
    def evaluate_response(self, question, response, is_coding_question=False):
        """Evaluate candidate's response to an interview question"""
        with self.lock:
            try:
                if not self.llm:
                    # Dummy scoring if no LLM available
                    return {
                        "score": 7.0, 
                        "feedback": "Response noted.", 
                        "strengths": [], 
                        "weaknesses": []
                    }
                
                # Store response
                self.interview_responses.append(response)
                
                # Create evaluation prompt
                system_prompt = config.SYSTEM_PROMPTS["response_evaluation"].format(role=self.current_role)
                
                # Different prompts for regular vs coding questions
                if is_coding_question:
                    prompt = f"""{system_prompt}

Question (Coding Challenge):
{question}

Candidate's Solution:
{response}

Evaluate this solution based on correctness, efficiency, code quality, and problem-solving approach.
Rate the solution on a scale of 0-10 and provide brief, constructive feedback.

Evaluation:"""
                else:
                    prompt = f"""{system_prompt}

Question:
{question}

Candidate's Response:
{response}

Evaluate this response based on relevance, clarity, depth, technical accuracy, and completeness.
Rate the response on a scale of 0-10 and provide brief, constructive feedback.

Evaluation:"""

                # Generate evaluation
                result = self.llm(prompt, max_tokens=300, temperature=0.3)
                
                if result and "choices" in result and len(result["choices"]) > 0:
                    evaluation_text = result["choices"][0]["text"].strip()
                    
                    # Parse the evaluation
                    score_match = re.search(r"(\d+(?:\.\d+)?)\s*\/\s*10", evaluation_text)
                    score = float(score_match.group(1)) if score_match else 7.0
                    
                    # Extract feedback comment
                    feedback = ""
                    lines = evaluation_text.split('\n')
                    for line in lines:
                        if "score" not in line.lower() and len(line.strip()) > 10:
                            feedback = line.strip()
                            break
                    
                    # Extract strengths and weaknesses if mentioned
                    strengths = []
                    weaknesses = []
                    
                    strength_section = re.search(r"Strengths?:(.*?)(?:Weakness|$)", evaluation_text, re.DOTALL)
                    if strength_section:
                        strength_text = strength_section.group(1).strip()
                        # Extract bullet points or sentences
                        strengths = [s.strip("- ").strip() for s in re.split(r"\n-|\n•|\n\*", strength_text) if s.strip()]
                    
                    weakness_section = re.search(r"Weakness(?:es)?:(.*?)(?:$)", evaluation_text, re.DOTALL)
                    if weakness_section:
                        weakness_text = weakness_section.group(1).strip()
                        # Extract bullet points or sentences
                        weaknesses = [w.strip("- ").strip() for w in re.split(r"\n-|\n•|\n\*", weakness_text) if w.strip()]
                    
                    evaluation = {
                        "score": score,
                        "feedback": feedback,
                        "strengths": strengths,
                        "weaknesses": weaknesses
                    }
                    
                    # Store evaluation
                    self.response_evaluations.append(evaluation)
                    
                    return evaluation
                
                # Default evaluation if parsing fails
                default_eval = {
                    "score": 7.0,
                    "feedback": "The response addresses the question adequately.",
                    "strengths": [],
                    "weaknesses": []
                }
                
                self.response_evaluations.append(default_eval)
                return default_eval
                
            except Exception as e:
                logger.error(f"Error evaluating response: {e}")
                default_eval = {
                    "score": 7.0,
                    "feedback": "Response received.",
                    "strengths": [],
                    "weaknesses": []
                }
                self.response_evaluations.append(default_eval)
                return default_eval
    
    def generate_final_evaluation(self):
        """Generate a comprehensive final evaluation of the interview"""
        with self.lock:
            try:
                if not self.llm or not self.interview_responses:
                    return {
                        "overall_score": 7.0,
                        "summary": "Interview completed.",
                        "strengths": ["Communication"],
                        "areas_for_improvement": ["Technical depth"],
                        "recommendation": "Consider"
                    }
                
                # Calculate average score from individual responses
                scores = [eval.get("score", 7.0) for eval in self.response_evaluations]
                avg_score = sum(scores) / len(scores) if scores else 7.0
                
                # Build a transcript of the interview
                transcript = []
                for i, (question, response) in enumerate(zip(self.asked_questions, self.interview_responses)):
                    transcript.append(f"Q{i+1}: {question}")
                    response_summary = response[:300] + "..." if len(response) > 300 else response
                    transcript.append(f"A{i+1}: {response_summary}")
                
                # Create the evaluation prompt
                system_prompt = config.SYSTEM_PROMPTS["final_evaluation"].format(role=self.current_role)
                
                prompt = f"""{system_prompt}

Role: {self.current_role}
Resume Skills: {', '.join(list(self.resume_skills)[:15])}

Interview Transcript:
{chr(10).join(transcript)}

Average Question Score: {avg_score:.1f}/10

Based on this interview, provide:
1. Overall assessment and score (0-10)
2. Key strengths demonstrated
3. Areas for improvement
4. Final recommendation (Strongly Recommend, Recommend, Consider, or Do Not Recommend)

Final Evaluation:"""

                # Generate evaluation
                result = self.llm(prompt, max_tokens=1000, temperature=0.3)
                
                if result and "choices" in result and len(result["choices"]) > 0:
                    evaluation_text = result["choices"][0]["text"].strip()
                    
                    # Parse overall score
                    score_match = re.search(r"(?:Overall (?:Score|Assessment|Rating):|Score:)\s*(\d+(?:\.\d+)?)\s*\/\s*10", 
                                           evaluation_text, re.IGNORECASE)
                    overall_score = float(score_match.group(1)) if score_match else avg_score
                    
                    # Extract summary
                    summary = ""
                    lines = evaluation_text.split('\n')
                    for i, line in enumerate(lines):
                        if ("overall" in line.lower() or "summary" in line.lower()) and i < len(lines) - 1:
                            next_line = lines[i+1].strip()
                            if len(next_line) > 10 and ":" not in next_line:
                                summary = next_line
                                break
                    
                    if not summary:
                        # Take the first substantial paragraph
                        paragraphs = evaluation_text.split('\n\n')
                        for para in paragraphs:
                            if len(para.strip()) > 20 and ":" not in para:
                                summary = para.strip()
                                break
                    
                    # Extract strengths
                    strengths = []
                    strength_section = re.search(r"(?:Key )?Strengths?:(.*?)(?:Areas?|Weakness|Improvement|Recommendation|$)", 
                                               evaluation_text, re.DOTALL | re.IGNORECASE)
                    if strength_section:
                        strength_text = strength_section.group(1).strip()
                        strengths = [s.strip("- ").strip() for s in re.split(r"\n-|\n•|\n\*|\d+\.", strength_text) if s.strip()]
                    
                    # Extract areas for improvement
                    improvements = []
                    improve_section = re.search(r"(?:Areas? for )?Improvement|Weakness(?:es)?:(.*?)(?:Recommendation|$)", 
                                              evaluation_text, re.DOTALL | re.IGNORECASE)
                    if improve_section:
                        improve_text = improve_section.group(1).strip()
                        improvements = [i.strip("- ").strip() for i in re.split(r"\n-|\n•|\n\*|\d+\.", improve_text) if i.strip()]
                    
                    # Extract recommendation
                    recommendation = "Consider"  # Default
                    rec_match = re.search(r"(?:Final )?Recommendation:?\s*(Strongly Recommend|Recommend|Consider|Do Not Recommend)", 
                                         evaluation_text, re.IGNORECASE)
                    if rec_match:
                        recommendation = rec_match.group(1)
                    
                    # Format final evaluation
                    final_eval = {
                        "overall_score": overall_score,
                        "summary": summary,
                        "strengths": strengths[:5],  # Limit to top 5 strengths
                        "areas_for_improvement": improvements[:5],  # Limit to top 5 improvements
                        "recommendation": recommendation,
                        "full_evaluation": evaluation_text
                    }
                    
                    return final_eval
                
                # Default evaluation if generation fails
                return {
                    "overall_score": avg_score,
                    "summary": "The candidate completed the interview process.",
                    "strengths": ["Communication skills"],
                    "areas_for_improvement": ["Technical depth"],
                    "recommendation": "Consider",
                    "full_evaluation": ""
                }
                
            except Exception as e:
                logger.error(f"Error generating final evaluation: {e}")
                return {
                    "overall_score": 7.0,
                    "summary": "Interview completed.",
                    "strengths": ["Communication"],
                    "areas_for_improvement": ["Technical depth"],
                    "recommendation": "Consider",
                    "full_evaluation": ""
                } 