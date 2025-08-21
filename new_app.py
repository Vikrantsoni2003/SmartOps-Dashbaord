"""
Unified Streamlit App (Fixed Version)
Sections included:
- Python Tasks
- SSH Remote Executor
- Secure File Manager
- AI Chatbot (Gemini AI)
- AWS Tasks
- Streamlit Project (MoodMate Full)
- Linux Tasks
- Machine Learning
- Windows Tasks

Save as: app.py
Run: streamlit run app.py

NOTE: Replace credentials/API keys where noted. Optional packages are imported safely.
"""

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from streamlit_drawable_canvas import st_canvas

import os
import sys
import time
import random
import requests
import shutil
import subprocess
import datetime
import pickle
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
import zipfile
import json
import base64

# Optional libraries (import safely)
try:
    import paramiko
except ImportError:
    paramiko = None

try:
    import pywhatkit
except ImportError:
    pywhatkit = None

try:
    import pyautogui
except ImportError:
    pyautogui = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    TwilioClient = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    KNNImputer = None
    OneHotEncoder = None
    StandardScaler = None
    train_test_split = None
    accuracy_score = None

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, initializers, optimizers
    from tensorflow.keras.datasets import mnist
except ImportError:
    tf = None
    layers = None
    models = None
    initializers = None
    optimizers = None
    mnist = None

# Image processing imports
try:
    import face_recognition
    import mediapipe as mp
    from PIL import Image
except ImportError:
    face_recognition = None
    mp = None
    Image = None

# AWS imports
try:
    import boto3
    import botocore
    from botocore.exceptions import ClientError
    from pymongo import MongoClient
except ImportError:
    boto3 = None
    botocore = None
    ClientError = None
    MongoClient = None

# Optional: suppress insecure request warnings if urllib3 available
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

# ---- App config ----
st.set_page_config(
    page_title="SmartOps Dashboard",
    page_icon="🤖🌸",
     layout="wide")

# ---- Header with Personal Info ----
st.markdown("""
<style>
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .name {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .bio {
        font-size: 1.2rem;
        opacity: 0.9;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    .tech-stack {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }

    .tech-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        backdrop-filter: blur(10px);
    }
</style>

<div class="header-container">
    <div class="name">Vikrant Soni</div>
    <div class="bio">MCA student passionate about AI, Cloud Computing, and innovative tech solutions.</div>
    <div class="tech-stack">
        <span class="tech-badge">🤖 AI/ML</span>
        <span class="tech-badge">☁️ Cloud Computing</span>
        <span class="tech-badge">💻 Full Stack Development</span>
        <span class="tech-badge">🚀 Innovation</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Display profile image using Streamlit
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("WhatsApp Image 2025-07-09 at 13.38.38_1e26c16f.jpg",
             width=120, caption="Vikrant Soni")

# ---- Utility: Lottie loader ----


def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ---- MoodMate data (Streamlit Project) ----
mood_data = {
    "Happy": {
        "movies": ["The Pursuit of Happyness", "Forrest Gump", "Sing", "Paddington 2", "Mamma Mia!"],
        "songs": ["Happy - Pharrell", "Best Day Of My Life", "Walking on Sunshine", "Good Life"],
        "quotes": ["Happiness is a direction, not a place.", "Smile, it confuses people.", "Be happy!"],
        "remedy": "Drink lemonade or mint tea to stay refreshed!"
    },
    "Sad": {
        "movies": ["Inside Out", "The Fault in Our Stars", "A Silent Voice"],
        "songs": ["Fix You - Coldplay", "Let Her Go - Passenger", "Someone Like You - Adele"],
        "quotes": ["Tough times never last.", "It's okay to not be okay."],
        "remedy": "Try chamomile tea and deep breathing."
    },
    "Anxious": {
        "movies": ["A Beautiful Mind", "Soul", "Good Will Hunting"],
        "songs": ["Weightless - Marconi Union", "Breathe Me - Sia"],
        "quotes": ["Just breathe.", "This too shall pass."],
        "remedy": "Try lavender tea."
    },
    "Bored": {
        "movies": ["Jumanji", "Zombieland", "Night at the Museum"],
        "songs": ["On Top of the World", "Uptown Funk", "Shut Up and Dance"],
        "quotes": ["Do something today your future self will thank you for."],
        "remedy": "Drink green tea and doodle!"
    }
}

# ---- Gemini AI Chatbot Class ----


class GeminiChatbot:
    """Main class for the Gemini AI chatbot system"""

    def __init__(self):
        """Initialize the chatbot system"""
        self.setup_gemini()
        self.setup_roles()

    def setup_gemini(self):
        """Setup Gemini AI configuration"""
        if genai:
            try:
                # Configure Gemini API
                GEMINI_API_KEY = "Your_api_key"
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                self.is_configured = True
            except Exception as e:
                st.error(f"Error configuring Gemini: {str(e)}")
                self.is_configured = False
        else:
            self.is_configured = False

    def setup_roles(self):
        """Setup chatbot roles and categories"""
        self.subcategory_roles = {
            "Mental Health Bot": "You are a kind and empathetic mental health assistant. Provide supportive, non-judgmental advice while encouraging professional help when needed.",
            "Finance Advisor": "You are a professional finance advisor helping users with investment strategies, savings plans, and financial planning. Always recommend consulting with licensed financial advisors for specific advice.",
            "Legal Help": "You are a legal assistant providing general information about laws and legal procedures. Always clarify that you cannot provide legal advice and recommend consulting with qualified attorneys.",
            "Tech Support": "You are a technical support assistant helping users troubleshoot software and hardware issues. Provide step-by-step solutions and recommend professional help for complex problems.",
            "Blog Post Writer": "You are a creative blog writer. Write engaging, informative, and SEO-friendly blog posts with proper structure and compelling content.",
            "News Headline Generator": "You generate catchy, informative, and accurate news headlines that capture attention while maintaining journalistic standards.",
            "Resume Builder": "You help users write professional resumes with proper formatting, action verbs, and industry-specific keywords.",
            "Contract Template Generator": "You draft general contract templates for businesses and freelancers. Always include disclaimers about legal review.",
            "Quiz Generator": "You are a quiz maker. Generate educational and fun quizzes with multiple choice questions and detailed explanations.",
            "MCQ Solver": "You are an exam assistant who solves MCQs with step-by-step explanations and educational insights.",
            "Notes Summarizer": "You summarize detailed notes into short, clear, and organized summaries while maintaining key information.",
            "Math Solver": "You solve math problems step by step with clear explanations, showing all work and providing educational insights.",
            "Pitch Deck Assistant": "You write compelling content for pitch decks and investor presentations with persuasive language and clear value propositions.",
            "Email Personalizer": "You write personalized emails for marketing or communication with appropriate tone and professional language.",
            "Invoice Creator": "You write invoice templates in plain text with proper formatting and all necessary business details.",
            "Excel Formula Generator": "You generate Microsoft Excel formulas from descriptions with explanations of how they work.",
            "Code Explainer": "You explain code in simple terms, breaking down complex concepts and highlighting important patterns.",
            "Bug Fix Assistant": "You help debug and fix code errors with detailed explanations and best practices.",
            "Code Converter": "You convert code from one language to another, like Python to JavaScript, with clear comments and explanations.",
            "Regex Generator": "You generate regular expressions based on user input with explanations and examples."
        }

        self.categories = {
            "Chatbots": [
                "Mental Health Bot", "Finance Advisor", "Legal Help", "Tech Support"
            ],
            "Content Generators": [
                "Blog Post Writer", "News Headline Generator", "Resume Builder", "Contract Template Generator"
            ],
            "Educational Tools": [
                "Quiz Generator", "MCQ Solver", "Notes Summarizer", "Math Solver"
            ],
            "Business Tools": [
                "Pitch Deck Assistant", "Email Personalizer", "Invoice Creator", "Excel Formula Generator"
            ],
            "Code Tools": [
                "Code Explainer", "Bug Fix Assistant", "Code Converter", "Regex Generator"
            ]
        }

    def generate_response(self, subcategory, prompt):
        """Generate response from Gemini AI"""
        if not self.is_configured:
            return "❌ Gemini AI is not configured. Please check your API key and internet connection."

        try:
            system_prompt = self.subcategory_roles.get(
                subcategory, "You are a helpful assistant.")
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

            response = self.model.generate_content(full_prompt)
            return response.text
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

    def get_categories(self):
        """Get available categories"""
        return list(self.categories.keys())

    def get_subcategories(self, category):
        """Get subcategories for a given category"""
        return self.categories.get(category, [])

# ---- AWS Utility functions ----


def boto3_client(service, region=None):
    if not boto3:
        return None

    # Get credentials from session state if available
    if "aws_credentials" in st.session_state and st.session_state.aws_credentials[
        "configured"]:
        creds = st.session_state.aws_credentials
        return boto3.client(
            service,
            aws_access_key_id=creds["access_key_id"],
            aws_secret_access_key=creds["secret_access_key"],
            region_name=region or creds["region"]
        )
    else:
        # Fallback to default boto3 behavior (environment variables, IAM roles,
        # etc.)
        return boto3.client(service, region_name=region)


def boto3_resource(service, region=None):
    if not boto3:
        return None

    # Get credentials from session state if available
    if "aws_credentials" in st.session_state and st.session_state.aws_credentials[
        "configured"]:
        creds = st.session_state.aws_credentials
        return boto3.resource(
            service,
            aws_access_key_id=creds["access_key_id"],
            aws_secret_access_key=creds["secret_access_key"],
            region_name=region or creds["region"]
        )
    else:
        # Fallback to default boto3 behavior (environment variables, IAM roles,
        # etc.)
        return boto3.resource(service, region_name=region)


# ---- Sidebar main menu ----
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    .sidebar-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        font-size: 1.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .section-item {
        background: rgba(255,255,255,0.1);
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        transition: all 0.3s ease;
    }

    .section-item:hover {
        background: rgba(255,255,255,0.2);
        transform: translateX(5px);
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-title">
    🚀 SmartOps Dashboard
</div>
""", unsafe_allow_html=True)

main_page = st.sidebar.selectbox("Choose Section", [
    "Python Tasks",
    "SSH Remote Executor", 
    "Secure File Manager",
    "AI Chatbot",
    "AWS Tasks",
    "Streamlit Project", 
    "Linux Tasks",
    # "Machine Learning", 
    "Windows Tasks",
])

# ------------------ PYTHON TASKS ------------------
if main_page == "Python Tasks":
    st.markdown("""
    <style>
        .section-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }

        .section-title {
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .section-subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
        }
    </style>

    <div class="section-header">
        <div class="section-title">🐍 Python Tasks</div>
        <div class="section-subtitle">Automation, AI, and Advanced Python Tools</div>
    </div>
    """, unsafe_allow_html=True)
    task = st.selectbox("Choose Task", [
        "WhatsApp Automation (pywhatkit)",
        "Send Email (pywhatkit.send_mail)",
        "Twilio Call",
        "System RAM Info",
        "Random Art (OpenCV)",
        "Swap Faces (OpenCV)",
        "Web Scraper -> PDF",
        "Instagram Photo Upload (instagrapi)",
        "Send SMS (Twilio)",
        "Anonymous WhatsApp via Twilio"
    ])
    st.markdown("---")

    # WhatsApp Automation
    if task == "WhatsApp Automation (pywhatkit)":
        st.subheader("Send WhatsApp Message (Direct)")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Method 1: Direct Send (pyautogui)")
            phone = st.text_input(
    "Recipient Phone Number (e.g. +911234567890)", "+91")
            message = st.text_area("Message", "Hello from Streamlit!")

            if st.button("📱 Send WhatsApp Message", type="primary"):
                if not pyautogui:
                    st.error(
                        "pyautogui not installed. Install via: pip install pyautogui")
                elif not phone or not message:
                    st.warning("Please enter both phone number and message")
                else:
                    try:
                        with st.spinner("Opening WhatsApp Web..."):
                            # Open WhatsApp Web
                            import webbrowser
                            webbrowser.open(
    f"https://web.whatsapp.com/send?phone={phone}&text={message}")

                            # Wait for page to load
                            time.sleep(5)

                            # Send message using pyautogui
                            pyautogui.press('enter')

                        st.success("✅ Message sent successfully!")
                        st.info(
                            "Note: Make sure WhatsApp Web is logged in and the chat is open")

                    except Exception as e:
                        st.error(f"❌ Error sending message: {e}")
                        st.info(
                            "Make sure WhatsApp Web is accessible and you're logged in")

        with col2:
            st.write("### Method 2: Scheduled Send (pywhatkit)")
            scheduled_phone = st.text_input(
    "Recipient Phone Number", "+91", key="scheduled_phone")
            scheduled_message = st.text_input(
    "Message", "Hello from Streamlit!", key="scheduled_message")
            hour = st.number_input(
    "Hour (24h)", 0, 23, 12, key="scheduled_hour")
            minute = st.number_input(
    "Minute", 0, 59, 0, key="scheduled_minute")

            if st.button("⏰ Schedule WhatsApp Message"):
                if not pywhatkit:
                    st.error(
                        "pywhatkit not installed. Install via: pip install pywhatkit")
                else:
                    try:
                        pywhatkit.sendwhatmsg(
                            scheduled_phone, scheduled_message, int(hour), int(minute))
                        st.success(
                            "✅ Message scheduled — a browser window will open at the scheduled time.")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        st.markdown("---")
        st.info("""
        **Instructions:**
        - **Method 1**: Opens WhatsApp Web directly and sends message
        - **Method 2**: Schedules message for later delivery
        - Make sure WhatsApp Web is logged in
        - Phone number should include country code (e.g., +91 for India)
        """)

    # Email Sender
    elif task == "Send Email (pywhatkit.send_mail)":
        st.subheader("Send Email (pywhatkit)")
        sender = st.text_input("Sender Email (Gmail recommended)")
        password = st.text_input(
    "App Password (Gmail app password)",
     type="password")
        receiver = st.text_input("Receiver Email")
        subject = st.text_input("Subject")
        body = st.text_area("Email Body")
        if st.button("Send Email"):
            if not pywhatkit:
                st.error("pywhatkit not installed.")
            else:
                try:
                    pywhatkit.send_mail(
    sender, password, subject, body, receiver)
                    st.success("Email sent successfully.")
                except Exception as e:
                    st.error(f"Failed: {e}")

    # Twilio Call
    elif task == "Twilio Call":
        st.subheader("Place a Call via Twilio")
        sid = st.text_input("Twilio Account SID")
        token = st.text_input("Twilio Auth Token", type="password")
        from_num = st.text_input("From (Twilio number, e.g. +1xxx)")
        to_num = st.text_input("To (recipient number)")
        if st.button("Call"):
            if not TwilioClient:
                st.error(
                    "twilio package not installed. Install via: pip install twilio")
            else:
                try:
                    client = TwilioClient(sid, token)
                    call = client.calls.create(
                        url="http://demo.twilio.com/docs/classic.mp3",
                        from_=from_num,
                        to=to_num
                    )
                    st.success(f"Call initiated. SID: {call.sid}")
                except Exception as e:
                    st.error(f"Call failed: {e}")

    # System RAM Info
    elif task == "System RAM Info":
        st.subheader("System Memory Information")
        if not psutil:
            st.error("psutil not installed. Install via: pip install psutil")
        else:
            mem = psutil.virtual_memory()
            st.metric("Total RAM (GB)", f"{mem.total / (1024**3):.2f}")
            st.metric("Available RAM (GB)", f"{mem.available / (1024**3):.2f}")
            st.metric("Used %", f"{mem.percent}%")

    # Random Art
    elif task == "Random Art (OpenCV)":
        st.subheader("Random Circle Art (OpenCV)")
        if np is None or cv2 is None:
            st.error(
                "numpy or opencv not installed. Install via: pip install numpy opencv-python")
        else:
            if st.button("Generate Art"):
                img = np.zeros((500, 500, 3), dtype=np.uint8)
                for _ in range(100):
                    center = np.random.randint(0, 500, 2)
                    radius = np.random.randint(5, 60)
                    color = np.random.randint(0, 255, 3).tolist()
                    cv2.circle(img, tuple(center), radius, color, -1)
                # convert BGR -> RGB for streamlit
                st.image(img[:, :, ::-1], caption="Random Circles",
                         use_column_width=True)

    # Swap Faces
    elif task == "Swap Faces (OpenCV)":
        st.subheader("🔄 Face Swap Tool")

        # Check for required packages
        if np is None or cv2 is None:
            st.error(
                "Required packages not installed. Install via: pip install numpy opencv-python")
            st.stop()

        # Custom CSS for face swap UI
        st.markdown("""
        <style>
            .face-swap-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 15px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;
            }

            .image-preview {
                border: 3px solid #667eea;
                border-radius: 10px;
                padding: 10px;
                background: white;
                margin: 1rem 0;
            }

            .result-container {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                padding: 1.5rem;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin: 1rem 0;
            }
        </style>
        """, unsafe_allow_html=True)

        # Header
        st.markdown("""
        <div class="face-swap-container">
            <h2>🔄 AI Face Swap Tool</h2>
            <p>Upload two images and swap faces using advanced computer vision techniques</p>
        </div>
        """, unsafe_allow_html=True)

        # Image upload section
        col1, col2 = st.columns(2)

        with col1:
            st.write("### 📸 Source Image (Face to be swapped)")
            source_image = st.file_uploader(
    "Upload source image", type=[
        'jpg', 'jpeg', 'png'], key="source")

            if source_image:
                source_img = Image.open(source_image)
                st.image(
    source_img,
    caption="Source Image",
     use_column_width=True)

        with col2:
            st.write("### 🎯 Target Image (Face to be replaced)")
            target_image = st.file_uploader(
    "Upload target image", type=[
        'jpg', 'jpeg', 'png'], key="target")

            if target_image:
                target_img = Image.open(target_image)
                st.image(
    target_img,
    caption="Target Image",
     use_column_width=True)

        # Face swap button
        if st.button("🔄 Swap Faces", type="primary", use_container_width=True):
            if source_image and target_image:
                try:
                    with st.spinner("🔄 Processing face swap..."):
                        # Convert PIL images to numpy arrays
                        source_array = np.array(source_img)
                        target_array = np.array(target_img)

                        # Convert RGB to BGR for OpenCV
                        source_bgr = cv2.cvtColor(
                            source_array, cv2.COLOR_RGB2BGR)
                        target_bgr = cv2.cvtColor(
                            target_array, cv2.COLOR_RGB2BGR)

                        # Load face detection model
                        face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

                        # Detect faces in both images
                        source_faces = face_cascade.detectMultiScale(
                            source_bgr, 1.1, 4)
                        target_faces = face_cascade.detectMultiScale(
                            target_bgr, 1.1, 4)

                        if len(source_faces) == 0:
                            st.error("❌ No face detected in source image!")
                            st.stop()

                        if len(target_faces) == 0:
                            st.error("❌ No face detected in target image!")
                            st.stop()

                        # Get the largest face from each image
                        source_face = max(
    source_faces, key=lambda x: x[2] * x[3])
                        target_face = max(
    target_faces, key=lambda x: x[2] * x[3])

                        # Extract face regions
                        (sx, sy, sw, sh) = source_face
                        (tx, ty, tw, th) = target_face

                        source_face_region = source_bgr[sy:sy + sh, sx:sx + sw]
                        target_face_region = target_bgr[ty:ty + th, tx:tx + tw]

                        # Resize source face to match target face size
                        source_face_resized = cv2.resize(
                            source_face_region, (tw, th))

                        # Create a mask for seamless cloning
                        mask = np.zeros((th, tw), dtype=np.uint8)
                        cv2.ellipse(
    mask, (tw // 2, th // 2), (tw // 2 - 10, th // 2 - 10), 0, 0, 360, 255, -1)

                        # Perform seamless cloning
                        center = (tx + tw // 2, ty + th // 2)
                        result = cv2.seamlessClone(
    source_face_resized, target_bgr, mask, center, cv2.NORMAL_CLONE)

                        # Convert back to RGB for display
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

                        # Display result
                        st.markdown("""
                        <div class="result-container">
                            <h3>✅ Face Swap Complete!</h3>
                            <p>Your face-swapped image is ready</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.image(
    result_rgb,
    caption="Face Swapped Result",
     use_column_width=True)

                        # Download button
                        result_pil = Image.fromarray(result_rgb)
                        img_byte_arr = BytesIO()
                        result_pil.save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()

                        st.download_button(
                            label="📥 Download Face Swapped Image",
                            data=img_byte_arr,
                            file_name="face_swapped_result.png",
                            mime="image/png",
                            use_container_width=True
                        )

                        st.success("🎉 Face swap completed successfully!")

                except Exception as e:
                    st.error(f"❌ Error during face swap: {str(e)}")
                    st.info(
                        "Make sure both images contain clear, front-facing faces")
            else:
                st.warning("⚠️ Please upload both source and target images")

        # Instructions
        st.markdown("---")
        st.info("""
        **📝 Instructions:**
        1. **Source Image**: Upload the image containing the face you want to swap
        2. **Target Image**: Upload the image where you want to place the face
        3. **Click "Swap Faces"**: The AI will detect faces and perform the swap
        4. **Download Result**: Save your face-swapped image

        **💡 Tips for best results:**
        - Use clear, front-facing photos
        - Ensure good lighting in both images
        - Faces should be roughly the same size
        - Avoid extreme angles or expressions
        """)

    # Web Scraper -> PDF
    elif task == "Web Scraper -> PDF":
        st.subheader("Scrape a webpage and save as PDF")
        url = st.text_input("Website URL (include http:// or https://)")
        if st.button("Scrape and Generate PDF"):
            if not BeautifulSoup or not FPDF:
                st.error(
                    "bs4 or fpdf not installed. Install via: pip install beautifulsoup4 fpdf2")
            else:
                if not url:
                    st.warning("Please enter a URL.")
                else:
                    try:
                        r = requests.get(url, timeout=10)
                        soup = BeautifulSoup(r.content, "html.parser")
                        title = soup.title.string.strip() if soup.title else "No Title"
                        text = soup.get_text(separator="\n", strip=True)
                        links = [a.get("href")
                                       for a in soup.find_all("a", href=True)]
                        links_text = "\n".join(links[:50])  # Limit links

                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        
                        # Handle encoding issues
                        safe_text = text.encode(
                            'latin-1', 'replace').decode('latin-1')[:8000]
                        safe_links = links_text.encode(
                            'latin-1', 'replace').decode('latin-1')[:3000]

                        pdf.multi_cell(0, 8, f"Website: {url}\n\nTitle: {
                                       title}\n\n--- Page Text ---\n{safe_text}\n\n--- Links ---\n{safe_links}")
                        path = "scraped_site.pdf"
                        pdf.output(path)
                        
                        with open(path, "rb") as f:
                            st.download_button(
    "Download PDF",
    f.read(),
    file_name="scraped_site.pdf",
     mime="application/pdf")
                        st.success("PDF generated successfully.")
                    except Exception as e:
                        st.error(f"Failed: {e}")

    # Instagram Photo Upload
    elif task == "Instagram Photo Upload (instagrapi)":
        st.subheader("Upload Photo to Instagram (instagrapi)")
        st.warning(
            "Note: Instagram automation may violate their terms of service. Use at your own risk.")
        ig_user = st.text_input("Instagram Username")
        ig_pass = st.text_input("Instagram Password", type="password")
        caption = st.text_input("Caption", "Hello from Streamlit!")
        uploaded = st.file_uploader(
    "Choose an image", type=[
        "jpg", "jpeg", "png"])
        if st.button("Upload"):
            if uploaded is None:
                st.warning("Upload an image first.")
            else:
                try:
                    from instagrapi import Client as IGClient
                    path = f"temp_{uploaded.name}"
                    with open(path, "wb") as f:
                        f.write(uploaded.getbuffer())
                    cl = IGClient()
                    cl.login(ig_user, ig_pass)
                    cl.photo_upload(path, caption)
                    os.remove(path)
                    st.success("Uploaded to Instagram successfully.")
                except ImportError:
                    st.error(
                        "instagrapi not installed. Install via: pip install instagrapi")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    # Send SMS (Twilio)
    elif task == "Send SMS (Twilio)":
        st.subheader("Send SMS via Twilio")
        sms_sid = st.text_input("Twilio SID")
        sms_token = st.text_input("Twilio Token", type="password")
        from_sms = st.text_input("From (Twilio number)")
        to_sms = st.text_input("To (recipient)")
        sms_body = st.text_area("Message", "Hello from Streamlit!")
        if st.button("Send SMS"):
            if not TwilioClient:
                st.error("twilio package not installed.")
            else:
                try:
                    client = TwilioClient(sms_sid, sms_token)
                    msg = client.messages.create(
    body=sms_body, from_=from_sms, to=to_sms)
                    st.success(f"SMS sent successfully. SID: {msg.sid}")
                except Exception as e:
                    st.error(f"Failed: {e}")

    # Anonymous WhatsApp via Twilio
    elif task == "Anonymous WhatsApp via Twilio":
        st.subheader("Send WhatsApp via Twilio (sandbox or approved sender)")
        acc_sid = st.text_input("Twilio SID")
        auth_token = st.text_input("Twilio Token", type="password")
        from_wh = st.text_input(
    "From (e.g. whatsapp:+14155238886)",
     value="whatsapp:+14155238886")
        to_wh = st.text_input(
    "To (e.g. whatsapp:+91XXXXXXXXXX)",
     value="whatsapp:+91")
        msg = st.text_area("Message")
        if st.button("Send WhatsApp"):
            if not TwilioClient:
                st.error("twilio package not installed.")
            else:
                try:
                    client = TwilioClient(acc_sid, auth_token)
                    message = client.messages.create(
                        from_=from_wh, body=msg, to=to_wh)
                    st.success(
    f"WhatsApp message sent successfully. SID: {
        message.sid}")
                except Exception as e:
                    st.error(f"Failed: {e}")

# ------------------ SSH Remote Executor ------------------
elif main_page == "SSH Remote Executor":
    st.title("🔐 SSH Remote Executor")
    if paramiko is None:
        st.error("paramiko not installed. Install via: pip install paramiko")
    else:
        ssh_host = st.text_input("SSH Host (IP or hostname)")
        ssh_user = st.text_input("SSH Username")
        ssh_pass = st.text_input("SSH Password", type="password")
        ssh_cmd = st.text_area("Command(s) to run", "ls -la")
        if st.button("Run over SSH"):
            if not all([ssh_host, ssh_user, ssh_pass, ssh_cmd]):
                st.warning("Please fill in all fields.")
            else:
                try:
                    ssh = paramiko.SSHClient()
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh.connect(
    ssh_host,
    username=ssh_user,
    password=ssh_pass,
     timeout=10)
                    stdin, stdout, stderr = ssh.exec_command(ssh_cmd)
                    out = stdout.read().decode()
                    err = stderr.read().decode()
                    if out:
                        st.code(out, language="bash")
                    if err:
                        st.error(f"Error output:\n{err}")
                    ssh.close()
                    st.success("Command executed successfully.")
                except Exception as e:
                    st.error(f"SSH Error: {e}")

# ------------------ Secure File Manager ------------------
elif main_page == "Secure File Manager":
    st.title("📂 Secure File Manager")
    directory = st.text_input("Enter Directory Path", os.getcwd())
    
    if directory and os.path.exists(directory):
        try:
            files = sorted(os.listdir(directory))
            search = st.text_input("Search files/folders")
            filtered = [f for f in files if search.lower() in f.lower()
                                                         ] if search else files
            file_types = []

            with st.expander("📁 Files & Folders", expanded=True):
                if not filtered:
                    st.info("No matching files found.")
                else:
                    for f in filtered:
                        full_path = os.path.join(directory, f)
                        try:
                            size_kb = os.path.getsize(
                                full_path) / 1024 if os.path.isfile(full_path) else 0
                            file_type = "📁 Folder" if os.path.isdir(
                                full_path) else f"📄 File ({Path(f).suffix or 'no ext'})"
                            st.write(f"{f} — {file_type} — {size_kb:.2f} KB")
                            
                            if os.path.isfile(full_path):
                                try:
                                    with open(full_path, "rb") as file:
                                        st.download_button(
                                            f"⬇ Download {f}", 
                                            data=file.read(), 
                                            file_name=f,
                                            key=f"download_{f}"
                                        )
                                    file_types.append(
                                        Path(f).suffix or 'no_ext')
                                except Exception as e:
                                    st.error(f"Cannot read {f}: {e}")
                        except Exception as e:
                            st.error(f"Error accessing {f}: {e}")

            # File Type Distribution
            st.markdown("### 📊 File Type Distribution")
            if file_types:
                type_counts = Counter(file_types)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(
    list(
        type_counts.values()), labels=list(
            type_counts.keys()), autopct="%1.1f%%")
                ax.set_title("File Type Distribution")
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No files to show distribution.")

            # File Upload
            uploaded = st.file_uploader("📤 Upload File")
            if uploaded:
                save_path = os.path.join(directory, uploaded.name)
                try:
                    with open(save_path, "wb") as f:
                        f.write(uploaded.getbuffer())
                    st.success(f"Uploaded '{uploaded.name}' successfully.")
                    
                    # Preview uploaded file
                    if uploaded.type and uploaded.type.startswith("image/"):
                        st.image(save_path, caption=uploaded.name)
                    elif uploaded.type and uploaded.type.startswith("text/"):
                        try:
                            with open(save_path, 'r', encoding='utf-8') as t:
                                content = t.read()
                                st.text_area(
                                    "Preview", content[:1000] + ("..." if len(content) > 1000 else ""), height=150)
                        except Exception as e:
                            st.warning(f"Cannot preview text file: {e}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

            # Dangerous operations
            if st.checkbox("⚠ Enable Deletion and Rename (Use with caution)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🗑 Delete File/Folder")
                    del_name = st.text_input("Name to Delete")
                    if st.button("🗑 Delete", type="secondary"):
                        if del_name:
                            try:
                                target = os.path.join(directory, del_name)
                                if os.path.isfile(target):
                                    os.remove(target)
                                    st.success(f"File '{del_name}' deleted.")
                                elif os.path.isdir(target):
                                    shutil.rmtree(target)
                                    st.success(
    f"Directory '{del_name}' deleted.")
                                else:
                                    st.warning("Target not found.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Deletion error: {e}")
                        else:
                            st.warning("Please enter a name to delete.")
                
                with col2:
                    st.subheader("✏ Rename File/Folder")
                    old = st.text_input("Old Name")
                    new = st.text_input("New Name")
                    if st.button("✏ Rename", type="secondary"):
                        if old and new:
                            try:
                                old_path = os.path.join(directory, old)
                                new_path = os.path.join(directory, new)
                                os.rename(old_path, new_path)
                                st.success(f"Renamed '{old}' to '{new}'.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Rename error: {e}")
                        else:
                            st.warning("Please enter both old and new names.")

            # Create new folder
            st.subheader("📦 Create New Folder")
            new_folder = st.text_input("New Folder Name")
            if st.button("📦 Create Folder"):
                if new_folder:
                    try:
                        folder_path = os.path.join(directory, new_folder)
                        os.makedirs(folder_path, exist_ok=True)
                        st.success(f"Folder '{new_folder}' created.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Folder creation error: {e}")
                else:
                    st.warning("Please enter a folder name.")
                    
        except PermissionError:
            st.error("Permission denied. Cannot access this directory.")
        except Exception as e:
            st.error(f"Error accessing directory: {e}")

# ------------------ AI Chatbot ------------------
elif main_page == "AI Chatbot":
    st.title("🤖 AI Chatbot Hub (Powered by Gemini)")
    
    # Check for required packages
    if not genai:
        st.error(
            "Google Generative AI not installed. Install via: pip install google-generativeai")
        st.stop()

    # Initialize chatbot in session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = GeminiChatbot()

    chatbot = st.session_state.chatbot

    # Custom CSS for modern UI
    st.markdown("""
    <style>
        .chatbot-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }

        .role-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        
        .response-box {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="chatbot-header">
        <h1>🤖 AI Tool Hub</h1>
        <p>Powered by Google Gemini AI - Your intelligent assistant for various tasks</p>
    </div>
    """, unsafe_allow_html=True)

    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 Choose Your Tool")

        # Category selection
        category = st.selectbox("Select Category", chatbot.get_categories())

        # Subcategory selection based on category
        subcategories = chatbot.get_subcategories(category)
        subcategory = st.selectbox("Select Tool", subcategories)

        # Display role description
        if subcategory:
            role_description = chatbot.subcategory_roles.get(subcategory, "")
            st.markdown(f"""
            <div class="role-card">
                <h4>🎭 {subcategory}</h4>
                <p>{role_description}</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.subheader("💬 Chat Interface")

        # Chat input
        user_prompt = st.text_area("Enter your prompt or question",
                                  placeholder="Describe what you need help with...",
                                  height=150)

        # Generate response button
        if st.button(
    "🚀 Generate Response",
    type="primary",
     use_container_width=True):
            if user_prompt.strip():
                with st.spinner("🤖 Generating response..."):
                    response = chatbot.generate_response(
                        subcategory, user_prompt)

                st.markdown("""
                <div class="response-box">
                    <h4>🤖 AI Response:</h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(response)

                # Add copy button
                st.button("📋 Copy Response", on_click=lambda: st.write(
                    "Response copied to clipboard!"))
                st.warning("Please enter a prompt or question")

    # Quick examples section
    st.markdown("---")
    st.subheader("💡 Quick Examples")

    examples = {
        "Mental Health Bot": "I'm feeling anxious about my upcoming presentation. Can you help me with some coping strategies?",
        "Finance Advisor": "I want to start investing $500 monthly. What are some good options for a beginner?",
        "Tech Support": "My computer is running very slow. What steps should I take to troubleshoot this?",
        "Blog Post Writer": "Write a blog post about the benefits of remote work for small businesses",
        "Code Explainer": "Explain this Python code: def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "Math Solver": "Solve: 2x + 5 = 13. Show all steps.",
        "Resume Builder": "Help me write a professional summary for a software developer position",
        "Quiz Generator": "Create a 5-question quiz about Python programming basics"
    }

    # Display examples in columns
    cols = st.columns(2)
    for i, (tool, example) in enumerate(examples.items()):
        with cols[i % 2]:
            with st.expander(f"📝 {tool}"):
                st.write(f"**Example:** {example}")
                if st.button(f"Use Example", key=f"example_{i}"):
                    st.session_state.example_prompt = example
                    st.rerun()

    # Features and capabilities
    st.markdown("---")
    st.subheader("✨ Features & Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🎭 **20+ Specialized Roles**
        - Mental Health Support
        - Financial Advisory
        - Legal Information
        - Technical Support
        - Content Creation
        """)

    with col2:
        st.markdown("""
        ### 🧠 **Advanced AI**
        - Powered by Google Gemini
        - Context-aware responses
        - Professional tone
        - Educational insights
        """)

    with col3:
        st.markdown("""
        ### 🚀 **Easy to Use**
        - Simple interface
        - Quick responses
        - Copy functionality
        - Example prompts
        """)

    # Usage tips
    st.markdown("---")
    st.subheader("💡 Usage Tips")

    st.info("""
    **For best results:**
    - Be specific in your prompts
    - Choose the most relevant tool for your task
    - Use the examples as starting points
    - The AI will provide professional, helpful responses
    - All responses are generated in real-time using Gemini AI
    """)

# ------------------ AWS Tasks ------------------
elif main_page == "AWS Tasks":
    st.title("☁️ AWS Tasks & Management")
    
    if not boto3:
        st.error("boto3 not installed. Install via: pip install boto3 pymongo")
        st.stop()

    # Initialize AWS credentials in session state
    if "aws_credentials" not in st.session_state:
        st.session_state.aws_credentials = {
            "access_key_id": "",
            "secret_access_key": "",
            "region": "us-east-1",
            "configured": False
        }

    aws_task = st.selectbox("Choose AWS Task", [
        "🔐 AWS Credentials Configuration",
        "📖 AWS Blogs & Case Studies",
        "🖥️ EC2 Management", 
        "📊 CloudWatch Logs",
        "💾 S3 Storage Classes",
        "🎤 Amazon Transcribe",
        "🗃️ MongoDB Connection",
        "🔗 S3 Presigned URLs"
    ])

    # AWS Credentials Configuration
    if aws_task == "🔐 AWS Credentials Configuration":
        st.subheader("🔐 AWS Credentials Configuration")
        
        st.info("""
        **Configure your AWS credentials to use all AWS services.**

        You can either:
        1. **Enter credentials manually** (for testing)
        2. **Use environment variables** (recommended for production)
        3. **Use AWS CLI configuration** (if already configured)
        """)

        # Check if credentials are already configured
        if st.session_state.aws_credentials["configured"]:
            st.success("✅ AWS credentials are configured!")
            st.info(
                f"**Region:** {st.session_state.aws_credentials['region']}")

            if st.button("🔄 Reconfigure Credentials"):
                st.session_state.aws_credentials["configured"] = False
                st.rerun()

        else:
            # Manual credentials input
            st.write("### Manual Credentials Setup")

            col1, col2 = st.columns(2)
            with col1:
                access_key = st.text_input("AWS Access Key ID",
                                         value=st.session_state.aws_credentials["access_key_id"],
                                         type="password",
                                         help="Your AWS Access Key ID")
                region = st.selectbox("AWS Region",
                                    ["us-east-1", "us-west-1", "us-west-2",
                                        "eu-west-1", "ap-south-1", "ap-southeast-1"],
                                    index=0 if st.session_state.aws_credentials["region"] == "us-east-1" else 1)

            with col2:
                secret_key = st.text_input("AWS Secret Access Key",
                                         value=st.session_state.aws_credentials["secret_access_key"],
                                         type="password",
                                         help="Your AWS Secret Access Key")

                # Test credentials button
                if st.button("🧪 Test Credentials", type="primary"):
                    if access_key and secret_key:
                        try:
                            # Test the credentials by creating a client
                            test_client = boto3.client('sts',
                                                      aws_access_key_id=access_key,
                                                      aws_secret_access_key=secret_key,
                                                      region_name=region)

                            # Get caller identity to verify credentials
                            identity = test_client.get_caller_identity()

                            st.success("✅ Credentials are valid!")
                            st.info(f"**Account ID:** {identity['Account']}")
                            st.info(f"**User ARN:** {identity['Arn']}")

                            # Save credentials to session state
                            st.session_state.aws_credentials = {
                                "access_key_id": access_key,
                                "secret_access_key": secret_key,
                                "region": region,
                                "configured": True
                            }

                            st.balloons()

                        except Exception as e:
                            st.error(f"❌ Invalid credentials: {str(e)}")
                    else:
                        st.warning(
                            "Please enter both Access Key ID and Secret Access Key")

            # Environment variables info
            st.write("### Environment Variables Setup")
            st.info("""
            **For production use, set these environment variables:**

            ```bash
            export AWS_ACCESS_KEY_ID="your_access_key"
            export AWS_SECRET_ACCESS_KEY="your_secret_key"
            export AWS_DEFAULT_REGION="us-east-1"
            ```

            **Or create ~/.aws/credentials file:**
            ```ini
            [default]
            aws_access_key_id = your_access_key
            aws_secret_access_key = your_secret_key
            region = us-east-1
            ```
            """)

            # Check if environment variables are set
            import os
            env_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
            env_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
            env_region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')

            if env_access_key and env_secret_key:
                st.success("✅ Environment variables are configured!")
                st.info(f"**Region:** {env_region}")

                if st.button("✅ Use Environment Variables"):
                    st.session_state.aws_credentials = {
                        "access_key_id": env_access_key,
                        "secret_access_key": env_secret_key,
                        "region": env_region,
                        "configured": True
                    }
                    st.rerun()
            else:
                st.warning("⚠️ Environment variables not found")

        # Security notice
        st.warning("""
        **Security Notice:**
        - Never commit credentials to version control
        - Use IAM roles when possible
        - Rotate credentials regularly
        - Use least privilege principle
        """)

    # Check if credentials are configured for other AWS tasks
    elif not st.session_state.aws_credentials["configured"]:
        st.error("❌ AWS credentials not configured!")
        st.info(
            "Please go to '🔐 AWS Credentials Configuration' first to set up your credentials.")
        st.stop()

    if aws_task == "📖 AWS Blogs & Case Studies":
        st.subheader("📖 AWS Blogs & Case Studies")

        # Blog 1: S3 Storage Classes
        st.markdown("---")
        st.markdown(
            "## 🗂️ **Unlocking Cost Optimization: A Deep Dive into Amazon S3 Storage Classes**")

        st.markdown("""
        In today's data-driven world, organizations are generating and storing massive amounts of data.
        Amazon S3, the industry-leading object storage service, offers multiple storage classes designed
        to optimize costs while maintaining the performance and durability your applications require.

        ### **Understanding S3 Storage Classes**

        **S3 Standard** serves as the default storage class for frequently accessed data. It provides
        millisecond access times and is ideal for active workloads, content distribution, and real-time
        applications. With 99.999999999% (11 9's) durability and 99.99% availability, it's perfect for
        mission-critical data that requires immediate access.

        **S3 Intelligent-Tiering** automatically moves objects between four access tiers based on
        changing access patterns. This hands-off approach eliminates the need for manual lifecycle
        management while optimizing costs. It's particularly beneficial for data with unpredictable
        access patterns, such as user-generated content or analytics datasets.

        **S3 Standard-IA (Infrequent Access)** is designed for data that's accessed less frequently
        but requires rapid retrieval when needed. With retrieval fees but lower storage costs than
        Standard, it's ideal for disaster recovery, long-term backups, and compliance data that must
        be readily available.

        **S3 Glacier** offers the most cost-effective storage for long-term archival data. With
        retrieval times ranging from minutes to hours, it's perfect for regulatory compliance,
        historical data analysis, and backup archives. The three retrieval options—Expedited,
        Standard, and Bulk—allow you to balance cost and speed based on your needs.

        ### **Real-World Applications**

        Consider a media streaming platform: frequently watched content stays in S3 Standard,
        while older shows move to Standard-IA. Rarely accessed documentaries can be archived in
        Glacier, dramatically reducing storage costs while maintaining accessibility.

        ### **Cost Optimization Strategy**

        Start with S3 Standard for new data, implement Intelligent-Tiering for unknown access
        patterns, and use lifecycle policies to automatically transition data to more cost-effective
        tiers as it ages. Monitor access patterns and adjust strategies based on actual usage data.

        **📌 Read More on LinkedIn:** [Amazon S3 Storage Classes Explained](https://www.linkedin.com/posts/vikrant-soni-ab861b330_amazon-s3-storage-classes-explained-optimize-activity-7355520201146208258-9wjf?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFNiqWwBMfKFniB-FZVLuk4Q8LU-9qCpwiI)
        """)

        # Blog 2: AWS Q Service
        st.markdown("---")
        st.markdown(
            "## 🤖 **Revolutionizing Enterprise AI: Amazon Q and the Future of Generative AI**")

        st.markdown("""
        The landscape of enterprise artificial intelligence is undergoing a revolutionary transformation,
        and Amazon Q stands at the forefront of this evolution. As organizations increasingly seek
        intelligent solutions to enhance productivity and decision-making, Amazon Q emerges as a
        game-changing generative AI assistant designed specifically for business environments.

        ### **What is Amazon Q?**

        Amazon Q is AWS's enterprise-focused generative AI assistant that leverages your organization's
        data, systems, and expertise to provide intelligent, contextual assistance. Unlike generic
        AI chatbots, Amazon Q understands your business context, security policies, and operational
        procedures, making it a truly integrated part of your workflow.

        ### **Key Capabilities and Benefits**

        **Intelligent Code Assistance:** Amazon Q excels at helping developers write, debug, and
        optimize code. It can analyze your existing codebase, suggest improvements, and even generate
        new functions based on your requirements. This capability significantly accelerates development
        cycles and reduces time-to-market for new features.

        **Business Intelligence and Analytics:** The assistant can interpret complex data sets,
        generate insights, and create visualizations that help stakeholders make informed decisions.
        It can answer questions about your business metrics, identify trends, and provide actionable
        recommendations.

        **Security and Compliance:** Amazon Q operates within your organization's security boundaries,
        ensuring that sensitive information remains protected. It can help with security assessments,
        compliance reporting, and best practice recommendations while maintaining data privacy.

        ### **Real-World Impact**

        Consider a financial services company implementing Amazon Q: developers can get instant
        assistance with AWS service integration, compliance officers can quickly generate regulatory
        reports, and business analysts can extract insights from complex financial data without
        extensive technical knowledge.

        ### **Integration and Deployment**

        Amazon Q integrates seamlessly with existing AWS services and can be customized to work
        with your specific tools and workflows. The service supports multiple deployment models,
        allowing organizations to choose the approach that best fits their security and operational
        requirements.

        As generative AI continues to evolve, Amazon Q represents a significant step toward
        democratizing AI capabilities within enterprises, making advanced AI assistance accessible
        to professionals across all departments and skill levels.

        **📌 Read More on LinkedIn:** [AWS Amazon Q Generative AI](https://www.linkedin.com/posts/vikrant-soni-ab861b330_aws-amazonq-generativeai-activity-7360310282084126720-lQF-?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFNiqWwBMfKFniB-FZVLuk4Q8LU-9qCpwiI)
        """)

        # Blog 3: AWS User Case Studies
        st.markdown("---")
        st.markdown(
            "## 🚀 **Digital Transformation Success Stories: How Organizations Are Leveraging AWS**")

        st.markdown("""
        The journey to digital transformation is unique for every organization, but the destination
        is universal: increased efficiency, enhanced customer experiences, and sustainable competitive
        advantages. AWS has been instrumental in enabling countless organizations to achieve these
        goals through innovative cloud solutions and scalable architectures.

        ### **Healthcare Revolution: Telemedicine Platform**

        A leading healthcare provider transformed their patient care delivery by migrating their
        telemedicine platform to AWS. By leveraging Amazon EC2 for scalable compute resources,
        Amazon RDS for secure patient data management, and Amazon CloudFront for global content
        delivery, they achieved 99.9% uptime and reduced infrastructure costs by 40%. The platform
        now serves over 2 million patients annually, with real-time video consultations and secure
        medical record access.

        **Key Technologies:** Amazon EC2, Amazon RDS, Amazon CloudFront, AWS Lambda

        ### **Financial Services Innovation: Digital Banking**

        A traditional bank modernized their core banking systems using AWS, implementing a
        microservices architecture that improved transaction processing speed by 60%. By utilizing
        Amazon Aurora for high-performance database operations, Amazon S3 for secure document
        storage, and AWS Lambda for event-driven processing, they reduced operational costs by
        35% while enhancing security and compliance capabilities.

        **Key Technologies:** Amazon Aurora, Amazon S3, AWS Lambda, Amazon API Gateway

        ### **Retail Transformation: E-commerce Platform**

        A global retailer achieved unprecedented scalability during peak shopping seasons by
        migrating their e-commerce platform to AWS. Using Amazon DynamoDB for high-performance
        data storage, Amazon S3 for content management, and Amazon CloudWatch for comprehensive
        monitoring, they handled 10x traffic increases without performance degradation. The
        platform now processes over 100,000 transactions per minute during peak periods.

        **Key Technologies:** Amazon DynamoDB, Amazon S3, Amazon CloudWatch, Amazon ElastiCache

        ### **Manufacturing Intelligence: IoT and Analytics**

        A manufacturing company implemented an IoT solution on AWS to monitor production
        equipment in real-time. By collecting sensor data through AWS IoT Core, processing
        it with Amazon Kinesis, and analyzing it with Amazon QuickSight, they achieved 25%
        improvement in operational efficiency and 30% reduction in maintenance costs through
        predictive maintenance capabilities.

        **Key Technologies:** AWS IoT Core, Amazon Kinesis, Amazon QuickSight, Amazon S3

        ### **Lessons Learned and Best Practices**

        These success stories highlight several common themes: the importance of starting with
        a clear strategy, the value of incremental migration approaches, and the critical role
        of security and compliance considerations. Organizations that succeed in their digital
        transformation journeys typically focus on building scalable, resilient architectures
        while maintaining strong governance and security practices.

        The future of digital transformation lies in continued innovation and adaptation. As
        organizations embrace emerging technologies like machine learning, serverless computing,
        and edge computing, AWS continues to provide the tools and services needed to turn
        transformation visions into reality.

        **📌 Read More on LinkedIn:** [AWS Digital Transformation Case Studies](https://www.linkedin.com/posts/vikrant-soni-ab861b330_aws-digitaltransformation-cloudleadership-activity-7355833366744616961-egKm?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFNiqWwBMfKFniB-FZVLuk4Q8LU-9qCpwiI)
        """)

        st.markdown("---")
        st.markdown("### 📚 **About These Articles**")
        st.info("""
        These articles are written by cloud experts and provide insights into AWS services,
        best practices, and real-world applications. Each article includes relevant keywords
        for SEO optimization and offers practical guidance for cloud professionals and organizations
        looking to leverage AWS for their digital transformation initiatives.
        """)

    elif aws_task == "🖥️ EC2 Management":
        st.subheader("EC2 Instance Management")

        # Show current region
        current_region = st.session_state.aws_credentials["region"]
        st.info(f"**Current Region:** {current_region}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Launch Instance")
            region = st.selectbox("Region",
                                ["us-east-1", "us-west-1", "us-west-2",
                                    "eu-west-1", "ap-south-1", "ap-southeast-1"],
                                index=0 if current_region == "us-east-1" else 1,
                                key="ec2_region")
            ami_id = st.text_input(
    "AMI ID", key="ami_id", placeholder="ami-12345678")
            instance_type = st.selectbox("Instance Type",
                                       ["t3.micro", "t3.small", "t3.medium",
                                           "t2.micro", "t2.small"],
                                       key="instance_type")
            
            if st.button("Launch Instance"):
                if ami_id:
                    try:
                        ec2 = boto3_client("ec2", region)
                        response = ec2.run_instances(ImageId=ami_id, InstanceType=instance_type, 
                                                   MinCount=1, MaxCount=1)
                        instance_id = response['Instances'][0]['InstanceId']
                        st.success(f"Instance launched: {instance_id}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Please provide AMI ID")
        
        with col2:
            st.write("### Manage Instance")
            instance_id = st.text_input(
    "Instance ID",
    key="manage_instance",
     placeholder="i-1234567890abcdef0")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Start"):
                    try:
                        ec2 = boto3_client("ec2", region)
                        ec2.start_instances(InstanceIds=[instance_id])
                        st.success("Start requested")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
                if st.button("Stop"):
                    try:
                        ec2 = boto3_client("ec2", region)
                        ec2.stop_instances(InstanceIds=[instance_id])
                        st.success("Stop requested")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col_b:
                if st.button("Reboot"):
                    try:
                        ec2 = boto3_client("ec2", region)
                        ec2.reboot_instances(InstanceIds=[instance_id])
                        st.success("Reboot requested")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
                if st.button("Terminate", type="secondary"):
                    try:
                        ec2 = boto3_client("ec2", region)
                        ec2.terminate_instances(InstanceIds=[instance_id])
                        st.success("Terminate requested")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # List instances
        st.write("### 📋 List Instances")
        if st.button("🔄 Refresh Instance List"):
            try:
                ec2 = boto3_client("ec2", region)
                instances = ec2.describe_instances()

                instance_list = []
                for reservation in instances['Reservations']:
                    for instance in reservation['Instances']:
                        instance_list.append({
                            'Instance ID': instance['InstanceId'],
                            'Type': instance['InstanceType'],
                            'State': instance['State']['Name'],
                            'Launch Time': instance['LaunchTime'].strftime('%Y-%m-%d %H:%M:%S'),
                            'Public IP': instance.get('PublicIpAddress', 'N/A'),
                            'Private IP': instance.get('PrivateIpAddress', 'N/A')
                        })

                if instance_list:
                    st.dataframe(instance_list, use_container_width=True)
                else:
                    st.info("No instances found in this region")

            except Exception as e:
                st.error(f"Error listing instances: {e}")

    elif aws_task == "📊 CloudWatch Logs":
        st.subheader("CloudWatch Logs Viewer")
        
        # Show current region
        current_region = st.session_state.aws_credentials["region"]
        st.info(f"**Current Region:** {current_region}")

        cw_region = st.selectbox("CloudWatch Region",
                               ["us-east-1", "us-west-1", "us-west-2",
                                   "eu-west-1", "ap-south-1", "ap-southeast-1"],
                               index=0 if current_region == "us-east-1" else 1)
        log_group = st.text_input("Log Group Name", "/aws/lambda/your-function",
                                 placeholder="/aws/lambda/your-function-name")
        minutes = st.number_input("Minutes to look back", 1, 1440, 60)
        
    if st.button("Fetch Logs", type="primary"):
        if not log_group:
            st.warning("Please enter a Log Group Name")
            st.stop()

        try:
            with st.spinner("Fetching logs from CloudWatch..."):
                logs = boto3_client("logs", cw_region)

                # First, verify the log group exists
                try:
                    log_groups = logs.describe_log_groups(
                        logGroupNamePrefix=log_group)
                    if not log_groups.get('logGroups'):
                        st.error(
                            f"Log group '{log_group}' not found in region {cw_region}")
                        st.info("Available log groups:")
                        all_groups = logs.describe_log_groups()
                        for group in all_groups.get(
                            'logGroups', [])[:10]:  # Show first 10
                            st.write(f"• {group['logGroupName']}")
                        st.stop()
                except Exception as e:
                    st.error(f"Error checking log group: {e}")
                    st.stop()

                # Get all log streams (not just 5)
                st.info("Fetching all log streams...")
                streams = logs.describe_log_streams(
                    logGroupName=log_group,
                    orderBy='LastEventTime',
                    descending=True
                )

                if not streams.get('logStreams'):
                    st.warning(
                        f"No log streams found in log group '{log_group}'")
                    st.stop()

                st.info(
                    f"Found {len(streams['logStreams'])} log streams. Fetching events...")

                # Calculate start time
                start_time = int(
                    (datetime.datetime.utcnow() -
                     datetime.timedelta(
                         minutes=minutes)).timestamp() *
                    1000)

                events_out = []
                total_streams_processed = 0

                # Process each stream to get all events
                for stream in streams.get('logStreams', []):
                    total_streams_processed += 1
                    if total_streams_processed % 10 == 0:  # Progress update every 10 streams
                        st.info(f"Processing stream {total_streams_processed}/{len(streams['logStreams'])}")

                    try:
                        # Get all events from this stream (not just 100)
                        events = logs.get_log_events(
                            logGroupName=log_group,
                            logStreamName=stream['logStreamName'],
                            startTime=start_time,
                            startFromHead=True  # Start from the beginning of the stream
                        )

                        # Add all events from this stream
                        stream_events = events.get('events', [])
                        events_out.extend(stream_events)

                        # Handle pagination if there are more events
                        while events.get('nextForwardToken'):
                            try:
                                events = logs.get_log_events(
                                    logGroupName=log_group,
                                    logStreamName=stream['logStreamName'],
                                    nextToken=events['nextForwardToken'],
                                    startFromHead=True
                                )
                                stream_events = events.get('events', [])
                                events_out.extend(stream_events)

                                # Safety check to prevent infinite loops
                                if len(stream_events) == 0:
                                    break
                            except Exception as e:
                                st.warning(
                                    f"Error paginating stream {stream['logStreamName']}: {e}")
                                break

                    except Exception as e:
                        st.warning(
                            f"Error processing stream {stream['logStreamName']}: {e}")
                        continue

                # Sort all events by timestamp
                events_out.sort(key=lambda x: x['timestamp'])

                if events_out:
                    st.success(f"✅ Successfully fetched {len(events_out)} log events from {total_streams_processed} streams!")

                    # Display all events (not just last 50)
                    st.write(f"**All {len(events_out)} log events (sorted by timestamp):**")

                    # Add a search/filter option
                    search_term = st.text_input("🔍 Filter logs (optional)", placeholder="Enter text to filter log messages")

                    # Filter events if search term is provided
                    filtered_events = events_out
                    if search_term:
                        filtered_events = [event for event in events_out if search_term.lower() in event['message'].lower()]
                        st.info(f"Showing {len(filtered_events)} events matching '{search_term}'")

                    # Display events with pagination for better performance
                    events_per_page = 100
                    total_pages = (len(filtered_events) + events_per_page - 1) // events_per_page

                    if total_pages > 1:
                        page = st.selectbox(f"Page (1-{total_pages})", range(1, total_pages + 1)) - 1
                        start_idx = page * events_per_page
                        end_idx = min(start_idx + events_per_page, len(filtered_events))
                        display_events = filtered_events[start_idx:end_idx]
                        st.info(f"Showing events {start_idx + 1}-{end_idx} of {len(filtered_events)}")
                    else:
                        display_events = filtered_events

                    # Display the events
                    for event in display_events:
                        timestamp = datetime.datetime.utcfromtimestamp(event['timestamp']/1000.0).strftime('%Y-%m-%d %H:%M:%S UTC')
                        st.text(f"📅 {timestamp} — {event['message']}")

                    # Add download option
                    if st.button("📥 Download Logs as JSON"):
                        log_data = {
                            "log_group": log_group,
                            "region": cw_region,
                            "total_events": len(events_out),
                            "events": [
                                {
                                    "timestamp": datetime.datetime.utcfromtimestamp(event['timestamp']/1000.0).isoformat(),
                                    "message": event['message']
                                }
                                for event in events_out
                            ]
                        }

                        json_str = json.dumps(log_data, indent=2)
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name=f"cloudwatch_logs_{log_group.replace('/', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                else:
                    st.info("No log events found in the specified time range")

        except Exception as e:
            st.error(f"❌ CloudWatch error: {e}")
            st.info("Please check:")
            st.info("• Log group name is correct")
            st.info("• Region is correct")
            st.info("• AWS credentials have CloudWatch Logs permissions")
            st.info("• Log group exists in the specified region")

    elif aws_task == "💾 S3 Storage Classes":
        st.subheader("S3 Storage Classes Demo")
        
        # Show current region
        current_region = st.session_state.aws_credentials["region"]
        st.info(f"**Current Region:** {current_region}")
        
        s3_region = st.selectbox("S3 Region", 
                               ["us-east-1", "us-west-1", "us-west-2", "eu-west-1", "ap-south-1", "ap-southeast-1"],
                               index=0 if current_region == "us-east-1" else 1)
        bucket_name = st.text_input("Bucket Name", placeholder="my-bucket-name")
        object_key = st.text_input("Object Key", "demo/test.txt", placeholder="folder/file.txt")
        content = st.text_area("Content to Upload", "Hello from Streamlit AWS demo")
        storage_class = st.selectbox("Storage Class", 
                                   ["STANDARD", "INTELLIGENT_TIERING", "STANDARD_IA", "GLACIER"])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Upload Object"):
                if bucket_name:
                    try:
                        s3 = boto3_client("s3", s3_region)
                        s3.put_object(Bucket=bucket_name, Key=object_key, 
                                    Body=content.encode('utf-8'), StorageClass=storage_class)
                        st.success(f"Uploaded {object_key} with storage class {storage_class}")
                    except Exception as e:
                        st.error(f"S3 error: {e}")
                else:
                    st.warning("Please provide bucket name")
        
        with col2:
            if st.button("List Buckets"):
                try:
                    s3 = boto3_client("s3", s3_region)
                    buckets = s3.list_buckets()
                    bucket_list = [bucket['Name'] for bucket in buckets['Buckets']]
                    if bucket_list:
                        st.write("**Available Buckets:**")
                        for bucket in bucket_list:
                            st.write(f"• {bucket}")
                    else:
                        st.info("No buckets found")
                except Exception as e:
                    st.error(f"Error listing buckets: {e}")

    elif aws_task == "🎤 Amazon Transcribe":
        st.subheader("Amazon Transcribe Demo")
        
        # Show current region
        current_region = st.session_state.aws_credentials["region"]
        st.info(f"**Current Region:** {current_region}")
        
        trans_region = st.selectbox("Transcribe Region", 
                                  ["us-east-1", "us-west-1", "us-west-2", "eu-west-1", "ap-south-1", "ap-southeast-1"],
                                  index=0 if current_region == "us-east-1" else 1)
        audio_bucket = st.text_input("Audio S3 Bucket", placeholder="my-audio-bucket")
        audio_key = st.text_input("Audio S3 Key", placeholder="audio/sample.mp3")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Transcription Job"):
                if audio_bucket and audio_key:
                    try:
                        transcribe = boto3_client("transcribe", trans_region)
                        job_name = f"job-{int(time.time())}"
                        media_uri = f"s3://{audio_bucket}/{audio_key}"
                        
                        transcribe.start_transcription_job(
                            TranscriptionJobName=job_name,
                            Media={'MediaFileUri': media_uri},
                            MediaFormat=audio_key.split('.')[-1],
                            LanguageCode="en-US"
                        )
                        st.success(f"Transcription job started: {job_name}")
                        st.info("Check AWS Console for job completion status")
                    except Exception as e:
                        st.error(f"Transcribe error: {e}")
                else:
                    st.warning("Please provide audio bucket and key")
        
        with col2:
            if st.button("List Transcription Jobs"):
                try:
                    transcribe = boto3_client("transcribe", trans_region)
                    jobs = transcribe.list_transcription_jobs(MaxResults=10)
                    
                    if jobs['TranscriptionJobSummaries']:
                        st.write("**Recent Transcription Jobs:**")
                        for job in jobs['TranscriptionJobSummaries']:
                            status_emoji = "✅" if job['TranscriptionJobStatus'] == 'COMPLETED' else "⏳"
                            st.write(f"{status_emoji} {job['TranscriptionJobName']} - {job['TranscriptionJobStatus']}")
                    else:
                        st.info("No transcription jobs found")
                except Exception as e:
                    st.error(f"Error listing jobs: {e}")

    elif aws_task == "🗃️ MongoDB Connection":
        st.subheader("MongoDB/DocumentDB Connection Test")
        
        mongo_uri = st.text_input("MongoDB URI", placeholder="mongodb://user:pass@host:port/db")
        
        if st.button("Test Connection"):
            if mongo_uri and MongoClient:
                try:
                    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
                    client.admin.command('ping')
                    st.success("✅ MongoDB connection successful")
                    
                    # Insert test document
                    db = client.get_database("demo_db")
                    collection = db.get_collection("test")
                    result = collection.insert_one({
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                        "note": "Test from Streamlit"
                    })
                    st.info(f"Inserted test document: {result.inserted_id}")
                except Exception as e:
                    st.error(f"MongoDB error: {e}")
            else:
                st.warning("Please provide MongoDB URI")

    elif aws_task == "🔗 S3 Presigned URLs":
        st.subheader("S3 Presigned URLs Generator")
        
        # Show current region
        current_region = st.session_state.aws_credentials["region"]
        st.info(f"**Current Region:** {current_region}")
        
        presign_region = st.selectbox("S3 Region", 
                                    ["us-east-1", "us-west-1", "us-west-2", "eu-west-1", "ap-south-1", "ap-southeast-1"],
                                    index=0 if current_region == "us-east-1" else 1,
                                    key="presign_region")
        presign_bucket = st.text_input("Bucket Name", key="presign_bucket", placeholder="my-bucket-name") 
        presign_key = st.text_input("Object Key", "uploads/demo.txt", key="presign_key", placeholder="folder/file.txt")
        expiry = st.number_input("Expiry (seconds)", 60, 86400, 3600)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Presigned URL"):
                if presign_bucket:
                    try:
                        s3 = boto3_client("s3", presign_region)
                        
                        # Presigned PUT URL
                        put_url = s3.generate_presigned_url('put_object',
                                                          Params={'Bucket': presign_bucket, 'Key': presign_key},
                                                          ExpiresIn=expiry)
                        st.write("**Presigned PUT URL:**")
                        st.code(put_url)
                        
                        # Presigned GET URL
                        get_url = s3.generate_presigned_url('get_object',
                                                          Params={'Bucket': presign_bucket, 'Key': presign_key},
                                                          ExpiresIn=expiry)
                        st.write("**Presigned GET URL:**")
                        st.code(get_url)
                        
                    except Exception as e:
                        st.error(f"Presigned URL error: {e}")
                else:
                    st.warning("Please provide bucket name")
        
        with col2:
            st.write("### Test Upload")
            demo_content = st.text_area("Test content to upload", value="Hello from Streamlit!")
            if st.button("Upload via Presigned URL"):
                if presign_bucket:
                    try:
                        s3 = boto3_client("s3", presign_region)
                        put_url = s3.generate_presigned_url('put_object',
                                                          Params={'Bucket': presign_bucket, 'Key': presign_key},
                                                          ExpiresIn=expiry)
                        
                        import requests
                        response = requests.put(put_url, data=demo_content.encode('utf-8'))
                        if response.status_code == 200:
                            st.success("✅ Upload successful!")
                        else:
                            st.error(f"❌ Upload failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"Upload error: {e}")
                else:
                    st.warning("Please provide bucket name")

# ------------------ Streamlit Project (MoodMate) ------------------
elif main_page == "Streamlit Project":
    st.title("🌸 MoodMate — Full")
    
    # Initialize session state for games
    if "ttt_board" not in st.session_state:
        st.session_state.ttt_board = [""] * 9
        st.session_state.ttt_current = "X"
    if "target_number" not in st.session_state:
        st.session_state.target_number = random.randint(1, 100)
    
    mood_menu = option_menu(
        menu_title=None,
        options=["Home", "Recommendations", "Breathing Tool", "Canvas", "Games"],
        icons=["house", "stars", "activity", "brush", "controller"],
        orientation="horizontal"
    )

    # Home
    if mood_menu == "Home":
        st.header("🌸 Welcome to MoodMate")
        st.image("https://images.unsplash.com/photo-1536859355448-76f92ebdc33d?auto=format&fit=crop&w=1470&q=80", 
                use_column_width=True, caption="Your mood companion")
        
        lottie = load_lottie_url("https://lottie.host/0ce497f1-f97d-4a40-8785-2c7ccf25b3f6/WFylOmQwRy.json")
        if lottie:
            st_lottie(lottie, height=220)
        else:
            st.info("🌟 Welcome to your personalized mood dashboard!")
            
        st.markdown("#### Your personalized mood-based activity dashboard.")
        st.markdown("Made with ❤ using Streamlit")

    # Recommendations
    elif mood_menu == "Recommendations":
        st.header("🌟 Mood-Based Recommendations")
        mood = st.radio("How are you feeling today?", list(mood_data.keys()), horizontal=True)
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎬 Movies")
            for m in mood_data[mood]["movies"]:
                st.write(f"• {m}")
            st.subheader("🎵 Songs")
            for s in mood_data[mood]["songs"]:
                st.write(f"• {s}")
        with col2:
            st.subheader("📖 Quotes")
            for q in mood_data[mood]["quotes"]:
                st.info(q)
            st.subheader("🍵 Herbal Remedy")
            st.success(mood_data[mood]["remedy"])

    # Breathing Tool
    elif mood_menu == "Breathing Tool":
        st.header("🧘 Guided Breathing")
        st.markdown("Take a few deep breaths with calming visuals and sound.")
        
        # Audio player
        st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3")
        
        lottie = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_lcszfgux.json")
        if lottie:
            st_lottie(lottie, height=300)
        else:
            st.markdown("🌊 Imagine calm waves flowing in and out...")
            
        st.info("Breathe in... Hold... Breathe out... Repeat 4–5 times.")
        
        with st.expander("⏱ Try 4-7-8 Breathing"):
            st.write("**Instructions:** Inhale for 4 sec, Hold for 7 sec, Exhale for 8 sec")
            if st.button("Start Timer (3 cycles)"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(3):
                    # Inhale
                    status_text.text(f"Cycle {i+1}/3 - Inhale (4 seconds)")
                    for j in range(4):
                        progress_bar.progress((i*19 + j) / 57)
                        time.sleep(1)
                    
                    # Hold
                    status_text.text(f"Cycle {i+1}/3 - Hold (7 seconds)")
                    for j in range(7):
                        progress_bar.progress((i*19 + 4 + j) / 57)
                        time.sleep(1)
                    
                    # Exhale
                    status_text.text(f"Cycle {i+1}/3 - Exhale (8 seconds)")
                    for j in range(8):
                        progress_bar.progress((i*19 + 11 + j) / 57)
                        time.sleep(1)
                
                progress_bar.progress(1.0)
                status_text.text("Completed! Well done! 🌟")
                st.success("Completed 3 breathing cycles. Well done!")

    # Canvas
    elif mood_menu == "Canvas":
        st.header("🎨 MoodMate Canvas")
        
        col1, col2 = st.columns(2)
        with col1:
            stroke_color = st.color_picker("Stroke color", "#000000")
            bg_color = st.color_picker("Background color", "#fffbe6")
        with col2:
            stroke_width = st.slider("Stroke width", 1, 25, 2)
            drawing_mode = st.selectbox("Drawing Tool", ["freedraw", "line", "rect", "circle", "transform"])
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=400,
            width=600,
            drawing_mode=drawing_mode,
            key="mood_canvas",
        )
        st.caption("🖌 Use different tools and colors for relaxation and creativity!")

    # Games
    elif mood_menu == "Games":
        st.header("🎮 MoodMate Games")
        game_choice = st.selectbox("Choose a game", ["Stone Paper Scissors", "Tic Tac Toe", "Guess the Number"])
        
        if game_choice == "Stone Paper Scissors":
            st.subheader("✂ Stone Paper Scissors")
            mode = st.radio("Mode", ["Player vs Computer", "Two Player"])
            choices = ["Stone", "Paper", "Scissors"]
            
            if mode == "Player vs Computer":
                user_choice = st.selectbox("Your Move", choices)
                if st.button("Play"):
                    comp = random.choice(choices)
                    st.write(f"**You:** {user_choice}")
                    st.write(f"**Computer:** {comp}")
                    
                    if user_choice == comp:
                        st.info("🤝 It's a draw!")
                    elif (user_choice == "Stone" and comp == "Scissors") or \
                         (user_choice == "Scissors" and comp == "Paper") or \
                         (user_choice == "Paper" and comp == "Stone"):
                        st.success("🎉 You win!")
                    else:
                        st.error("🤖 Computer wins!")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    p1 = st.selectbox("Player 1 Move", choices, key="p1")
                with col2:
                    p2 = st.selectbox("Player 2 Move", choices, key="p2")
                
                if st.button("Play Match"):
                    st.write(f"**Player 1:** {p1}")
                    st.write(f"**Player 2:** {p2}")
                    
                    if p1 == p2:
                        st.info("🤝 Draw match!")
                    elif (p1 == "Stone" and p2 == "Scissors") or \
                         (p1 == "Scissors" and p2 == "Paper") or \
                         (p1 == "Paper" and p2 == "Stone"):
                        st.success("🎉 Player 1 wins!")
                    else:
                        st.success("🎉 Player 2 wins!")
                        
        elif game_choice == "Tic Tac Toe":
            st.subheader("⭕ Tic Tac Toe")
            
            board = st.session_state.ttt_board
            current = st.session_state.ttt_current
            
            def check_winner(b):
                wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
                for i,j,k in wins:
                    if b[i] == b[j] == b[k] and b[i] != "":
                        return b[i]
                if "" not in b:
                    return "Draw"
                return None
            
            st.write(f"**Current Player:** {current}")
            
            # Create 3x3 grid
            for row in range(3):
                cols = st.columns(3)
                for col in range(3):
                    idx = row * 3 + col
                    with cols[col]:
                        if st.button(board[idx] or "⬜", key=f"ttt_{idx}", use_container_width=True):
                            if board[idx] == "":
                                board[idx] = current
                                winner = check_winner(board)
                                if winner:
                                    if winner == "Draw":
                                        st.info("🤝 It's a draw!")
                                    else:
                                        st.success(f"🎉 {winner} wins!")
                                    # Reset game
                                    st.session_state.ttt_board = [""] * 9
                                    st.session_state.ttt_current = "X"
                                    st.rerun()
                                else:
                                    st.session_state.ttt_current = "O" if current == "X" else "X"
                                    st.rerun()
            
            if st.button("Reset Game"):
                st.session_state.ttt_board = [""] * 9
                st.session_state.ttt_current = "X"
                st.rerun()
                
        elif game_choice == "Guess the Number":
            st.subheader("🔢 Guess the Number")
            st.write("I'm thinking of a number between 1 and 100!")
            
            guess = st.number_input("Your guess:", min_value=1, max_value=100, value=50)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Check Guess"):
                    target = st.session_state.target_number
                    if guess == target:
                        st.success(f"🎉 Correct! The number was {target}!")
                        st.balloons()
                        # Generate new number
                        st.session_state.target_number = random.randint(1, 100)
                    elif guess < target:
                        st.warning("📈 Too low! Try higher.")
                    else:
                        st.warning("📉 Too high! Try lower.")
            
            with col2:
                if st.button("New Game"):
                    st.session_state.target_number = random.randint(1, 100)
                    st.info("🎲 New number generated!")

# ------------------ Linux Tasks ------------------
elif main_page == "Linux Tasks":
    st.title("🐧 Linux Tasks")
    st.markdown("Collection of Linux tutorials and guides:")

    def linkedin_button(url, text):
        st.markdown(f"""
        <a href="{url}" target="_blank" style="text-decoration: none;">
            <div style="
                background-color: #0A66C2;
                color: white;
                padding: 10px 20px;
                border-radius: 8px;
                margin: 10px 0;
                display: inline-block;
                font-weight: bold;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: background-color 0.3s;
            ">
                🔗 {text}
            </div>
        </a>
        """, unsafe_allow_html=True)

    tasks = [
        {
            "title": "1. Companies Using Linux",
            "url": "https://www.linkedin.com/posts/vikrant-soni-ab861b330_linux-opensource-enterpriseit-activity-7355503627219726336-a0SY?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFNiqWwBMfKFniB-FZVLuk4Q8LU-9qCpwiI",
            "description": "Learn about major companies that rely on Linux for their operations."
        },
        {
            "title": "2. Linux GUI Programs & Underlying Commands",
            "url": "https://www.linkedin.com/posts/vikrant-soni-ab861b330_linux-linuxgui-opensourcetools-activity-7355504249566318594-z8AK?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFNiqWwBMfKFniB-FZVLuk4Q8LU-9qCpwiI",
            "description": "Discover the command-line equivalents of popular GUI applications."
        },
        {
            "title": "3. Change Program Icons in Linux",
            "url": "https://www.linkedin.com/posts/vikrant-soni-ab861b330_linux-opensource-customization-activity-7360362718031200256-KZ-p?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFNiqWwBMfKFniB-FZVLuk4Q8LU-9qCpwiI",
            "description": "Customize your Linux desktop by changing application icons."
        },
        {
            "title": "4. Terminal Messaging Automation",
            "url": "https://www.linkedin.com/posts/vikrant-soni-ab861b330_linux-terminal-gui-activity-7360365427920703488-fZYw?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFNiqWwBMfKFniB-FZVLuk4Q8LU-9qCpwiI",
            "description": "Send emails, WhatsApp messages, tweets, and SMS from the terminal."
        },
        {
            "title": "5. Additional Terminals and GUI Interfaces",
            "url": "https://www.linkedin.com/posts/vikrant-soni-ab861b330_linux-cli-automation-activity-7360366565000704001-tHU1?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFNiqWwBMfKFniB-FZVLuk4Q8LU-9qCpwiI",
            "description": "Explore different terminal emulators and file managers for Linux."
        },
        {
            "title": "6. Linux ctrl c and ctrl z",
            "url": "https://www.linkedin.com/posts/vikrant-soni-ab861b330_linuxtips-processmanagement-sigint-activity-7355505093284126720-QmTP?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFNiqWwBMfKFniB-FZVLuk4Q8LU-9qCpwiI",
            "description": "Learn about Linux commands and their usage."
        }
    ]

    for task in tasks:
        with st.expander(task["title"]):
            st.write(task["description"])
            linkedin_button(task["url"], "View Post on LinkedIn")



# ------------------ Windows Tasks ------------------
elif main_page == "Windows Tasks":
    st.title("🪟 Windows Automation Tasks")
    st.warning("⚠️ These tasks will execute system commands. Use with caution!")

    windows_task = st.selectbox("Choose a Windows Task:", [
        "Open Notepad", "Open Calculator", "Open Command Prompt", "Open File Explorer",
        "Open Chrome", "Open MS Word", "Open MS Excel", "Open Specific Folder",
        "Play Music Folder", "System Information", "Lock Screen"
    ])

    if windows_task == "Open Notepad":
        if st.button("🗒️ Open Notepad"):
            try:
                os.system("notepad")
                st.success("Notepad opened successfully!")
            except Exception as e:
                st.error(f"Failed to open Notepad: {e}")

    elif windows_task == "Open Calculator":
        if st.button("🧮 Open Calculator"):
            try:
                os.system("calc")
                st.success("Calculator opened successfully!")
            except Exception as e:
                st.error(f"Failed to open Calculator: {e}")

    elif windows_task == "Open Command Prompt":
        if st.button("💻 Open Command Prompt"):
            try:
                os.system("start cmd")
                st.success("Command Prompt opened successfully!")
            except Exception as e:
                st.error(f"Failed to open Command Prompt: {e}")

    elif windows_task == "Open File Explorer":
        if st.button("📁 Open File Explorer"):
            try:
                os.system("explorer")
                st.success("File Explorer opened successfully!")
            except Exception as e:
                st.error(f"Failed to open File Explorer: {e}")

    elif windows_task == "Open Chrome":
        if st.button("🌐 Open Chrome Browser"):
            try:
                os.system("start chrome")
                st.success("Chrome browser opened successfully!")
            except Exception as e:
                st.error(f"Failed to open Chrome: {e}")

    elif windows_task == "Open MS Word":
        if st.button("📝 Open Microsoft Word"):
            try:
                os.system("start winword")
                st.success("Microsoft Word opened successfully!")
            except Exception as e:
                st.error(f"Failed to open MS Word: {e}")

    elif windows_task == "Open MS Excel":
        if st.button("📊 Open Microsoft Excel"):
            try:
                os.system("start excel")
                st.success("Microsoft Excel opened successfully!")
            except Exception as e:
                st.error(f"Failed to open MS Excel: {e}")

    elif windows_task == "Open Specific Folder":
        folder_path = st.text_input("Enter full folder path:", placeholder="C:\\Users\\YourName\\Documents")
        if st.button("📂 Open Folder"):
            if folder_path:
                if os.path.exists(folder_path):
                    try:
                        os.startfile(folder_path)
                        st.success(f"Folder '{folder_path}' opened successfully!")
                    except Exception as e:
                        st.error(f"Failed to open folder: {e}")
                else:
                    st.error("❌ Path not found. Please check the folder path.")
            else:
                st.warning("Please enter a folder path.")

    elif windows_task == "Play Music Folder":
        music_dir = st.text_input("Music folder path:", value="C:\\Users\\Public\\Music")
        if st.button("🎵 Open Music Folder"):
            try:
                if os.path.exists(music_dir):
                    os.startfile(music_dir)
                    st.success("🎵 Music folder opened successfully!")
                else:
                    st.error("❌ Music folder not found. Please check the path.")
            except Exception as e:
                st.error(f"❌ Unable to open music folder: {e}")

    elif windows_task == "System Information":
        if st.button("ℹ️ Show System Information"):
            try:
                import platform
                st.write("**System Information:**")
                st.write(f"- **OS**: {platform.system()} {platform.release()}")
                st.write(f"- **Version**: {platform.version()}")
                st.write(f"- **Architecture**: {platform.architecture()[0]}")
                st.write(f"- **Processor**: {platform.processor()}")
                st.write(f"- **Machine**: {platform.machine()}")
                st.write(f"- **Node**: {platform.node()}")
                
                if psutil:
                    st.write(f"- **CPU Cores**: {psutil.cpu_count()}")
                    st.write(f"- **RAM**: {psutil.virtual_memory().total / (1024**3):.2f} GB")
                    st.write(f"- **Disk Usage**: {psutil.disk_usage('/').total / (1024**3):.2f} GB")
                
            except Exception as e:
                st.error(f"Failed to get system information: {e}")

    elif windows_task == "Lock Screen":
        st.warning("⚠️ This will lock your computer screen!")
        if st.button("🔒 Lock Screen", type="secondary"):
            try:
                os.system("rundll32.exe user32.dll,LockWorkStation")
                st.info("System locked.")
            except Exception as e:
                st.error(f"Failed to lock screen: {e}")

# Footer
st.markdown("---")
st.markdown("""
<style>
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 3rem;
    }
    
    .footer-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .footer-content {
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    
    .footer-item {
        text-align: center;
    }
    
    .footer-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
</style>

<div class="footer">
    <div class="footer-title">🚀 SmartOps Dashboard</div>
    <div class="footer-content">
        <div class="footer-item">
            <div class="footer-icon">⚡</div>
            <div>Built with Streamlit</div>
        </div>
        <div class="footer-item">
            <div class="footer-icon">🤖</div>
            <div>AI-Powered Tools</div>
        </div>
        <div class="footer-item">
            <div class="footer-icon">☁️</div>
            <div>Cloud Integration</div>
        </div>
        <div class="footer-item">
            <div class="footer-icon">💡</div>
            <div>Innovation First</div>
        </div>
    </div>
    <div style="margin-top: 1rem; opacity: 0.8;">
        💡 <strong>Tip:</strong> Install required packages using pip for full functionality
    </div>
</div>
""", unsafe_allow_html=True)

# Display current time in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
    <div style="font-size: 0.9rem; color: white; opacity: 0.8;">🕐 Current Time</div>
    <div style="font-size: 1.1rem; color: white; font-weight: bold;">
        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</div>
""", unsafe_allow_html=True)
