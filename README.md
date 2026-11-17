🧠 SYNAPTIQ PROJECT DOCUMENTATION
1. 📌 Project Title

Synaptiq — AI-Powered Cognitive Assessment and Alzheimer’s Support System

2. 🧠 Abstract

Synaptiq is an AI-assisted healthcare application designed to analyze cognitive health and assist in early detection of Alzheimer’s-related symptoms. The system integrates rule-based cognitive assessment, interactive visualization, multilingual AI chatbot support, and voice-based interaction to provide an intelligent and user-friendly diagnostic experience.

It aims to make early cognitive screening more accessible, interactive, and interpretable for both patients and healthcare professionals.

3. 🎯 Objectives
To detect early signs of cognitive decline using MMSE and CDR scores
To provide an AI-powered conversational assistant for medical queries
To visualize cognitive progression in an intuitive timeline format
To build interactive cognitive training games for memory assessment
To support multilingual and voice-based interaction
4. ⚙️ System Architecture
🧩 Architecture Flow:
User Input
   ↓
Streamlit UI Layer
   ↓
Rule-Based Cognitive Model (MMSE → Stage Classification)
   ↓
LLM Layer (Groq / Llama-3)
   ↓
Visualization Engine (Timeline, Charts, Games)
   ↓
Final Output (Diagnosis + AI Response + Interaction UI)
5. 🧠 Modules Description
5.1 🏠 Dashboard Module
Displays patient cognitive summary
Shows MMSE, CDR, and risk confidence
Provides overall diagnostic snapshot
5.2 🧠 Risk Timeline Module
Converts MMSE score into 7-stage cognitive classification
Visualizes progression from:
No Decline → Very Severe Decline
Interactive slider-based exploration
5.3 🎤 Voice Assistant Module
Converts speech to text using Whisper model
Processes query using LLM (Llama 3.1 via Groq)
Returns medical responses in natural language
5.4 💬 Multilingual Chatbot
Supports multiple languages (English, Hindi, Kannada, Tamil)
Uses LLM for Alzheimer-specific responses
Maintains chat history using session state
5.5 🧩 Cognitive Memory Game
Cognitive training mini-game
Includes:
Memory preview phase
Hidden recall phase
Tracks:
attempts
score
accuracy
5.6 🔐 Authentication System
Simple session-based login system
Role-based access (Doctor / Patient / Admin concept ready)
6. 🤖 AI / ML Methodology
🧠 Cognitive Classification Model

Currently uses a rule-based ML simulation model:

MMSE Score → Cognitive Stage Mapping
MMSE Range	Stage
27–30	No Decline
24–26	Very Mild
20–23	Mild
15–19	Moderate
10–14	Moderately Severe
5–9	Severe
<5	Very Severe
🧠 AI Chat System
Uses Groq API (Llama 3.1 model)
Prompt-engineered for Alzheimer-focused responses
Supports multilingual output
🎤 Speech Processing
Whisper Large-v3 model for transcription
Converts speech → text → AI response
7. 📊 Visualization Techniques
📈 Cognitive timeline (7-stage progression axis)
📊 Metric cards (MMSE, CDR, Risk score)
🎮 Interactive memory game UI
💬 Chat-based conversational UI
🎤 Voice interaction feedback system
8. 🛠️ Tech Stack
Frontend / UI:
Streamlit
AI Models:
Llama 3.1 (Groq API)
Whisper Large-v3
Backend Logic:
Python
Session State Management
Libraries:
Streamlit
Groq SDK
NumPy
Pandas
Tempfile
9. 🚀 Key Features
🧠 Alzheimer risk stage prediction
🎤 Voice-based AI assistant
💬 Multilingual chatbot
🧩 Cognitive memory game
📊 Interactive timeline visualization
🔐 Login authentication system
10. 📌 Limitations
Requires internet for LLM API
Limited clinical validation dataset
11. 🔮 Future Scope
Integration with real ML models (Random Forest / XGBoost)
MRI / brain scan image analysis
Wearable device integration
Real-time cognitive tracking
Hospital EMR integration
Offline medical assistant mode
12. 🧾 Conclusion

Synaptiq demonstrates how AI can be used to assist in early cognitive screening and Alzheimer’s support. By combining machine learning logic, generative AI, speech processing, and interactive visualization, it provides a comprehensive cognitive health assistant system that is both accessible and scalable.

