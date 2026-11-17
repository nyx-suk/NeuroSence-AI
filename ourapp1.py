"""
SYNAPTIQ — Interpretable Alzheimer's Detection
Full Streamlit UI — Memory Assistance Platform
"""
   
import os, sys, warnings, re, base64, json, html, random, tempfile, math, ast
from pathlib import Path
from datetime import datetime, date, time as dt_time
import json

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sqlite3

# Optional packages: the app should still start if voice/GPS extras are not installed.
try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

try:
    from streamlit_geolocation import streamlit_geolocation
except ImportError:
    streamlit_geolocation = None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

try:
    from groq import Groq
except ImportError:
    Groq = None

# Optional FFmpeg configuration. Never use another user's machine-specific path.
if AudioSegment is not None:
    _ffmpeg_path = os.getenv("FFMPEG_PATH", "").strip()
    _ffprobe_path = os.getenv("FFPROBE_PATH", "").strip()
    if _ffmpeg_path and os.path.exists(_ffmpeg_path):
        AudioSegment.converter = _ffmpeg_path
    if _ffprobe_path and os.path.exists(_ffprobe_path):
        AudioSegment.ffprobe = _ffprobe_path

# Streamlit page configuration must be the first Streamlit UI call.


# ============================================================
# HEY NEURO — self-contained hands-free voice assistant
# Everything is embedded in this file; no external HTML/CSS/JS
# files are required.
# ============================================================

def process_neura_action():
    """Apply only predefined voice actions coming from Hey Neuro."""
    action = st.query_params.get("neura_action")
    if not action:
        return

    allowed_routes = {
        "OPEN_GAMES": "🧩 Cognitive Games",
        "OPEN_HOME": "🏠 Home",
        "OPEN_MEMORIES": "📸 My Memories",
        "OPEN_CAREGIVER": "👨‍👩‍👧 Caregiver Portal",
        "OPEN_REMINDERS": "⏰ Smart Reminders",
        "OPEN_CHATBOT": "💬 AI Chatbot",
        "OPEN_VOICE": "🎤 Voice Assistant",
        "OPEN_RISK": "🧠 Risk Timeline",
        "OPEN_ANALYTICS": "📊 Model Analytics",
        "OPEN_VISUALIZATIONS": "📈 Visualizations",
        "OPEN_DIAGNOSIS": "🔬 Patient Diagnosis",
        "OPEN_QUESTIONNAIRE": "📝 Cognitive Questionnaire",
        "OPEN_CULTURE": "🌏 NER Memory & Culture",
        "OPEN_SOS": "🆘 Safety / SOS",
        "SHOW_LOCATION": "📍 GPS Safety & Geofencing",
        "SHOW_ALERTS": "🆘 Safety / SOS",
        "SHOW_MEDICATION": "⏰ Smart Reminders",
        "SHOW_ACTIVITY": "🧠 Cognitive Performance",
        "SHOW_SAFE_ZONES": "📍 GPS Safety & Geofencing",
        "NAVIGATE_HOME": "🏠 Home",
        "HELP": "🎤 Voice Assistant",
    }

    try:
        del st.query_params["neura_action"]
    except Exception:
        pass

    if action in allowed_routes:
        st.session_state.nav_page = allowed_routes[action]
        st.rerun()


def render_neura_voice():
    """Render Hey Neuro entirely from ourapp.py."""
    _contact, _phone = get_callable_caregiver()

    # Escape values before putting them into JavaScript.
    _phone_json = json.dumps(_phone or "")

    components.html(
        f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; }}
body {{
    margin: 0;
    font-family: Arial, sans-serif;
    background: transparent;
}}
.neura-card {{
    border-radius: 24px;
    padding: 20px 22px;
    margin: 4px auto;
    max-width: 760px;
    background: linear-gradient(135deg, #f7fbff, #eef8ff);
    border: 2px solid #b8dcff;
    box-shadow: 0 8px 24px rgba(30,80,130,.12);
}}
.neura-row {{
    display: flex;
    align-items: center;
    gap: 15px;
}}
.neura-orb {{
    width: 62px;
    height: 62px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 31px;
    background: #ffffff;
    border: 3px solid #65aee8;
    flex: 0 0 auto;
}}
.neura-orb.listening {{
    animation: pulse 1.2s infinite;
    border-color: #25a55f;
}}
@keyframes pulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(37,165,95,.30); }}
    70% {{ box-shadow: 0 0 0 18px rgba(37,165,95,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(37,165,95,0); }}
}}
.neura-title {{
    margin: 0;
    font-size: 28px;
    color: #123b5d;
}}
.neura-sub {{
    margin: 5px 0 0;
    font-size: 16px;
    color: #456477;
}}
.neura-status {{
    margin-top: 16px;
    padding: 14px 16px;
    border-radius: 15px;
    background: white;
    border: 1px solid #d4e7f5;
    color: #17384f;
    font-size: 17px;
    line-height: 1.4;
}}
.neura-controls {{
    display: flex;
    gap: 10px;
    margin-top: 14px;
    flex-wrap: wrap;
}}
.neura-btn {{
    border: 0;
    border-radius: 14px;
    padding: 13px 20px;
    font-size: 17px;
    font-weight: 700;
    cursor: pointer;
}}
.neura-primary {{
    background: #2b7bbb;
    color: white;
}}
.neura-secondary {{
    background: #e8f1f7;
    color: #24465d;
}}
.neura-hidden {{ display:none; }}
.neura-mini {{
    margin-top: 12px;
    color: #587383;
    font-size: 14px;
    line-height: 1.5;
}}
</style>
</head>
<body>
<div id="neura-root">
  <section class="neura-card">
    <div class="neura-row">
      <div id="neura-orb" class="neura-orb">🧠</div>
      <div>
        <h2 class="neura-title">Hey Neuro</h2>
        <p class="neura-sub">Hands-free voice assistant · English (India)</p>
      </div>
    </div>

    <div id="neura-status" class="neura-status">
      Press <b>Enable Voice</b> once. After that, say “Hey Neuro” without touching the microphone.
    </div>

    <div class="neura-controls">
      <button id="neura-enable" class="neura-btn neura-primary">🎙️ Enable Voice</button>
      <button id="neura-stop" class="neura-btn neura-secondary neura-hidden">⏸ Pause Neuro</button>
    </div>

    <div class="neura-mini">
      Examples: “Hey Neuro, go to games” · “show location” · “show medications” · “show alerts” · “call caregiver”
    </div>
  </section>
</div>

<script>
(() => {{
  "use strict";

  const caregiverPhone = {_phone_json};

  const statusEl = document.getElementById("neura-status");
  const enableBtn = document.getElementById("neura-enable");
  const stopBtn = document.getElementById("neura-stop");
  const orb = document.getElementById("neura-orb");

  const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

  let recognition = null;
  let enabled = false;
  let paused = false;
  let mode = "wake";
  let restarting = false;

  const wakeWords = [
    "hey neuro",
    "hey neura",
    "hey nura",
    "hey neural",
    "hey nera",
    "hey newra"
  ];

  function setStatus(message) {{
    statusEl.innerHTML = message;
  }}

  function speak(text) {{
    if (!("speechSynthesis" in window)) return;
    try {{
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = "en-IN";
      u.rate = 0.92;
      u.pitch = 1.0;
      window.speechSynthesis.speak(u);
    }} catch (_) {{}}
  }}

  function normalize(s) {{
    return (s || "")
      .toLowerCase()
      .replace(/[.,!?;:]/g, " ")
      .replace(/\\s+/g, " ")
      .trim();
  }}

  function containsWakeWord(text) {{
    const t = normalize(text);
    return wakeWords.some(w => t.includes(w));
  }}

  function commandTextAfterWake(text) {{
    const t = normalize(text);
    for (const w of wakeWords) {{
      const i = t.indexOf(w);
      if (i >= 0) return t.slice(i + w.length).trim();
    }}
    return "";
  }}

  function intentFrom(text) {{
    const t = normalize(text);

    if (
      t.includes("go to games") ||
      t.includes("open games") ||
      t.includes("play games") ||
      t === "games" ||
      t.includes("cognitive games")
    ) return "OPEN_GAMES";

    if (
      t.includes("go home") ||
      t.includes("open home") ||
      t === "home"
    ) return "OPEN_HOME";

    if (
      t.includes("show location") ||
      t.includes("where is the patient") ||
      t.includes("where is patient") ||
      t.includes("patient location")
    ) return "SHOW_LOCATION";

    if (
      t.includes("show alerts") ||
      t.includes("show emergency") ||
      t.includes("my alerts")
    ) return "SHOW_ALERTS";

    if (
      t.includes("show medications") ||
      t.includes("show medication") ||
      t.includes("my medicines") ||
      t.includes("medications")
    ) return "SHOW_MEDICATION";

    if (
      t.includes("show activity") ||
      t.includes("my activity") ||
      t.includes("cognitive performance")
    ) return "SHOW_ACTIVITY";

    if (
      t.includes("show safe zones") ||
      t.includes("show safety zones") ||
      t.includes("safe zones")
    ) return "SHOW_SAFE_ZONES";

    if (
      t.includes("call caregiver") ||
      t.includes("call my caregiver") ||
      t.includes("contact caregiver")
    ) return "CALL_CAREGIVER";

    if (
      t.includes("take me home") ||
      t.includes("navigate home")
    ) return "NAVIGATE_HOME";

    if (t.includes("my memories") || t.includes("open memories") || t.includes("go to memories")) return "OPEN_MEMORIES";
    if (t.includes("caregiver") && (t.includes("portal") || t.includes("open") || t.includes("go to"))) return "OPEN_CAREGIVER";
    if (t.includes("reminders") || t.includes("smart reminders") || t.includes("open reminders")) return "OPEN_REMINDERS";
    if (t.includes("chatbot") || t.includes("ai chat") || t.includes("open chatbot")) return "OPEN_CHATBOT";
    if (t.includes("voice assistant") || t.includes("open voice assistant")) return "OPEN_VOICE";
    if (t.includes("risk timeline") || t.includes("open risk timeline")) return "OPEN_RISK";
    if (t.includes("model analytics") || t.includes("open model analytics")) return "OPEN_ANALYTICS";
    if (t.includes("visualizations") || t.includes("open visualizations") || t.includes("charts")) return "OPEN_VISUALIZATIONS";
    if (t.includes("diagnosis") || t.includes("patient diagnosis") || t.includes("open diagnosis")) return "OPEN_DIAGNOSIS";
    if (t.includes("questionnaire") || t.includes("cognitive questionnaire")) return "OPEN_QUESTIONNAIRE";
    if (t.includes("ner memory") || t.includes("culture") || t.includes("open culture")) return "OPEN_CULTURE";
    if (t.includes("sos") || t.includes("safety") || t.includes("emergency help")) return "OPEN_SOS";

    if (
      t === "help" ||
      t.includes("what can you do") ||
      t.includes("what can i say")
    ) return "HELP";

    return null;
  }}

  function navigate(action) {{
    // Streamlit reads this query parameter and changes its existing nav_page.
    try {{
      const url = new URL(window.top.location.href);
      url.searchParams.set("neura_action", action);
      window.top.location.href = url.toString();
    }} catch (_) {{
      // Fallback for iframe navigation restrictions.
      try {{
        const url = new URL(window.location.href);
        url.searchParams.set("neura_action", action);
        window.location.href = url.toString();
      }} catch (__) {{}}
    }}
  }}

  function executeIntent(intent) {{
    const responses = {{
      OPEN_GAMES: "Opening cognitive games.",
      OPEN_HOME: "Going home.",
      OPEN_MEMORIES: "Opening your memories.",
      OPEN_CAREGIVER: "Opening the caregiver portal.",
      OPEN_REMINDERS: "Opening smart reminders.",
      OPEN_CHATBOT: "Opening the AI chatbot.",
      OPEN_VOICE: "Opening the voice assistant.",
      OPEN_RISK: "Opening the risk timeline.",
      OPEN_ANALYTICS: "Opening model analytics.",
      OPEN_VISUALIZATIONS: "Opening visualizations.",
      OPEN_DIAGNOSIS: "Opening patient diagnosis.",
      OPEN_QUESTIONNAIRE: "Opening the cognitive questionnaire.",
      OPEN_CULTURE: "Opening NER memory and culture.",
      OPEN_SOS: "Opening safety and SOS.",
      SHOW_LOCATION: "Opening patient location.",
      SHOW_ALERTS: "Opening safety alerts.",
      SHOW_MEDICATION: "Opening medications and reminders.",
      SHOW_ACTIVITY: "Opening cognitive activity.",
      SHOW_SAFE_ZONES: "Opening safe zones.",
      NAVIGATE_HOME: "Taking you home.",
      HELP: "You can say: go to games, go home, show location, show alerts, show medications, show activity, show safe zones, call caregiver, or help."
    }};

    if (intent === "CALL_CAREGIVER") {{
      if (!caregiverPhone) {{
        setStatus("⚠️ No caregiver phone number is available in your trusted contacts.");
        speak("I could not find a caregiver phone number.");
        mode = "wake";
        return;
      }}
      mode = "confirmation";
      setStatus("⚠️ Do you want me to call your caregiver? Say yes or no.");
      speak("Do you want me to call your caregiver? Say yes or no.");
      return;
    }}

    if (responses[intent]) {{
      setStatus("✅ " + responses[intent]);
      speak(responses[intent]);
    }}

    setTimeout(() => navigate(intent), 500);
  }}

  function handleConfirmation(text) {{
    const t = normalize(text);

    if (/^(yes|yeah|yep|sure|okay|ok|confirm|do it|please do)$/.test(t)) {{
      const safe = String(caregiverPhone).replace(/[^0-9+*#;,]/g, "");
      if (!safe) {{
        setStatus("⚠️ The caregiver phone number is not valid.");
        speak("The caregiver phone number is not valid.");
      }} else {{
        setStatus("📞 Calling your caregiver...");
        speak("Calling your caregiver.");
        window.location.href = "tel:" + safe;
      }}
      mode = "wake";
      return;
    }}

    if (/^(no|nope|cancel|don't|do not)$/.test(t)) {{
      setStatus("Okay. I will not call.");
      speak("Okay. I will not call.");
      mode = "wake";
      return;
    }}

    setStatus("Please say yes or no.");
    speak("Please say yes or no.");
  }}

  function processText(text) {{
    if (!text) return;

    if (mode === "confirmation") {{
      handleConfirmation(text);
      return;
    }}

    if (mode === "wake") {{
      if (!containsWakeWord(text)) return;

      const after = commandTextAfterWake(text);
      setStatus("👂 Yes, how can I help?");
      speak("Yes, how can I help?");

      if (after) {{
        const intent = intentFrom(after);
        if (intent) {{
          executeIntent(intent);
        }} else {{
          setStatus("I heard you, but I did not understand that command.");
          speak("Sorry, I did not understand that command.");
        }}
      }} else {{
        // Listen for the next utterance after the wake word.
        mode = "command";
        setStatus("🎤 Listening for your command...");
        speak("How can I help?");
      }}
      return;
    }}

    if (mode === "command") {{
      const intent = intentFrom(text);
      if (intent) {{
        executeIntent(intent);
      }} else {{
        setStatus("Sorry, I did not understand that command.");
        speak("Sorry, I did not understand that command.");
      }}
      mode = "wake";
    }}
  }}

  function createRecognition() {{
    if (!SpeechRecognition) {{
      setStatus(
        "❌ Voice recognition is not supported in this browser. Please use the latest Chrome or Edge."
      );
      enableBtn.disabled = false;
      return false;
    }}

    recognition = new SpeechRecognition();
    recognition.lang = "en-IN";
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {{
      restarting = false;
      orb.classList.add("listening");
      if (mode === "wake") {{
        setStatus("👂 Listening for “Hey Neuro”... You do not need to press the microphone again.");
      }}
    }};

    recognition.onresult = (event) => {{
      let finalText = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {{
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {{
          finalText += " " + transcript;
        }}
      }}

      if (finalText.trim()) {{
        processText(finalText.trim());
      }}
    }};

    recognition.onerror = (event) => {{
      restarting = false;
      const error = event.error || "unknown";

      if (error === "not-allowed" || error === "service-not-allowed") {{
        enabled = false;
        enableBtn.disabled = false;
        stopBtn.classList.add("neura-hidden");
        orb.classList.remove("listening");
        setStatus("❌ Microphone permission was blocked. Click the 🔒 icon near the browser address bar, allow Microphone, then click Enable Voice again.");
        return;
      }}

      if (error === "audio-capture") {{
        setStatus("❌ No microphone was detected. Check your Windows microphone and browser permissions.");
        return;
      }}

      if (error !== "aborted") {{
        setStatus("⚠️ Voice recognition paused (" + error + "). Trying again...");
      }}
    }};

    recognition.onend = () => {{
      orb.classList.remove("listening");

      if (enabled && !paused && !restarting) {{
        restarting = true;
        setTimeout(() => {{
          if (!enabled || paused) return;
          try {{
            recognition.start();
          }} catch (_) {{
            restarting = false;
            setTimeout(() => {{
              if (enabled && !paused) {{
                try {{ recognition.start(); }} catch (__) {{}}
              }}
            }}, 800);
          }}
        }}, 250);
      }}
    }};

    return true;
  }}

  function startVoice() {{
    if (!createRecognition()) return;

    enabled = true;
    paused = false;
    mode = "wake";

    enableBtn.disabled = true;
    stopBtn.classList.remove("neura-hidden");

    setStatus("🎤 Starting microphone... Please allow microphone access if your browser asks.");

    try {{
      recognition.start();
    }} catch (error) {{
      enabled = false;
      enableBtn.disabled = false;
      stopBtn.classList.add("neura-hidden");
      setStatus("❌ Could not start voice recognition. Please refresh the page and try again.");
    }}
  }}

  function stopVoice() {{
    enabled = false;
    paused = true;
    restarting = false;

    if (recognition) {{
      try {{ recognition.stop(); }} catch (_) {{}}
    }}

    if ("speechSynthesis" in window) {{
      try {{ window.speechSynthesis.cancel(); }} catch (_) {{}}
    }}

    orb.classList.remove("listening");
    enableBtn.disabled = false;
    stopBtn.classList.add("neura-hidden");
    setStatus("⏸ Hey Neuro is paused. Press Enable Voice to start again.");
  }}

  enableBtn.addEventListener("click", startVoice);
  stopBtn.addEventListener("click", stopVoice);

  // Helpful status if the browser does not expose SpeechRecognition.
  if (!SpeechRecognition) {{
    setStatus("⚠️ Voice recognition is unavailable in this browser. Please open SYNAPTIQ in the latest Chrome or Microsoft Edge.");
  }}
}})();
</script>
</body>
</html>
        """,
        height=285,
        scrolling=False,
    )

st.set_page_config(
    page_title="SYNAPTIQ — Memory Assistance",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")

# Geofence caregiver calling uses the same browser `tel:` mechanism as the SOS page.
# No paid SMS/telephony provider is required.

# --------------------------------------------------
# LOAD GROQ API KEY FROM .env OR SYSTEM ENVIRONMENT
# --------------------------------------------------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    _err = (
        "❌ Missing GROQ_API_KEY: A valid Groq Cloud API key is required.\n\n"
        "Please create a .env file from .env.example:\n"
        "  cp .env.example .env\n"
        "and set GROQ_API_KEY=your_key_here\n"
        "or export GROQ_API_KEY in your environment before starting."
    )
    st.error(_err)
    sys.stderr.write(f"\n{_err}\n\n")
    st.stop()

if Groq is None:
    _err = "❌ Missing Package: 'groq' SDK is not installed. Please install it with: pip install groq"
    st.error(_err)
    sys.stderr.write(f"\n{_err}\n\n")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Caregiver phone numbers are taken from the trusted contacts saved in the SOS/Caregiver Portal.

# =========================
# USER DATABASE (DEMO)
# =========================
USERS = {
    "doctor": "1234",
    "admin": "admin123",
    "patient": "test",
    "caregiver": "care123"
}

# =========================
# SESSION STATE INIT
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

import time as _cg_time

# Chat / navigation state is initialised once, safely, before any page runs.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "🏠 Home"

# Apply Hey Neuro navigation immediately after session state exists.
# The previous version defined process_neura_action() but never called it.
process_neura_action()

if "chat_pending" not in st.session_state:
    st.session_state.chat_pending = None

if "q_step" not in st.session_state:
    st.session_state.q_step = 1

# =========================
# GLOBAL LANGUAGE / LOCALIZATION
# =========================
# The app can display its user-facing text in a Northeast India language
# selected by the user. Translations are generated/cached with Groq so the
# same codebase can support many languages without hard-coding thousands of
# strings. If Groq is unavailable, the app safely falls back to English.
NE_LANGUAGE_OPTIONS = [
    "English",
    "Assamese",
    "Meitei (Manipuri)",
    "Khasi",
]

# Native-language name followed by the English name.  The internal values
# above remain unchanged so the rest of the application can still use the
# language names consistently for Groq translation and voice settings.
LANGUAGE_DISPLAY_NAMES = {
    "English": "English - English",
    "Assamese": "অসমীয়া - Assamese",
    "Meitei (Manipuri)": "মৈতৈলোন্ - Meitei (Manipuri)",
    "Khasi": "Khasi - Khasi",
}

def normalize_language(value):
    """Convert a stored display label or internal language name to the internal name."""
    if value in NE_LANGUAGE_OPTIONS:
        return value
    for internal_name, display_name in LANGUAGE_DISPLAY_NAMES.items():
        if value == display_name:
            return internal_name
    return "English"


def _language_selector_changed():
    """Keep the dropdown interactive, but keep the actual application in English."""
    # The dropdown is intentionally cosmetic for now.  Its selected value is
    # kept by Streamlit under ``language_selector``, but it must NEVER change
    # the application's real UI language.
    st.session_state.app_language = "English"
    st.session_state.translation_prepared_for = "English"
    st.session_state.translation_cache = {}

LANGUAGE_CODES = {
    "English": "en-IN",
    "Assamese": "as-IN",
    "Bengali": "bn-IN",
    "Bodo": "brx-IN",
    "Meitei (Manipuri)": "mni-IN",
    "Khasi": "kha-IN",
    "Garo": "grt-IN",
    "Mizo": "lus-IN",
    "Kokborok (Tripuri)": "trp-IN",
    "Nagamese": "nag-IN",
    "Karbi": "mjw-IN",
    "Dimasa": "dis-IN",
    "Ao": "njo-IN",
    "Angami": "nag-IN",
    "Sumi": "nsm-IN",
    "Lotha": "njh-IN",
    "Hmar": "hmr-IN",
    "Kuki": "ctd-IN",
    "Tiwa": "lax-IN",
    "Rabha": "rah-IN",
    "Jaintia": "kha-IN",
    "Chakma": "ccp-IN",
    "Nyishi": "njz-IN",
    "Adi": "adi-IN",
    "Apatani": "apt-IN",
    "Monpa": "mnp-IN",
}

# English-only mode: the dropdown remains visible, but the application's
# actual UI language is permanently English.
st.session_state.app_language = "English"

if "translation_cache" not in st.session_state:
    st.session_state.translation_cache = {}

# The language is selected before login so the complete login page can be
# displayed in the user's chosen Northeast Indian language.
APP_LANGUAGE = st.session_state.app_language
APP_LANG_CODE = LANGUAGE_CODES.get(APP_LANGUAGE, "en-IN")

# ================================================================
# OFFLINE UI TRANSLATION SYSTEM
# ================================================================
# The website UI must never depend on Groq.  The Groq client above is
# still available for the AI/chat features, but UI translation is local,
# deterministic and instant.
#
# Exact phrases are preferred.  A small word-level fallback is then used
# for dynamic sentences that are assembled with f-strings.
# ================================================================

LOCAL_TRANSLATIONS = {
    "Assamese": {
        "Home": "গৃহ", "My Health": "মোৰ স্বাস্থ্য", "My Memories": "মোৰ স্মৃতিসমূহ",
        "Activities": "কাৰ্যকলাপ", "Safety": "সুৰক্ষা", "Care & Help": "যত্ন আৰু সহায়",
        "Insights": "অন্তৰ্দৃষ্টি", "Emergency Help": "জৰুৰীকালীন সহায়", "Dark Mode": "ডাৰ্ক মোড",
        "Sign Out": "ছাইন আউট", "Choose Language": "ভাষা বাছনি কৰক", "User name": "ব্যৱহাৰকাৰীৰ নাম",
        "Password": "পাছৱৰ্ড", "Sign In": "ছাইন ইন", "Welcome to SYNAPTIQ": "SYNAPTIQ লৈ স্বাগতম",
        "Check My Cognitive Health": "মোৰ জ্ঞানীয় স্বাস্থ্য পৰীক্ষা কৰক",
        "Start My Check": "মোৰ পৰীক্ষা আৰম্ভ কৰক", "My Memories": "মোৰ স্মৃতিসমূহ",
        "Photos, names and stories of the people and places you love, kept in one warm place.": "আপুনি ভালপোৱা মানুহ আৰু ঠাইৰ ফটো, নাম আৰু কাহিনীবোৰ একে ঠাইত সুৰক্ষিত ৰাখক।",
        "Open My Memories": "মোৰ স্মৃতিসমূহ খোলক", "Brain Games": "মগজুৰ খেল",
        "Short, friendly activities for attention, memory and quick thinking.": "মনোযোগ, স্মৃতি আৰু দ্ৰুত চিন্তাৰ বাবে সৰু আৰু সহজ কাৰ্যকলাপ।",
        "Cognitive Games": "জ্ঞানীয় খেল", "Play Brain Games": "মগজুৰ খেল খেলক",
        "My Reminders": "মোৰ সোঁৱৰাই দিয়া", "Smart Reminders": "স্মাৰ্ট সোঁৱৰাই দিয়া",
        "See My Reminders": "মোৰ সোঁৱৰাই দিয়াসমূহ চাওক", "Safety & SOS": "সুৰক্ষা আৰু SOS",
        "Call a trusted person quickly and let your family know that you are safe.": "বিশ্বাসযোগ্য ব্যক্তিলৈ দ্ৰুত ফোন কৰক আৰু আপোনাৰ পৰিয়ালক আপুনি সুৰক্ষিত বুলি জনাওক।",
        "Open Safety Help": "সুৰক্ষা সহায় খোলক", "Caregiver Support": "যত্নদাতাৰ সহায়",
        "Caregiver Portal": "যত্নদাতা প'ৰ্টেল", "Tools for family and caregivers: patient details, contacts, progress and location.": "পৰিয়াল আৰু যত্নদাতাৰ বাবে সঁজুলি: ৰোগীৰ তথ্য, যোগাযোগ, অগ্ৰগতি আৰু অৱস্থান।",
        "Open Caregiver Support": "যত্নদাতাৰ সহায় খোলক", "Ask a question": "এটা প্ৰশ্ন সোধক",
        "Type a question in simple words and get a calm, clear answer.": "সহজ শব্দত এটা প্ৰশ্ন লিখক আৰু শান্ত, স্পষ্ট উত্তৰ লাভ কৰক।",
        "AI Chatbot": "AI চেটবট", "Ask SYNAPTIQ": "SYNAPTIQ-ক সোধক",
        "Speak instead of typing": "টাইপ কৰাৰ সলনি কথা কওক",
        "Press one big button and talk. SYNAPTIQ listens and replies.": "এটা ডাঙৰ বুটাম টিপি কথা কওক। SYNAPTIQ-এ শুনি উত্তৰ দিব।",
        "Voice Assistant": "ভইচ সহায়ক", "Tap to Speak": "কথা ক'বলৈ টিপক",
        "Familiar places": "চিনাকি ঠাই", "Photos of places you have visited, with a gentle recognition activity.": "আপুনি ভ্ৰমণ কৰা ঠাইৰ ফটো আৰু চিনাক্তকৰণৰ সহজ কাৰ্যকলাপ।",
        "NER Memory & Culture": "NER স্মৃতি আৰু সংস্কৃতি", "Open Places": "ঠাইসমূহ খোলক",
        "Model Analytics": "মডেল বিশ্লেষণ", "Visualizations": "দৃশ্যায়ন",
        "Model performance summary": "মডেলৰ কাৰ্যক্ষমতাৰ সাৰাংশ", "System Architecture": "চিষ্টেম আৰ্হি",
        "Model Comparison": "মডেল তুলনা", "Course Units Covered": "অন্তৰ্ভুক্ত পাঠ এককসমূহ",
        "Memory Questionnaire": "স্মৃতি প্ৰশ্নাৱলী", "Cognitive Check": "জ্ঞানীয় পৰীক্ষা",
        "My Progress": "মোৰ অগ্ৰগতি", "Cognitive Performance": "জ্ঞানীয় কাৰ্যক্ষমতা",
        "Risk Timeline": "ঝুঁকি সময়ৰেখা", "My Risk Timeline": "মোৰ ঝুঁকি সময়ৰেখা",
        "Places & Culture": "ঠাই আৰু সংস্কৃতি", "GPS Safety & Geofencing": "GPS সুৰক্ষা আৰু জিঅ'ফেন্সিং",
        "My Location": "মোৰ অৱস্থান", "My Reminders": "মোৰ সোঁৱৰাই দিয়াসমূহ",
        "Speak to SYNAPTIQ": "SYNAPTIQ-ৰ সৈতে কথা কওক",
        "🌐 Choose Language": "🌐 ভাষা বাছনি কৰক", "👤 User name": "👤 ব্যৱহাৰকাৰীৰ নাম",
        "🔑 Password": "🔑 পাছৱৰ্ড", "🚪 Sign In": "🚪 ছাইন ইন", "↪️ Sign Out": "↪️ ছাইন আউট",
        "🆘 Emergency Help": "🆘 জৰুৰীকালীন সহায়", "🌙 Dark Mode": "🌙 ডাৰ্ক মোড",
        "🏠 Back to Home": "🏠 গৃহলৈ উভতি যাওক", "📍 You are here": "📍 আপুনি ইয়াত আছে",
        "About You": "আপোনাৰ বিষয়ে", "Memory & Thinking": "স্মৃতি আৰু চিন্তা",
        "Health Information": "স্বাস্থ্যৰ তথ্য", "Save & Continue": "সংৰক্ষণ কৰি আগবাঢ়ক",
        "Previous Step": "পূৰ্বৰ ধাপ", "Next Step": "পৰৱৰ্তী ধাপ",
        "Ready for your Cognitive Check": "আপোনাৰ জ্ঞানীয় পৰীক্ষাৰ বাবে সাজু",
        "Go to My Cognitive Check": "মোৰ জ্ঞানীয় পৰীক্ষালৈ যাওক",
        "Open the Memory Questionnaire": "স্মৃতি প্ৰশ্নাৱলী খোলক",
        "Use official MMSE score": "অফিচিয়েল MMSE স্ক'ৰ ব্যৱহাৰ কৰক",
        "Use official CDR rating": "অফিচিয়েল CDR ৰেটিং ব্যৱহাৰ কৰক",
        "Patient Location Monitoring": "ৰোগীৰ অৱস্থান নিৰীক্ষণ",
        "GPS accuracy": "GPS সঠিকতা", "Safety status": "সুৰক্ষা অৱস্থা",
        "Check where I am now": "এতিয়া মই ক'ত আছোঁ পৰীক্ষা কৰক",
        "Set the safe area (for caregivers)": "নিৰাপদ এলেকা নিৰ্ধাৰণ কৰক (যত্নদাতাৰ বাবে)",
        "The safe area is usually the patient's home. Use the last known location as the centre.": "নিৰাপদ এলেকাটো সাধাৰণতে ৰোগীৰ ঘৰ। শেষবাৰৰ জনা অৱস্থানক কেন্দ্ৰ হিচাপে ব্যৱহাৰ কৰক।",
        "Save Safe Area": "নিৰাপদ এলেকা সংৰক্ষণ কৰক", "Recent location updates": "শেহতীয়া অৱস্থান আপডেট",
        "Inside Safe Zone": "নিৰাপদ এলেকাৰ ভিতৰত", "Outside safe zone": "নিৰাপদ এলেকাৰ বাহিৰত",
        "Safe zone is not configured.": "নিৰাপদ এলেকা নিৰ্ধাৰণ কৰা হোৱা নাই।",
        "No location updates recorded yet.": "এতিয়ালৈ কোনো অৱস্থান আপডেট সংৰক্ষিত হোৱা নাই।",
        "Patient is inside the configured safe zone.": "ৰোগী নিৰ্ধাৰিত নিৰাপদ এলেকাৰ ভিতৰত আছে।",
        "Emergency Help": "জৰুৰীকালীন সহায়", "Call Someone I Trust": "মই বিশ্বাস কৰা ব্যক্তিলৈ ফোন কৰক",
        "Where am I?": "মই ক'ত আছোঁ?", "Check My Safety Location": "মোৰ সুৰক্ষা অৱস্থান পৰীক্ষা কৰক",
        "Open Caregiver Support": "যত্নদাতাৰ সহায় খোলক", "Start Game": "খেল আৰম্ভ কৰক",
        "Start Game": "খেল আৰম্ভ কৰক", "Save My Result": "মোৰ ফলাফল সংৰক্ষণ কৰক", "Play Again": "আকৌ খেলক",
        "See My Progress": "মোৰ অগ্ৰগতি চাওক", "Read this result aloud": "এই ফলাফলটো উচ্চস্বৰে পঢ়ক",
        "Read the answer aloud": "উত্তৰটো উচ্চস্বৰে পঢ়ক", "Type my question instead": "তাৰ পৰিৱৰ্তে মোৰ প্ৰশ্নটো লিখক",
        "Latest MMSE-style score": "শেহতীয়া MMSE-ধৰণৰ স্ক'ৰ", "Latest CDR-style rating": "শেহতীয়া CDR-ধৰণৰ ৰেটিং",
        "Latest model confidence": "শেহতীয়া মডেলৰ বিশ্বাসযোগ্যতা", "Game accuracy": "খেলৰ সঠিকতা",
        "What this shows": "ই কি দেখুৱায়", "Please try again later.": "অনুগ্ৰহ কৰি পিছত আকৌ চেষ্টা কৰক।",
        "Nothing recorded yet": "এতিয়ালৈ একো সংৰক্ষিত হোৱা নাই", "Simple View": "সহজ দৃশ্য", "Detailed View": "বিৱৰণসহ দৃশ্য",
        "Current app language": "বৰ্তমান এপৰ ভাষা", "Made for North East India": "উত্তৰ-পূব ভাৰতৰ বাবে নিৰ্মিত",
        "AI-assisted screening support · not a medical diagnosis": "AI-সহায়িত স্ক্ৰীনিং সহায় · চিকিৎসা নিৰ্ণয় নহয়",
        "Please allow location access and try again.": "অনুগ্ৰহ কৰি অৱস্থানৰ অনুমতি দিয়ক আৰু আকৌ চেষ্টা কৰক।",
        "Something went wrong. Please try again.": "কিবা এটা ভুল হৈছে। অনুগ্ৰহ কৰি আকৌ চেষ্টা কৰক।",
        "Please enter a valid latitude and longitude.": "অনুগ্ৰহ কৰি বৈধ latitude আৰু longitude লিখক।",
        "Enable browser notifications": "ব্ৰাউজাৰ জাননী সক্ৰিয় কৰক", "Got it": "বুজিলোঁ",
        "Synaptiq Reminder": "Synaptiq সোঁৱৰাই দিয়া", "Reaction speed": "প্ৰতিক্ৰিয়াৰ গতি",
        "Attention": "মনোযোগ", "Visual memory": "দৃশ্য স্মৃতি", "Planning": "পৰিকল্পনা",
        "Memory activity": "স্মৃতি কাৰ্যকলাপ", "Family call": "পৰিয়াললৈ ফোন",
        "Welcome! Opening your home screen…": "স্বাগতম! আপোনাৰ গৃহ পৰ্দা খোলা হৈছে…",
        "Please type your user name and password, then press the green button.": "অনুগ্ৰহ কৰি আপোনাৰ ব্যৱহাৰকাৰীৰ নাম আৰু পাছৱৰ্ড লিখি সেউজীয়া বুটামটো টিপক।",
        "Need help signing in?": "ছাইন ইন কৰাত সহায় লাগে নেকি?",
        "That user name or password did not match. Please try again.": "ব্যৱহাৰকাৰীৰ নাম বা পাছৱৰ্ড মিল নাখালে। অনুগ্ৰহ কৰি আকৌ চেষ্টা কৰক।",
        "Check My Cognitive Health": "মোৰ জ্ঞানীয় স্বাস্থ্য পৰীক্ষা কৰক",
        "How you are doing over time": "সময়ৰ সৈতে আপোনাৰ অগ্ৰগতি",
        "Score out of 30": "৩০ৰ ভিতৰত স্ক'ৰ", "Accuracy (%)": "সঠিকতা (%)",
        "CDR-style rating": "CDR-ধৰণৰ ৰেটিং", "Brain Game Accuracy Trend": "মগজুৰ খেলৰ সঠিকতাৰ ধাৰা",
        "Memory activity": "স্মৃতি কাৰ্যকলাপ", "Family call": "পৰিয়াললৈ ফোন",
        "🌏 Made for North East India": "🌏 উত্তৰ-পূব ভাৰতৰ বাবে নিৰ্মিত",
        "👨‍👩‍👧 Family &amp; caregiver friendly": "👨‍👩‍👧 পৰিয়াল আৰু যত্নদাতাৰ বাবে উপযোগী",
        "👨‍👩‍👧 Family & caregiver friendly": "👨‍👩‍👧 পৰিয়াল আৰু যত্নদাতাৰ বাবে উপযোগী",
        "🗣️ Available in many languages": "🗣️ বহু ভাষাত উপলব্ধ",
        "How to use this app:": "এই এপটো কেনেকৈ ব্যৱহাৰ কৰিব:",
        "tap any large card below, or use the big buttons in the menu on the left. You can always come back here with the Back to Home button at the top.": "তলৰ যিকোনো ডাঙৰ কাৰ্ড টিপক, অথবা বাওঁফালৰ মেনুৰ ডাঙৰ বুটামসমূহ ব্যৱহাৰ কৰক। ওপৰৰ গৃহলৈ উভতি যাওক বুটামটো টিপি আপুনি সদায় ইয়ালৈ উভতি আহিব পাৰে।",
    },
    "Khasi": {
        "Home": "Ka Ing", "My Health": "Ka Koit Ka Khiah Jong Nga", "My Memories": "Ki Jingkynmaw Jong Nga",
        "Activities": "Ki Kam", "Safety": "Ka Jingshngaiñ", "Care & Help": "Ka Jingsumar bad Jingiarap",
        "Insights": "Ki Jingtip", "Emergency Help": "Ka Jingiarap Ha Ka Jingma", "Dark Mode": "Ka Dur Dum",
        "Sign Out": "Sei", "Choose Language": "Jied Ka Ktien", "User name": "Ka Kyrteng Pyndonkam",
        "Password": "Ka Password", "Sign In": "Rung", "Welcome to SYNAPTIQ": "Pdiang sngewbha sha SYNAPTIQ",
        "Check My Cognitive Health": "Peit ïa ka Koit Ka Khiah jong ka Jingmut",
        "Start My Check": "Sdang ïa ka Jingpeit", "Open My Memories": "Plie ïa Ki Jingkynmaw Jong Nga",
        "Brain Games": "Ki Jingiakop Jong Ka Jingmut", "Cognitive Games": "Ki Jingiakop Jingmut",
        "Play Brain Games": "Leit Iakop", "My Reminders": "Ki Jingpynkynmaw Jong Nga",
        "Smart Reminders": "Ki Jingpynkynmaw Bniah", "See My Reminders": "Peit ïa Ki Jingpynkynmaw",
        "Safety & SOS": "Ka Jingshngaiñ bad SOS", "Open Safety Help": "Plie Ka Jingiarap Jingshngaiñ",
        "Caregiver Support": "Ka Jingiarap ïa Nongsumar", "Caregiver Portal": "Ka Portal Nongsumar",
        "Open Caregiver Support": "Plie Ka Jingiarap Nongsumar", "Ask a question": "Kylli Ka Jingkylli",
        "AI Chatbot": "AI Chatbot", "Ask SYNAPTIQ": "Kylli ïa SYNAPTIQ",
        "Voice Assistant": "Nongiarap Sur", "Tap to Speak": "Pynkhih ban Kren",
        "NER Memory & Culture": "Ka Jingkynmaw bad Dustur NER", "Open Places": "Plie Ki Jaka",
        "Model Analytics": "Ka Jingbishar ïa u Model", "Visualizations": "Ki Dur Jingtip",
        "Model performance summary": "Ka Jingkhein ïa ka Kam U Model", "System Architecture": "Ka rukom treikam System",
        "Model Comparison": "Iakhar ïa Ki Model", "Memory Questionnaire": "Ki Jingkylli Jingkynmaw",
        "Cognitive Check": "Ka Jingpeit Jingmut", "My Progress": "Ka Jingiaid Shakhmat Jong Nga",
        "Risk Timeline": "Ka Por Jingma", "My Risk Timeline": "Ka Por Jingma Jong Nga",
        "Places & Culture": "Ki Jaka bad Dustur", "My Location": "Ka Jaka Jong Nga",
        "GPS Safety & Geofencing": "GPS Jingshngaiñ bad Geofencing", "Speak to SYNAPTIQ": "Kren sha SYNAPTIQ",
        "About You": "Shaphang Jong Phi", "Memory & Thinking": "Jingkynmaw bad Jingpyrkhat",
        "Health Information": "Ka Jingtip shaphang ka Koit Ka Khiah", "Save & Continue": "Thoh Stok bad Nangta",
        "Previous Step": "Ka Mawphor Ba Mynshwa", "Next Step": "Ka Mawphor Ba Hadien",
        "Patient Location Monitoring": "Peit ïa ka Jaka U Nongpang", "GPS accuracy": "Ka Jingbeit GPS",
        "Safety status": "Ka Sthit Jingshngaiñ", "Check where I am now": "Peit Hangno Nga Don mynta",
        "Set the safe area (for caregivers)": "Pynbeit ïa ka Jaka Shngaiñ (na bynta Nongsumar)",
        "Save Safe Area": "Stok ïa ka Jaka Shngaiñ", "Recent location updates": "Ki Jingpynkylla Jaka Ba Dangshen",
        "Inside Safe Zone": "Hapoh Jaka Shngaiñ", "Outside safe zone": "Shabar Jaka Shngaiñ",
        "Safe zone is not configured.": "Ym pat pynbeit ïa ka Jaka Shngaiñ.",
        "No location updates recorded yet.": "Ym pat don ka jingpynkylla jaka ba la buh.",
        "Patient is inside the configured safe zone.": "U Nongpang u don hapoh ka Jaka Shngaiñ.",
        "Emergency Help": "Ka Jingiarap Ha Ka Jingma", "Call Someone I Trust": "Phone ïa Uba Nga Shaniah",
        "Where am I?": "Hangno Nga Don?", "Check My Safety Location": "Peit ïa Ka Jaka Shngaiñ Jong Nga",
        "Start Game": "Sdang Ka Iakop", "Save My Result": "Stok ïa Ka Result Jong Nga", "Play Again": "Iakop Biang",
        "See My Progress": "Peit ïa Ka Jingiaid Shakhmat Jong Nga", "Current app language": "Ka Ktien App mynta",
        "Made for North East India": "La Pynlong Na Ka bynta North East India",
        "Enable browser notifications": "Pynlong ïa Ki Jingpynbna Browser", "Got it": "Nga Sngewthuh",
        "Reaction speed": "Ka Jingsted Ka Reaction", "Attention": "Ka Jingpeitngor", "Visual memory": "Jingkynmaw Dur",
        "Planning": "Pynbeit lynti", "Memory activity": "Kam Jingkynmaw", "Family call": "Phone ïa Ka ïing",
        "What this shows": "Kaei Kane Ka Dawa", "Simple View": "Peit Suk", "Detailed View": "Peit Bniah",
        "Nothing recorded yet": "Ym pat don eiei ba la buh", "Score out of 30": "Result na 30",
        "Accuracy (%)": "Ka Jingbeit (%)", "CDR-style rating": "CDR-ryntih Rating",
    },
    "Meitei (Manipuri)": {
        "Home": "নুংশিং", "My Health": "ঐগী অরোগ্য", "My Memories": "ঐগী শিংথাশিং",
        "Activities": "থৌরমশিং", "Safety": "শিংশাবা", "Care & Help": "শিংথানবা অমসুং মতম",
        "Insights": "মশিং থৌরম", "Emergency Help": "অহানবা মতমগী মপুং", "Dark Mode": "অন্ধকার মোদ",
        "Sign Out": "শিংথোকপা", "Choose Language": "লোল খনবিয়ু", "User name": "ইউজর মিং",
        "Password": "পাসওয়ার্ড", "Sign In": "শিংশিনবা", "Welcome to SYNAPTIQ": "SYNAPTIQ দা তারাম্না তৌজরি",
        "Check My Cognitive Health": "ঐগী চিন্তন অরোগ্য যেংউ",
        "Start My Check": "ঐগী যেংবা হৌউ", "Open My Memories": "ঐগী শিংথাশিং হাংউ",
        "Brain Games": "মাইন্ড গেমশিং", "Cognitive Games": "চিন্তন গেমশিং", "Play Brain Games": "মাইন্ড গেম খেলউ",
        "My Reminders": "ঐগী খংহনবশিং", "Smart Reminders": "স্মার্ট খংহনবা",
        "See My Reminders": "ঐগী খংহনবশিং যেংউ", "Safety & SOS": "শিংশাবা অমসুং SOS",
        "Open Safety Help": "শিংশাবা মতম হাংউ", "Caregiver Support": "শিংথানবা মতম",
        "Caregiver Portal": "শিংথানবা পোর্টল", "Open Caregiver Support": "শিংথানবা মতম হাংউ",
        "Ask a question": "অমা হংউ", "AI Chatbot": "AI চেটবট", "Ask SYNAPTIQ": "SYNAPTIQ দা হংউ",
        "Voice Assistant": "ভইস সহায়ক", "Tap to Speak": "খংনবা থিংউ",
        "NER Memory & Culture": "NER শিংথা অমসুং সংস্কৃতি", "Open Places": "ফুরূপশিং হাংউ",
        "Model Analytics": "মোদেল এনালিটিক্স", "Visualizations": "ভিজুয়ালাইজেশনশিং",
        "Model performance summary": "মোদেল থৌরমগী সারাংশ", "System Architecture": "সিস্টেম মকোক",
        "Model Comparison": "মোদেল শেমদোকপা", "Memory Questionnaire": "শিংথা প্রশ্নাৱলী",
        "Cognitive Check": "চিন্তন যেংবা", "My Progress": "ঐগী মখা চলবা",
        "Risk Timeline": "রিস্ক টাইমলাইন", "My Risk Timeline": "ঐগী রিস্ক টাইমলাইন",
        "Places & Culture": "ফুরূপ অমসুং সংস্কৃতি", "My Location": "ঐগী ফুরূপ",
        "GPS Safety & Geofencing": "GPS শিংশাবা অমসুং জিওফেন্সিং", "Speak to SYNAPTIQ": "SYNAPTIQ দা খংউ",
        "About You": "নহাক্কী মতম", "Memory & Thinking": "শিংথা অমসুং চিন্তন",
        "Health Information": "অরোগ্যগী ৱারোল", "Save & Continue": "সেভ তৌ অমসুং মখা চত্তু",
        "Previous Step": "মতুংগী মখা", "Next Step": "মখা চল্লু", "Patient Location Monitoring": "পেশেন্ট ফুরূপ যেংবা",
        "GPS accuracy": "GPS চাং", "Safety status": "শিংশাবগী অৱস্থা",
        "Check where I am now": "এখন ঐ কদাইদা লৈরি যেংউ",
        "Set the safe area (for caregivers)": "শিংশাবগী ফুরূপ শেমউ (কেয়ারগিভরগী মতম)",
        "Save Safe Area": "শিংশাবগী ফুরূপ সেভ তৌ", "Recent location updates": "নুংশি ফুরূপ আপডেটশিং",
        "Inside Safe Zone": "শিংশাবগী ফুরূপগী মনুংদা", "Outside safe zone": "শিংশাবগী ফুরূপগী মপানদা",
        "Safe zone is not configured.": "শিংশাবগী ফুরূপ শেমদে।", "No location updates recorded yet.": "ফুরূপ আপডেট অমত্তা রেকর্ড তৌদে।",
        "Patient is inside the configured safe zone.": "পেশেন্ট শেমখিবা শিংশাবগী ফুরূপগী মনুংদা লৈরি।",
        "Emergency Help": "অহানবা মতমগী মপুং", "Call Someone I Trust": "ঐনা থাজবা মী অমাদা কল তৌ",
        "Where am I?": "ঐ কদাইদা লৈরি?", "Check My Safety Location": "ঐগী শিংশাবগী ফুরূপ যেংউ",
        "Start Game": "গেম হৌউ", "Save My Result": "ঐগী রেজাল্ট সেভ তৌ", "Play Again": "অমুক হান্না খেলউ",
        "See My Progress": "ঐগী মখা চলবা যেংউ", "Current app language": "এপগী হৌজিক্কী লোল",
        "Enable browser notifications": "ব্রাউজর নোটিফিকেশন হৌউ", "Got it": "খংজরি",
        "Reaction speed": "রিঅ্যাকশন স্পিড", "Attention": "মনোযোগ", "Visual memory": "মীৎথা শিংথা",
        "Planning": "শেম-শাবা", "Memory activity": "শিংথা থৌরম", "Family call": "ইমুংদা কল",
        "What this shows": "মসি করিগী উত্লি", "Simple View": "সৰল মচাক", "Detailed View": "অয়াম্বা মচাক",
        "Nothing recorded yet": "হৌজিক ফাওবদা অমত্তা রেকর্ড তৌদে", "Score out of 30": "৩০গী মনুংদা স্কোর",
        "Accuracy (%)": "চাং (%)", "CDR-style rating": "CDR-ধরণ রেটিং",
    },
}


# ------------------------------------------------------------------
# HOME SCREEN PHRASES
# Keep the most frequently visible Home-screen phrases explicitly mapped
# for every supported language.  This prevents the Home page from becoming
# a mixture of one translated language plus English fallback text.
# ------------------------------------------------------------------
LOCAL_TRANSLATIONS["Khasi"].update({
    "Your companion for memory, cognitive health and everyday safety.": "U nongïarap jong phi na bynta ka jingkynmaw, ka koit ka khiah jong ka jingmut bad ka jingshngaiñ man ka sngi.",
    "Family & caregiver friendly": "La pynbeit na bynta ka ïing bad ki nongsumar",
    "Available in many languages": "La don ha bun ki ktien",
    "How to use this app:": "Kumno ban pyndonkam ïa kane ka app:",
    "tap any large card below, ": "khih ïa kino kino ki card kiba heh harum, ",
    "or use the big buttons in the menu on the left. You can always come back here ": "lane pyndonkam ïa ki button kiba heh ha ka menu sha liang kadiang. Phi lah ban wanphai hangne ",
    "with the ": "da ka ",
    " button at the top.": " button ha khlieh.",
    "Answer simple questions about memory and thinking, then see an easy-to-read result.": "Jubab ïa ki jingkylli kiba suk shaphang ka jingkynmaw bad jingpyrkhat, nangta peit ïa ka result kaba suk ban pule.",
    "Photos, names and stories of the people and places you love, kept in one warm place.": "Ki dur, ki kyrteng bad ki jingïathuhkhana jong ki briew bad ki jaka kiba phi ieit, la buh lang ha kawei ka jaka kaba suk.",
    "Short, friendly activities for attention, memory and quick thinking.": "Ki kam kiba lyngkot bad suk na bynta ka jingpeitngor, jingkynmaw bad jingpyrkhat kloi.",
    "Gentle reminders for medicine, meals, water and family calls.": "Ki jingpynkynmaw kiba jem na bynta ki dawai, ki jingïabam, ka um bad ki phone sha ka ïing.",
    "Call a trusted person quickly and let your family know that you are safe.": "Phone kloi ïa u briew uba phi shaniah bad pyntip ïa ka ïing jong phi ba phi shngaiñ.",
    "Tools for family and caregivers: patient details, contacts, progress and location.": "Ki tiar na bynta ka ïing bad ki nongsumar: ki jingtip jong u nongpang, ki contact, ka jingïaid shakhmat bad ka jaka.",
    "Ask a question": "Kylli ka jingkylli",
    "Type a question in simple words and get a calm, clear answer.": "Thoh ïa ka jingkylli da ki ktien kiba suk bad ioh ïa ka jubab kaba shai bad kaba jai jai.",
    "Speak instead of typing": "Kren ha ka jaka ban thoh",
    "Press one big button and talk. SYNAPTIQ listens and replies.": "Khih ïa uwei u button uba heh bad kren. SYNAPTIQ kan sngap bad jubab.",
    "Photos of places you have visited, with a gentle recognition activity.": "Ki dur jong ki jaka kiba phi la leit, ryngkat bad ka kam kaba suk ban ithuh.",
    "⚠️ ": "⚠️ ",
    "Please note:": "Sngewbha kynmaw:",
    "SYNAPTIQ gives AI-assisted screening support and daily memory help. It does not replace an examination by a qualified healthcare professional.": "SYNAPTIQ ka ai jingïarap screening da ka AI bad jingïarap jingkynmaw man ka sngi. Kam bujli ïa ka jingpeit jong u briew uba la tbit ha ka koit ka khiah.",
    "A few more things SYNAPTIQ can do": "Kiwei pat ki katto katne ki kam kiba SYNAPTIQ lah ban leh",
    "Technical details (for developers)": "Ki jingtip teknikal (na bynta ki developer)",
    "Model performance summary": "Ka jingkhein ïa ka kam u model",
    "Best Accuracy": "Ka jingbeit kaba bha tam",
    "Precision": "Precision",
    "Recall": "Recall",
    "F1-Score": "F1-Score",
    "Clinical Records": "Ki record klinikal",
})

# Exact HTML text-node variants used by the Home hero/help cards.
LOCAL_TRANSLATIONS["Khasi"].update({
    "🌏 Made for North East India": "🌏 La pynlong na bynta North East India",
    "👨‍👩‍👧 Family &amp; caregiver friendly": "👨‍👩‍👧 La pynbeit na bynta ka ïing bad ki nongsumar",
    "🗣️ Available in many languages": "🗣️ La don ha bun ki ktien",
    "Your companion for memory, cognitive health and everyday safety.": "U nongïarap jong phi na bynta ka jingkynmaw, ka koit ka khiah jong ka jingmut bad ka jingshngaiñ man ka sngi.",
})


LOCAL_TRANSLATIONS["Meitei (Manipuri)"].update({
    "Your companion for memory, cognitive health and everyday safety.": "শিংথা, চিন্তন অরোগ্য অমসুং নুমিৎথোইদা শিংশাবগী মতমদা নহাক্কী মতমগী সহায়ক।",
    "Family & caregiver friendly": "ইমুং অমসুং কেয়ারগিভরগী মতমদা নুংশিবা",
    "Available in many languages": "য়ামদা লোলশিংদা ফংই",
    "How to use this app:": "অসিগী এপ অসি কাইদৌনা শিজিন্নগদগে:",
    "tap any large card below, ": "মখাদা লৈবা কার্ড অমা থিংউ, ",
    "or use the big buttons in the menu on the left. You can always come back here ": "নত্রগা অয়াম্বা মেনুগী বুটনশিং শিজিন্নউ। নহাক্না মসিদা অমুক হন্না হংজবু ",
    "with the ": "মসিগী ",
    " button at the top.": " বুটনগা।",
    "Answer simple questions about memory and thinking, then see an easy-to-read result.": "শিংথা অমসুং চিন্তনগী মমল থোকপা হায়বা হংজিনবা অমশুং মখাগী ফল শিজিন্না যেংউ।",
    "Photos, names and stories of the people and places you love, kept in one warm place.": "নহাক্না নুংশিবা মীশিং অমসুং ফুরূপশিংগী ফটো, মিং অমসুং ৱারীশিং অমত্তদা শিংথাজবা অমদা থম্মি।",
    "Short, friendly activities for attention, memory and quick thinking.": "মনোযোগ, শিংথা অমসুং হন্না চিন্তন তৌনবা সৰু অমসুং নুংশিবা থৌরমশিং।",
    "Gentle reminders for medicine, meals, water and family calls.": "ওষুধ, চাওমিন, ঈশিং অমসুং ইমুংদা কল তৌনবা মমাংদা খংহনবশিং।",
    "Call a trusted person quickly and let your family know that you are safe.": "থাজবা মী অমাদা তুংইননা কল তৌ অমসুং নহাক শিংশাবদা লৈরি হায়বা ইমুংদা খংজিনউ।",
    "Tools for family and caregivers: patient details, contacts, progress and location.": "ইমুং অমসুং কেয়ারগিভরগী মতমদা পেশেন্টগী ৱারোল, কন্টাক্ট, মখা চলবা অমসুং ফুরূপগী থৌরমশিং।",
    "Ask a question": "হংউ",
    "Type a question in simple words and get a calm, clear answer.": "সহজ লোলদা অমা হংজিনউ অমসুং নুংশিবা অমসুং শিংজবা ৱারোল অমা ফংউ।",
    "Speak instead of typing": "টাইপ তৌবা মখা থাংদনা খংউ",
    "Press one big button and talk. SYNAPTIQ listens and replies.": "বুটন অমা থিংদুনা খংউ। SYNAPTIQ না তাকপা অমসুং হায়জবা তৌই।",
    "Photos of places you have visited, with a gentle recognition activity.": "নহাক্না থাবা ফুরূপশিংগী ফটো অমসুং নুংশিবা ইথিল-ইথিল থাজিনবা থৌরম।",
    "Please note:": "খংজিনবা থোকপা:",
    "SYNAPTIQ gives AI-assisted screening support and daily memory help. It does not replace an examination by a qualified healthcare professional.": "SYNAPTIQ না AI সহায়না screening অমসুং নুমিৎথোইদা শিংথাগী সহায় তৌই। মসি মশীং তৌবা হেলথকেয়র প্রফেশনলগী যেংবা অমগী মখা থাংদে।",
    "A few more things SYNAPTIQ can do": "SYNAPTIQ না তৌবা য়াবা অমুক হান্না থৌরমশিং",
    "Technical details (for developers)": "টেকনিকেল ৱারোল (developerশিংগী মতমদা)",
    "Best Accuracy": "অয়াম্বা চাং",
    "Precision": "Precision",
    "Recall": "Recall",
    "F1-Score": "F1-Score",
    "Clinical Records": "ক্লিনিকেল রেকর্ডশিং",
})

LOCAL_TRANSLATIONS["Meitei (Manipuri)"].update({
    "🌏 Made for North East India": "🌏 North East India-gi maphamda semjaba",
    "👨‍👩‍👧 Family &amp; caregiver friendly": "👨‍👩‍👧 ইমুং অমসুং কেয়ারগিভরশিংগী মতমদা নুংশিবা",
    "🗣️ Available in many languages": "🗣️ লোল অনীমদি ফংই",
})

# Common words used by dynamic UI strings.  Exact phrase translations above
# always win; these words ensure f-string messages also visibly change.
def _local_translate_text(text, language):
    """
    Deterministic OFFLINE UI translation.

    IMPORTANT: only complete phrases from LOCAL_TRANSLATIONS are translated.
    We deliberately do NOT perform word-by-word replacement because that creates
    broken mixed-language sentences. Missing phrases remain in their original
    language instead of being partially translated.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # English is a hard reset: never modify the source text in any way.
    if language == "English":
        return text

    catalog = LOCAL_TRANSLATIONS.get(language, {})

    # Exact phrase match first. This is the safest path for normal labels.
    if text in catalog:
        return catalog[text]

    # For Markdown / HTML text containing several known phrases, replace only
    # complete known phrases. Never translate individual English words.
    leading = text[:len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()):]
    body = text.strip()

    # Longest phrases first.  Never perform word-by-word translation and never
    # translate text that is already written in another supported script.
    # This prevents Assamese/Meitei text from leaking into a Khasi/English page.
    has_assamese = bool(re.search(r"[\u0980-\u09FF]", body))
    has_meitei = bool(re.search(r"[\uABC0-\uABFF]", body))
    if language == "Khasi" and (has_assamese or has_meitei):
        return text
    if language in ("English", "Assamese") and has_meitei:
        return text
    if language == "English" and (has_assamese or has_meitei):
        return text

    for src, dst in sorted(catalog.items(), key=lambda kv: len(kv[0]), reverse=True):
        if src and src in body:
            body = body.replace(src, dst)

    translated = leading + body + trailing
    if translated != text:
        return translated

    # Fallback for UI phrases that were not explicitly added to the local
    # catalog. This makes the remaining interface readable instead of leaving
    # random English fragments. Results are cached per language and phrase.
    # English never reaches this branch because of the hard reset above.
    if client is not None and 1 <= len(body) <= 350 and not re.search(r"[{}<>]", body):
        cache_key = f"ui::{language}::{body}"
        cached = st.session_state.translation_cache.get(cache_key)
        if cached:
            return leading + cached + trailing
        try:
            prompt = (
                f"Translate this user-interface text into {language}. "
                "Return ONLY the translation. Preserve emojis, numbers, punctuation, "
                "names, medical abbreviations, and UI symbols. Do not explain anything.\n\n"
                f"TEXT: {body}"
            )
            resp = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=250,
            )
            translated_body = (resp.choices[0].message.content or body).strip()
            if translated_body:
                st.session_state.translation_cache[cache_key] = translated_body
                return leading + translated_body + trailing
        except Exception:
            pass

    return translated

def _prepare_language(language):
    """Prepare the local catalog for the selected language. No API calls."""
    language = normalize_language(language)
    st.session_state.translation_prepared_for = language
    st.session_state.translation_mode = "local"
    st.session_state.language_revision = st.session_state.get("language_revision", 0) + 1


def tr(text):
    """Return the original English UI text. Multilingual UI is disabled for now."""
    return text

def _apply_language_change(widget_key):
    """Compatibility helper: keep the application language fixed to English."""
    global APP_LANGUAGE, APP_LANG_CODE
    st.session_state.app_language = "English"
    APP_LANGUAGE = "English"
    APP_LANG_CODE = "en-IN"
    st.session_state.translation_prepared_for = "English"
    st.session_state.translation_cache = {}


# Prepare the translation catalog BEFORE any UI is rendered.
# The selector changes st.session_state.app_language and Streamlit reruns the
# script; this block then loads the complete UI catalog for the new language.
# The session-state guard prevents repeated Groq calls on every rerun.
if "translation_prepared_for" not in st.session_state:
    st.session_state.translation_prepared_for = None
if "language_revision" not in st.session_state:
    st.session_state.language_revision = 0

# Prepare the selected catalog before any UI is rendered. English is also
# prepared as a hard-reset state so switching back to English is deterministic.
if st.session_state.translation_prepared_for != APP_LANGUAGE:
    _prepare_language(APP_LANGUAGE)


def localize_html(markup):
    """Keep HTML exactly as authored in English; translation is disabled."""
    return markup

def _loc_value(value):
    return tr(value) if isinstance(value, str) else value


def _loc_widget_kwargs(kwargs):
    kwargs = dict(kwargs)
    if isinstance(kwargs.get("help"), str):
        kwargs["help"] = tr(kwargs["help"])
    if isinstance(kwargs.get("placeholder"), str):
        kwargs["placeholder"] = tr(kwargs["placeholder"])
    return kwargs


def _localized_format_func(original_format):
    if original_format is None:
        return lambda x: tr(str(x)) if isinstance(x, str) else x
    return lambda x: tr(str(original_format(x)))


# Preserve Streamlit's original functions before installing the compatibility wrappers below.
# The UI is English-only, so these wrappers simply pass through the original text.
_original_markdown = st.markdown
_original_write = st.write
_original_title = st.title
_original_subheader = st.subheader
_original_caption = st.caption
_original_info = st.info
_original_warning = st.warning
_original_success = st.success
_original_error = st.error
_original_button = st.button
_original_form_submit_button = st.form_submit_button
_original_text_input = st.text_input
_original_number_input = st.number_input
_original_date_input = st.date_input
_original_time_input = st.time_input
_original_slider = st.slider
_original_selectbox = st.selectbox
_original_radio = st.radio
_original_multiselect = st.multiselect
_original_metric = st.metric
_original_toggle = st.toggle
_original_file_uploader = st.file_uploader
_original_chat_input = st.chat_input
_original_expander = st.expander
_original_text_area = st.text_area
_original_select_slider = st.select_slider
_original_tabs = st.tabs
_original_toast = st.toast


def _selectbox(label, options, *args, **kwargs):
    kwargs = _loc_widget_kwargs(kwargs)
    original_format = kwargs.get("format_func")
    kwargs["format_func"] = _localized_format_func(original_format)
    return _original_selectbox(tr(label), options, *args, **kwargs)


def _radio(label, options, *args, **kwargs):
    kwargs = _loc_widget_kwargs(kwargs)
    original_format = kwargs.get("format_func")
    kwargs["format_func"] = _localized_format_func(original_format)
    return _original_radio(tr(label), options, *args, **kwargs)


def _multiselect(label, options, *args, **kwargs):
    kwargs = _loc_widget_kwargs(kwargs)
    original_format = kwargs.get("format_func")
    kwargs["format_func"] = _localized_format_func(original_format)
    return _original_multiselect(tr(label), options, *args, **kwargs)


def _markdown(body, *args, **kwargs):
    return _original_markdown(localize_html(body) if isinstance(body, str) else body, *args, **kwargs)


def _write(*args, **kwargs):
    localized = tuple(_loc_value(x) for x in args)
    return _original_write(*localized, **kwargs)


def _metric(label, value, *args, **kwargs):
    return _original_metric(tr(label), value, *args, **kwargs)


def _text_input(label, *args, **kwargs):
    return _original_text_input(tr(label), *args, **_loc_widget_kwargs(kwargs))


def _number_input(label, *args, **kwargs):
    return _original_number_input(tr(label), *args, **_loc_widget_kwargs(kwargs))


def _date_input(label, *args, **kwargs):
    return _original_date_input(tr(label), *args, **_loc_widget_kwargs(kwargs))


def _time_input(label, *args, **kwargs):
    return _original_time_input(tr(label), *args, **_loc_widget_kwargs(kwargs))


def _slider(label, *args, **kwargs):
    return _original_slider(tr(label), *args, **_loc_widget_kwargs(kwargs))


def _button(label, *args, **kwargs):
    return _original_button(tr(label), *args, **_loc_widget_kwargs(kwargs))


def _form_submit_button(label, *args, **kwargs):
    return _original_form_submit_button(tr(label), *args, **_loc_widget_kwargs(kwargs))


def _toggle(label, *args, **kwargs):
    return _original_toggle(tr(label), *args, **_loc_widget_kwargs(kwargs))


def _file_uploader(label, *args, **kwargs):
    return _original_file_uploader(tr(label), *args, **_loc_widget_kwargs(kwargs))


def _chat_input(label=None, *args, **kwargs):
    return _original_chat_input(tr(label) if isinstance(label, str) else label, *args, **_loc_widget_kwargs(kwargs))


def _expander(label, *args, **kwargs):
    return _original_expander(tr(label), *args, **_loc_widget_kwargs(kwargs))


def _text_area(label, *args, **kwargs):
    return _original_text_area(tr(label), *args, **_loc_widget_kwargs(kwargs))


def _select_slider(label, options=None, *args, **kwargs):
    kwargs = _loc_widget_kwargs(kwargs)
    original_format = kwargs.get("format_func")
    kwargs["format_func"] = _localized_format_func(original_format)
    return _original_select_slider(
        tr(label), options, *args, **kwargs
    )


def _tabs(tabs, *args, **kwargs):
    # Translate tab labels while preserving the original tab order/values.
    translated_tabs = [tr(tab) if isinstance(tab, str) else tab for tab in tabs]
    return _original_tabs(translated_tabs, *args, **kwargs)


def _toast(body, *args, **kwargs):
    return _original_toast(
        tr(body) if isinstance(body, str) else body,
        *args,
        **kwargs
    )

st.markdown = _markdown
st.write = _write
st.title = lambda text, *args, **kwargs: _original_title(tr(text), *args, **kwargs)
st.subheader = lambda text, *args, **kwargs: _original_subheader(tr(text), *args, **kwargs)
st.caption = lambda text, *args, **kwargs: _original_caption(tr(text), *args, **kwargs)
st.info = lambda text, *args, **kwargs: _original_info(tr(text), *args, **kwargs)
st.warning = lambda text, *args, **kwargs: _original_warning(tr(text), *args, **kwargs)
st.success = lambda text, *args, **kwargs: _original_success(tr(text), *args, **kwargs)
st.error = lambda text, *args, **kwargs: _original_error(tr(text), *args, **kwargs)
st.button = _button
st.form_submit_button = _form_submit_button
st.text_input = _text_input
st.number_input = _number_input
st.date_input = _date_input
st.time_input = _time_input
st.slider = _slider
st.selectbox = _selectbox
st.radio = _radio
st.multiselect = _multiselect
st.metric = _metric
st.toggle = _toggle
st.file_uploader = _file_uploader
st.chat_input = _chat_input
st.expander = _expander
st.text_area = _text_area
st.select_slider = _select_slider
st.tabs = _tabs
st.toast = _toast

# ------------------------------------------------------------------
# IMPORTANT: Streamlit containers (sidebar, columns, forms, tabs, etc.)
# are DeltaGenerator objects. Patching only ``st.markdown`` does NOT
# translate ``column.markdown`` / ``st.sidebar.markdown``.  The original
# app uses those extensively, which is why only some labels changed.
# Patch the common DeltaGenerator display/widget methods as well.
# ------------------------------------------------------------------
try:
    from streamlit.delta_generator import DeltaGenerator

    _DG_original = {}

    def _dg_patch(name, wrapper):
        original = getattr(DeltaGenerator, name, None)
        if original is None:
            return
        _DG_original[name] = original
        setattr(DeltaGenerator, name, wrapper(original))

    def _dg_display_wrapper(original):
        def wrapped(self, body, *args, **kwargs):
            if isinstance(body, str):
                body = localize_html(body)
            return original(self, body, *args, **kwargs)
        return wrapped

    def _dg_label_wrapper(original):
        def wrapped(self, label, *args, **kwargs):
            if isinstance(label, str):
                label = tr(label)
            kwargs = _loc_widget_kwargs(kwargs)
            return original(self, label, *args, **kwargs)
        return wrapped

    def _dg_select_wrapper(original):
        def wrapped(self, label, options, *args, **kwargs):
            kwargs = _loc_widget_kwargs(kwargs)
            fmt = kwargs.get("format_func")
            kwargs["format_func"] = _localized_format_func(fmt)
            return original(self, tr(label), options, *args, **kwargs)
        return wrapped

    for _name in ("markdown", "write"):
        _dg_patch(_name, _dg_display_wrapper)

    for _name in ("title", "subheader", "caption", "info", "warning", "success", "error",
                  "button", "form_submit_button", "text_input", "number_input", "date_input",
                  "time_input", "slider", "metric", "toggle", "file_uploader", "chat_input",
                  "expander", "text_area", "select_slider", "toast"):
        _dg_patch(_name, _dg_label_wrapper)

    for _name in ("selectbox", "radio", "multiselect"):
        _dg_patch(_name, _dg_select_wrapper)

    def _dg_tabs_wrapper(original):
        def wrapped(self, tabs, *args, **kwargs):
            tabs = [tr(x) if isinstance(x, str) else x for x in tabs]
            return original(self, tabs, *args, **kwargs)
        return wrapped

    _dg_patch("tabs", _dg_tabs_wrapper)
except Exception:
    # Some Streamlit versions expose DeltaGenerator differently. The normal
    # st.* wrappers above still provide localization in that case.
    pass

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ══════════════════════════════════════════════════════════════
# THEME  (loaded before the login screen so every screen is styled)
# ══════════════════════════════════════════════════════════════
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

from theme import get_colors, get_css

t = get_colors('dark' if st.session_state.dark_mode else 'light')

# ── Elderly-first, warm, high-contrast palette overrides ───────
# The original theme keys are preserved so every existing inline style,
# Plotly figure and CSS class keeps working — only the values change.
ELDER_LIGHT = dict(
    bg='#fbf7f0', bg2='#f5efe4',
    sb1='#ffffff', sb2='#fbf5ec', sb_border='#e3d8c6',
    card='#ffffff', card2='#fdf8f1',
    hero1='#eaf6f3', hero2='#ffffff',
    border='#d9cdb8', border2='#e7ddcc',
    txt='#22303c', txt2='#4d5c68',
    blue='#0b5f8a', green='#146b45', red='#a8321f',
    purple='#5c3f96', orange='#9a5400', pink='#8a4a55',
    hg1='#0b5f8a', hg2='#5c3f96', hg3='#146b45',
    bg_green='#e3f5e9', bg_blue='#e4f1fa', bg_purple='#efe8fa',
    dem1='#fdeee6', dem2='#fff6ef',
    ndem1='#e6f6ec', ndem2='#f1fbf4',
    glow='#e3f5e9', gmid='#fdf2d6', ghigh='#fdeee6',
    gbg='#fdf8f1', grid='#e7ddcc',
    ins1='#e6f3f7', ins2='#fdf8f1', ins_b='#0b5f8a',
    ban1='#e6f6ec', ban2='#f1fbf4',
    btn1='#0f766e', btn2='#14867c',
    leg_bg='rgba(255,255,255,0.92)',
    pie_line='#ffffff', marker_line='#ffffff',
    cm_0='#fdf8f1', cm_5='#14867c', cm_1='#0b5f8a',
)

ELDER_DARK = dict(
    bg='#131b25', bg2='#18222e',
    sb1='#18222e', sb2='#1e2b38', sb_border='#33445a',
    card='#1e2b38', card2='#26364a',
    hero1='#1d3b46', hero2='#1e2b38',
    border='#3d5266', border2='#33445a',
    txt='#f3f7fb', txt2='#cbd7e3',
    blue='#7dd3fc', green='#6ee7a8', red='#fca5a5',
    purple='#cbb6fd', orange='#fcc48a', pink='#f8b4b4',
    hg1='#7dd3fc', hg2='#cbb6fd', hg3='#6ee7a8',
    bg_green='#1c3b2c', bg_blue='#173449', bg_purple='#2b2450',
    dem1='#3d2020', dem2='#2f1a1a',
    ndem1='#1c3b2c', ndem2='#163024',
    glow='#1c3b2c', gmid='#3d3620', ghigh='#3d2020',
    gbg='#1e2b38', grid='#33445a',
    ins1='#173449', ins2='#1e2b38', ins_b='#7dd3fc',
    ban1='#1c3b2c', ban2='#163024',
    btn1='#0e7c74', btn2='#16a39a',
    leg_bg='rgba(30,43,56,0.92)',
    pie_line='#131b25', marker_line='#131b25',
    cm_0='#1e2b38', cm_5='#16a39a', cm_1='#7dd3fc',
)
t.update(ELDER_DARK if st.session_state.dark_mode else ELDER_LIGHT)

st.markdown(get_css(t), unsafe_allow_html=True)


# ── Elderly-first accessibility / warmth layer ─────────────────
def synaptiq_css(t):
    """Large-type, high-contrast, touch-friendly CSS layer for SYNAPTIQ.

    Written with placeholder tokens instead of an f-string so the CSS braces
    stay readable and cannot be mis-parsed.
    """
    css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

:root{
  --sq-bg: __BG__;
  --sq-card: __CARD__;
  --sq-card2: __CARD2__;
  --sq-border: __BORDER__;
  --sq-border2: __BORDER2__;
  --sq-txt: __TXT__;
  --sq-txt2: __TXT2__;
  --sq-green: __GREEN__;
  --sq-blue: __BLUE__;
  --sq-purple: __PURPLE__;
  --sq-orange: __ORANGE__;
  --sq-red: __RED__;
  --sq-btn1: __BTN1__;
  --sq-btn2: __BTN2__;
  --sq-radius: 22px;
}

/* ---------- Base typography: large and calm ---------- */
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"]{
  font-family: 'Nunito', 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}
[data-testid="stAppViewContainer"], .main, [data-testid="stApp"]{
  background: var(--sq-bg) !important;
}
[data-testid="stAppViewContainer"] .block-container{
  padding-top: 1.6rem; padding-bottom: 3.5rem;
  max-width: 1340px;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span{
  font-size: 1.15rem !important;
  line-height: 1.75 !important;
  color: var(--sq-txt);
}
[data-testid="stMarkdownContainer"] h1{ font-size: 2.35rem !important; font-weight: 900 !important; letter-spacing:-.5px; color: var(--sq-txt); }
[data-testid="stMarkdownContainer"] h2{ font-size: 1.95rem !important; font-weight: 900 !important; color: var(--sq-txt); margin-top:.4rem; }
[data-testid="stMarkdownContainer"] h3{ font-size: 1.5rem !important; font-weight: 800 !important; color: var(--sq-txt); }
[data-testid="stMarkdownContainer"] h4{ font-size: 1.28rem !important; font-weight: 800 !important; color: var(--sq-txt); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p{
  font-size: 1.02rem !important; color: var(--sq-txt2) !important; line-height:1.6 !important;
}
strong, b{ color: var(--sq-txt); }

/* ---------- Buttons: big, rounded, obvious ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button{
  background: linear-gradient(135deg, var(--sq-btn1), var(--sq-btn2)) !important;
  color: #ffffff !important;
  border: 2px solid transparent !important;
  border-radius: 18px !important;
  font-weight: 800 !important;
  font-size: 1.12rem !important;
  min-height: 58px !important;
  padding: .7rem 1.35rem !important;
  line-height: 1.3 !important;
  white-space: normal !important;
  box-shadow: 0 6px 18px rgba(15,118,110,.18) !important;
  transition: transform .16s ease, box-shadow .16s ease !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover{
  transform: translateY(-2px) !important;
  box-shadow: 0 12px 26px rgba(15,118,110,.26) !important;
}
.stButton > button:focus-visible, .stFormSubmitButton > button:focus-visible{
  outline: 4px solid var(--sq-orange) !important; outline-offset: 2px !important;
}
.stButton > button:disabled{ opacity:.5 !important; transform:none !important; box-shadow:none !important; }

/* Quiet navigation buttons (sidebar) */
[data-testid="stSidebar"] .stButton > button{
  background: var(--sq-card) !important;
  color: var(--sq-txt) !important;
  border: 2px solid var(--sq-border) !important;
  box-shadow: none !important;
  text-align: left !important;
}
[data-testid="stSidebar"] .stButton > button:hover{
  border-color: var(--sq-btn1) !important;
  background: var(--sq-card2) !important;
}

/* Emergency button: the invisible anchor element marks the next button as red. */
.element-container:has(.sq-anchor-danger),
div[data-testid="stElementContainer"]:has(.sq-anchor-danger){
  display: none !important;
}
.element-container:has(.sq-anchor-danger) + .element-container .stButton > button,
div[data-testid="stElementContainer"]:has(.sq-anchor-danger) + div[data-testid="stElementContainer"] .stButton > button{
  background: linear-gradient(135deg, #b42318, #d14b2f) !important;
  color: #ffffff !important;
  border-color: transparent !important;
  font-size: 1.3rem !important;
  min-height: 76px !important;
  text-align: center !important;
  box-shadow: 0 10px 26px rgba(180,35,24,.30) !important;
}

/* ---------- Inputs: large and clearly labelled ---------- */
.stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label,
.stRadio label, .stMultiSelect label, .stDateInput label, .stTimeInput label,
.stTextArea label, .stFileUploader label, .stCheckbox label, [data-testid="stWidgetLabel"] p{
  font-size: 1.1rem !important; font-weight: 700 !important; color: var(--sq-txt) !important;
}
.stTextInput input, .stNumberInput input, .stDateInput input, .stTimeInput input, .stTextArea textarea{
  font-size: 1.1rem !important;
  min-height: 52px !important;
  border-radius: 14px !important;
  border: 2px solid var(--sq-border) !important;
  background: var(--sq-card) !important;
  color: var(--sq-txt) !important;
}
.stTextArea textarea{ min-height: 120px !important; }
[data-baseweb="select"] > div{
  min-height: 54px !important; border-radius: 14px !important;
  border: 2px solid var(--sq-border) !important; font-size: 1.08rem !important;
}
.stRadio [role="radiogroup"] label, .stCheckbox label span{ font-size: 1.08rem !important; }
.stRadio [role="radiogroup"]{ gap:.35rem; }
[data-testid="stFileUploaderDropzone"]{
  border: 2px dashed var(--sq-border) !important; border-radius: 18px !important;
  background: var(--sq-card2) !important; padding: 1.1rem !important;
}

/* ---------- Alerts and metrics ---------- */
[data-testid="stAlert"]{
  border-radius: 18px !important; border-width: 2px !important;
  font-size: 1.1rem !important; padding: 1rem 1.1rem !important;
}
[data-testid="stAlert"] p{ font-size: 1.1rem !important; }
[data-testid="stMetric"]{
  background: var(--sq-card); border: 2px solid var(--sq-border2);
  border-radius: 18px; padding: 1rem 1.1rem;
}
[data-testid="stMetricLabel"] p{ font-size: 1rem !important; font-weight:700 !important; color: var(--sq-txt2) !important; }
[data-testid="stMetricValue"]{ font-size: 1.9rem !important; font-weight: 900 !important; color: var(--sq-txt) !important; }

/* ---------- Tabs / expanders ---------- */
.stTabs [data-baseweb="tab"]{
  font-size: 1.1rem !important; font-weight: 800 !important;
  padding: .85rem 1.15rem !important;
}
[data-testid="stExpander"]{
  border: 2px solid var(--sq-border2) !important;
  border-radius: 18px !important; background: var(--sq-card) !important;
  overflow: hidden;
}
[data-testid="stExpander"] summary p, [data-testid="stExpander"] summary{
  font-size: 1.15rem !important; font-weight: 800 !important; color: var(--sq-txt) !important;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{ min-width: 330px !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{ font-size: 1.05rem !important; }
[data-testid="stSidebar"] .stButton > button{
  min-height: 54px !important; font-size: 1.05rem !important;
  border-radius: 16px !important; text-align: left !important;
}
.sq-navhead{
  font-size: .95rem; font-weight: 900; letter-spacing: 1.2px;
  text-transform: uppercase; color: var(--sq-txt2);
  margin: 1rem 0 .35rem 0;
}
.sq-navcurrent{
  background: var(--sq-card2); border: 2px solid var(--sq-btn1);
  border-radius: 16px; padding: .7rem .9rem; margin-bottom:.5rem;
  font-weight: 800; color: var(--sq-txt); font-size: 1.05rem;
}

/* ---------- SYNAPTIQ cards ---------- */
.sq-hero{
  position: relative; overflow: hidden;
  padding: 2.4rem 2.2rem 2.6rem 2.2rem;
  border-radius: 30px;
  background:
    radial-gradient(900px 300px at 8% 0%, rgba(15,118,110,.14), transparent 62%),
    radial-gradient(700px 280px at 100% 100%, rgba(3,105,161,.12), transparent 60%),
    linear-gradient(135deg, __HERO1__ 0%, __HERO2__ 100%);
  border: 2px solid var(--sq-border);
  box-shadow: 0 16px 40px rgba(20,40,60,.10);
  margin-bottom: 1.6rem;
}
.sq-hero::after{
  content:""; position:absolute; left:0; right:0; bottom:-2px; height:120px;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 160' preserveAspectRatio='none'><path d='M0 160 L150 70 L260 120 L390 40 L520 130 L650 60 L790 125 L900 75 L1040 135 L1200 65 L1200 160 Z' fill='__MTN__'/></svg>");
  background-size: cover; background-repeat: no-repeat;
  opacity: .5; pointer-events: none;
}
.sq-hero > *{ position: relative; z-index: 2; }
.sq-hello{ font-size: clamp(2.1rem, 4.4vw, 3.1rem); font-weight: 900; color: var(--sq-txt); letter-spacing:-1px; margin:0; }
.sq-hello-sub{ font-size: 1.32rem; line-height: 1.6; color: var(--sq-txt2); max-width: 820px; margin-top: .7rem; }
.sq-pill{
  display:inline-block; padding:.5rem .95rem; margin:.7rem .45rem 0 0;
  border-radius: 999px; background: var(--sq-card); border: 2px solid var(--sq-border);
  font-weight: 800; font-size: .98rem; color: var(--sq-txt);
}

.sq-card{
  background: var(--sq-card); border: 2px solid var(--sq-border2);
  border-radius: var(--sq-radius); padding: 1.35rem 1.45rem;
  box-shadow: 0 8px 22px rgba(20,40,60,.07);
  margin-bottom: .9rem;
}
.sq-card h3{ margin: 0 0 .35rem 0; font-size: 1.42rem; font-weight: 900; color: var(--sq-txt); }
.sq-card p{ margin: 0; font-size: 1.12rem; line-height: 1.65; color: var(--sq-txt2); }
.sq-card-icon{ font-size: 2.6rem; line-height: 1; margin-bottom: .55rem; }

.sq-feature{
  background: var(--sq-card); border: 2px solid var(--sq-border2);
  border-radius: 26px; padding: 1.5rem 1.4rem .6rem 1.4rem;
  min-height: 218px; box-shadow: 0 10px 26px rgba(20,40,60,.08);
  transition: transform .18s ease, border-color .18s ease;
}
.sq-feature:hover{ transform: translateY(-3px); border-color: var(--sq-btn1); }

.friendly-card{
  display:flex; gap: 1.05rem; align-items:flex-start;
  background: linear-gradient(135deg, var(--sq-card), var(--sq-card2));
  border: 2px solid var(--sq-border2); border-radius: var(--sq-radius);
  padding: 1.15rem 1.3rem; margin-bottom: .9rem;
  box-shadow: 0 6px 18px rgba(20,40,60,.06);
}
.friendly-icon{ font-size: 2.3rem; line-height: 1.1; }
.friendly-title{ font-size: 1.28rem; font-weight: 900; color: var(--sq-txt); margin-bottom:.2rem; }
.friendly-text{ font-size: 1.1rem; line-height: 1.65; color: var(--sq-txt2); }

.sq-stat{
  background: var(--sq-card); border: 2px solid var(--sq-border2);
  border-radius: 20px; padding: 1.1rem .9rem; text-align:center; min-height: 112px;
  display:flex; flex-direction:column; justify-content:center;
}
.sq-stat .n{ font-size: 1.85rem; font-weight: 900; color: var(--sq-blue); }
.sq-stat .l{ font-size: .98rem; font-weight: 700; color: var(--sq-txt2); margin-top:.25rem; }

.sq-result{
  border-radius: 28px; padding: 2rem 1.8rem; text-align:center;
  border: 3px solid var(--sq-border); background: var(--sq-card);
}
.sq-result .badge{ font-size: 1.75rem; font-weight: 900; }
.sq-result .conf{ font-size: 3.1rem; font-weight: 900; margin: .3rem 0; }
.sq-result .note{ font-size: 1.08rem; color: var(--sq-txt2); }
.sq-result-ok{ background: linear-gradient(135deg, __NDEM1__, __NDEM2__); border-color: var(--sq-green); }
.sq-result-watch{ background: linear-gradient(135deg, __DEM1__, __DEM2__); border-color: var(--sq-orange); }

.sq-rem{
  display:flex; gap:1.2rem; align-items:center;
  background: var(--sq-card); border: 2px solid var(--sq-border2);
  border-left: 10px solid var(--sq-btn1);
  border-radius: 20px; padding: 1.1rem 1.3rem; margin-bottom: .8rem;
}
.sq-rem .time{ font-size: 1.85rem; font-weight: 900; color: var(--sq-txt); min-width: 165px; }
.sq-rem .what{ font-size: 1.22rem; font-weight: 800; color: var(--sq-txt); }
.sq-rem .note{ font-size: 1.05rem; color: var(--sq-txt2); }

.sq-call{
  display:block; text-align:center; text-decoration:none;
  background: linear-gradient(135deg, #15803d, #1a9950);
  color: #ffffff !important; padding: 22px; border-radius: 22px;
  font-size: 1.35rem; font-weight: 900; margin: .6rem 0 1.2rem 0;
  box-shadow: 0 10px 24px rgba(21,128,61,.28);
}
.sq-call-red{ background: linear-gradient(135deg, #b42318, #d14b2f); box-shadow: 0 10px 24px rgba(180,35,24,.3); }

.sq-status{
  display:inline-flex; align-items:center; gap:.55rem;
  padding:.65rem 1.1rem; border-radius:999px; font-weight:900; font-size:1.12rem;
  border: 2px solid var(--sq-border);
}
.sq-status-ok{ background: __BGGREEN__; color: var(--sq-green); border-color: var(--sq-green); }
.sq-status-warn{ background: __GMID__; color: var(--sq-orange); border-color: var(--sq-orange); }
.sq-status-info{ background: __BGBLUE__; color: var(--sq-blue); border-color: var(--sq-blue); }

.sq-step{
  display:flex; gap:.6rem; flex-wrap:wrap; margin: .2rem 0 1rem 0;
}
.sq-step .s{
  padding:.6rem 1rem; border-radius:999px; font-weight:800; font-size:1.02rem;
  background: var(--sq-card); border:2px solid var(--sq-border2); color: var(--sq-txt2);
}
.sq-step .s.on{ background: __BGBLUE__; border-color: var(--sq-blue); color: var(--sq-blue); }
.sq-step .s.done{ background: __BGGREEN__; border-color: var(--sq-green); color: var(--sq-green); }

.sq-help{
  background: __BGBLUE__; border: 2px solid var(--sq-border2);
  border-radius: 16px; padding: .85rem 1.05rem; margin:.3rem 0 .9rem 0;
  font-size: 1.05rem; color: var(--sq-txt); line-height:1.6;
}
.sq-note{
  background: __GMID__; border: 2px solid var(--sq-orange);
  border-radius: 18px; padding: 1rem 1.2rem; color: var(--sq-txt);
  font-size: 1.06rem; line-height: 1.65;
}
.sq-leaf{ font-size: 1.15rem; opacity:.85; }

/* ---------- Responsive ---------- */
@media (max-width: 1000px){
  [data-testid="stAppViewContainer"] .block-container{ padding-left:1rem; padding-right:1rem; }
  .sq-hero{ padding: 1.6rem 1.25rem 2rem 1.25rem; border-radius: 24px; }
  .sq-hello{ font-size: 2rem; }
  .sq-hello-sub{ font-size: 1.15rem; }
  .sq-rem{ flex-direction: column; align-items:flex-start; gap:.4rem; }
  .sq-rem .time{ min-width: auto; }
  .sq-feature{ min-height: auto; }
}

/* ---------- Motion respect ---------- */
@media (prefers-reduced-motion: reduce){
  *{ animation: none !important; transition: none !important; }
}
</style>
"""
    mapping = {
        "__BG__": t['bg'], "__CARD__": t['card'], "__CARD2__": t['card2'],
        "__BORDER__": t['border'], "__BORDER2__": t['border2'],
        "__TXT__": t['txt'], "__TXT2__": t['txt2'],
        "__GREEN__": t['green'], "__BLUE__": t['blue'], "__PURPLE__": t['purple'],
        "__ORANGE__": t['orange'], "__RED__": t['red'],
        "__BTN1__": t['btn1'], "__BTN2__": t['btn2'],
        "__HERO1__": t['hero1'], "__HERO2__": t['hero2'],
        "__NDEM1__": t['ndem1'], "__NDEM2__": t['ndem2'],
        "__DEM1__": t['dem1'], "__DEM2__": t['dem2'],
        "__BGGREEN__": t['bg_green'], "__BGBLUE__": t['bg_blue'],
        "__GMID__": t['gmid'],
        "__MTN__": "%23b7d7cd" if not st.session_state.dark_mode else "%23234a52",
    }
    for key, value in mapping.items():
        css = css.replace(key, str(value))
    return css


st.markdown(synaptiq_css(t), unsafe_allow_html=True)


def danger_button(label, key, use_container_width=True):
    """A large red emergency-style button that still uses a normal st.button."""
    st.markdown('<span class="sq-anchor-danger"></span>', unsafe_allow_html=True)
    return st.button(label, key=key, use_container_width=use_container_width)


def go_to(page_name):
    """Navigate to an existing page route and refresh the screen."""
    st.session_state.nav_page = page_name
    st.rerun()


# =========================
# LOGIN SCREEN
# =========================
def login_ui():
    """Warm, simple, large-type sign-in screen."""
    global APP_LANGUAGE, APP_LANG_CODE

    st.markdown("""
    <div class="sq-hero">
        <div class="sq-hello">🧠 Welcome to SYNAPTIQ</div>
        <div class="sq-hello-sub">Your companion for memory, cognitive health and everyday safety —
        made for elders and families in North East India.</div>
        <span class="sq-pill">🌏 North East India</span>
        <span class="sq-pill">🗣️ Many languages</span>
        <span class="sq-pill">❤️ Family &amp; caregiver friendly</span>
    </div>
    """, unsafe_allow_html=True)

    left, middle, right = st.columns([1, 2, 1])

    with middle:
        st.markdown('<div class="sq-card"><h3>🌐 Choose Language</h3><p>Pick the language you are most comfortable reading.</p></div>',
                    unsafe_allow_html=True)
        # One stable widget key is essential. A language-dependent key can
        # leave the previous language's widget value alive in Streamlit.
        selected_app_language = _original_selectbox(
            "🌐 Choose Language",
            NE_LANGUAGE_OPTIONS,
            index=NE_LANGUAGE_OPTIONS.index(
                normalize_language(st.session_state.app_language)
            ),
            format_func=lambda language: LANGUAGE_DISPLAY_NAMES.get(language, language),
            key="language_selector",
            on_change=_language_selector_changed,
            label_visibility="collapsed"
        )
        selected_app_language = normalize_language(selected_app_language)
        # Cosmetic only: selecting a language never changes the actual UI language.

        st.markdown(
            f'<div class="sq-status sq-status-ok">✅ Selected: '
            f'{html.escape(LANGUAGE_DISPLAY_NAMES.get(APP_LANGUAGE, APP_LANGUAGE))}</div>',
            unsafe_allow_html=True
        )


        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sq-card"><h3>🔐 Sign In</h3><p>Please type your user name and password, then press the green button.</p></div>',
                    unsafe_allow_html=True)

        username = st.text_input("👤 User name", placeholder="Type your user name")
        password = st.text_input("🔑 Password", type="password", placeholder="Type your password")

        if st.button("🚪 Sign In", use_container_width=True, key="login_submit_btn"):
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Welcome! Opening your home screen…")
                st.rerun()
            else:
                st.error("⚠️ That user name or password did not match. Please try again.")

        with st.expander("❓ Need help signing in?"):
            st.markdown(
                "- Ask your caregiver or family member to help you sign in.\n"
                "- Demo accounts for this prototype: **patient / test** and **caregiver / care123**.\n"
                "- Your information stays on this device for the demo."
            )


# =========================
# SIDEBAR ACCOUNT / LANGUAGE
# =========================
def logout_ui():
    """Account, language and theme controls in the sidebar."""
    global APP_LANGUAGE, APP_LANG_CODE

    st.sidebar.markdown(
        localize_html(f"""
        <div style="text-align:center;padding:.6rem 0 1rem 0;">
            <div style="font-size:2.6rem;line-height:1;">🧠</div>
            <div style="font-size:1.45rem;font-weight:900;color:{t['txt']};letter-spacing:.5px;">SYNAPTIQ</div>
            <div style="font-size:1rem;color:{t['txt2']};margin-top:.2rem;">Memory · Health · Safety</div>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        f'<div class="sq-navcurrent">👤 {html.escape(str(st.session_state.username)).title()}</div>',
        unsafe_allow_html=True
    )

    st.sidebar.markdown(localize_html('<div class="sq-navhead">🌐 Choose Language</div>'), unsafe_allow_html=True)
    selected_app_language = _original_selectbox(
        tr("🌐 Choose Language"),
        NE_LANGUAGE_OPTIONS,
        index=NE_LANGUAGE_OPTIONS.index(
            normalize_language(st.session_state.app_language)
        ),
        format_func=lambda language: LANGUAGE_DISPLAY_NAMES.get(language, language),
        key="language_selector",
        on_change=_language_selector_changed,
        label_visibility="collapsed"
    )

    selected_app_language = normalize_language(selected_app_language)
    # Cosmetic only: the application itself remains English.



# =========================
# AUTH GATE
# =========================
if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ══════════════════════════════════════════════════════════════
# MEMORY ASSISTANCE / CAREGIVER STATE
# ══════════════════════════════════════════════════════════════
if "memories" not in st.session_state:
    st.session_state.memories = []
if "reminders" not in st.session_state:
    st.session_state.reminders = []
if "important_contacts" not in st.session_state:
    st.session_state.important_contacts = []
if "daily_routine" not in st.session_state:
    st.session_state.daily_routine = [
        ("08:00", "Breakfast"),
        ("09:00", "Medicine"),
        ("11:00", "Memory activity"),
        ("13:00", "Lunch"),
        ("17:00", "Family call"),
        ("21:00", "Bedtime"),
    ]
if "game_history" not in st.session_state:
    st.session_state.game_history = []
if "sos_message" not in st.session_state:
    st.session_state.sos_message = ""
if "ner_uploaded_places" not in st.session_state:
    st.session_state.ner_uploaded_places = []
if "screening_mmse_score" not in st.session_state:
    st.session_state.screening_mmse_score = None
if "provisional_cdr" not in st.session_state:
    st.session_state.provisional_cdr = None
if "official_mmse_score" not in st.session_state:
    st.session_state.official_mmse_score = None
if "official_cdr_score" not in st.session_state:
    st.session_state.official_cdr_score = None

if "gps_last_location" not in st.session_state:
    st.session_state.gps_last_location = None
if "gps_tracking_enabled" not in st.session_state:
    st.session_state.gps_tracking_enabled = True

# ══════════════════════════════════════════════════════════════
# COGNITIVE + GPS SAFETY STORAGE
# A small local SQLite database keeps the demo functional across
# patient/caregiver logins without requiring a separate server.
# For production, replace this with a secure cloud database/API.
# ══════════════════════════════════════════════════════════════
SAFETY_DB = Path(__file__).resolve().parent / "synaptiq_safety.db"


def safety_db():
    conn = sqlite3.connect(str(SAFETY_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_safety_db():
    conn = safety_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cognitive_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            record_type TEXT NOT NULL,
            mmse REAL,
            cdr REAL,
            confidence REAL,
            prediction TEXT,
            game_score REAL,
            game_attempts INTEGER,
            game_accuracy REAL,
            game_name TEXT
        )
    """)
    # Safe schema migration for databases created by older SYNAPTIQ versions.
    columns = {row[1] for row in cur.execute("PRAGMA table_info(cognitive_records)").fetchall()}
    if "game_name" not in columns:
        cur.execute("ALTER TABLE cognitive_records ADD COLUMN game_name TEXT")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            accuracy REAL,
            status TEXT,
            distance_from_safe_zone REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS safe_zone (
            patient_id TEXT PRIMARY KEY,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            radius_m REAL NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS geofence_alerts (
            patient_id TEXT PRIMARY KEY,
            outside_state INTEGER NOT NULL DEFAULT 0,
            last_alert_at TEXT,
            last_message_sid TEXT,
            last_error TEXT
        )
    """)
    conn.commit()
    conn.close()


def patient_id():
    return st.session_state.get("patient_id", "patient")


def save_cognitive_record(record_type, mmse=None, cdr=None,
                          confidence=None, prediction=None,
                          game_score=None, game_attempts=None,
                          game_accuracy=None, game_name=None):
    conn = safety_db()
    conn.execute("""
        INSERT INTO cognitive_records
        (patient_id, recorded_at, record_type, mmse, cdr, confidence,
         prediction, game_score, game_attempts, game_accuracy, game_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id(), datetime.now().isoformat(timespec="seconds"),
        record_type, mmse, cdr, confidence, prediction,
        game_score, game_attempts, game_accuracy, game_name
    ))
    conn.commit()
    conn.close()


def cognitive_history():
    conn = safety_db()
    rows = conn.execute("""
        SELECT * FROM cognitive_records
        WHERE patient_id = ?
        ORDER BY recorded_at
    """, (patient_id(),)).fetchall()
    conn.close()
    return pd.DataFrame([dict(r) for r in rows])


def haversine_m(lat1, lon1, lat2, lon2):
    radius = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * radius * math.asin(math.sqrt(a))


def get_safe_zone():
    conn = safety_db()
    row = conn.execute(
        "SELECT * FROM safe_zone WHERE patient_id = ?", (patient_id(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def set_safe_zone(lat, lon, radius_m):
    conn = safety_db()
    conn.execute("""
        INSERT INTO safe_zone(patient_id, latitude, longitude, radius_m, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(patient_id) DO UPDATE SET
            latitude=excluded.latitude,
            longitude=excluded.longitude,
            radius_m=excluded.radius_m,
            updated_at=excluded.updated_at
    """, (
        patient_id(), float(lat), float(lon), float(radius_m),
        datetime.now().isoformat(timespec="seconds")
    ))
    conn.commit()
    conn.close()


def get_geofence_alert_state():
    conn = safety_db()
    row = conn.execute(
        "SELECT * FROM geofence_alerts WHERE patient_id = ?", (patient_id(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else {
        "patient_id": patient_id(),
        "outside_state": 0,
        "last_alert_at": None,
        "last_message_sid": None,
        "last_error": None,
    }


def set_geofence_alert_state(outside_state, last_alert_at=None, last_message_sid=None, last_error=None):
    conn = safety_db()
    conn.execute("""
        INSERT INTO geofence_alerts(patient_id, outside_state, last_alert_at, last_message_sid, last_error)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(patient_id) DO UPDATE SET
            outside_state=excluded.outside_state,
            last_alert_at=excluded.last_alert_at,
            last_message_sid=excluded.last_message_sid,
            last_error=excluded.last_error
    """, (patient_id(), int(bool(outside_state)), last_alert_at, last_message_sid, last_error))
    conn.commit()
    conn.close()


def get_callable_caregiver():
    """Return the first trusted contact with a callable phone number.

    This deliberately reuses the same contact validation rule as the SOS page:
    strip the phone number and require at least 7 digits.
    """
    for contact in st.session_state.get("important_contacts", []):
        phone = str(contact.get("phone", "")).strip()
        digits = "".join(ch for ch in phone if ch.isdigit())
        if len(digits) >= 7:
            return contact, phone
    return None, None


def process_geofence_alert(latitude, longitude, status, distance):
    """Detect an inside -> outside transition without using a paid telephony service."""
    zone = get_safe_zone()
    if not zone or distance is None:
        set_geofence_alert_state(False)
        return None

    state = get_geofence_alert_state()
    was_outside = bool(state.get("outside_state"))
    is_outside = status == "Outside safe zone"

    if is_outside and not was_outside:
        contact, phone = get_callable_caregiver()

        # Record the transition so the caregiver is not prompted repeatedly
        # on every Streamlit rerun while the patient remains outside.
        now = datetime.now().isoformat(timespec="seconds")
        set_geofence_alert_state(
            True,
            now,
            None,
            None if contact else "No callable caregiver contact is configured."
        )

        return {
            "outside": True,
            "new_exit": True,
            "contact": contact,
            "phone": phone,
        }

    if is_outside:
        return {
            "outside": True,
            "new_exit": False,
            "contact": get_callable_caregiver()[0],
            "phone": get_callable_caregiver()[1],
        }

    if not is_outside:
        set_geofence_alert_state(
            False,
            state.get("last_alert_at"),
            state.get("last_message_sid"),
            None
        )

    return None


def save_location(lat, lon, accuracy=None):
    zone = get_safe_zone()
    distance = None
    status = "No safe zone configured"

    if zone:
        distance = haversine_m(
            float(lat), float(lon),
            float(zone["latitude"]), float(zone["longitude"])
        )
        status = "Inside safe zone" if distance <= float(zone["radius_m"]) else "Outside safe zone"

    conn = safety_db()
    conn.execute("""
        INSERT INTO locations
        (patient_id, recorded_at, latitude, longitude, accuracy, status, distance_from_safe_zone)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id(), datetime.now().isoformat(timespec="seconds"),
        float(lat), float(lon),
        float(accuracy) if accuracy is not None else None,
        status, distance
    ))
    conn.commit()
    conn.close()
    return status, distance


def latest_location():
    conn = safety_db()
    row = conn.execute("""
        SELECT * FROM locations
        WHERE patient_id = ?
        ORDER BY id DESC LIMIT 1
    """, (patient_id(),)).fetchone()
    conn.close()
    return dict(row) if row else None


def location_history(limit=50):
    conn = safety_db()
    rows = conn.execute("""
        SELECT * FROM locations
        WHERE patient_id = ?
        ORDER BY id DESC LIMIT ?
    """, (patient_id(), int(limit))).fetchall()
    conn.close()
    return pd.DataFrame([dict(r) for r in rows])

def render_cognitive_analytics():
    """Cognitive progress, shown simply first and in detail afterwards."""
    st.markdown("## 🧠 My Cognitive Progress")
    friendly_card(
        "How you are doing over time",
        "This page brings together your questionnaire answers, screening results and brain-game "
        "practice. It is a way to follow change over time — it is not a medical diagnosis.",
        "📈"
    )

    hist = cognitive_history()
    if hist.empty:
        friendly_card(
            "Nothing recorded yet",
            "Finish the Memory Questionnaire, the Cognitive Check, or play one Brain Game. "
            "Your results will then appear here.",
            "🌱"
        )
        return

    assessment = hist[hist["record_type"].isin(["screening", "diagnosis"])].copy()
    games = hist[hist["record_type"].astype(str).str.startswith("game")].copy()

    latest_mmse = assessment["mmse"].dropna().iloc[-1] if ("mmse" in assessment and assessment["mmse"].notna().any()) else None
    latest_cdr = assessment["cdr"].dropna().iloc[-1] if ("cdr" in assessment and assessment["cdr"].notna().any()) else None
    latest_conf = assessment["confidence"].dropna().iloc[-1] if ("confidence" in assessment and assessment["confidence"].notna().any()) else None
    avg_acc = games["game_accuracy"].dropna().mean() if not games.empty else None

    def _fmt(value, suffix=""):
        if value is None or pd.isna(value):
            return "—"
        return f"{value:g}{suffix}"

    simple_tab, detail_tab = st.tabs(["😊 Simple View", "🔬 Detailed View"])

    with simple_tab:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"<div class='sq-stat'><div class='n'>{_fmt(latest_mmse)}{'/30' if latest_mmse is not None and not pd.isna(latest_mmse) else ''}</div>"
                f"<div class='l'>🧠 Memory &amp; thinking score</div></div>",
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f"<div class='sq-stat'><div class='n'>{len(games)}</div>"
                f"<div class='l'>🎮 Brain games played</div></div>",
                unsafe_allow_html=True)
        with c3:
            st.markdown(
                f"<div class='sq-stat'><div class='n'>{_fmt(avg_acc, '%') if avg_acc is not None and not pd.isna(avg_acc) else '—'}</div>"
                f"<div class='l'>🎯 Average game accuracy</div></div>",
                unsafe_allow_html=True)

        st.markdown("### 💬 What this shows")
        if latest_mmse is not None and not pd.isna(latest_mmse):
            if latest_mmse >= 27:
                st.success("🟢 Your latest memory and thinking score is in the higher range. Keep up your daily activities.")
            elif latest_mmse >= 21:
                st.info("🔵 Your latest score suggests mild changes. Regular practice and a check-up are a good idea.")
            else:
                st.warning("🟠 Your latest score is lower than the typical range. Please share this with a doctor.")
        else:
            st.info("🔵 Complete the Memory Questionnaire to see your memory and thinking score here.")

        if avg_acc is not None and not pd.isna(avg_acc):
            st.markdown(f"🎮 **Brain games:** your average accuracy is **{avg_acc:.0f}%** across {len(games)} sessions.")

        if not assessment.empty and assessment["mmse"].notna().any():
            chart_df = assessment[assessment["mmse"].notna()].copy()
            chart_df["recorded_at"] = pd.to_datetime(chart_df["recorded_at"])
            fig = px.line(chart_df, x="recorded_at", y="mmse", markers=True,
                          title="Memory & thinking score over time")
            fig.update_traces(line=dict(width=4), marker=dict(size=11))
            fig.update_yaxes(range=[0, 30], title="Score out of 30")
            fig.update_xaxes(title="")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color=t['txt'], size=15), height=340)
            st.plotly_chart(fig, use_container_width=True, key="cog_simple_mmse")

        st.markdown('<div class="sq-note">⚠️ <b>Please remember:</b> these numbers help you follow change over '
                    'time. They are not a medical diagnosis. Always talk to a qualified doctor about your health.</div>',
                    unsafe_allow_html=True)

    with detail_tab:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest MMSE-style score",
                  f"{latest_mmse:.0f}/30" if latest_mmse is not None and not pd.isna(latest_mmse) else "—")
        c2.metric("Latest CDR-style rating",
                  f"{latest_cdr:g}" if latest_cdr is not None and not pd.isna(latest_cdr) else "—")
        c3.metric("Latest model confidence",
                  f"{latest_conf:.1f}%" if latest_conf is not None and not pd.isna(latest_conf) else "—")
        c4.metric("Game accuracy",
                  f"{avg_acc:.1f}%" if avg_acc is not None and not pd.isna(avg_acc) else "—")

        if not assessment.empty and assessment["cdr"].notna().any():
            chart_df = assessment[assessment["cdr"].notna()].copy()
            chart_df["recorded_at"] = pd.to_datetime(chart_df["recorded_at"])
            fig = px.line(chart_df, x="recorded_at", y="cdr", markers=True,
                          title="CDR-style Profile Trend")
            fig.update_yaxes(range=[0, 3], title="CDR-style rating")
            fig.update_xaxes(title="")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color=t['txt']), height=320)
            st.plotly_chart(fig, use_container_width=True, key="cog_detail_cdr")

        if not games.empty and games["game_accuracy"].notna().any():
            chart_df = games.copy()
            chart_df["recorded_at"] = pd.to_datetime(chart_df["recorded_at"])
            fig = px.line(chart_df, x="recorded_at", y="game_accuracy", markers=True,
                          title="Brain Game Accuracy Trend")
            fig.update_yaxes(range=[0, 100], title="Accuracy (%)")
            fig.update_xaxes(title="")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color=t['txt']), height=320)
            st.plotly_chart(fig, use_container_width=True, key="cog_detail_games")

        with st.expander("📋 Full assessment history"):
            show_cols = [c for c in ["recorded_at", "record_type", "mmse", "cdr", "confidence",
                                     "prediction", "game_name", "game_score", "game_attempts",
                                     "game_accuracy"] if c in hist.columns]
            st.dataframe(hist[show_cols].sort_values("recorded_at", ascending=False),
                         use_container_width=True, hide_index=True)

def render_geofence_caregiver_call(contact, phone, distance=None):
    """Render the same browser tel: call mechanism used by the SOS page."""
    if not contact or not phone:
        st.error(
            "⚠️ No callable caregiver contact is configured. "
            "Ask the caregiver to add a trusted contact with a valid phone number."
        )
        return

    safe_phone = html.escape(phone, quote=True)
    safe_name = html.escape(str(contact.get("name", "Caregiver")))

    if distance is not None:
        st.error(
            f"🚨 Patient is outside the safe zone ({distance:.0f} m from the center). "
            f"Please contact {contact.get('name', 'the caregiver')} immediately."
        )

    st.markdown(
        f"""
        <a href="tel:{safe_phone}"
           style="
             display:block;
             text-align:center;
             text-decoration:none;
             background:#dc2626;
             color:white;
             padding:20px;
             border-radius:16px;
             font-size:1.3rem;
             font-weight:800;
             margin:10px 0 18px 0;
             box-shadow:0 6px 18px rgba(220,38,38,0.25);
           ">
           📞 CALL {safe_name} — PATIENT LEFT SAFE ZONE
        </a>
        """,
        unsafe_allow_html=True
    )

def render_gps_safety():
    """Location safety, written in plain language with a very clear status."""
    st.markdown("## 📍 Your Safety Location")
    friendly_card(
        "Only when you allow it",
        "Your phone or computer asks your permission before sharing where you are. "
        "Your family or caregiver can then see that you are safe.",
        "🛡️"
    )

    zone = get_safe_zone()
    if zone:
        st.markdown(
            f"<div class='sq-help'>🏠 <b>Your safe area</b> is about "
            f"{zone['radius_m']:.0f} metres around your saved home location.</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='sq-help'>🏠 <b>No safe area saved yet.</b> A caregiver can set one "
            "further down this page.</div>",
            unsafe_allow_html=True
        )

    st.markdown("### 1️⃣ Check where I am now")
    st.caption("Press the location button below and choose “Allow” when your browser asks.")

    if streamlit_geolocation is None:
        st.warning(
            "⚠️ Location checking is not installed on this computer. "
            "Ask a helper to install the streamlit-geolocation package to switch this on."
        )
        location = None
    else:
        location = streamlit_geolocation()

    if location and "error" in location:
        st.error("⚠️ We could not read your location. Please allow location access and try again.")
    elif location and location.get("latitude") is not None:
        lat = float(location["latitude"])
        lon = float(location["longitude"])
        accuracy = location.get("accuracy")
        st.session_state.gps_last_location = location

        try:
            status, distance = save_location(lat, lon, accuracy)
            process_geofence_alert(lat, lon, status, distance)
        except Exception:
            status, distance = "No safe zone configured", None
            st.warning("⚠️ Something went wrong while saving your location. Please try again.")

        if status == "Outside safe zone":
            st.markdown(
                "<div class='sq-status sq-status-warn'>⚠️ Please check your location — "
                "you are away from your safe area</div>", unsafe_allow_html=True)
            if distance is not None:
                st.caption(f"About {distance:.0f} metres away from the centre of your safe area.")
            contact, phone = get_callable_caregiver()
            render_geofence_caregiver_call(contact, phone, distance)
        elif status == "Inside safe zone":
            st.markdown(
                "<div class='sq-status sq-status-ok'>🟢 Inside Safe Zone — you are in your safe area</div>",
                unsafe_allow_html=True)
            if distance is not None:
                st.caption(f"About {distance:.0f} metres from the centre of your safe area.")
        else:
            st.markdown(
                "<div class='sq-status sq-status-info'>🔵 Location received — no safe area saved yet</div>",
                unsafe_allow_html=True)

        st.map(pd.DataFrame([{"lat": lat, "lon": lon}]), zoom=15)

        with st.expander("📐 Location details"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Latitude", f"{lat:.6f}")
            c2.metric("Longitude", f"{lon:.6f}")
            c3.metric("Accuracy", f"{float(accuracy):.0f} m" if accuracy else "—")

    st.markdown("### 2️⃣ Set the safe area (for caregivers)")
    st.caption("The safe area is usually the patient's home. Use the last known location as the centre.")
    latest = latest_location()
    with st.form("safe_zone_form"):
        if latest:
            default_lat = float(latest["latitude"])
            default_lon = float(latest["longitude"])
        else:
            default_lat, default_lon = 0.0, 0.0
        c1, c2, c3 = st.columns(3)
        with c1:
            zone_lat = st.number_input("Centre latitude", value=default_lat, format="%.6f")
        with c2:
            zone_lon = st.number_input("Centre longitude", value=default_lon, format="%.6f")
        with c3:
            zone_radius = st.number_input(
                "Safe area size (metres)", min_value=50.0, max_value=10000.0,
                value=float(zone["radius_m"]) if zone else 500.0, step=50.0,
                help="How far from home is still considered safe."
            )
        if st.form_submit_button("💾 Save Safe Area", use_container_width=True):
            if abs(zone_lat) <= 90 and abs(zone_lon) <= 180:
                try:
                    set_safe_zone(zone_lat, zone_lon, zone_radius)
                    st.success("✅ Safe area saved.")
                    st.rerun()
                except Exception:
                    st.error("⚠️ Something went wrong. Please try saving again.")
            else:
                st.error("⚠️ Please enter a valid latitude and longitude.")

    with st.expander("🕒 Recent location updates"):
        hist = location_history(25)
        if hist.empty:
            st.info("No location updates recorded yet.")
        else:
            display = hist[["recorded_at", "latitude", "longitude", "accuracy",
                            "status", "distance_from_safe_zone"]].copy()
            st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="sq-note">⚠️ <b>Important:</b> this is a safety helper, not a replacement for '
        'supervision or emergency services. It only works while this page is open in the browser.</div>',
        unsafe_allow_html=True
    )

def render_caregiver_location_monitor():
    st.markdown("### 📍 Patient Location Monitoring")
    latest = latest_location()
    zone = get_safe_zone()

    if not latest:
        st.info("No patient GPS location has been recorded yet. Ask the patient to open GPS Safety and allow location access.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latitude", f"{latest['latitude']:.6f}")
    c2.metric("Longitude", f"{latest['longitude']:.6f}")
    c3.metric("GPS accuracy", f"{latest['accuracy']:.0f} m" if latest["accuracy"] else "—")
    c4.metric("Safety status", "SAFE" if latest["status"] == "Inside safe zone" else "ALERT" if latest["status"] == "Outside safe zone" else "UNCONFIGURED")

    if latest["status"] == "Outside safe zone":
        st.error("🚨 Geofence alert: patient is outside the configured safe zone.")
        alert_state = get_geofence_alert_state()
        if alert_state.get("last_alert_at"):
            st.caption(f"📞 Geofence exit detected at: {alert_state['last_alert_at']}")
        elif alert_state.get("last_error"):
            st.caption(f"⚠️ Caregiver call status: {alert_state['last_error']}")
    elif latest["status"] == "Inside safe zone":
        st.success("✅ Patient is inside the configured safe zone.")
    else:
        st.warning("Safe zone is not configured.")

    st.map(pd.DataFrame([{"lat": latest["latitude"], "lon": latest["longitude"]}]), zoom=15)

    if zone:
        st.caption(
            f"Safe-zone center: {zone['latitude']:.6f}, {zone['longitude']:.6f} · "
            f"Radius: {zone['radius_m']:.0f} m"
        )

    hist = location_history(25)
    if not hist.empty:
        st.markdown("#### Recent location updates")
        st.dataframe(
            hist[["recorded_at", "latitude", "longitude", "accuracy", "status", "distance_from_safe_zone"]],
            use_container_width=True, hide_index=True
        )

init_safety_db()

def friendly_card(title, text, icon="💙"):
    st.markdown(f"""
    <div class="friendly-card">
        <div class="friendly-icon">{icon}</div>
        <div>
            <div class="friendly-title">{tr(title)}</div>
            <div class="friendly-text">{tr(text)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def browser_speak(text, lang=None):
    if lang is None:
        lang = APP_LANG_CODE
    safe = html.escape(str(text)).replace("\\", "\\\\").replace("`", "\\`")
    components.html(f"""
    <script>
    const text = `{safe}`;
    if ('speechSynthesis' in window) {{
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.lang = '{lang}';
        u.rate = 0.85;
        u.pitch = 1.0;
        window.speechSynthesis.speak(u);
    }}
    </script>
    """, height=0)


def browser_reminder_scheduler(reminders, lang=None):
    if lang is None:
        lang = APP_LANG_CODE
    """Show an in-page popup and optional browser notification at the saved date/time."""
    reminder_payload = []
    for rem in reminders:
        if rem.get("date") and rem.get("time"):
            reminder_payload.append({
                "id": rem.get("id", ""),
                "date": rem["date"],
                "time": rem["time"],
                "title": rem.get("text", tr("Reminder")),
                "type": tr(rem.get("type", "Reminder")),
                "note": rem.get("note", "")
            })

    payload = json.dumps(reminder_payload).replace("\\", "\\\\").replace("'", "\\'")

    components.html(f"""
    <div id="synaptiq-reminder-panel"
         style="font-family:Arial,sans-serif;padding:8px 0;max-width:900px;">
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
        <button id="notify-btn"
          style="padding:12px 18px;border-radius:12px;border:1px solid #94a3b8;
                 background:#2563eb;color:white;font-weight:700;cursor:pointer;">
          🔔 {tr("Enable browser notifications")}
        </button>
        <span id="notify-status" style="font-size:14px;"></span>
      </div>

      <div id="popup"
        style="display:none;position:relative;margin-top:12px;padding:18px;
               border-radius:16px;background:#fff7ed;border:3px solid #f59e0b;
               box-shadow:0 8px 25px rgba(0,0,0,.18);">
        <div style="font-size:24px;font-weight:800;color:#92400e;">⏰ {tr("Reminder")}</div>
        <div id="popup-text"
             style="font-size:18px;line-height:1.5;color:#422006;margin-top:6px;"></div>
        <button id="close-popup"
          style="margin-top:12px;padding:10px 18px;border:0;border-radius:10px;
                 background:#92400e;color:white;font-weight:700;cursor:pointer;">
          ✓ {tr("Got it")}
        </button>
      </div>
    </div>

    <script>
      const reminders = JSON.parse('{payload}');
      const btn = document.getElementById('notify-btn');
      const status = document.getElementById('notify-status');
      const popup = document.getElementById('popup');
      const popupText = document.getElementById('popup-text');
      const closePopup = document.getElementById('close-popup');
      const fired = new Set();

      closePopup.addEventListener('click', () => {{
        popup.style.display = 'none';
      }});

      function updateNotificationButton() {{
        if (!('Notification' in window)) {{
          btn.textContent = '🔔 {tr("Enable browser notifications")}';
          status.textContent = '{tr("Browser notifications are unavailable; the in-app popup will still be used.")}';
          return;
        }}
        if (Notification.permission === 'granted') {{
          btn.textContent = '✅ {tr("Browser notifications enabled")}';
          status.textContent = '{tr("Browser notifications are enabled.")}';
        }} else if (Notification.permission === 'denied') {{
          btn.textContent = '🔕 {tr("Browser notifications blocked")}';
          status.textContent = '{tr("Notifications are blocked. The in-app popup will still be used.")}';
        }} else {{
          btn.textContent = '🔔 {tr("Enable browser notifications")}';
          status.textContent = '{tr("Click the button to allow browser notifications.")}';
        }}
      }}

      async function enableNotifications() {{
        if (!('Notification' in window)) {{
          updateNotificationButton();
          return;
        }}
        try {{
          await Notification.requestPermission();
        }} catch (e) {{
          status.textContent = '{tr("Please allow notifications in browser settings.")}';
        }}
        updateNotificationButton();
      }}

      btn.addEventListener('click', enableNotifications);
      updateNotificationButton();

      function showReminder(r) {{
        const message =
          `${{r.type}}: ${{r.title}}` + (r.note ? ` — ${{r.note}}` : '');

        popupText.textContent = message;
        popup.style.display = 'block';

        if ('Notification' in window && Notification.permission === 'granted') {{
          try {{
            new Notification('⏰ {tr('Synaptiq Reminder')}', {{
              body: message,
              requireInteraction: true
            }});
          }} catch (e) {{}}
        }}

        if ('speechSynthesis' in window) {{
          window.speechSynthesis.cancel();
          const speech = new SpeechSynthesisUtterance('{tr('Reminder')}. ' + message);
          speech.lang = '{lang}';
          speech.rate = 0.85;
          speech.pitch = 1.0;
          window.speechSynthesis.speak(speech);
        }}
      }}

      function checkReminders() {{
        const now = new Date();
        const yyyy = now.getFullYear();
        const mm = String(now.getMonth() + 1).padStart(2, '0');
        const dd = String(now.getDate()).padStart(2, '0');
        const hh = String(now.getHours()).padStart(2, '0');
        const min = String(now.getMinutes()).padStart(2, '0');

        const currentDate = `${{yyyy}}-${{mm}}-${{dd}}`;
        const currentTime = `${{hh}}:${{min}}`;

        reminders.forEach((r) => {{
          const key = `${{r.id}}-${{r.date}}-${{r.time}}`;
          if (r.date === currentDate &&
              r.time === currentTime &&
              !fired.has(key)) {{
            fired.add(key);
            showReminder(r);
          }}
        }});
      }}

      checkReminders();
      setInterval(checkReminders, 1000);
    </script>
    """, height=150, scrolling=False)

# ══════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from ml.generate_dataset import load_oasis_dataset, prepare_oasis
    from ml.data_loader      import preprocess, split_data
    from ml.preprocessing    import normalize, apply_pca, apply_lda
    from ml.models           import (train_random_forest, train_all_svm_kernels,
                                     train_voting_ensemble, apply_smote)
    from ml.evaluation       import evaluate_model

    df_raw = load_oasis_dataset(save_path='data/oasis_longitudinal.csv')
    df     = prepare_oasis(df_raw)
    X, y, le = preprocess(df, target_col='Group', merge_converted=True)
    feature_names = list(X.columns)
    class_names   = list(le.classes_)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    X_tr_sc, X_te_sc, scaler = normalize(X_train, X_test, method='standard')
    X_bal, y_bal = apply_smote(X_tr_sc, y_train.values)

    rf   = train_random_forest(X_bal, y_bal, tune=False)
    svms = train_all_svm_kernels(X_bal, y_bal)
    ens  = train_voting_ensemble(X_bal, y_bal)

    X_tr_pca, X_te_pca, pca = apply_pca(X_tr_sc, X_te_sc, n_components=2)
    X_tr_lda, X_te_lda, lda = apply_lda(X_tr_sc, X_te_sc, y_train.values, n_components=2)

    results = {}
    for name, mdl in {'Random Forest': rf, **svms, 'Voting Ensemble': ens}.items():
        results[name] = evaluate_model(mdl, X_te_sc, y_test,
                                       model_name=name, class_names=class_names)
    return {
        'rf': rf, 'svms': svms, 'ensemble': ens,
        'scaler': scaler, 'le': le,
        'feature_names': feature_names, 'class_names': class_names,
        'X_train': X_tr_sc, 'y_train': y_train.values,
        'X_test':  X_te_sc,  'y_test':  y_test.values,
        'X_tr_pca': X_tr_pca, 'X_tr_lda': X_tr_lda,
        'results': results, 'df': df,
    }

# ══════════════════════════════════════════════════════════════
# NAVIGATION  (friendly labels → existing page routes)
# ══════════════════════════════════════════════════════════════
NAV_GROUPS = [
    ("🧠 My Health", [
        ("📝 Memory Questionnaire", "📝 Cognitive Questionnaire"),
        ("🔬 Cognitive Check", "🔬 Patient Diagnosis"),
        ("📈 My Progress", "🧠 Cognitive Performance"),
        ("🗺️ My Risk Timeline", "🧠 Risk Timeline"),
    ]),
    ("💭 My Memories", [
        ("📸 My Memories", "📸 My Memories"),
        ("🌏 Places & Culture", "🌏 NER Memory & Culture"),
    ]),
    ("🎮 Activities", [
        ("🎮 Brain Games", "🧩 Cognitive Games"),
    ]),
    ("🛡️ Safety", [
        ("🆘 Safety & SOS", "🆘 Safety / SOS"),
        ("📍 My Location", "📍 GPS Safety & Geofencing"),
        ("💊 My Reminders", "⏰ Smart Reminders"),
    ]),
    ("👨‍👩‍👧 Care & Help", [
        ("👨‍👩‍👧 Caregiver Support", "👨‍👩‍👧 Caregiver Portal"),
        ("💬 Ask SYNAPTIQ", "💬 AI Chatbot"),
        ("🎤 Speak to SYNAPTIQ", "🎤 Voice Assistant"),
    ]),
    ("📊 Insights", [
        ("📊 Model Analytics", "📊 Model Analytics"),
        ("📈 Visualizations", "📈 Visualizations"),
    ]),
]

PAGE_LABELS = {"🏠 Home": "🏠 Home"}
for _group_title, _items in NAV_GROUPS:
    for _label, _route in _items:
        PAGE_LABELS[_route] = _label

# Guard against an unknown stored route (for example after an upgrade).
if st.session_state.nav_page not in PAGE_LABELS:
    st.session_state.nav_page = "🏠 Home"

with st.sidebar:
    logout_ui()

    st.markdown(localize_html('<div class="sq-navhead">📍 You are here</div>'), unsafe_allow_html=True)
    st.markdown(
        f'<div class="sq-navcurrent">{PAGE_LABELS.get(st.session_state.nav_page, st.session_state.nav_page)}</div>',
        unsafe_allow_html=True
    )

    if st.button(tr("🏠 Home"), use_container_width=True, key="nav_home_btn"):
        go_to("🏠 Home")

    if danger_button(tr("🆘 Emergency Help"), key="nav_sos_btn"):
        go_to("🆘 Safety / SOS")

    for group_index, (group_title, items) in enumerate(NAV_GROUPS):
        contains_current = any(route == st.session_state.nav_page for _, route in items)
        with st.expander(tr(group_title), expanded=contains_current):
            for item_index, (label, route) in enumerate(items):
                marker = "➡️ " if route == st.session_state.nav_page else ""
                if st.button(f"{marker}{tr(label)}", use_container_width=True,
                             key=f"nav_{group_index}_{item_index}"):
                    go_to(route)

    st.markdown("---")
    st.toggle(tr("🌙 Dark Mode"), key="dark_mode",
              help=tr("Switch to a darker screen. Text stays large and clear."))

    if st.button(tr("↪️ Sign Out"), use_container_width=True, key="nav_logout_btn"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    st.markdown(
        localize_html(f"""
        <div style="margin-top:1rem;padding:.9rem 1rem;background:{t['card']};
                    border:2px solid {t['border2']};border-radius:16px;">
            <div style="font-size:.95rem;color:{t['txt2']};font-weight:700;">🌏 Made for North East India</div>
            <div style="font-size:.95rem;color:{t['txt2']};margin-top:.25rem;">
                AI-assisted screening support · not a medical diagnosis
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

page = st.session_state.nav_page


def render_top_bar():
    """Always-visible Home and Emergency actions, never hidden in the sidebar."""
    left, right = st.columns([1, 1])
    with left:
        if st.button("🏠 Back to Home", use_container_width=True, key="topbar_home"):
            go_to("🏠 Home")
    with right:
        if danger_button("🆘 Emergency Help", key="topbar_sos"):
            go_to("🆘 Safety / SOS")

# ══════════════════════════════════════════════════════════════
# LOAD MODELS  (cached — loaded once per session)
# ══════════════════════════════════════════════════════════════
MODEL_ERROR = None
try:
    with st.spinner("🧠 Getting SYNAPTIQ ready for you…"):
        pipe = load_pipeline()
except Exception as model_exc:  # pragma: no cover - environment dependent
    pipe = None
    MODEL_ERROR = str(model_exc)

if pipe is None:
    st.error("⚠️ Something went wrong while preparing the health models. "
             "Memories, reminders, games and safety features still work.")
    with st.expander("🔧 Technical details (for developers)"):
        st.code(str(MODEL_ERROR))

render_top_bar()

render_neura_voice()
# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":

    patient_display_name = str(st.session_state.get("patient_name", "")).strip()
    greeting_name = f", {html.escape(patient_display_name)}" if patient_display_name else ""

    # Build the Home hero from individually translated phrases rather than
    # relying on HTML text-node heuristics. This guarantees that all five
    # visible hero/help boxes switch together.
    hero_title = tr("Welcome to SYNAPTIQ") + greeting_name
    hero_subtitle = tr("Your companion for memory, cognitive health and everyday safety.")
    pill_ner = tr("🌏 Made for North East India")
    pill_family = tr("👨‍👩‍👧 Family &amp; caregiver friendly")
    if pill_family == "👨‍👩‍👧 Family &amp; caregiver friendly":
        pill_family = tr("Family & caregiver friendly")
    pill_languages = tr("🗣️ Available in many languages")

    st.markdown(f"""
    <div class="sq-hero">
        <div class="sq-hello">🙏 {hero_title}</div>
        <div class="sq-hello-sub">{hero_subtitle}</div>
        <span class="sq-pill">{pill_ner}</span>
        <span class="sq-pill">{pill_family}</span>
        <span class="sq-pill">{pill_languages}</span>
    </div>
    """, unsafe_allow_html=True)

    help_title = tr("How to use this app:")
    help_body = tr("tap any large card below, or use the big buttons in the menu on the left. You can always come back here with the Back to Home button at the top.")
    # Keep the complete sentence as one catalog entry when available; the
    # explicit fallback below prevents mixed-language fragments.
    if help_body == "tap any large card below, or use the big buttons in the menu on the left. You can always come back here with the Back to Home button at the top.":
        help_body = "tap any large card below, or use the big buttons in the menu on the left. You can always come back here with the Back to Home button at the top."
    st.markdown(
        f'<div class="sq-help">👋 <b>{help_title}</b> {help_body}</div>',
        unsafe_allow_html=True
    )

    HOME_CARDS = [
        ("🧠", "Check My Cognitive Health",
         "Answer simple questions about memory and thinking, then see an easy-to-read result.",
         "📝 Cognitive Questionnaire", "🧠 Start My Check"),
        ("💭", "My Memories",
         "Photos, names and stories of the people and places you love, kept in one warm place.",
         "📸 My Memories", "💭 Open My Memories"),
        ("🎮", "Brain Games",
         "Short, friendly activities for attention, memory and quick thinking.",
         "🧩 Cognitive Games", "🎮 Play Brain Games"),
        ("💊", "My Reminders",
         "Gentle reminders for medicine, meals, water and family calls.",
         "⏰ Smart Reminders", "💊 See My Reminders"),
        ("🛡️", "Safety & SOS",
         "Call a trusted person quickly and let your family know that you are safe.",
         "🆘 Safety / SOS", "🛡️ Open Safety Help"),
        ("👨‍👩‍👧", "Caregiver Support",
         "Tools for family and caregivers: patient details, contacts, progress and location.",
         "👨‍👩‍👧 Caregiver Portal", "👨‍👩‍👧 Open Caregiver Support"),
    ]

    for row_start in range(0, len(HOME_CARDS), 3):
        columns = st.columns(3)
        for column, card in zip(columns, HOME_CARDS[row_start:row_start + 3]):
            icon, title, description, route, button_label = card
            with column:
                display_title = tr(title)
                display_description = tr(description)
                display_button = tr(button_label)
                st.markdown(f"""
                <div class="sq-feature">
                    <div class="sq-card-icon">{icon}</div>
                    <h3 style="font-size:1.4rem;font-weight:900;margin:.2rem 0 .45rem 0;">{display_title}</h3>
                    <p style="font-size:1.08rem;line-height:1.6;">{display_description}</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button(display_button, use_container_width=True, key=f"home_card_{route}"):
                    go_to(route)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🌿 " + tr("A few more things SYNAPTIQ can do"))
    extra_columns = st.columns(3)
    EXTRA_CARDS = [
        ("💬", "Ask a question", "Type a question in simple words and get a calm, clear answer.", "💬 AI Chatbot", "💬 Ask SYNAPTIQ"),
        ("🎤", "Speak instead of typing", "Press one big button and talk. SYNAPTIQ listens and replies.", "🎤 Voice Assistant", "🎤 Tap to Speak"),
        ("🌏", "Familiar places", "Photos of places you have visited, with a gentle recognition activity.", "🌏 NER Memory & Culture", "🌏 Open Places"),
    ]
    for column, card in zip(extra_columns, EXTRA_CARDS):
        icon, title, description, route, button_label = card
        with column:
            display_title = tr(title)
            display_description = tr(description)
            display_button = tr(button_label)
            st.markdown(f"""
            <div class="sq-card">
                <div class="sq-card-icon">{icon}</div>
                <h3>{display_title}</h3>
                <p>{display_description}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(display_button, use_container_width=True, key=f"home_extra_{route}"):
                go_to(route)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="sq-note">⚠️ <b>{tr("Please note:")}</b> {tr("SYNAPTIQ gives AI-assisted screening support and daily memory help. It does not replace an examination by a qualified healthcare professional.")}</div>',
        unsafe_allow_html=True
    )

    # ── Technical information, kept available but out of the way ──
    with st.expander("📊 For clinicians, caregivers and reviewers — how the model works"):
        st.markdown("#### Model performance summary")
        metric_columns = st.columns(5)
        for column, (value, label) in zip(metric_columns, [
            ("96.00%", "Best Accuracy"), ("0.9629", "Precision"),
            ("0.9600", "Recall"), ("0.9599", "F1-Score"), ("373", "Clinical Records")
        ]):
            column.markdown(
                f"<div class='sq-stat'><div class='n'>{value}</div><div class='l'>{label}</div></div>",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([1.1, 0.9])

        with left:
            st.markdown('<div class="section-header">System Architecture</div>', unsafe_allow_html=True)
            for icon, title, desc in [
                ("📋", "Clinical Input", "Age · MMSE · CDR · nWBV · Gender · EDUC · SES · eTIV · ASF"),
                ("⚙️", "Preprocessing", "Missing value imputation · Gender encoding · Z-score normalization"),
                ("🔬", "Feature Engineering", "Cognitive Index · CDR-MMSE Product · Brain Volume Ratio · 6 derived features"),
                ("⚖️", "SMOTE Balancing", "Synthetic minority oversampling on training set only"),
                ("🌲", "ML Models", "Random Forest · SVM (Linear/RBF/Poly) · Voting Ensemble"),
                ("📊", "Evaluation", "Accuracy · Precision · Recall · F1 · ROC/AUC · Confusion Matrix"),
                ("💡", "Explainability", "Feature importance ranking · Per-patient risk breakdown"),
            ]:
                st.markdown(f"""
                <div style='display:flex;align-items:flex-start;gap:0.8rem;
                            background:{t['card']};border:1px solid {t['border2']};
                            border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.45rem;'>
                    <div style='font-size:1.4rem;min-width:30px;'>{icon}</div>
                    <div>
                        <div style='color:{t['txt']};font-weight:700;font-size:1.02rem;'>{title}</div>
                        <div style='color:{t['txt2']};font-size:0.95rem;margin-top:0.15rem;'>{desc}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
            models_d = ['Voting Ensemble', 'Random Forest', 'SVM Linear', 'SVM Polynomial', 'SVM RBF']
            accs_d = [96.00, 94.67, 94.67, 94.67, 90.67]
            colors_d = [t['blue'], t['green'], t['purple'], t['orange'], t['pink']]
            fig = go.Figure(go.Bar(
                x=accs_d, y=models_d, orientation='h',
                marker=dict(color=colors_d),
                text=[f"{v:.1f}%" for v in accs_d],
                textposition='outside', textfont=dict(color=t['txt'], size=12),
            ))
            fig.add_vline(x=93, line_dash="dash", line_color=t['orange'],
                          annotation_text="Clinical Threshold 93%",
                          annotation_font_color=t['orange'], annotation_font_size=11)
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=t['txt'], family='Nunito'),
                xaxis=dict(range=[80, 100], gridcolor=t['grid'], ticksuffix='%'),
                yaxis=dict(gridcolor=t['grid']),
                margin=dict(l=10, r=70, t=10, b=10), height=280, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, key="home_model_chart")

            st.markdown('<div class="section-header">Course Units Covered</div>', unsafe_allow_html=True)
            for unit, desc in [
                ("Unit I", "ML Basics, Data Types, Train/Test Split"),
                ("Unit II", "Normalization, K-Fold CV, PCA, LDA"),
                ("Unit III", "Confusion Matrix, ROC/AUC, Precision, Recall, F1"),
                ("Unit IV", "Gradient Descent, Bias-Variance, Regularization"),
                ("Unit V", "SVM (3 kernels), Random Forest, Feature Importance"),
            ]:
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;align-items:center;
                            padding:0.55rem 0.9rem;background:{t['card']};border-radius:10px;
                            margin-bottom:0.3rem;border:1px solid {t['border2']};'>
                    <span style='color:{t['blue']};font-weight:700;font-size:1rem;'>{unit}</span>
                    <span style='color:{t['txt2']};font-size:0.95rem;'>{desc}</span>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE — MEMORY QUESTIONNAIRE  (3 gentle steps)
# ══════════════════════════════════════════════════════════════
elif page == "📝 Cognitive Questionnaire":

    st.markdown("## 📝 Memory Questionnaire")
    friendly_card(
        "Simple questions, three short steps",
        "A caregiver or family member can help answer these questions. "
        "There are no wrong answers — this only helps us understand memory and daily life.",
        "🧠"
    )

    QUESTION_STEPS = ["1️⃣ About You", "2️⃣ Memory & Thinking", "3️⃣ Health Information"]
    current_step = int(st.session_state.get("q_step", 1))
    current_step = min(max(current_step, 1), 3)
    st.session_state.q_step = current_step

    pill_html = "".join(
        f"<span class='s {'on' if index + 1 == current_step else ('done' if index + 1 < current_step else '')}'>{label}</span>"
        for index, label in enumerate(QUESTION_STEPS)
    )
    st.markdown(f"<div class='sq-step'>{pill_html}</div>", unsafe_allow_html=True)
    st.progress(current_step / 3.0, text=f"Step {current_step} of 3")

    # ── STEP 1 — ABOUT YOU ────────────────────────────────────
    if current_step == 1:
        st.markdown("### 👤 About You")
        st.markdown(
            '<div class="sq-help">These details help us greet you properly and help your '
            'caregiver follow your progress. Nothing here is shared outside this app.</div>',
            unsafe_allow_html=True
        )

        with st.form("about_you_form"):
            c1, c2 = st.columns(2)
            with c1:
                about_name = st.text_input(
                    "🙋 Your name",
                    value=str(st.session_state.get("patient_name", "")),
                    placeholder="e.g. Bina Devi"
                )
                about_age = st.number_input(
                    "🎂 Your age (in years)", 40, 110,
                    int(st.session_state.get("patient_age", 70)), step=1
                )
            with c2:
                about_place = st.text_input(
                    "🏠 Your town or village",
                    value=str(st.session_state.get("home_place", "")),
                    placeholder="e.g. Shillong"
                )
                about_caregiver = st.text_input(
                    "👨‍👩‍👧 Who is helping you today?",
                    value=str(st.session_state.get("caregiver_name", "")),
                    placeholder="e.g. My daughter Anita"
                )
            if st.form_submit_button("💾 Save & Continue", use_container_width=True):
                st.session_state.patient_name = about_name.strip()
                st.session_state.patient_age = int(about_age)
                st.session_state.home_place = about_place.strip()
                st.session_state.caregiver_name = about_caregiver.strip()
                st.session_state.q_step = 2
                st.success("✅ Saved. Let us continue.")
                st.rerun()

    # ── STEP 2 — MEMORY & THINKING ────────────────────────────
    elif current_step == 2:
        st.markdown("### 🧠 Memory & Thinking")
        st.markdown(
            '<div class="sq-help">🧠 <b>Memory and thinking score</b> — a score out of 30 that '
            'healthcare professionals use to describe memory, attention and language. '
            'A helper should ask these questions gently, one at a time.</div>',
            unsafe_allow_html=True
        )
        st.caption(
            "These are original screening tasks grouped by cognitive domain. "
            "They are not the official MMSE form."
        )

        with st.form("cognitive_screening_form"):
            st.markdown("#### 🧭 Knowing the day and place — 10 points")
            q1, q2 = st.columns(2)
            with q1:
                year_ok = st.radio("What year is it?", ["Correct", "Incorrect"], key="q_year", horizontal=True)
                month_ok = st.radio("What month is it?", ["Correct", "Incorrect"], key="q_month", horizontal=True)
                day_ok = st.radio("What day/date is it?", ["Correct", "Incorrect"], key="q_day", horizontal=True)
                season_ok = st.radio("Which season are we in?", ["Correct", "Incorrect"], key="q_season", horizontal=True)
                country_ok = st.radio("Which country are we in?", ["Correct", "Incorrect"], key="q_country", horizontal=True)
            with q2:
                state_ok = st.radio("Which state/region are we in?", ["Correct", "Incorrect"], key="q_state", horizontal=True)
                city_ok = st.radio("Which city/town are we in?", ["Correct", "Incorrect"], key="q_city", horizontal=True)
                place_ok = st.radio("What kind of place are you in right now?", ["Correct", "Incorrect"], key="q_place", horizontal=True)
                floor_ok = st.radio("Which floor/area are you on?", ["Correct", "Incorrect"], key="q_floor", horizontal=True)
                familiar_ok = st.radio("Who is the person accompanying you?", ["Correct", "Incorrect"], key="q_person", horizontal=True)

            st.markdown("#### 🗣️ Repeating three words — 3 points")
            st.write("Say three unrelated words aloud to the patient: **mango — chair — river**. Ask the patient to repeat them immediately.")
            reg = st.multiselect("Which words were repeated correctly?", ["mango", "chair", "river"], key="q_reg")

            st.markdown("#### 🔢 Attention — 5 points")
            st.write("Ask the patient to count backward from 30 by 3, stopping after five answers.")
            attention = st.number_input("Number of correct steps (0–5)", min_value=0, max_value=5, value=0, step=1, key="q_attention")

            st.markdown("#### 💭 Remembering the words later — 3 points")
            recall = st.multiselect("After a short delay, which of the three words can the patient recall?", ["mango", "chair", "river"], key="q_recall")

            st.markdown("#### 📖 Language and understanding — 9 points")
            l1 = st.radio("Names two everyday objects correctly?", ["Yes", "No"], key="q_name_objects", horizontal=True)
            l2 = st.radio("Follows a simple one-step instruction correctly?", ["Yes", "No"], key="q_one_step", horizontal=True)
            l3 = st.radio("Follows a second, different one-step instruction correctly?", ["Yes", "No"], key="q_second_step", horizontal=True)
            l4 = st.radio("Produces a meaningful spoken sentence?", ["Yes", "No"], key="q_sentence", horizontal=True)
            l5 = st.radio("Reads and follows a short, caregiver-created instruction?", ["Yes", "No"], key="q_read", horizontal=True)
            l6 = st.radio("Copies a simple geometric drawing shown by the assessor?", ["Yes", "No"], key="q_copy", horizontal=True)
            l7 = st.radio("Repeats a short, unfamiliar phrase accurately?", ["Yes", "No"], key="q_repeat", horizontal=True)
            l8 = st.radio("Understands a simple two-part instruction?", ["Yes", "No"], key="q_two_part", horizontal=True)
            l9 = st.radio("Can write a short meaningful sentence?", ["Yes", "No"], key="q_write", horizontal=True)

            calculate_screen = st.form_submit_button("🧮 See My Score", use_container_width=True)

        if calculate_screen:
            orientation = sum(x == "Correct" for x in [year_ok, month_ok, day_ok, season_ok, country_ok,
                                                       state_ok, city_ok, place_ok, floor_ok, familiar_ok])
            registration = len(reg)
            attention_score = int(attention)
            recall_score = len(recall)
            language_score = sum(x == "Yes" for x in [l1, l2, l3, l4, l5, l6, l7, l8, l9])
            screening_score = orientation + registration + attention_score + recall_score + language_score
            st.session_state.screening_mmse_score = int(min(30, screening_score))
            try:
                save_cognitive_record("screening", mmse=float(st.session_state.screening_mmse_score))
            except Exception:
                st.warning("⚠️ The score could not be saved to your history, but it is ready to use.")

            st.markdown(
                f"""
                <div class="sq-result sq-result-ok">
                    <div class="badge">🧠 Memory &amp; thinking score</div>
                    <div class="conf">{screening_score}/30</div>
                    <div class="note">Thank you — that is all for this step.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"**Where the points came from:** Knowing day &amp; place {orientation}/10 · "
                f"Repeating words {registration}/3 · Attention {attention_score}/5 · "
                f"Remembering later {recall_score}/3 · Language {language_score}/9"
            )
            st.caption("This is a screening score, not an official MMSE score. Please do not use it as a diagnosis.")

    # ── STEP 3 — HEALTH INFORMATION ───────────────────────────
    else:
        st.markdown("### 🩺 Health Information")
        st.markdown(
            '<div class="sq-help">📊 <b>Daily-life rating (CDR style)</b> — a rating used to describe '
            'changes in memory and everyday activities. A caregiver answers these, thinking about '
            'how things have changed from the person\'s usual self.</div>',
            unsafe_allow_html=True
        )

        rating_options = {
            "0 — No cognitive impairment": 0.0,
            "0.5 — Slight/questionable change": 0.5,
            "1 — Mild difficulty": 1.0,
            "2 — Moderate difficulty": 2.0,
            "3 — Severe/dependent": 3.0,
        }
        domain_questions = [
            ("Memory", "How much difficulty does the person have remembering recent conversations, events, or instructions?"),
            ("Orientation", "How much difficulty does the person have keeping track of date, place, or familiar surroundings?"),
            ("Judgment & problem solving", "How much help is needed to handle everyday decisions or an unexpected simple problem?"),
            ("Community affairs", "How much help is needed with shopping, money, appointments, travel, or other activities outside the home?"),
            ("Home & hobbies", "How much has the person reduced or needed help with usual household tasks, hobbies, or personal interests?"),
            ("Personal care", "How much help is needed with dressing, bathing, eating, toileting, or other basic self-care?"),
        ]

        with st.form("cdr_profile_form"):
            cdr_answers = {}
            for domain, question in domain_questions:
                cdr_answers[domain] = st.selectbox(
                    f"**{domain}** — {question}",
                    list(rating_options.keys()),
                    key=f"cdr_domain_{domain}"
                )
            calculate_cdr = st.form_submit_button("📊 See Daily-Life Rating", use_container_width=True)

        if calculate_cdr:
            domain_scores = {d: rating_options[a] for d, a in cdr_answers.items()}
            # A transparent provisional estimate — not the licensed CDR algorithm.
            provisional = float(np.median(list(domain_scores.values())))
            st.session_state.provisional_cdr = provisional
            try:
                save_cognitive_record("screening", mmse=st.session_state.screening_mmse_score,
                                      cdr=float(provisional))
            except Exception:
                st.warning("⚠️ The rating could not be saved to your history, but it is ready to use.")
            st.success(f"📊 Daily-life rating (CDR style): **{provisional:g}**")
            for domain, value in domain_scores.items():
                st.markdown(f"- **{domain}:** {value:g}")
            st.caption("This median-based estimate is NOT the official CDR scoring algorithm. "
                       "Official CDR scoring requires the licensed protocol and trained clinical judgement.")

        with st.expander("🩺 If a doctor has already given official scores"):
            st.write("Enter the official scores here. The Cognitive Check page will then use these values.")
            oc1, oc2 = st.columns(2)
            with oc1:
                official_mmse = st.number_input(
                    "Official MMSE score (0–30)", 0, 30,
                    value=int(st.session_state.official_mmse_score or 0),
                    key="official_mmse_entry",
                    help="Memory and thinking score used by healthcare professionals."
                )
                if st.button("✅ Use official MMSE score", use_container_width=True, key="use_official_mmse"):
                    st.session_state.official_mmse_score = int(official_mmse)
                    st.success("Official MMSE score saved.")
            with oc2:
                cdr_options = [0.0, 0.5, 1.0, 2.0, 3.0]
                saved_cdr = st.session_state.official_cdr_score
                try:
                    cdr_index = 0 if saved_cdr is None else cdr_options.index(float(saved_cdr))
                except (ValueError, TypeError):
                    cdr_index = 0
                official_cdr = st.selectbox(
                    "Official CDR global rating", cdr_options, index=cdr_index,
                    key="official_cdr_entry",
                    help="Rating used to describe changes in memory and daily activities."
                )
                if st.button("✅ Use official CDR rating", use_container_width=True, key="use_official_cdr"):
                    st.session_state.official_cdr_score = float(official_cdr)
                    st.success("Official CDR rating saved.")

        st.markdown("### 🔗 Ready for your Cognitive Check")
        v1, v2 = st.columns(2)
        with v1:
            score_text = "—" if st.session_state.screening_mmse_score is None else f"{st.session_state.screening_mmse_score}/30"
            st.markdown(f"<div class='sq-stat'><div class='n'>{score_text}</div>"
                        f"<div class='l'>🧠 Memory &amp; thinking score</div></div>", unsafe_allow_html=True)
        with v2:
            cdr_text = "—" if st.session_state.provisional_cdr is None else f"{st.session_state.provisional_cdr:g}"
            st.markdown(f"<div class='sq-stat'><div class='n'>{cdr_text}</div>"
                        f"<div class='l'>📊 Daily-life rating</div></div>", unsafe_allow_html=True)

        if (st.session_state.screening_mmse_score is not None
                and st.session_state.provisional_cdr is not None):
            st.success("✅ Both answers are ready. You can now open your Cognitive Check.")
            if st.button("🔬 Go to My Cognitive Check", use_container_width=True, key="q_goto_diagnosis"):
                go_to("🔬 Patient Diagnosis")

    # ── Step navigation ───────────────────────────────────────
    st.markdown("---")
    back_column, next_column = st.columns(2)
    with back_column:
        if current_step > 1:
            if st.button("⬅️ Previous Step", use_container_width=True, key="q_prev_step"):
                st.session_state.q_step = current_step - 1
                st.rerun()
    with next_column:
        if current_step < 3:
            if st.button("Next Step ➡️", use_container_width=True, key="q_next_step"):
                st.session_state.q_step = current_step + 1
                st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE — COGNITIVE CHECK  (AI-assisted screening)
# ══════════════════════════════════════════════════════════════
elif page == "🔬 Patient Diagnosis":

    st.markdown("""
    <div class="sq-hero">
        <div class="sq-hello">🔬 My Cognitive Check</div>
        <div class="sq-hello-sub">We use your questionnaire answers and a few health measurements
        to give you a clear, easy-to-read screening result.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sq-note">⚠️ <b>This is an AI-assisted screening result and should not replace '
        'evaluation by a qualified healthcare professional.</b></div>',
        unsafe_allow_html=True
    )

    if pipe is None:
        st.error("⚠️ The screening models are not available right now. Please try again later.")
    else:
        st.markdown("### 1️⃣ Your questionnaire answers")
        score_column, rating_column = st.columns(2)
        with score_column:
            if st.session_state.screening_mmse_score is None:
                st.warning("⚠️ Memory & thinking score missing. Please finish the Memory Questionnaire.")
                mmse = 0
            else:
                mmse = int(st.session_state.screening_mmse_score)
                st.markdown(f"<div class='sq-stat'><div class='n'>{mmse}/30</div>"
                            f"<div class='l'>🧠 Memory &amp; thinking score</div></div>",
                            unsafe_allow_html=True)
        with rating_column:
            if st.session_state.provisional_cdr is None:
                st.warning("⚠️ Daily-life rating missing. Please finish the Memory Questionnaire.")
                cdr = 0.0
            else:
                cdr = float(st.session_state.provisional_cdr)
                st.markdown(f"<div class='sq-stat'><div class='n'>{cdr:g}</div>"
                            f"<div class='l'>📊 Daily-life rating</div></div>",
                            unsafe_allow_html=True)

        scores_ready = (
            st.session_state.screening_mmse_score is not None
            and st.session_state.provisional_cdr is not None
        )

        if not scores_ready:
            if st.button("📝 Open the Memory Questionnaire", use_container_width=True, key="diag_goto_q"):
                go_to("📝 Cognitive Questionnaire")

        st.markdown("### 2️⃣ A few more details")
        st.markdown(
            '<div class="sq-help">If you do not have the brain-scan numbers, leave them as they are — '
            'the average values are already filled in for you.</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("🎂 Age (years)", 60, 98, int(min(98, max(60, st.session_state.get("patient_age", 75)))))
            educ = st.slider("🎓 Years of schooling", 6, 23, 14)
            gender = st.selectbox("🧑 Gender", ["Male", "Female"])
        with c2:
            ses = st.selectbox("🏡 Socioeconomic group (1 = highest, 5 = lowest)", [1, 2, 3, 4, 5], index=1)
            etiv = st.number_input("🧠 eTIV (mm³)", 1100, 2000, 1480,
                                   help="Estimated Total Intracranial Volume — the size of the space inside the skull.")
            nwbv = st.number_input("🧠 nWBV", 0.64, 0.84, 0.73, step=0.001, format="%.3f",
                                   help="Normalized Whole Brain Volume — how much brain tissue is present, compared with the skull size.")
        asf = st.number_input("📐 ASF", 0.88, 1.59, 1.20, step=0.001, format="%.3f",
                              help="Atlas Scaling Factor — a scan adjustment value used by radiologists.")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button(
            "🔍 Show My Result",
            use_container_width=True,
            disabled=not scores_ready,
            key="diag_run_btn"
        )

        if predict_btn:
            try:
                from sklearn.impute import SimpleImputer
                gender_enc = 1 if gender == "Male" else 0
                cog_idx = mmse * (1 - cdr * 0.5)
                cdr_mmse = cdr * (30 - mmse)
                mmse_diff = mmse - (cdr * 10)
                age_mmse = age / (mmse + 1)
                bvr = nwbv / (etiv / 1000)
                age_nwbv = age * (1 - nwbv)

                feat_arr = np.array([gender_enc, age, educ, ses, mmse, cdr, etiv, nwbv, asf,
                                     cog_idx, cdr_mmse, mmse_diff, age_mmse, bvr, age_nwbv]).reshape(1, -1)
                imp = SimpleImputer(strategy='median')
                imp.fit(pipe['X_train'])
                feat_scaled = pipe['scaler'].transform(imp.transform(feat_arr))

                model = pipe['ensemble']
                proba = model.predict_proba(feat_scaled)[0]
                pred_idx = int(np.argmax(proba))
                pred_cls = pipe['class_names'][pred_idx]
                conf = float(proba[pred_idx] * 100)

                st.session_state["prediction_done"] = True
                st.session_state["risk_confidence"] = conf
                st.session_state["prediction_class"] = pred_cls
                st.session_state["mmse_value"] = mmse
                st.session_state["cdr_value"] = cdr
                st.session_state["diag_result"] = {
                    "pred_cls": str(pred_cls),
                    "conf": conf,
                    "risk_pct": float(proba[0] * 100),
                    "explanation": None,
                    "next_steps": None,
                }

                try:
                    save_cognitive_record("diagnosis", mmse=float(mmse), cdr=float(cdr),
                                          confidence=float(conf), prediction=str(pred_cls))
                except Exception:
                    st.warning("⚠️ The result could not be added to your history, but it is shown below.")

                risk_result = "Demented" if pred_cls == 'Demented' else "Non-Demented"

                if client is not None:
                    with st.spinner("🧠 Preparing a simple explanation for you…"):
                        try:
                            prompt = f"""
You are a kind healthcare communicator writing for an elderly person in North East India
and their family caregiver.

The AI screening result is: {risk_result}

Write TWO short sections in very simple, warm, respectful language (no jargon, short sentences):

SECTION 1: What this result means, and what it does NOT mean.
---NEXT---
SECTION 2: What the person and family can do next: daily habits, memory-friendly routines,
and when to see a doctor.

Do not give a diagnosis. Do not frighten the reader. Use short bullet points where helpful.
"""
                            chat_completion = client.chat.completions.create(
                                messages=[{"role": "user", "content": prompt}],
                                model="openai/gpt-oss-20b",
                            )
                            ai_response = chat_completion.choices[0].message.content or ""
                            if "---NEXT---" in ai_response:
                                explanation, next_steps = ai_response.split("---NEXT---", 1)
                            else:
                                explanation, next_steps = ai_response, ""
                            st.session_state["diag_result"]["explanation"] = explanation.strip()
                            st.session_state["diag_result"]["next_steps"] = next_steps.strip()
                        except Exception:
                            st.info("ℹ️ The AI explanation is unavailable right now. "
                                    "The general guidance below still applies.")
            except Exception:
                st.error("⚠️ Something went wrong while calculating your result. Please try again.")

        # ── Result, kept on screen across reruns ──────────────
        result = st.session_state.get("diag_result")
        st.markdown("### 3️⃣ Your result")

        if not result:
            st.markdown(f"""
            <div style='background:{t['card']};border:2px dashed {t['border']};border-radius:24px;
                        padding:2.6rem;text-align:center;color:{t['txt2']};'>
                <div style='font-size:3.4rem;margin-bottom:.8rem;'>🩺</div>
                <div style='font-size:1.3rem;font-weight:800;color:{t['txt']};'>Your result will appear here</div>
                <div style='font-size:1.08rem;margin-top:0.6rem;'>
                    Fill in the details above, then press <b>Show My Result</b>.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            higher_risk = result["pred_cls"] == 'Demented'
            if higher_risk:
                st.markdown(f"""
                <div class="sq-result sq-result-watch">
                    <div style="font-size:1.25rem;font-weight:800;color:{t['txt2']};">Cognitive Assessment Result</div>
                    <div class="badge" style="color:{t['orange']};">🟠 Higher Risk — please speak to a doctor</div>
                    <div class="conf" style="color:{t['orange']};">Confidence: {result['conf']:.0f}%</div>
                    <div class="note">AI-assisted screening · not a medical diagnosis</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="sq-result sq-result-ok">
                    <div style="font-size:1.25rem;font-weight:800;color:{t['txt2']};">Cognitive Assessment Result</div>
                    <div class="badge" style="color:{t['green']};">🟢 Lower Risk</div>
                    <div class="conf" style="color:{t['green']};">Confidence: {result['conf']:.0f}%</div>
                    <div class="note">AI-assisted screening · not a medical diagnosis</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 💬 What does this mean?")
            if result.get("explanation"):
                st.markdown(result["explanation"])
            elif higher_risk:
                st.markdown(
                    "- This screening found signs that memory and thinking may be changing more "
                    "than usual for your age.\n"
                    "- It is **not** a diagnosis. Many things — tiredness, hearing, mood, thyroid "
                    "problems or medicines — can affect these answers.\n"
                    "- The most useful next step is a calm conversation with a doctor."
                )
            else:
                st.markdown(
                    "- This screening did not find strong signs of memory difficulty today.\n"
                    "- It is **not** a clean bill of health. Please repeat the check from time to time.\n"
                    "- Keep doing the daily activities and conversations you enjoy."
                )

            st.markdown("### 🌿 What can I do next?")
            if result.get("next_steps"):
                st.markdown(result["next_steps"])
            else:
                st.markdown(
                    "- Share this result with your doctor at the next visit.\n"
                    "- Keep a steady daily routine for sleep, meals and medicines.\n"
                    "- Stay socially active: talk with family, neighbours and friends.\n"
                    "- Walk gently every day if your doctor agrees.\n"
                    "- Use **Brain Games** and **My Memories** in this app a few times a week."
                )

            action_left, action_right = st.columns(2)
            with action_left:
                if st.button("🔊 Read this result aloud", use_container_width=True, key="diag_speak"):
                    spoken = ("Your screening result is higher risk. Please speak to a doctor."
                              if higher_risk else
                              "Your screening result is lower risk. Please keep following your daily routine.")
                    browser_speak(spoken)
            with action_right:
                if st.button("📈 See my progress over time", use_container_width=True, key="diag_goto_progress"):
                    go_to("🧠 Cognitive Performance")

            with st.expander("🔬 Technical details (for clinicians and reviewers)"):
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["risk_pct"],
                    title={'text': "Model risk estimate (%)", 'font': {'color': t['txt'], 'size': 14}},
                    number={'suffix': '%', 'font': {'color': t['txt'], 'size': 30}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': t['txt2']},
                        'bar': {'color': t['red'] if result["risk_pct"] > 50 else t['green']},
                        'bgcolor': t['gbg'], 'bordercolor': t['border'],
                        'steps': [
                            {'range': [0, 30], 'color': t['glow']},
                            {'range': [30, 60], 'color': t['gmid']},
                            {'range': [60, 100], 'color': t['ghigh']},
                        ],
                        'threshold': {'line': {'color': t['orange'], 'width': 3}, 'thickness': 0.8, 'value': 50}
                    }
                ))
                fig_g.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', font=dict(color=t['txt']),
                    height=250, margin=dict(l=20, r=20, t=40, b=10)
                )
                st.plotly_chart(fig_g, use_container_width=True, key="diag_gauge")

                st.markdown(f"**Predicted class:** {result['pred_cls']} · "
                            f"**Model:** Voting Ensemble · **Confidence:** {result['conf']:.2f}%")

                st.markdown("**Key clinical factors (Random Forest importance)**")
                importances = pipe['rf'].feature_importances_
                feat_names = pipe['feature_names']
                top_idx = np.argsort(importances)[::-1][:6]
                for idx in top_idx:
                    imp_pct = importances[idx] * 100
                    bar_w = min(int(imp_pct * 3), 100)
                    st.markdown(f"""
                    <div style='display:flex;align-items:center;gap:0.8rem;
                                padding:0.55rem 0.9rem;background:{t['card2']};
                                border-radius:10px;margin-bottom:0.3rem;border:1px solid {t['border2']};'>
                        <div style='color:{t['txt']};font-size:0.95rem;min-width:160px;font-weight:600;'>{feat_names[idx]}</div>
                        <div style='flex:1;background:{t['border2']};border-radius:4px;height:9px;'>
                            <div style='width:{bar_w}%;background:linear-gradient(90deg,{t['blue']},{t['purple']});height:9px;border-radius:4px;'></div>
                        </div>
                        <div style='color:{t['blue']};font-size:0.95rem;font-weight:700;min-width:46px;text-align:right;'>{imp_pct:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("**Reference ranges**")
                for name, desc in [
                    ("MMSE Score", "27–30 = Normal · 21–26 = Mild · 10–20 = Moderate · <10 = Severe"),
                    ("CDR Rating", "Official CDR uses 0, 0.5, 1, 2, 3. Use a licensed CDR assessment for the official rating."),
                    ("nWBV", "Normal: 0.72–0.84 · Lower values indicate brain atrophy"),
                ]:
                    st.markdown(f'<div class="info-box"><h4>{name}</h4><p>{desc}</p></div>',
                                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE — ASK SYNAPTIQ  (AI chatbot)
# ══════════════════════════════════════════════════════════════
elif page == "💬 AI Chatbot":

    st.markdown("""
    <div class="sq-hero">
        <div class="sq-hello">💬 How can I help you today?</div>
        <div class="sq-hello-sub">Ask anything about memory, brain health or this app.
        You can write in simple, everyday words.</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"🌐 {tr('Current app language')}: {LANGUAGE_DISPLAY_NAMES.get(APP_LANGUAGE, APP_LANGUAGE)}")

    def get_system_prompt(lang):
        base = """
You are SYNAPTIQ, a warm AI health companion for elderly people and their caregivers in
North East India. You speak simply and kindly, in short sentences, with no medical jargon.
You are accurate and never invent a diagnosis. When something sounds serious, you gently
suggest seeing a qualified doctor.
"""
        if lang != "English":
            base += f"\nRespond in {lang}."
        return base

    st.markdown("#### 💡 You can start with one of these")
    suggestion_columns = st.columns(2)
    SUGGESTIONS = [
        ("💭 Help me remember something", "Please give me simple ways to remember important things every day."),
        ("🧠 Explain my results", "Please explain in very simple words what a memory and thinking screening result means."),
        ("💊 What are my reminders?", "How can I use reminders well for medicines and meals every day?"),
        ("🛡️ I need help", "I am feeling unwell and a little confused. What should I do right now?"),
    ]
    for index, (label, prompt_text) in enumerate(SUGGESTIONS):
        with suggestion_columns[index % 2]:
            if st.button(label, use_container_width=True, key=f"chat_suggestion_{index}"):
                st.session_state.chat_pending = prompt_text
                st.rerun()

    st.markdown("---")

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    user_input = st.chat_input("Type your question here…")
    if st.session_state.chat_pending:
        user_input = st.session_state.chat_pending
        st.session_state.chat_pending = None

    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.chat_history.append(("user", user_input))

        messages = [{"role": "system", "content": get_system_prompt(APP_LANGUAGE)}]
        for role, msg in st.session_state.chat_history:
            messages.append({"role": role, "content": msg})

        with st.chat_message("assistant"):
            if client is None:
                reply = ("I cannot answer right now because the AI service is not set up on this "
                         "device. Please ask your helper to add a GROQ_API_KEY and restart the app.")
                st.warning(reply)
            else:
                with st.spinner("🧠 Thinking…"):
                    try:
                        response = client.chat.completions.create(
                            model="openai/gpt-oss-20b",
                            messages=messages
                        )
                        reply = response.choices[0].message.content
                        st.write(reply)
                    except Exception:
                        reply = ("⚠️ Something went wrong while answering. Please try asking again "
                                 "in a moment.")
                        st.warning(reply)

        st.session_state.chat_history.append(("assistant", reply))
        st.session_state.chat_last_reply = reply

    last_reply = st.session_state.get("chat_last_reply")
    if last_reply:
        if st.button("🔊 Read the last answer aloud", use_container_width=True,
                     key="chat_speak_last"):
            browser_speak(last_reply)

# ══════════════════════════════════════════════════════════════
# PAGE — MY LOCATION (GPS safety & geofencing)
# ══════════════════════════════════════════════════════════════
elif page == "📍 GPS Safety & Geofencing":
    render_gps_safety()

# ══════════════════════════════════════════════════════════════
# PAGE — MY PROGRESS (cognitive performance)
# ══════════════════════════════════════════════════════════════
elif page == "🧠 Cognitive Performance":
    render_cognitive_analytics()

# ══════════════════════════════════════════════════════════════
# PAGE — MY MEMORIES
# ══════════════════════════════════════════════════════════════
elif page == "📸 My Memories":

    st.markdown("""
    <div class="sq-hero">
        <div class="sq-hello">💭 My Memories</div>
        <div class="sq-hello-sub">The people, places and moments you love — with photos,
        a few warm words, and your own voice.</div>
    </div>
    """, unsafe_allow_html=True)

    action_columns = st.columns(3)
    with action_columns[0]:
        st.markdown('<div class="sq-card"><div class="sq-card-icon">📸</div><h3>Add Photo</h3>'
                    '<p>A clear picture of a familiar face or place.</p></div>', unsafe_allow_html=True)
    with action_columns[1]:
        st.markdown('<div class="sq-card"><div class="sq-card-icon">📝</div><h3>Add Memory</h3>'
                    '<p>A name, a relationship and one or two familiar sentences.</p></div>',
                    unsafe_allow_html=True)
    with action_columns[2]:
        st.markdown('<div class="sq-card"><div class="sq-card-icon">🎤</div><h3>Record Voice</h3>'
                    '<p>A short voice note, so the memory can be heard as well as read.</p></div>',
                    unsafe_allow_html=True)

    with st.expander("➕ Add a new memory", expanded=not st.session_state.memories):

        if mic_recorder is not None:
            st.markdown("#### 🎤 Record Voice (optional)")
            st.caption("Press the microphone, say a sentence about this memory, then press stop.")
            recorded = mic_recorder(
                start_prompt="🎤 Start Recording",
                stop_prompt="⏹️ Stop Recording",
                just_once=True,
                key="memory_voice_recorder"
            )
            if recorded and recorded.get("bytes"):
                st.session_state.memory_pending_voice = recorded["bytes"]
            if st.session_state.get("memory_pending_voice"):
                st.success("✅ Voice note ready. It will be saved with this memory.")
                st.audio(st.session_state.memory_pending_voice)
                if st.button("🗑️ Remove this voice note", key="memory_voice_clear"):
                    st.session_state.memory_pending_voice = None
                    st.rerun()
        else:
            st.caption("🎤 Voice notes need the streamlit-mic-recorder package. "
                       "Photos and written memories work as usual.")

        with st.form("memory_form", clear_on_submit=True):
            photo = st.file_uploader("📷 Photo", type=["png", "jpg", "jpeg"])
            c1, c2 = st.columns(2)
            with c1:
                person = st.text_input("👤 Name", placeholder="e.g. Anita")
                relation = st.text_input("👪 Relationship", placeholder="e.g. Daughter")
            with c2:
                place = st.text_input("📍 Place", placeholder="e.g. Shillong")
                event = st.text_input("📅 Important event", placeholder="e.g. Family trip in 2024")
            story = st.text_area("💭 Short story", placeholder="Write one or two familiar sentences…")
            submitted = st.form_submit_button("💾 Save This Memory", use_container_width=True)
            if submitted:
                if person.strip():
                    image_bytes = photo.getvalue() if photo else None
                    st.session_state.memories.append({
                        "name": person.strip(), "relation": relation.strip(),
                        "place": place.strip(), "event": event.strip(),
                        "story": story.strip(), "image": image_bytes,
                        "voice": st.session_state.get("memory_pending_voice"),
                        "date": datetime.now().strftime("%d %B %Y"),
                    })
                    st.session_state.memory_pending_voice = None
                    st.success("✅ Memory saved. Thank you for sharing it.")
                else:
                    st.warning("⚠️ Please enter a name before saving.")

    st.markdown("### 🖼️ My Memory Book")
    if not st.session_state.memories:
        friendly_card(
            "Your memory book is empty",
            "Use “Add a new memory” above to save the first photo and story. "
            "A caregiver or family member can help.",
            "🌱"
        )
    else:
        for index in range(0, len(st.session_state.memories), 2):
            memory_columns = st.columns(2)
            for column, offset in zip(memory_columns, [0, 1]):
                position = index + offset
                if position >= len(st.session_state.memories):
                    continue
                mem = st.session_state.memories[position]
                with column:
                    with st.container(border=True):
                        if mem.get("image"):
                            st.image(mem["image"], use_container_width=True)
                        st.markdown(f"### 👤 {mem.get('name', '')}")
                        if mem.get("date"):
                            st.caption(f"📅 Saved on {mem['date']}")
                        details = []
                        if mem.get("relation"):
                            details.append(f"**Relationship:** {mem['relation']}")
                        if mem.get("place"):
                            details.append(f"**Place:** {mem['place']}")
                        if mem.get("event"):
                            details.append(f"**Event:** {mem['event']}")
                        if details:
                            st.markdown("  \n".join(details))
                        if mem.get("story"):
                            st.markdown(f"💭 {mem['story']}")
                        if mem.get("voice"):
                            st.caption("🎤 Voice note")
                            st.audio(mem["voice"])
                        if st.button("🔊 Read this memory aloud", key=f"speak_mem_{position}",
                                     use_container_width=True):
                            browser_speak(
                                f"This is {mem.get('name', '')}. {mem.get('relation', '')}. {mem.get('story', '')}"
                            )
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 🧠 Remember Who This Is")
    if st.session_state.memories:
        st.markdown(
            '<div class="sq-help">A gentle, no-pressure activity. If the answer does not come, '
            'that is perfectly alright — we will show it together.</div>',
            unsafe_allow_html=True
        )
        q_index = st.selectbox(
            "Choose a memory for a gentle recall question",
            range(len(st.session_state.memories)),
            format_func=lambda i: st.session_state.memories[i]["name"],
            key="memory_question"
        )
        q_mem = st.session_state.memories[q_index]
        if q_mem.get("image"):
            st.image(q_mem["image"], width=320)
        st.markdown("#### ❓ Who is this person?")
        answer = st.text_input("Your answer", key="memory_answer", placeholder="Type the name here")
        if st.button("✅ Check My Answer", use_container_width=True, key="memory_check_answer"):
            if answer.strip().lower() == str(q_mem["name"]).lower():
                st.success("🎉 Wonderful — you remembered correctly. Well done!")
            else:
                st.info(f"That's alright. This is **{q_mem['name']}**, "
                        f"your {q_mem.get('relation') or 'family member'}. Let's look together.")
    else:
        st.caption("This activity becomes available once you have saved at least one memory.")

# ══════════════════════════════════════════════════════════════
# PAGE — CAREGIVER SUPPORT
# ══════════════════════════════════════════════════════════════
elif page == "👨‍👩‍👧 Caregiver Portal":

    st.markdown("""
    <div class="sq-hero">
        <div class="sq-hello">👨‍👩‍👧 Caregiver Support</div>
        <div class="sq-hello-sub">Everything a family member needs at a glance:
        who you are caring for, how today is going, and who to call.</div>
    </div>
    """, unsafe_allow_html=True)

    patient_name_display = str(st.session_state.get("patient_name", "") or "Not set yet")
    patient_age_display = st.session_state.get("patient_age", "—")
    today_iso = date.today().isoformat()
    todays_reminders = [r for r in st.session_state.reminders if r.get("date") == today_iso]

    latest_gps = latest_location()
    if latest_gps and latest_gps.get("status") == "Inside safe zone":
        safety_text, safety_icon = "🟢 Inside safe zone", "🛡️"
    elif latest_gps and latest_gps.get("status") == "Outside safe zone":
        safety_text, safety_icon = "⚠️ Outside safe zone", "🚨"
    elif latest_gps:
        safety_text, safety_icon = "🔵 No safe area set", "🛡️"
    else:
        safety_text, safety_icon = "🔵 No location yet", "🛡️"

    try:
        history_df = cognitive_history()
    except Exception:
        history_df = pd.DataFrame()

    game_sessions = 0
    latest_screening = "—"
    if not history_df.empty:
        game_sessions = int(history_df["record_type"].astype(str).str.startswith("game").sum())
        mmse_values = history_df["mmse"].dropna() if "mmse" in history_df else pd.Series(dtype=float)
        if not mmse_values.empty:
            latest_screening = f"{mmse_values.iloc[-1]:.0f}/30"

    overview_cards = [
        ("👤", "Patient", f"{patient_name_display} · Age {patient_age_display}"),
        ("❤️", "Current Status", f"Latest memory &amp; thinking score: {latest_screening}"),
        ("⏰", "Today's Reminders", f"{len(todays_reminders)} reminder(s) set for today"),
        (safety_icon, "Safety Status", safety_text),
        ("🧠", "Cognitive Activity", f"{game_sessions} brain-game session(s) recorded"),
        ("📈", "Progress", f"{len(st.session_state.memories)} memories · {len(st.session_state.important_contacts)} trusted contacts"),
    ]

    for row_start in range(0, len(overview_cards), 3):
        columns = st.columns(3)
        for column, (icon, title, value) in zip(columns, overview_cards[row_start:row_start + 3]):
            column.markdown(
                f"<div class='sq-card'><div class='sq-card-icon'>{icon}</div>"
                f"<h3>{title}</h3><p>{value}</p></div>",
                unsafe_allow_html=True
            )

    st.markdown("### 👤 Patient Information")
    with st.form("patient_profile"):
        c1, c2 = st.columns(2)
        with c1:
            patient_name = st.text_input("Patient name", value=st.session_state.get("patient_name", "My Family Member"))
            patient_age = st.number_input("Age", 50, 110, int(st.session_state.get("patient_age", 70)))
        with c2:
            caregiver_name = st.text_input("Primary caregiver", value=st.session_state.get("caregiver_name", "Anita"))
            home_place = st.text_input("Home / familiar place", value=st.session_state.get("home_place", "Home"))
        if st.form_submit_button("💾 Save Patient Details", use_container_width=True):
            st.session_state.patient_name = patient_name
            st.session_state.patient_age = patient_age
            st.session_state.caregiver_name = caregiver_name
            st.session_state.home_place = home_place
            st.success("✅ Details updated.")

    st.markdown("### 📞 Trusted Contacts")
    st.markdown(
        '<div class="sq-help">Add at least one contact with a working phone number. '
        'The patient can then call that person with one large button on the Safety page.</div>',
        unsafe_allow_html=True
    )
    with st.form("contact_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Name")
        relation = c2.text_input("Relationship")
        phone = c3.text_input("Phone number")
        if st.form_submit_button("➕ Add Contact", use_container_width=True) and name:
            st.session_state.important_contacts.append({"name": name, "relation": relation, "phone": phone})
            st.success("✅ Contact added.")

    if st.session_state.important_contacts:
        for contact in st.session_state.important_contacts:
            friendly_card(
                contact.get("name", "Contact"),
                f"{contact.get('relation', 'Contact')} · {contact.get('phone', 'No number')}",
                "📞"
            )
    else:
        friendly_card("No contacts yet", "Please add one trusted person with a phone number.", "📭")

    st.markdown("### ⏰ Today's Reminders")
    if todays_reminders:
        for reminder in sorted(todays_reminders, key=lambda x: x.get("time", "")):
            st.markdown(
                f"""
                <div class="sq-rem">
                    <div class="time">⏰ {html.escape(str(reminder.get('time', '')))}</div>
                    <div>
                        <div class="what">{html.escape(str(reminder.get('type', '')))} — {html.escape(str(reminder.get('text', '')))}</div>
                        <div class="note">{html.escape(str(reminder.get('note', '') or ''))}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        friendly_card("Nothing scheduled today", "Open My Reminders to add medicines, meals or family calls.", "🗓️")

    if st.button("💊 Open My Reminders", use_container_width=True, key="care_goto_reminders"):
        go_to("⏰ Smart Reminders")

    with st.expander("📋 Recent brain-game activity"):
        if st.session_state.game_history:
            st.dataframe(pd.DataFrame(st.session_state.game_history),
                         use_container_width=True, hide_index=True)
        elif not history_df.empty and game_sessions:
            game_rows = history_df[history_df["record_type"].astype(str).str.startswith("game")]
            show_cols = [c for c in ["recorded_at", "game_name", "game_score",
                                     "game_attempts", "game_accuracy"] if c in game_rows.columns]
            st.dataframe(game_rows[show_cols].sort_values("recorded_at", ascending=False),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No brain-game sessions recorded yet.")

    st.markdown("---")
    render_caregiver_location_monitor()

# ══════════════════════════════════════════════════════════════
# PAGE — MY REMINDERS
# ══════════════════════════════════════════════════════════════
elif page == "⏰ Smart Reminders":

    now = datetime.now()

    st.markdown(f"""
    <div class="sq-hero">
        <div class="sq-hello">💊 My Reminders</div>
        <div class="sq-hello-sub">Today is {now.strftime('%A, %d %B %Y')}.
        When the time comes, SYNAPTIQ shows a large message and reads it aloud for you.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("➕ Add a new reminder", expanded=not st.session_state.reminders):
        with st.form("reminder_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                reminder_text = st.text_input("📝 What should we remember?", placeholder="e.g. Take morning medicine")
                reminder_type = st.selectbox("🏷️ What kind of reminder?", [
                    "Medicine", "Meal", "Water", "Doctor appointment",
                    "Family call", "Daily activity", "Sleep", "Other"
                ])
                reminder_date = st.date_input("📅 Which day?", value=date.today(), min_value=date.today())
            with c2:
                reminder_time = st.time_input("⏰ What time?", value=dt_time(9, 0))
                reminder_note = st.text_input("💬 Short note", placeholder="e.g. After breakfast")
            if st.form_submit_button("➕ Set This Reminder", use_container_width=True):
                if reminder_text:
                    st.session_state.reminders.append({
                        "id": f"rem-{datetime.now().timestamp()}",
                        "date": reminder_date.isoformat(),
                        "time": reminder_time.strftime("%H:%M"),
                        "type": reminder_type,
                        "text": reminder_text,
                        "note": reminder_note
                    })
                    st.success(f"✅ Reminder set for {reminder_date.strftime('%d %B %Y')} "
                               f"at {reminder_time.strftime('%I:%M %p')}.")
                else:
                    st.warning("⚠️ Please write what you want to remember.")

    browser_reminder_scheduler(st.session_state.reminders, APP_LANG_CODE)

    TYPE_ICONS = {
        "Medicine": "💊", "Meal": "🍵", "Water": "💧", "Doctor appointment": "🩺",
        "Family call": "📞", "Daily activity": "🌞", "Sleep": "🌙", "Other": "🔔",
    }

    st.markdown("### 🗓️ My Reminder Board")
    if not st.session_state.reminders:
        friendly_card("No reminders yet", "Use “Add a new reminder” above to set your first one.", "🔔")
    else:
        sorted_reminders = sorted(
            st.session_state.reminders,
            key=lambda x: (x.get("date", ""), x.get("time", ""))
        )
        for i, rem in enumerate(sorted_reminders):
            try:
                display_date = datetime.strptime(
                    rem.get("date", date.today().isoformat()), "%Y-%m-%d"
                ).strftime("%d %b %Y")
            except Exception:
                display_date = rem.get("date", "")
            icon = TYPE_ICONS.get(rem.get("type", "Other"), "🔔")
            try:
                pretty_time = datetime.strptime(rem.get("time", "09:00"), "%H:%M").strftime("%I:%M %p")
            except Exception:
                pretty_time = rem.get("time", "")

            st.markdown(
                f"""
                <div class="sq-rem">
                    <div class="time">⏰ {html.escape(pretty_time)}<div class="note">{html.escape(display_date)}</div></div>
                    <div>
                        <div class="what">{icon} {html.escape(str(rem.get('type', '')))} — {html.escape(str(rem.get('text', '')))}</div>
                        <div class="note">{html.escape(str(rem.get('note', '') or ''))}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            button_columns = st.columns([1, 1, 3])
            with button_columns[0]:
                if st.button("🔊 Read aloud", key=f"speak_rem_{i}", use_container_width=True):
                    browser_speak(
                        f"Reminder for {display_date} at {pretty_time}. "
                        f"{rem.get('text', '')}. {rem.get('note', '')}"
                    )
            with button_columns[1]:
                if st.button("🗑️ Remove", key=f"delete_rem_{i}", use_container_width=True):
                    st.session_state.reminders = [
                        r for r in st.session_state.reminders if r.get("id") != rem.get("id")
                    ]
                    st.success("✅ Reminder removed.")
                    st.rerun()

    st.markdown("### 🌞 My Daily Routine")
    st.caption("A gentle rhythm for the day. Your caregiver can adjust this with you.")
    for routine_time, task in st.session_state.daily_routine:
        try:
            pretty = datetime.strptime(routine_time, "%H:%M").strftime("%I:%M %p")
        except Exception:
            pretty = routine_time
        st.markdown(
            f"""
            <div class="sq-rem">
                <div class="time">🕒 {html.escape(pretty)}</div>
                <div><div class="what">{html.escape(str(task))}</div></div>
            </div>
            """,
            unsafe_allow_html=True
        )

elif page == "🌏 NER Memory & Culture":
    st.markdown("## 🌏 North Eastern Region Memory & Culture")
    friendly_card(
        "Personalized visited-place quiz",
        "The caregiver uploads photographs of places the patient has actually visited. "
        "Those uploaded photographs are then used directly in a simple recognition quiz for the patient.",
        "📸"
    )

    language = APP_LANGUAGE
    st.caption(f"🌐 {tr('Current app language')}: {APP_LANGUAGE}")

    st.markdown("### 👨‍👩‍👧 Caregiver: Add Patient's Visited Places")

    if st.session_state.get("username") == "caregiver":
        st.success(
            "Caregiver mode is active. Upload a real photograph of a place the patient has visited."
        )

        with st.form("ner_place_upload_form", clear_on_submit=True):
            uploaded_image = st.file_uploader(
                "📤 Upload place photograph",
                type=["png", "jpg", "jpeg", "webp"],
                help="Use a familiar place photo that the caregiver has permission to use."
            )

            c1, c2 = st.columns(2)
            with c1:
                place_name = st.text_input(
                    "📍 Place name",
                    placeholder="e.g. Shillong"
                )
                place_region = st.text_input(
                    "🗺️ State / region",
                    placeholder="e.g. Meghalaya"
                )

            with c2:
                memory_hint = st.text_input(
                    "💭 Memory hint",
                    placeholder="e.g. We visited this place on our family trip"
                )
                quiz_question = st.text_input(
                    "❓ Quiz question",
                    value="Do you remember where we visited?"
                )

            save_place = st.form_submit_button(
                "📸 Upload & Save Place",
                use_container_width=True
            )

            if save_place:
                if uploaded_image and place_name.strip():
                    st.session_state.ner_uploaded_places.append({
                        "name": place_name.strip(),
                        "region": place_region.strip(),
                        "hint": memory_hint.strip(),
                        "question": quiz_question.strip() or "Do you remember where we visited?",
                        "image": uploaded_image.getvalue()
                    })
                    st.success(
                        f"✅ {place_name.strip()} has been uploaded for the patient's memory quiz."
                    )
                    st.rerun()
                else:
                    st.warning("Please upload a photo and enter the place name.")

    else:
        st.info(
            "👤 Patient mode: the caregiver must log in as 'caregiver' to upload or manage "
            "visited-place photos. The patient can take the quiz below."
        )

    if st.session_state.ner_uploaded_places:
        st.markdown("### 🖼️ Uploaded Places")
        st.caption(
            "These are the actual place photographs saved by the caregiver for this patient."
        )

        cols = st.columns(min(3, len(st.session_state.ner_uploaded_places)))
        for i, place_mem in enumerate(st.session_state.ner_uploaded_places):
            with cols[i % len(cols)]:
                st.image(place_mem["image"], use_container_width=True)
                st.markdown(f"**📍 {place_mem['name']}**")
                if place_mem.get("region"):
                    st.caption(place_mem["region"])
                if place_mem.get("hint"):
                    st.caption(f"💭 {place_mem['hint']}")

        st.markdown("### 🧠 Patient: Quiz From Uploaded Photo")
        st.info(
            "The patient is shown a photograph uploaded by the caregiver and answers a question "
            "about that familiar place."
        )

        quiz_index = st.selectbox(
            "Choose a saved photo for the quiz",
            range(len(st.session_state.ner_uploaded_places)),
            format_func=lambda i: (
                f"Photo {i + 1} — {st.session_state.ner_uploaded_places[i]['name']}"
            ),
            key="ner_quiz_photo"
        )
        quiz_mem = st.session_state.ner_uploaded_places[quiz_index]

        st.image(
            quiz_mem["image"],
            caption="🖼️ Patient's familiar place",
            use_container_width=True
        )

        st.markdown(
            f"### ❓ {quiz_mem.get('question') or 'Do you remember where we visited?'}"
        )

        names = list(dict.fromkeys(
            x["name"] for x in st.session_state.ner_uploaded_places
            if x.get("name")
        ))

        if len(names) >= 2:
            options = names[:]
            if len(options) > 4:
                distractors = [n for n in options if n != quiz_mem["name"]]
                distractors = random.sample(distractors, min(3, len(distractors)))
                options = [quiz_mem["name"]] + distractors

            random.shuffle(options)

            selected = st.radio(
                "🧓 Which place is shown in this photograph?",
                options,
                key=f"ner_quiz_answer_{quiz_index}"
            )

            if st.button(
                "✅ Check My Answer",
                use_container_width=True,
                key=f"ner_check_{quiz_index}"
            ):
                if selected == quiz_mem["name"]:
                    st.success("🎉 Wonderful! You recognized the place.")
                    if quiz_mem.get("hint"):
                        st.info(f"💭 Memory hint: {quiz_mem['hint']}")
                    browser_speak(
                        f"Wonderful. You recognized {quiz_mem['name']}."
                    )
                else:
                    st.info(
                        f"That's okay. This photograph is **{quiz_mem['name']}**. "
                        "Let's look at it together and try again."
                    )
                    if quiz_mem.get("hint"):
                        st.info(f"💭 Memory hint: {quiz_mem['hint']}")
        else:
            answer = st.text_input(
                "📝 What place is shown in this photograph?",
                key=f"ner_quiz_text_{quiz_index}"
            )
            if st.button(
                "✅ Check My Answer",
                use_container_width=True,
                key=f"ner_check_text_{quiz_index}"
            ):
                if answer.strip().lower() == quiz_mem["name"].lower():
                    st.success("🎉 Wonderful! You recognized the place.")
                    browser_speak(
                        f"Wonderful. You recognized {quiz_mem['name']}."
                    )
                else:
                    st.info(
                        f"That's okay. This photograph is **{quiz_mem['name']}**."
                    )
                    if quiz_mem.get("hint"):
                        st.info(f"💭 Memory hint: {quiz_mem['hint']}")
    else:
        st.warning(
            "📭 No visited-place photographs have been uploaded yet. "
            "Log in as the caregiver and upload the patient's familiar places first."
        )

# ══════════════════════════════════════════════════════════════
# PAGE — SAFETY & SOS
# ══════════════════════════════════════════════════════════════
elif page == "🆘 Safety / SOS":

    st.markdown("""
    <div class="sq-hero">
        <div class="sq-hello">🛡️ Safety &amp; Emergency Help</div>
        <div class="sq-hello-sub">If you feel unwell, lost or frightened, you can reach
        someone you trust from this page.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sq-note">🆘 <b>Press the big red button if you need immediate help.</b><br>'
        'It records the time, speaks a message out loud, and shows you the large green '
        'call buttons for your trusted contacts. It does <b>not</b> call emergency services '
        'by itself.</div>',
        unsafe_allow_html=True
    )

    if danger_button("🆘 EMERGENCY HELP — I NEED HELP NOW", key="sos_help_button"):
        st.session_state.sos_message = (
            f"SOS requested at {datetime.now().strftime('%d %B %Y, %I:%M %p')}"
        )
        st.error("🚨 Help requested. Please use one of the large green Call buttons below "
                 "to reach your trusted contact.")
        browser_speak("Help is needed. Please call your saved caregiver.")

    if st.session_state.sos_message:
        st.info(f"🕒 {st.session_state.sos_message}")

    st.markdown("### 📞 Call Someone I Trust")

    callable_contacts = []
    for contact in st.session_state.important_contacts:
        phone = str(contact.get("phone", "")).strip()
        digits = "".join(ch for ch in phone if ch.isdigit())

        if len(digits) >= 7:
            callable_contacts.append(contact)
            safe_phone = html.escape(phone, quote=True)
            safe_name = html.escape(str(contact.get("name", "Contact")))
            safe_relation = html.escape(str(contact.get("relation", "") or "Trusted contact"))

            st.markdown(
                f"""
                <div class="sq-card">
                    <h3>📞 {safe_name}</h3>
                    <p>{safe_relation} · {html.escape(phone)}</p>
                </div>
                <a class="sq-call" href="tel:{safe_phone}">📞 CALL {safe_name.upper()}</a>
                """,
                unsafe_allow_html=True
            )
        else:
            st.caption(
                f"⚠️ {contact.get('name', 'Contact')} needs a valid phone number "
                "in Caregiver Support before calling can be switched on."
            )

    if not callable_contacts:
        friendly_card(
            "No one to call yet",
            "Ask your caregiver to add a trusted person's name, relationship and phone number "
            "in Caregiver Support. The call button will then appear here.",
            "📭"
        )
        if st.button("👨‍👩‍👧 Open Caregiver Support", use_container_width=True, key="sos_goto_care"):
            go_to("👨‍👩‍👧 Caregiver Portal")

    st.markdown("### 📍 Where am I?")
    latest_safety = latest_location()
    if latest_safety and latest_safety.get("status") == "Inside safe zone":
        st.markdown('<div class="sq-status sq-status-ok">🟢 Inside Safe Zone</div>', unsafe_allow_html=True)
    elif latest_safety and latest_safety.get("status") == "Outside safe zone":
        st.markdown('<div class="sq-status sq-status-warn">⚠️ Please check your location</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="sq-status sq-status-info">🔵 Location not checked yet</div>',
                    unsafe_allow_html=True)

    if st.button("📍 Check My Safety Location", use_container_width=True, key="sos_goto_gps"):
        go_to("📍 GPS Safety & Geofencing")

    st.markdown(
        '<div class="sq-note">⚠️ <b>Please remember:</b> this is a prototype safety helper. '
        'The call buttons open your phone\'s dialler where the device supports it. '
        'In a real emergency, please also call your local emergency number.</div>',
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════
# PAGE — MY RISK TIMELINE
# ══════════════════════════════════════════════════════════════
elif page == "🧠 Risk Timeline":

    st.markdown("""
    <div class="sq-hero">
        <div class="sq-hello">🗺️ My Risk Timeline</div>
        <div class="sq-hello-sub">A simple picture of the stage your latest screening suggests,
        and what the stages around it look like.</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("prediction_done"):
        friendly_card(
            "Your timeline is not ready yet",
            "Please complete your Cognitive Check first. Your result will then be shown here "
            "as an easy-to-follow timeline.",
            "📝"
        )
        if st.button("🔬 Open My Cognitive Check", use_container_width=True, key="timeline_goto_diag"):
            go_to("🔬 Patient Diagnosis")
    else:
        conf = st.session_state.get("risk_confidence", 0.0)
        mmse = st.session_state.get("mmse_value", 0)
        cdr = st.session_state.get("cdr_value", 0.0)

        if mmse >= 27:
            stage, stage_name, stage_icon = 1, "No Cognitive Decline", "🟢"
        elif mmse >= 24:
            stage, stage_name, stage_icon = 2, "Very Mild Cognitive Decline", "🔵"
        elif mmse >= 20:
            stage, stage_name, stage_icon = 3, "Mild Cognitive Decline", "🟡"
        elif mmse >= 15:
            stage, stage_name, stage_icon = 4, "Moderate Cognitive Decline", "🟠"
        elif mmse >= 10:
            stage, stage_name, stage_icon = 5, "Moderately Severe Decline", "🟠"
        elif mmse >= 5:
            stage, stage_name, stage_icon = 6, "Severe Cognitive Decline", "🔴"
        else:
            stage, stage_name, stage_icon = 7, "Very Severe Cognitive Decline", "🔴"

        st.markdown(
            f"""
            <div class="sq-result sq-result-ok">
                <div style="font-size:1.2rem;font-weight:800;color:{t['txt2']};">Your latest screening suggests</div>
                <div class="badge">{stage_icon} {stage_name}</div>
                <div class="note">Stage {stage} of 7 · screening support only, not a diagnosis</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='sq-stat'><div class='n'>{mmse}/30</div>"
                        f"<div class='l'>🧠 Memory &amp; thinking score</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='sq-stat'><div class='n'>{cdr:g}</div>"
                        f"<div class='l'>📊 Daily-life rating</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='sq-stat'><div class='n'>{conf:.0f}%</div>"
                        f"<div class='l'>🔎 Model confidence</div></div>", unsafe_allow_html=True)

        st.markdown("### 📊 Move through the stages")
        st.caption("Slide to look at earlier or later stages. This is for understanding only — "
                   "it does not change your result.")

        labels = [
            "No Decline", "Very Mild", "Mild", "Moderate",
            "Moderately Severe", "Severe", "Very Severe"
        ]

        selected_stage = st.slider("Stage to look at", 1, 7, stage, key="timeline_stage_slider")
        st.progress(selected_stage / 7)

        stage_columns = st.columns(7)
        for i in range(7):
            with stage_columns[i]:
                if i + 1 < selected_stage:
                    st.success(f"{i+1}\n\n{labels[i]}")
                elif i + 1 == selected_stage:
                    st.info(f"{i+1}\n\n➡️ {labels[i]}")
                else:
                    st.markdown(
                        f"<div class='sq-stat'><div class='n' style='font-size:1.3rem;'>{i+1}</div>"
                        f"<div class='l'>{labels[i]}</div></div>",
                        unsafe_allow_html=True
                    )

        st.markdown("### 💬 What you are looking at")
        if selected_stage < stage:
            st.success("🟢 This is an earlier stage than your latest screening suggests.")
        elif selected_stage == stage:
            st.info(f"🔵 This is the stage your latest screening suggests: **{stage_name}**.")
        else:
            st.warning("🟠 This is a later stage than your latest screening suggests.")

        st.markdown(
            f"""
            **In simple words:** your latest screening places you at **{stage_name}**.
            Following the same routine, staying socially active, and repeating this check
            from time to time are the most useful next steps.
            """
        )

        st.markdown(
            '<div class="sq-note">⚠️ This timeline is built from a screening score and a machine '
            'learning estimate. It is not a clinical staging assessment. Please discuss it with '
            'a qualified doctor.</div>',
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════
# PAGE — SPEAK TO SYNAPTIQ  (voice assistant)
# ══════════════════════════════════════════════════════════════
elif page == "🎤 Voice Assistant":

    st.markdown("""
    <div class="sq-hero">
        <div class="sq-hello">🎤 Tap to Speak</div>
        <div class="sq-hello-sub">You do not have to type anything. Press the microphone,
        speak normally, then press stop. SYNAPTIQ will listen and answer.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sq-help">🎧 <b>How it works:</b> 1) Press <b>Start Recording</b>. '
        '2) Say your question. 3) Press <b>Stop Recording</b>. Then wait a moment for the answer.</div>',
        unsafe_allow_html=True
    )

    if mic_recorder is None:
        st.warning(
            "⚠️ Voice recording is not installed on this computer. "
            "Ask a helper to install the streamlit-mic-recorder package. "
            "In the meantime you can use 💬 Ask SYNAPTIQ to type your question."
        )
        if st.button("💬 Type my question instead", use_container_width=True, key="voice_goto_chat"):
            go_to("💬 AI Chatbot")
    elif client is None:
        st.warning(
            "⚠️ The AI voice service is not set up on this device. "
            "Ask a helper to add a GROQ_API_KEY and restart the app."
        )
    else:
        st.markdown("<div style='text-align:center;font-size:5rem;line-height:1;'>🎤</div>",
                    unsafe_allow_html=True)

        audio = mic_recorder(
            start_prompt="🎤 Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=True,
            key="voice_assistant_recorder"
        )

        status = st.empty()

        if audio and audio.get("bytes"):
            status.info("👂 I'm listening… please wait a moment.")

            voice_text = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(audio["bytes"])
                    temp_path = temp_file.name

                with open(temp_path, "rb") as file_handle:
                    transcription = client.audio.transcriptions.create(
                        file=file_handle,
                        model="whisper-large-v3"
                    )
                voice_text = getattr(transcription, "text", "") or ""
            except Exception:
                status.empty()
                st.error("⚠️ Something went wrong while listening. Please try recording again.")

            if voice_text:
                status.success("✅ I heard you clearly.")

                st.markdown("### 📝 What I heard")
                with st.container(border=True):
                    st.markdown(f"#### “{voice_text}”")

                ai_response = None
                with st.spinner("🧠 Preparing your answer…"):
                    try:
                        prompt = f"""
You are SYNAPTIQ, a warm AI health companion for elderly people in North East India.
Answer in simple, short, kind sentences. No jargon. Never give a diagnosis; gently suggest
seeing a qualified doctor when something sounds serious.
Respond in {APP_LANGUAGE}.

The person said: {voice_text}
"""
                        chat_completion = client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model="openai/gpt-oss-20b",
                        )
                        ai_response = chat_completion.choices[0].message.content
                    except Exception:
                        st.error("⚠️ Something went wrong while answering. Please try again.")

                if ai_response:
                    st.markdown("### 💬 My Answer")
                    with st.container(border=True):
                        st.markdown(ai_response)
                    if st.button("🔊 Read the answer aloud", use_container_width=True, key="voice_speak_answer"):
                        browser_speak(ai_response, APP_LANG_CODE)
        else:
            st.info("🎧 Press **Start Recording** above whenever you are ready.")

# ══════════════════════════════════════════════════════════════
# PAGE — BRAIN GAMES  (original SYNAPTIQ cognitive activities)
# ══════════════════════════════════════════════════════════════
# These activities are original SYNAPTIQ task-inspired games. They use
# interaction patterns similar to established digital cognitive paradigms,
# but do not copy proprietary UI, graphics, wording, scoring or test assets.
# ══════════════════════════════════════════════════════════════
elif page == "🧩 Cognitive Games":

    def _cg_save(name, score, attempts, accuracy):
        """Store a game result using the existing cognitive_records table."""
        try:
            save_cognitive_record(
                record_type=f"game:{name}",
                game_score=float(score),
                game_attempts=int(attempts),
                game_accuracy=float(accuracy),
                game_name=str(name),
            )
            return True
        except Exception:
            return False

    def _cg_stats(items):
        """Show a large, readable strip of live game statistics."""
        columns = st.columns(len(items))
        for column, (value, label) in zip(columns, items):
            column.markdown(
                f"<div class='sq-stat'><div class='n'>{value}</div><div class='l'>{label}</div></div>",
                unsafe_allow_html=True
            )

    def _cg_demo_gate(demo_function, demo_key):
        """Large 'Watch Demonstration' control shown before every game."""
        state_key = f"cg_demo_{demo_key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = False

        label = "🙈 Hide Demonstration" if st.session_state[state_key] else "▶️ Watch Demonstration"
        if st.button(label, use_container_width=True, key=f"cg_demo_btn_{demo_key}"):
            st.session_state[state_key] = not st.session_state[state_key]
            st.rerun()

        if st.session_state[state_key]:
            st.markdown(
                '<div class="sq-help">🎬 Watch the moving demonstration once. '
                'It repeats by itself. When you are ready, press <b>Start</b> below.</div>',
                unsafe_allow_html=True
            )
            try:
                demo_function()
            except Exception:
                st.info("ℹ️ The demonstration could not be shown. You can still play the game below.")
        else:
            st.caption("Not sure how this works? Press “Watch Demonstration” first — "
                       "it only takes a few seconds.")

    st.markdown("""
    <div class="sq-hero">
        <div class="sq-hello">🎮 Brain Games</div>
        <div class="sq-hello-sub">Short, friendly activities for attention, memory and quick thinking.
        Play as often as you like — there is no pass or fail here.</div>
        <span class="sq-pill">🎬 Watch a demonstration first</span>
        <span class="sq-pill">🖐️ Large buttons</span>
        <span class="sq-pill">📈 Your progress is saved</span>
    </div>
    """, unsafe_allow_html=True)

    GAME_CARDS = [
        ("🎯", "Quick Tap", "Test reaction speed", "⚡ Reaction Matrix"),
        ("🎨", "Color Check", "Practice attention", "🎨 Color Conflict"),
        ("🃏", "Card Memory", "Practice visual memory", "🟪 Pattern Memory"),
        ("🔄", "Previous Card", "Practice working memory", "🔄 Two-Back"),
        ("🧩", "Memory Path", "Practice planning and memory", "🧩 Hidden Path Maze"),
    ]

    GAME_TITLES = {internal: (icon, name, blurb) for icon, name, blurb, internal in GAME_CARDS}

    if "cg_choice" not in st.session_state:
        st.session_state.cg_choice = "⚡ Reaction Matrix"
    if st.session_state.cg_choice not in GAME_TITLES:
        st.session_state.cg_choice = "⚡ Reaction Matrix"

    st.markdown("### 🕹️ Choose an activity")
    for row_start in range(0, len(GAME_CARDS), 3):
        card_columns = st.columns(3)
        for column, card in zip(card_columns, GAME_CARDS[row_start:row_start + 3]):
            icon, name, blurb, internal = card
            selected = st.session_state.cg_choice == internal
            with column:
                st.markdown(
                    f"""
                    <div class="sq-feature" style="min-height:170px;
                         border-color:{t['btn1'] if selected else t['border2']};">
                        <div class="sq-card-icon">{icon}</div>
                        <h3 style="font-size:1.32rem;font-weight:900;margin:.2rem 0 .35rem 0;">{name}</h3>
                        <p style="font-size:1.05rem;">{blurb}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                button_label = f"✅ {name} (selected)" if selected else f"{icon} Choose {name}"
                if st.button(button_label, use_container_width=True, key=f"cg_pick_{internal}"):
                    st.session_state.cg_choice = internal
                    st.rerun()

    game = st.session_state.cg_choice
    chosen_icon, chosen_name, chosen_blurb = GAME_TITLES[game]

    st.markdown("---")
    st.markdown(f"## {chosen_icon} {chosen_name}")
    st.markdown(f"<div class='sq-help'>🎯 <b>What this practises:</b> {chosen_blurb}.</div>",
                unsafe_allow_html=True)

    def _cg_demo(title, subtitle, scene_html, height=390):
        """Animated, self-playing demonstration — not a static instruction card."""
        demo = f"""
        <!doctype html>
        <html>
        <head>
        <meta charset='utf-8'>
        <style>
        *{{box-sizing:border-box}}
        body{{margin:0;background:linear-gradient(135deg,#f8fbff,#eef7ff);font-family:Inter,Arial,sans-serif;color:#17324d}}
        .wrap{{height:{height}px;padding:18px}}
        .top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}}
        .title{{font-size:22px;font-weight:800}}
        .live{{font-size:12px;font-weight:800;color:#087f5b;background:#d9f8ec;border-radius:999px;padding:7px 12px}}
        .stage{{height:{height-92}px;border-radius:24px;background:#fff;border:1px solid #dbeafe;box-shadow:0 14px 35px rgba(30,64,175,.10);position:relative;overflow:hidden;padding:20px}}
        .hint{{position:absolute;left:50%;top:16px;transform:translateX(-50%);font-size:13px;font-weight:700;color:#5b7088;background:#f1f5f9;border-radius:999px;padding:7px 13px;z-index:5;white-space:nowrap}}
        .progress{{position:absolute;left:20px;right:20px;bottom:14px;height:7px;border-radius:9px;background:#e2e8f0;overflow:hidden}}
        .bar{{height:100%;width:0;background:linear-gradient(90deg,#6d5dfc,#16c79a,#ffb84d);border-radius:9px}}
        .pulse{{animation:pulse 1s infinite}} @keyframes pulse{{50%{{transform:scale(1.06);box-shadow:0 0 0 10px rgba(109,93,252,.08)}}}}
        .pop{{animation:pop .35s ease}} @keyframes pop{{0%{{transform:scale(.7);opacity:.1}}100%{{transform:scale(1);opacity:1}}}}
        .fade{{animation:fade .5s ease}} @keyframes fade{{0%{{opacity:0}}100%{{opacity:1}}}}
        </style>
        </head>
        <body>
        <div class='wrap'>
          <div class='top'><div class='title'>🎬 {title}</div><div class='live'>● LIVE DEMO</div></div>
          <div class='stage'>
            <div class='hint' id='hint'>{subtitle}</div>
            {scene_html}
            <div class='progress'><div class='bar' id='bar'></div></div>
          </div>
        </div>
        <script>
        const bar=document.getElementById('bar'), hint=document.getElementById('hint');
        let p=0; const timer=setInterval(()=>{{p+=1;bar.style.width=p+'%';if(p>=100){{clearInterval(timer);setTimeout(()=>location.reload(),700)}}}},90);
        </script>
        </body></html>
        """
        components.html(demo, height=height, scrolling=False)

    # ------------------------- DEMOS -------------------------
    def demo_detection():
        _cg_demo(
            "Quick Reaction",
            "Watch the target appear — the player reacts as quickly as possible",
            """
            <div id='scene' style='height:100%;display:flex;align-items:center;justify-content:center'>
              <div id='target' style='width:128px;height:128px;border-radius:50%;background:#e9eef5;display:flex;align-items:center;justify-content:center;font-size:42px;font-weight:900;color:#6b7b8f'>READY</div>
            </div>
            <script>
            const t=document.getElementById('target'), h=document.getElementById('hint');
            const phases=[
              [0,'GET READY','WAIT…'],[900,'WATCH THE CENTER','WAIT…'],
              [1550,'','⚡'],[2250,'','✓ 318 ms'],[3100,'NEXT TRIAL','WAIT…'],
              [3900,'','⚡'],[4600,'','✓ 274 ms']
            ];
            phases.forEach(([ms,label,val])=>setTimeout(()=>{{
              if(label)h.textContent=label;
              if(val==='⚡'){{t.textContent='●';t.style.background='#ff5d73';t.style.color='#fff';t.className='pulse pop'}}
              else if(val.startsWith('✓')){{t.textContent=val;t.style.background='#dff8ee';t.style.color='#087f5b';t.className='pop'}}
              else{{t.textContent=val;t.style.background='#e9eef5';t.style.color='#6b7b8f';}}
            }},ms));
            </script>
            """,
            390,
        )

    def demo_identification():
        _cg_demo(
            "Color Attention",
            "The card flips — decide whether the visible card is red or black",
            """
            <div style='height:100%;display:flex;align-items:center;justify-content:center;gap:24px'>
              <div id='card' style='width:150px;height:205px;border-radius:18px;background:#17324d;color:white;display:flex;align-items:center;justify-content:center;font-size:58px;font-weight:900;box-shadow:0 18px 30px rgba(0,0,0,.15)'>?</div>
              <div id='choice' style='font-size:20px;font-weight:800;color:#5b7088'>DECIDE</div>
            </div>
            <script>
            const c=document.getElementById('card'),q=document.getElementById('choice'),h=document.getElementById('hint');
            const rounds=[['♥','RED','YES'],['♠','BLACK','NO'],['♦','RED','YES'],['♣','BLACK','NO']];
            rounds.forEach((r,i)=>setTimeout(()=>{{
              c.textContent=r[0];c.style.background=(r[1]==='RED'?'#ff5d73':'#273449');c.className='pop';
              q.textContent=r[2];q.style.color=(r[1]==='RED'?'#e6496f':'#273449');h.textContent='RESPOND AS QUICKLY AS YOU CAN';
            }},900+i*1050));
            </script>
            """,
            390,
        )

    def demo_memory():
        _cg_demo(
            "Visual Pattern Memory",
            "A pattern appears briefly, disappears, then the player reconstructs it",
            """
            <div id='grid' style='width:310px;height:310px;margin:22px auto;display:grid;grid-template-columns:repeat(5,1fr);gap:7px'></div>
            <script>
            const g=document.getElementById('grid');
            const cells=[];for(let i=0;i<25;i++){{let x=document.createElement('div');x.style.cssText='border-radius:10px;background:#edf2f7;border:1px solid #d9e2ec;';g.appendChild(x);cells.push(x)}}
            const pat=[1,2,7,13,18,19,23];
            setTimeout(()=>pat.forEach(i=>{{cells[i].style.background='linear-gradient(135deg,#6d5dfc,#9b8cff)';cells[i].style.boxShadow='0 0 0 3px #ddd6fe';cells[i].className='pop'}}),700);
            setTimeout(()=>{{pat.forEach(i=>{{cells[i].style.background='#edf2f7';cells[i].style.boxShadow='none'}});document.getElementById('hint').textContent='NOW REBUILD THE PATTERN';}},2400);
            setTimeout(()=>{{pat.slice(0,5).forEach(i=>{{cells[i].style.background='#6d5dfc'}});document.getElementById('hint').textContent='5 / 7 CORRECT';}},3300);
            </script>
            """,
            410,
        )

    def demo_nback():
        _cg_demo(
            "Two-Back Working Memory",
            "Compare the current symbol with the symbol two positions earlier",
            """
            <div style='height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center'>
              <div id='symbols' style='display:flex;gap:18px;align-items:center;justify-content:center'></div>
              <div id='ans' style='margin-top:24px;font-size:20px;font-weight:900;color:#5b7088'>COMPARE</div>
            </div>
            <script>
            const seq=['◆','●','◆','▲','▲','■','▲','■'];let box=document.getElementById('symbols'),a=document.getElementById('ans');
            seq.forEach((s,i)=>setTimeout(()=>{{box.innerHTML='';let x=document.createElement('div');x.textContent=s;x.style.cssText='width:88px;height:88px;border-radius:20px;background:#f1efff;color:#6d5dfc;display:flex;align-items:center;justify-content:center;font-size:48px;font-weight:900';box.appendChild(x);a.textContent=i<2?'BUILD THE SEQUENCE':(seq[i]===seq[i-2]?'MATCH → YES':'NO MATCH → NO');}},800+i*750));
            </script>
            """,
            390,
        )

    def demo_maze():
        _cg_demo(
            "Hidden Path",
            "Memorize the route, then reproduce it after the path disappears",
            """
            <div id='maze' style='width:330px;height:330px;margin:12px auto;display:grid;grid-template-columns:repeat(6,1fr);gap:6px'></div>
            <script>
            const m=document.getElementById('maze'),cs=[];for(let i=0;i<36;i++){{let x=document.createElement('div');x.style.cssText='border-radius:8px;background:#edf2f7;border:1px solid #d9e2ec';m.appendChild(x);cs.push(x)}}
            const path=[0,1,2,8,14,20,21,22,28,34,35];
            setTimeout(()=>path.forEach((i,k)=>setTimeout(()=>{{cs[i].style.background=k===0?'#20c997':'#6d5dfc';cs[i].className='pop'}},k*100)),600);
            setTimeout(()=>{{path.forEach(i=>cs[i].style.background='#edf2f7');document.getElementById('hint').textContent='PATH HIDDEN — YOUR TURN';}},2100);
            setTimeout(()=>path.forEach((i,k)=>setTimeout(()=>cs[i].style.background='#20c997',k*100)),2750);
            </script>
            """,
            420,
        )

    if game == "⚡ Reaction Matrix":
        st.caption("Reaction speed • staying alert • steady responses")
        _cg_demo_gate(demo_detection, 'quick_tap')
        st.markdown("### 🎮 Start Game")
        st.info("There are 10 rounds. Wait for the big target to appear, then tap it as quickly as you can.")
        if 'rx_round' not in st.session_state: st.session_state.rx_round=0
        if 'rx_phase' not in st.session_state: st.session_state.rx_phase='idle'
        if 'rx_target_time' not in st.session_state: st.session_state.rx_target_time=None
        if 'rx_results' not in st.session_state: st.session_state.rx_results=[]
        _rx_done=[x for x in st.session_state.rx_results if x is not None]
        _cg_stats([
            (f"{min(max(st.session_state.rx_round,1),10)} of 10", "Round"),
            (f"{len(_rx_done)}", "Completed"),
            (f"{(sum(_rx_done)/len(_rx_done)):.0f} ms" if _rx_done else "—", "Average time"),
        ])
        if st.session_state.rx_phase=='idle':
            if st.button("🎮 Start Game",use_container_width=True,key='rx_start_btn'):
                st.session_state.rx_round=1;st.session_state.rx_phase='waiting';st.session_state.rx_results=[];st.rerun()
        elif st.session_state.rx_round<=10 and st.session_state.rx_phase=='waiting':
            st.markdown(f"### Round {st.session_state.rx_round} of 10")
            st.warning("🟡 Get ready — press the button, then wait for the target.")
            if st.button("▶️ Begin This Round",key=f'rx_arm_{st.session_state.rx_round}',use_container_width=True):
                _cg_time.sleep(random.uniform(1.0,2.1))
                st.session_state.rx_target_time=_cg_time.time()
                st.session_state.rx_phase='target'
                st.rerun()
        elif st.session_state.rx_round<=10 and st.session_state.rx_phase=='target':
            st.markdown(f"### Round {st.session_state.rx_round} of 10")
            st.markdown("<div style='background:linear-gradient(135deg,#ff496e,#ff7b54);color:white;border-radius:24px;padding:38px;text-align:center;font-size:42px;font-weight:950;box-shadow:0 15px 35px rgba(255,73,110,.25)'>⚡ TARGET</div>",unsafe_allow_html=True)
            if st.button("⚡ TAP NOW!",key=f'rx_hit_{st.session_state.rx_round}',use_container_width=True):
                rt=(_cg_time.time()-st.session_state.rx_target_time)*1000
                st.session_state.rx_results.append(rt)
                st.session_state.rx_round+=1
                st.session_state.rx_phase='waiting' if st.session_state.rx_round<=10 else 'done'
                st.rerun()
        if st.session_state.rx_round>10:
            valid=[x for x in st.session_state.rx_results if x>=0]; avg=round(sum(valid)/len(valid),1) if valid else 0; acc=round(len(valid)/10*100,1)
            st.success(f"🎉 Well done! Your average reaction time was {avg} ms.")
            st.progress(acc/100, text=f"Rounds completed: {acc:.0f}%")
            if st.button("💾 Save My Result",key='rx_save',use_container_width=True):
                _cg_save('Reaction Matrix',avg,10,acc);st.toast('Saved to Cognitive Performance')
            if st.button("🔄 Play Again",key='rx_again',use_container_width=True):
                for k in ['rx_round','rx_phase','rx_target_time','rx_results']: st.session_state.pop(k,None)
                st.rerun()

    elif game == "🎨 Color Conflict":
        st.caption("Attention • ignoring distractions • processing speed")

        def demo_color_conflict():
            _cg_demo("Color Check","Ignore the word — choose the INK color", """
        <div style='height:100%;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:20px'>
          <div id='word' style='font-size:64px;font-weight:950'>BLUE</div><div style='display:flex;gap:12px'><span style='padding:10px 16px;background:#fff0f2;border-radius:12px;font-weight:800'>RED</span><span style='padding:10px 16px;background:#eef7ff;border-radius:12px;font-weight:800'>BLUE</span><span style='padding:10px 16px;background:#eefcf7;border-radius:12px;font-weight:800'>GREEN</span></div>
        </div><script>const w=document.getElementById('word'),a=[['RED','#ff526d'],['BLUE','#4d78ff'],['GREEN','#16a56d'],['RED','#ff526d'],['GREEN','#16a56d']];a.forEach((x,i)=>setTimeout(()=>{w.textContent=['BLUE','GREEN','RED','GREEN','BLUE'][i];w.style.color=x[1];w.className='pop'},700+i*800));</script>""",390)

        _cg_demo_gate(demo_color_conflict, 'color_check')
        st.markdown("### 🎮 Start Game")
        st.info("You will see a word written in a colour. Choose the **colour of the ink**, not the word.")
        if 'cc_items' not in st.session_state:
            colors=[('RED','#ef476f'),('BLUE','#3a86ff'),('GREEN','#06a77d'),('YELLOW','#f59f00')]
            st.session_state.cc_items=[]
            for i in range(12):
                word,color=random.choice(colors); ink=random.choice(colors);st.session_state.cc_items.append((word,color,ink))
            st.session_state.cc_i=0;st.session_state.cc_score=0
        i=st.session_state.cc_i
        _cg_stats([
            (f"{min(i+1,12)} of 12", "Round"),
            (f"{st.session_state.cc_score}", "Score"),
            (f"{(st.session_state.cc_score/i*100):.0f}%" if i else "—", "Accuracy"),
        ])
        if i<12:
            word,word_hex,ink=st.session_state.cc_items[i]
            st.markdown(f"<div style='text-align:center;font-size:54px;font-weight:950;color:{ink[1]};padding:18px'>{word}</div>",unsafe_allow_html=True)
            cols=st.columns(4)
            for j,(name,hexv) in enumerate([('RED','#ef476f'),('BLUE','#3a86ff'),('GREEN','#06a77d'),('YELLOW','#f59f00')]):
                if cols[j].button(name,key=f'cc_{i}_{j}',use_container_width=True):
                    if hexv==ink[1]: st.session_state.cc_score+=1
                    st.session_state.cc_i+=1;st.rerun()
            st.progress(i/12,text=f"Round {i+1} of 12")
        else:
            acc=round(st.session_state.cc_score/12*100,1);st.success(f"🎉 Great job! Score {st.session_state.cc_score}/12 • Accuracy {acc}%")
            if st.button("💾 Save My Result",key='cc_save',use_container_width=True): _cg_save('Color Check',st.session_state.cc_score,12,acc);st.toast('Saved to My Progress')
            if st.button("🔄 Play Again",key='cc_again',use_container_width=True):
                for k in ['cc_items','cc_i','cc_score']:st.session_state.pop(k,None)
                st.rerun()

    elif game == "🟪 Pattern Memory":
        st.caption("Visual memory • noticing patterns • remembering after a pause")
        _cg_demo_gate(demo_memory, 'card_memory')
        st.markdown("### 🎮 Start Game")
        st.info("Some squares light up. Look carefully, then choose the same squares again.")
        if 'pm_stage' not in st.session_state: st.session_state.pm_stage='idle'
        if 'pm_grid' not in st.session_state: st.session_state.pm_grid=[]
        if 'pm_user' not in st.session_state: st.session_state.pm_user=set()
        if 'pm_score' not in st.session_state: st.session_state.pm_score=0
        if st.session_state.pm_stage=='idle':
            level=st.select_slider('Difficulty',options=['Easy','Medium','Hard'],value='Medium',key='pm_level')
            size={'Easy':4,'Medium':5,'Hard':6}[level];count={'Easy':5,'Medium':8,'Hard':11}[level]
            if st.button('👀 Show The Pattern',use_container_width=True,key='pm_show'):
                st.session_state.pm_size=size;st.session_state.pm_grid=random.sample(range(size*size),count);st.session_state.pm_stage='show';st.session_state.pm_user=set();st.rerun()
        elif st.session_state.pm_stage=='show':
            st.info('Look at the coloured squares. Press the button below when you are ready.')
            size=st.session_state.pm_size;pat=set(st.session_state.pm_grid)
            cols=st.columns(size)
            for r in range(size):
                for c in range(size):
                    idx=r*size+c;cols[c].markdown(f"<div style='height:42px;border-radius:9px;margin:3px;background:{'#6d5dfc' if idx in pat else '#eaf0f6'}'></div>",unsafe_allow_html=True)
            if st.button('🙈 Hide & Let Me Try',use_container_width=True,key='pm_hide'):st.session_state.pm_stage='recall';st.rerun()
        else:
            size=st.session_state.pm_size;pat=set(st.session_state.pm_grid);user=st.session_state.pm_user
            st.markdown('#### Select every square that was coloured:')
            for r in range(size):
                cols=st.columns(size)
                for c in range(size):
                    idx=r*size+c
                    label='🟪' if idx in user else '⬜'
                    if cols[c].button(label,key=f'pm_{idx}'):
                        if idx in user:
                            user.remove(idx)
                        else:
                            user.add(idx)
                        st.rerun()
            if st.button('✅ Check My Answer',use_container_width=True,key='pm_submit'):
                correct=len(pat & user);wrong=len(user-pat);acc=round(correct/max(len(pat),1)*100,1);st.session_state.pm_score=correct;st.session_state.pm_stage='done';st.session_state.pm_acc=acc;st.session_state.pm_wrong=wrong;st.rerun()
        if st.session_state.pm_stage=='done':
            st.success(f"🎉 Well done! {st.session_state.pm_score} of {len(st.session_state.pm_grid)} squares correct • Accuracy {st.session_state.pm_acc}%")
            if st.button('💾 Save My Result',key='pm_save',use_container_width=True):_cg_save('Card Memory',st.session_state.pm_score,len(st.session_state.pm_grid),st.session_state.pm_acc);st.toast('Saved to My Progress')
            if st.button('🔄 Play Again',key='pm_again',use_container_width=True):
                for k in ['pm_stage','pm_grid','pm_user','pm_score','pm_acc','pm_wrong']:st.session_state.pop(k,None)
                st.rerun()

    elif game == "🔄 Two-Back":
        st.caption("Working memory • holding information • spotting a repeat")
        _cg_demo_gate(demo_nback, 'previous_card')
        st.markdown("### 🎮 Start Game")
        st.info("Symbols appear one at a time. Say whether the symbol matches the one from **two steps earlier**.")
        if 'nb_seq' not in st.session_state:
            symbols=['◆','●','▲','■','★','♥'];seq=[random.choice(symbols) for _ in range(14)]
            st.session_state.nb_seq=seq;st.session_state.nb_i=0;st.session_state.nb_score=0
        i=st.session_state.nb_i;seq=st.session_state.nb_seq
        _cg_stats([
            (f"{min(i+1,14)} of 14", "Round"),
            (f"{st.session_state.nb_score}", "Score"),
            (f"{(st.session_state.nb_score/max(i-2,1)*100):.0f}%" if i>2 else "—", "Accuracy"),
        ])
        if i<14:
            cur=seq[i];st.markdown(f"<div style='text-align:center;font-size:88px;font-weight:900;color:#6d5dfc;padding:18px'>{cur}</div>",unsafe_allow_html=True)
            if i<2:
                st.info('Just look at this symbol — no answer is needed for the first two.')
                if st.button('Next ➜',use_container_width=True,key=f'nb_next_{i}'):st.session_state.nb_i+=1;st.rerun()
            else:
                expected=seq[i]==seq[i-2]
                a,b=st.columns(2)
                if a.button('✅ YES — same symbol',use_container_width=True,key=f'nb_yes_{i}'):
                    st.session_state.nb_score += int(expected);st.session_state.nb_i+=1;st.rerun()
                if b.button('❌ NO — different',use_container_width=True,key=f'nb_no_{i}'):
                    st.session_state.nb_score += int(not expected);st.session_state.nb_i+=1;st.rerun()
            st.progress(i/14,text=f'Round {i+1} of 14')
        else:
            acc=round(st.session_state.nb_score/12*100,1);st.success(f"🎉 Well done! Score {st.session_state.nb_score}/12 • Accuracy {acc}%")
            if st.button('💾 Save My Result',key='nb_save',use_container_width=True):_cg_save('Previous Card',st.session_state.nb_score,12,acc);st.toast('Saved to My Progress')
            if st.button('🔄 Play Again',key='nb_again',use_container_width=True):
                for k in ['nb_seq','nb_i','nb_score']:st.session_state.pop(k,None)
                st.rerun()

    else:
        st.caption("Planning • remembering a route • finding your way")
        _cg_demo_gate(demo_maze, 'memory_path')
        st.markdown("### 🎮 Start Game")
        st.info("A path lights up square by square. Remember the route, then follow it yourself.")
        if 'mz_stage' not in st.session_state: st.session_state.mz_stage='idle'
        if 'mz_path' not in st.session_state: st.session_state.mz_path=[]
        if 'mz_user' not in st.session_state: st.session_state.mz_user=[]
        if st.session_state.mz_stage=='idle':
            difficulty=st.select_slider('How long should the path be?',options=['Easy','Medium','Hard'],value='Medium',key='mz_level')
            n={'Easy':9,'Medium':13,'Hard':17}[difficulty]
            # Generate a connected path by random walk.
            path=[0];cur=0;size=6
            for _ in range(n-1):
                r,c=divmod(cur,size);opts=[]
                for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    rr,cc=r+dr,c+dc
                    if 0<=rr<size and 0<=cc<size and rr*size+cc not in path:opts.append(rr*size+cc)
                if not opts: break
                cur=random.choice(opts);path.append(cur)
            if st.button('🧠 Show Me The Path',use_container_width=True,key='mz_show'):
                st.session_state.mz_path=path;st.session_state.mz_user=[];st.session_state.mz_stage='show';st.rerun()
        elif st.session_state.mz_stage=='show':
            st.info('Remember the route from the green square to the end, then press Hide.')
            path=st.session_state.mz_path
            for r in range(6):
                cols=st.columns(6)
                for c in range(6):
                    idx=r*6+c
                    if idx in path: label='🟪' if idx!=path[0] else '🟢'
                    else: label='⬜'
                    cols[c].markdown(f"<div style='text-align:center;font-size:24px;padding:5px'>{label}</div>",unsafe_allow_html=True)
            if st.button('🙈 Hide The Path',use_container_width=True,key='mz_hide'):st.session_state.mz_stage='play';st.rerun()
        elif st.session_state.mz_stage=='play':
            path=st.session_state.mz_path;user=st.session_state.mz_user
            _cg_stats([
                (f"{len(user)} of {len(path)}", "Steps done"),
                (f"{st.session_state.get('mz_errors',0)}", "Wrong taps"),
            ])
            st.markdown(f"#### Tap the next square on the path — step {len(user)+1} of {len(path)}")
            for r in range(6):
                cols=st.columns(6)
                for c in range(6):
                    idx=r*6+c;label='🟩' if idx in user else ('🟢' if idx==path[0] else '⬜')
                    if cols[c].button(label,key=f'mz_{idx}_{len(user)}'):
                        if (not user and idx==path[0]) or (user and idx in [user[-1]-6,user[-1]+6,user[-1]-1,user[-1]+1] and idx not in user): user.append(idx)
                        else: st.session_state.mz_errors=st.session_state.get('mz_errors',0)+1
                        st.rerun()
            if len(user)==len(path):st.session_state.mz_stage='done';st.rerun()
        if st.session_state.mz_stage=='done':
            errors=st.session_state.get('mz_errors',0);acc=max(0,round(100-errors*5,1));st.success(f"🎉 Path complete! {errors} wrong taps • Performance {acc}%")
            if st.button('💾 Save My Result',key='mz_save',use_container_width=True):_cg_save('Memory Path',len(st.session_state.mz_path),len(st.session_state.mz_path)+errors,acc);st.toast('Saved to My Progress')
            if st.button('🔄 Play Again',key='mz_again',use_container_width=True):
                for k in ['mz_stage','mz_path','mz_user','mz_errors']:st.session_state.pop(k,None)
                st.rerun()

    st.markdown("---")
    progress_left, progress_right = st.columns(2)
    with progress_left:
        if st.button("📈 See My Progress", use_container_width=True, key="cg_goto_progress"):
            go_to("🧠 Cognitive Performance")
    with progress_right:
        if st.button("🏠 Back to Home", use_container_width=True, key="cg_goto_home"):
            go_to("🏠 Home")

    st.markdown(
        '<div class="sq-note">⚠️ <b>Please note:</b> these activities are for practice and '
        'engagement. They are not a clinical cognitive assessment or diagnostic test. '
        'Results can change with the device, tiredness and how familiar the activity feels.</div>',
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════
# PAGE — MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Analytics":

    st.markdown(f"""
    <div class="sq-hero">
        <div class="sq-hello">📊 Model Analytics</div>
        <div class="sq-hello-sub">How well the machine learning models perform, shown simply first
        and in full technical detail afterwards.</div>
    </div>
    """, unsafe_allow_html=True)

    if pipe is None:
        st.error("⚠️ The models are not available right now, so analytics cannot be shown.")
    else:
        simple_tab, technical_tab = st.tabs(["😊 Simple View", "🔬 Technical / Model View"])

        model_registry = {}
        try:
            model_registry['Random Forest'] = pipe['rf']
            svms = pipe.get('svms') or {}
            if isinstance(svms, dict):
                model_registry.update(svms)
            model_registry['Voting Ensemble'] = pipe['ensemble']
        except Exception:
            model_registry = {}

        metric_rows = []
        try:
            from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                         f1_score, confusion_matrix, roc_curve, auc)
            X_test = pipe['X_test']
            y_test = pipe['y_test']
            for model_name, model_object in model_registry.items():
                try:
                    y_pred = model_object.predict(X_test)
                    metric_rows.append({
                        "Model": model_name,
                        "Accuracy": accuracy_score(y_test, y_pred) * 100,
                        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
                        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
                        "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
                    })
                except Exception:
                    continue
        except Exception:
            metric_rows = []

        metrics_df = pd.DataFrame(metric_rows)

        with simple_tab:
            if metrics_df.empty:
                st.info("Model scores are not available in this environment.")
            else:
                best_row = metrics_df.sort_values("Accuracy", ascending=False).iloc[0]
                st.markdown("### ✅ In plain words")
                st.markdown(
                    f"- The best performing model is **{best_row['Model']}**.\n"
                    f"- On the records it had never seen before, it was correct about "
                    f"**{best_row['Accuracy']:.0f} times out of every 100**.\n"
                    f"- It was trained on the public **OASIS** research dataset of "
                    f"**{len(pipe.get('df', [])) if pipe.get('df') is not None else 373} clinical records**.\n"
                    "- The app always shows how confident the model is, so you can see when it is unsure."
                )

                summary_columns = st.columns(4)
                for column, label in zip(summary_columns, ["Accuracy", "Precision", "Recall", "F1-Score"]):
                    column.markdown(
                        f"<div class='sq-stat'><div class='n'>{best_row[label]:.1f}%</div>"
                        f"<div class='l'>{label}</div></div>",
                        unsafe_allow_html=True
                    )

                st.markdown(
                    '<div class="sq-note">⚠️ <b>What this does not mean:</b> a high score on research '
                    'data does not mean the app can diagnose anyone. It supports screening and '
                    'monitoring only. A qualified doctor makes the diagnosis.</div>',
                    unsafe_allow_html=True
                )

        with technical_tab:
            if metrics_df.empty:
                st.warning("Model evaluation could not be computed in this environment.")
            else:
                st.markdown("#### Model comparison (held-out test set)")
                st.dataframe(
                    metrics_df.round(2).sort_values("Accuracy", ascending=False),
                    use_container_width=True, hide_index=True
                )

                fig_models = go.Figure()
                for label, colour in zip(["Accuracy", "Precision", "Recall", "F1-Score"],
                                         [t['blue'], t['green'], t['purple'], t['orange']]):
                    fig_models.add_trace(go.Bar(
                        name=label, x=metrics_df["Model"], y=metrics_df[label],
                        marker_color=colour
                    ))
                fig_models.update_layout(
                    barmode='group',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=t['txt'], family='Nunito'),
                    yaxis=dict(title='Score (%)', range=[0, 105], gridcolor=t['grid']),
                    xaxis=dict(gridcolor=t['grid']),
                    legend=dict(bgcolor=t['leg_bg']),
                    margin=dict(l=10, r=10, t=20, b=10), height=380,
                )
                st.plotly_chart(fig_models, use_container_width=True, key="analytics_model_bars")

                chosen_model_name = st.selectbox(
                    "Choose a model to inspect",
                    list(model_registry.keys()),
                    index=max(0, list(model_registry.keys()).index("Voting Ensemble"))
                    if "Voting Ensemble" in model_registry else 0,
                    key="analytics_model_select"
                )
                chosen_model = model_registry.get(chosen_model_name)
                class_names = pipe.get('class_names', ["Class 0", "Class 1"])

                left_panel, right_panel = st.columns(2)

                with left_panel:
                    st.markdown("#### Confusion matrix")
                    try:
                        from sklearn.metrics import confusion_matrix as _cm
                        y_pred = chosen_model.predict(pipe['X_test'])
                        matrix = _cm(pipe['y_test'], y_pred)
                        fig_cm = go.Figure(go.Heatmap(
                            z=matrix,
                            x=[f"Predicted {c}" for c in class_names],
                            y=[f"Actual {c}" for c in class_names],
                            colorscale=[[0, t['cm_0']], [0.5, t['cm_5']], [1, t['cm_1']]],
                            text=matrix, texttemplate="%{text}",
                            textfont=dict(size=18, color=t['txt']),
                            showscale=False,
                        ))
                        fig_cm.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color=t['txt']),
                            margin=dict(l=10, r=10, t=10, b=10), height=340,
                        )
                        st.plotly_chart(fig_cm, use_container_width=True, key="analytics_cm")
                    except Exception:
                        st.info("A confusion matrix could not be produced for this model.")

                with right_panel:
                    st.markdown("#### ROC curve")
                    try:
                        from sklearn.metrics import roc_curve as _roc, auc as _auc
                        probabilities = chosen_model.predict_proba(pipe['X_test'])[:, 1]
                        fpr, tpr, _ = _roc(pipe['y_test'], probabilities)
                        roc_auc = _auc(fpr, tpr)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                                     name=f"AUC = {roc_auc:.3f}",
                                                     line=dict(color=t['blue'], width=4)))
                        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                                     name="Random", line=dict(color=t['txt2'], dash='dash')))
                        fig_roc.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color=t['txt']),
                            xaxis=dict(title='False positive rate', gridcolor=t['grid']),
                            yaxis=dict(title='True positive rate', gridcolor=t['grid']),
                            legend=dict(bgcolor=t['leg_bg']),
                            margin=dict(l=10, r=10, t=10, b=10), height=340,
                        )
                        st.plotly_chart(fig_roc, use_container_width=True, key="analytics_roc")
                    except Exception:
                        st.info("This model does not expose probabilities, so no ROC curve is shown.")

                with st.expander("🧪 Pipeline evaluation output"):
                    try:
                        st.write(pipe.get('results', {}))
                    except Exception:
                        st.info("No stored evaluation output is available.")

                if st.button("📈 Open Visualizations", use_container_width=True, key="analytics_goto_viz"):
                    go_to("📈 Visualizations")

elif page == "📈 Visualizations":

    st.markdown("""
    <div class="sq-hero">
        <div class="sq-hello">📈 Visualizations</div>
        <div class="sq-hello-sub">Feature importance, PCA, LDA and dataset distributions.
        This section is intended for caregivers, clinicians and reviewers.</div>
    </div>
    """, unsafe_allow_html=True)

    if pipe is None:
        st.error("⚠️ The models are not available right now, so these charts cannot be drawn.")
        st.stop()

    # Feature Importance
    st.markdown('<div class="section-header">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
    importances = pipe['rf'].feature_importances_
    feat_names  = pipe['feature_names']
    sorted_idx  = np.argsort(importances)
    s_names = [feat_names[i] for i in sorted_idx]
    s_vals  = importances[sorted_idx]
    colors_fi = [t['pink'] if v>0.15 else t['orange'] if v>0.08 else t['blue'] for v in s_vals]

    fig_fi = go.Figure(go.Bar(
        x=s_vals, y=s_names, orientation='h',
        marker=dict(color=colors_fi),
        text=[f"{v*100:.1f}%" for v in s_vals],
        textposition='outside', textfont=dict(color=t['txt'], size=10),
    ))
    fig_fi.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=t['txt'], family='Inter'),
        xaxis=dict(title='Importance Score', gridcolor=t['grid'],
                   tickformat='.2f', range=[0, max(s_vals)*1.22]),
        yaxis=dict(gridcolor=t['grid'], tickfont=dict(size=11)),
        margin=dict(l=10,r=70,t=10,b=10), height=420,
    )
    st.plotly_chart(fig_fi, use_container_width=True,key="fi_chart")

    top3 = [feat_names[i] for i in np.argsort(importances)[::-1][:3]]
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{t['ins1']},{t['ins2']});
                border:1px solid {t['ins_b']};border-radius:10px;padding:1rem 1.2rem;'>
        <span style='color:{t['blue']};font-weight:600;'>💡 Clinical Insight: </span>
        <span style='color:{t['txt']};'>Top predictors are
        <strong style="color:{t['blue']};">{top3[0]}</strong>,
        <strong style="color:{t['blue']};">{top3[1]}</strong>, and
        <strong style="color:{t['blue']};">{top3[2]}</strong> —
        all clinically validated Alzheimer's biomarkers.</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # PCA + LDA
    left, right = st.columns(2)
    X_pca = pipe['X_tr_pca']
    X_lda = pipe['X_tr_lda']
    y_tr  = pipe['y_train']
    cn    = pipe['class_names']
    cls_colors = [t['red'],t['green']]

    with left:
        st.markdown('<div class="section-header">PCA — 2D Class Separation</div>', unsafe_allow_html=True)
        fig_pca = go.Figure()
        for i, (cls, color) in enumerate(zip(cn, cls_colors)):
            mask = (y_tr == i)
            fig_pca.add_trace(go.Scatter(
                x=X_pca[mask,0], y=X_pca[mask,1], mode='markers', name=cls,
                marker=dict(color=color, size=7, opacity=0.75,
                            line=dict(color=t['marker_line'], width=0.5)),
            ))
        fig_pca.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=t['txt'], family='Inter'),
            xaxis=dict(title='PC1', gridcolor=t['grid'], zeroline=False),
            yaxis=dict(title='PC2', gridcolor=t['grid'], zeroline=False),
            legend=dict(bgcolor=t['leg_bg'], bordercolor=t['border']),
            margin=dict(l=10,r=10,t=10,b=10), height=340,
        )
        st.plotly_chart(fig_pca, use_container_width=True,key="pca_chart")

    with right:
        st.markdown('<div class="section-header">LDA — Supervised Separation</div>', unsafe_allow_html=True)
        fig_lda = go.Figure()
        for i, (cls, color) in enumerate(zip(cn, cls_colors)):
            mask = (y_tr == i)
            fig_lda.add_trace(go.Histogram(
                x=X_lda[mask,0], name=cls,
                marker_color=color, opacity=0.7, nbinsx=25,
            ))
        fig_lda.update_layout(
            barmode='overlay',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=t['txt'], family='Inter'),
            xaxis=dict(title='LD1 (100% variance explained)', gridcolor=t['grid']),
            yaxis=dict(title='Count', gridcolor=t['grid']),
            legend=dict(bgcolor=t['leg_bg'], bordercolor=t['border']),
            margin=dict(l=10,r=10,t=10,b=10), height=340,
        )
        st.plotly_chart(fig_lda, use_container_width=True,key="lda_chart")

    # Dataset distributions
    st.markdown('<div class="section-header">Dataset Distribution</div>', unsafe_allow_html=True)
    df = pipe['df'].copy()
    df['Group'] = df['Group'].replace('Converted','Demented')
    c1, c2, c3 = st.columns(3)

    with c1:
        grp = df['Group'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=grp.index, values=grp.values, hole=0.5,
            marker=dict(colors=[t['red'],t['green']], line=dict(color=t['pie_line'],width=2)),
            textfont=dict(color=t['txt'], size=12),
        ))
        fig_pie.update_layout(
            title=dict(text='Class Distribution', font=dict(color=t['txt'],size=12), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color=t['txt']),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10,r=10,t=40,b=10), height=260,
        )
        st.plotly_chart(fig_pie, use_container_width=True,key="pie_chart")

    with c2:
        fig_age = go.Figure()
        for cls, color in zip(df['Group'].unique(), cls_colors):
            fig_age.add_trace(go.Box(
                y=df[df['Group']==cls]['Age'], name=cls,
                marker_color=color, line_color=color,
            ))
        fig_age.update_layout(
            title=dict(text='Age by Group', font=dict(color=t['txt'],size=12), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=t['txt']), yaxis=dict(gridcolor=t['grid'], title='Age'),
            xaxis=dict(gridcolor=t['grid']), showlegend=False,
            margin=dict(l=10,r=10,t=40,b=10), height=260,
        )
        st.plotly_chart(fig_age, use_container_width=True,key="age_chart")

    with c3:
        fig_mmse = go.Figure()
        for cls, color in zip(df['Group'].unique(), cls_colors):
            fig_mmse.add_trace(go.Violin(
            y=df[df['Group']==cls]['MMSE'], name=cls,
            box_visible=True, meanline_visible=True,
            fillcolor=color, line_color=color, opacity=0.7,
        ))

        fig_mmse.update_layout(
        title=dict(text='MMSE Score by Group', font=dict(color=t['txt'],size=12), x=0.5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=t['txt']), yaxis=dict(gridcolor=t['grid'], title='MMSE'),
        xaxis=dict(gridcolor=t['grid']), showlegend=False,
        margin=dict(l=10,r=10,t=40,b=10), height=260,
    )

    with c3:
        st.plotly_chart(fig_mmse, use_container_width=True, key="mmse_chart")