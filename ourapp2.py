"""
SYNAPTIQ — Interpretable Alzheimer's Detection
Full Streamlit UI — Memory Assistance Platform
"""
   
from sklearn import base
from streamlit_mic_recorder import mic_recorder
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
from pydub import AudioSegment
AudioSegment.converter = r"C:\Users\shiva\Downloads\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"


AudioSegment.ffprobe = r"C:\Users\shiva\Downloads\ffmpeg-8.1.1-essentials_build\bin\ffprobe.exe"
# ADD THIS LINE
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment

import tempfile

import streamlit as st
import streamlit.components.v1 as components

# Streamlit page configuration must be set before other Streamlit UI calls.
st.set_page_config(
    page_title="SYNAPTIQ — Memory Assistance",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys, warnings, re
from dotenv import load_dotenv
from datetime import datetime, date, time as dt_time
from pathlib import Path
from groq import Groq

# --------------------------------------------------
# LOAD GROQ API KEY FROM .env OR SYSTEM ENVIRONMENT
# --------------------------------------------------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
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

client = Groq(api_key=GROQ_API_KEY)

import base64
import json
import html
import random
warnings.filterwarnings('ignore')

import streamlit as st

# =========================
# USER DATABASE (DEMO)
# =========================
USERS = {
    "cg": "1234",
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
    "Bengali",
    "Bodo",
    "Meitei (Manipuri)",
    "Khasi",
    "Garo",
    "Mizo",
    "Kokborok (Tripuri)",
    "Nagamese",
    "Karbi",
    "Dimasa",
    "Ao",
    "Angami",
    "Sumi",
    "Lotha",
    "Hmar",
    "Kuki",
    "Tiwa",
    "Rabha",
    "Jaintia",
    "Chakma",
    "Nyishi",
    "Adi",
    "Apatani",
    "Monpa",
]

# Native-language name followed by the English name.  The internal values
# above remain unchanged so the rest of the application can still use the
# language names consistently for Groq translation and voice settings.
LANGUAGE_DISPLAY_NAMES = {
    "English": "English - English",
    "Assamese": "অসমীয়া - Assamese",
    "Bengali": "বাংলা - Bengali",
    "Bodo": "बर' - Bodo",
    "Meitei (Manipuri)": "মৈতৈলোন্ - Meitei (Manipuri)",
    "Khasi": "Khasi - Khasi",
    "Garo": "A·chik - Garo",
    "Mizo": "Mizo - Mizo",
    "Kokborok (Tripuri)": "Kokborok - Kokborok (Tripuri)",
    "Nagamese": "Nagamese - Nagamese",
    "Karbi": "Karbi - Karbi",
    "Dimasa": "Dimasa - Dimasa",
    "Ao": "Ao - Ao",
    "Angami": "Angami - Angami",
    "Sumi": "Sumi - Sumi",
    "Lotha": "Lotha - Lotha",
    "Hmar": "Hmar - Hmar",
    "Kuki": "Kuki - Kuki",
    "Tiwa": "Tiwa - Tiwa",
    "Rabha": "Rabha - Rabha",
    "Jaintia": "Pnar - Jaintia",
    "Chakma": "Changma - Chakma",
    "Nyishi": "Nyishi - Nyishi",
    "Adi": "Adi - Adi",
    "Apatani": "Tanii - Apatani",
    "Monpa": "Monpa - Monpa",
}

def normalize_language(value):
    """Convert a stored display label or internal language name to the internal name."""
    if value in NE_LANGUAGE_OPTIONS:
        return value
    for internal_name, display_name in LANGUAGE_DISPLAY_NAMES.items():
        if value == display_name:
            return internal_name
    return "English"

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

if "app_language" not in st.session_state:
    st.session_state.app_language = "English"
else:
    st.session_state.app_language = normalize_language(st.session_state.app_language)

if "translation_cache" not in st.session_state:
    st.session_state.translation_cache = {}

# The language is selected before login so the complete login page can be
# displayed in the user's chosen Northeast Indian language.
APP_LANGUAGE = st.session_state.app_language
APP_LANG_CODE = LANGUAGE_CODES.get(APP_LANGUAGE, "en-IN")


def tr(text):
    """Translate short user-facing text and cache it for this session."""
    if text is None or not isinstance(text, str) or APP_LANGUAGE == "English":
        return text
    text = str(text)
    if not text.strip():
        return text
    cache_key = (APP_LANGUAGE, text)
    if cache_key in st.session_state.translation_cache:
        return st.session_state.translation_cache[cache_key]
    if client is None:
        return text
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a UI translator for an elderly-friendly memory assistance app. "
                        f"Translate the user's text into {APP_LANGUAGE}. "
                        "Preserve emojis, numbers, punctuation, Markdown markers, names, and technical terms. "
                        "Keep the same meaning and approximate length. Do not translate code, variable names, URLs, or model names. "
                        "Do not add explanations. Return only the translated text."
                    )
                },
                {"role": "user", "content": text}
            ]
        )
        translated = response.choices[0].message.content.strip()
        st.session_state.translation_cache[cache_key] = translated
        return translated
    except Exception:
        return text


def localize_html(markup):
    """Translate visible HTML text nodes while leaving CSS/JS/attributes intact."""
    if APP_LANGUAGE == "English" or not isinstance(markup, str):
        return markup
    low = markup.lower()
    if "<style" in low or "<script" in low or ("@import" in markup and "{" in markup):
        return markup
    parts = re.split(r"(<[^>]+>)", markup)
    for i, part in enumerate(parts):
        if "<" not in part and ">" not in part and part.strip():
            parts[i] = tr(part)
    return "".join(parts)


# Keep Streamlit's internal return values unchanged while translating labels
# and displayed options. This means existing comparisons such as
# gender == "Male" and answer == "Yes" continue to work.
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

# =========================
# LOGIN SCREEN (CARD UI)
# =========================
def login_ui():

    # Choose the application language BEFORE entering credentials.
    # The selected language is then used for the entire login page and,
    # after login, for the rest of the application.
    global APP_LANGUAGE, APP_LANG_CODE
    selected_app_language = _original_selectbox(
        "🌐 Choose your language",
        NE_LANGUAGE_OPTIONS,
        index=NE_LANGUAGE_OPTIONS.index(
            normalize_language(st.session_state.app_language)
        ),
        format_func=lambda language: LANGUAGE_DISPLAY_NAMES.get(language, language),
        key="prelogin_language_selector"
    )
    if selected_app_language != st.session_state.app_language:
        st.session_state.app_language = selected_app_language
        APP_LANGUAGE = selected_app_language
        APP_LANG_CODE = LANGUAGE_CODES.get(APP_LANGUAGE, "en-IN")
        st.rerun()

    if APP_LANGUAGE != "English" and client is None:
        st.warning("Translation needs GROQ_API_KEY. Add it to your .env file and restart Streamlit.")

    st.markdown("## 🧠 Synaptiq AI Health System")

    st.markdown("### 🔐 Secure Login Portal")

    st.write("Enter your credentials to access the AI diagnostic dashboard.")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        with st.container():

            st.markdown("### 👤 Login Card")

            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")

            st.markdown("---")

            login_btn = st.button("🚀 Access System")

            if login_btn:

                if username in USERS and USERS[username] == password:

                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Login successful. Redirecting...")
                    st.rerun()

                else:
                    st.error("Invalid credentials. Try again.")

# =========================
# LOGOUT UI (SIDEBAR)
# =========================
def logout_ui():

    # Allow the language to be changed after login as well.
    global APP_LANGUAGE, APP_LANG_CODE
    selected_app_language = _original_selectbox(
        tr("🌐 App language"),
        NE_LANGUAGE_OPTIONS,
        index=NE_LANGUAGE_OPTIONS.index(
            normalize_language(st.session_state.app_language)
        ),
        format_func=lambda language: LANGUAGE_DISPLAY_NAMES.get(language, language),
        key="postlogin_language_selector"
    )
    if selected_app_language != st.session_state.app_language:
        st.session_state.app_language = selected_app_language
        APP_LANGUAGE = selected_app_language
        APP_LANG_CODE = LANGUAGE_CODES.get(APP_LANGUAGE, "en-IN")
        st.rerun()

    if APP_LANGUAGE != "English" and client is None:
        st.sidebar.warning(tr("Translation is unavailable until GROQ_API_KEY is configured."))

    st.sidebar.markdown(tr("### 🧠 Synaptiq Panel"))

    st.sidebar.success(tr(f"Logged in as: {st.session_state.username}"))

    if st.sidebar.button(tr("Logout")):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# =========================
# AUTH GATE
# =========================
if not st.session_state.logged_in:

    login_ui()
    st.stop()

else:

    logout_ui()

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Theme state ────────────────────────────────────────────────
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

from theme import get_colors, get_css
t = get_colors('dark' if st.session_state.dark_mode else 'light')

# ── Custom CSS (theme-aware) ───────────────────────────────────
st.markdown(get_css(t), unsafe_allow_html=True)
st.markdown("""
<style>

/* MAIN GLASS CARD */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);

    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.15);

    padding: 1.5rem;
    margin-bottom: 1rem;

    box-shadow:
        0 8px 32px rgba(0,0,0,0.25),
        inset 0 1px 1px rgba(255,255,255,0.08);

    transition: all 0.35s ease;
}

/* HOVER EFFECT */
.glass-card:hover {
    transform: translateY(-6px) scale(1.01);

    border: 1px solid rgba(56,189,248,0.5);

    box-shadow:
        0 12px 40px rgba(56,189,248,0.25),
        0 0 20px rgba(139,92,246,0.2);

    cursor: pointer;
}

/* CARD TITLE */
.glass-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.5rem;
}

/* CARD TEXT */
.glass-text {
    color: #cbd5e1;
    font-size: 0.9rem;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)

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


def browser_reminder_scheduler(reminders):
    """Reliable browser-side reminder engine.

    It shows an in-page popup even when browser notification permission is denied.
    If the user enables notifications, it also sends a browser notification.
    localStorage prevents the same reminder from firing repeatedly after rerenders.
    """
    reminder_payload = []
    for rem in reminders:
        if rem.get("date") and rem.get("time"):
            reminder_payload.append({
                "id": rem.get("id", ""),
                "date": rem["date"],
                "time": rem["time"],
                "title": rem.get("text", "Reminder"),
                "type": rem.get("type", "Reminder"),
                "note": rem.get("note", "")
            })
    payload = json.dumps(reminder_payload).replace("\\", "\\\\").replace("'", "\\'")
    components.html(f"""
    <div id="synaptiq-reminder-root" style="font-family:Arial,sans-serif;position:relative;">
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <button id="notify-btn" style="padding:9px 14px;border-radius:10px;border:1px solid #bbb;background:#fff;cursor:pointer;font-weight:600;">🔔 Enable browser notifications</button>
        <span id="notify-status" style="font-size:13px;color:#555;"></span>
      </div>
      <div id="reminder-popup" style="display:none;position:fixed;left:12px;right:12px;top:12px;z-index:999999;background:#fff;border:3px solid #ff4b4b;border-radius:16px;padding:18px;box-shadow:0 10px 35px rgba(0,0,0,.25);">
        <div style="font-size:22px;font-weight:800;margin-bottom:8px;">⏰ Synaptiq Reminder</div>
        <div id="reminder-message" style="font-size:17px;line-height:1.5;"></div>
        <button id="dismiss-reminder" style="margin-top:14px;padding:9px 18px;border:0;border-radius:9px;background:#ff4b4b;color:white;font-weight:700;cursor:pointer;">Dismiss</button>
      </div>
    </div>
    <script>
      const reminders = JSON.parse('{payload}');
      const btn = document.getElementById('notify-btn');
      const status = document.getElementById('notify-status');
      const popup = document.getElementById('reminder-popup');
      const messageBox = document.getElementById('reminder-message');
      const dismissBtn = document.getElementById('dismiss-reminder');
      const storageKey = 'synaptiq_fired_reminders';

      function getFired() {{
        try {{ return new Set(JSON.parse(localStorage.getItem(storageKey) || '[]')); }}
        catch(e) {{ return new Set(); }}
      }}
      function saveFired(set) {{
        try {{ localStorage.setItem(storageKey, JSON.stringify(Array.from(set).slice(-100))); }} catch(e) {{}}
      }}
      let fired = getFired();

      async function enableNotifications() {{
        if (!('Notification' in window)) {{
          status.textContent = 'In-page popups are enabled; this browser does not support notifications.';
          return;
        }}
        try {{
          const permission = await Notification.requestPermission();
          status.textContent = permission === 'granted'
            ? '✓ Browser notifications enabled.'
            : 'Browser permission denied. In-page popups will still work.';
        }} catch (e) {{
          status.textContent = 'In-page popups will still work.';
        }}
      }}
      btn.addEventListener('click', enableNotifications);
      dismissBtn.addEventListener('click', () => {{ popup.style.display = 'none'; }});

      function showReminder(r) {{
        const message = `${{r.type}}: ${{r.title}}${{r.note ? '. ' + r.note : ''}}`;
        messageBox.textContent = message;
        popup.style.display = 'block';
        try {{
          if ('Notification' in window && Notification.permission === 'granted') {{
            new Notification('⏰ Synaptiq Reminder', {{ body: message, requireInteraction: true }});
          }}
        }} catch(e) {{}}
        try {{
          if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(message);
            u.rate = 0.9;
            window.speechSynthesis.speak(u);
          }}
        }} catch(e) {{}}
      }}

      function checkReminders() {{
        const now = new Date();
        const currentDate = `${{now.getFullYear()}}-${{String(now.getMonth()+1).padStart(2,'0')}}-${{String(now.getDate()).padStart(2,'0')}}`;
        const currentTime = `${{String(now.getHours()).padStart(2,'0')}}:${{String(now.getMinutes()).padStart(2,'0')}}`;
        reminders.forEach((r) => {{
          const key = `${{r.id}}-${{r.date}}-${{r.time}}`;
          if (r.date === currentDate && r.time === currentTime && !fired.has(key)) {{
            fired.add(key);
            saveFired(fired);
            showReminder(r);
          }}
        }});
      }}
      checkReminders();
      setInterval(checkReminders, 1000);
    </script>
    """, height=170, scrolling=False)


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
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:1rem 0 1.5rem 0;'>
        <div style='font-size:2.5rem;'>🧠</div>
        <div style='font-size:1.2rem; font-weight:700; color:{t['txt']};'>SYNAPTIQ</div>
        <div style='font-size:0.75rem; color:{t['txt2']}; margin-top:0.2rem;'>Memory Assistance & Cognitive Support</div>
    </div>
    """, unsafe_allow_html=True)

    st.toggle("🌙 Dark Mode", key="dark_mode")
    st.sidebar.caption(f"🌐 {tr('Language')}: {APP_LANGUAGE}")

    page = st.radio("Navigation", [
        "🏠 Home",
        "📝 Cognitive Questionnaire",
        "🔬 Patient Diagnosis",
        "📸 My Memories",
        "👨‍👩‍👧 Caregiver Portal",
        "⏰ Smart Reminders",
        "🌏 NER Memory & Culture",
        "🆘 Safety / SOS",
        "🧠 Risk Timeline",
        "💬 AI Chatbot",
        "🎤 Voice Assistant",
        "🧩 Cognitive Games",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"""
    <div style='padding:0.8rem; background:{t['card']}; border-radius:8px; border:1px solid {t['border2']}; margin-bottom:0.8rem;'>
        <div style='font-size:0.7rem; color:{t['txt2']}; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem;'>Best Accuracy</div>
        <div style='color:{t['green']}; font-size:1.2rem; font-weight:700;'>96.00%</div>
        <div style='color:{t['txt2']}; font-size:0.75rem;'>Voting Ensemble</div>
    </div>
   
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════
with st.spinner("🧠 Loading Synaptiq AI models..."):
    pipe = load_pipeline()


# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":



    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800;900&family=JetBrains+Mono:wght@500&display=swap');

    .hero-card {
    position: relative;
    margin: 1.25rem 0 2rem 0;
    padding: 2.75rem 2.25rem;
    border-radius: 24px;
    background:
        radial-gradient(1200px 400px at 0% 0%, rgba(99,102,241,0.18), transparent 60%),
        radial-gradient(900px 350px at 100% 100%, rgba(16,185,129,0.16), transparent 60%),
        linear-gradient(135deg, #0b1220 0%, #111a2e 60%, #0b1220 100%);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow:
        0 20px 60px -20px rgba(0,0,0,0.6),
        inset 0 1px 0 rgba(255,255,255,0.06);
    overflow: hidden;
    font-family: 'Inter', sans-serif;
}

.hero-card::before {
    content: "";
    position: absolute;
    inset: -2px;
    background: conic-gradient(from 180deg at 50% 50%,
        rgba(99,102,241,0.35), rgba(16,185,129,0.25),
        rgba(168,85,247,0.30), rgba(99,102,241,0.35));
    filter: blur(40px);
    opacity: 0.35;
    z-index: 0;
    animation: spin 18s linear infinite;
}

.hero-card > * { position: relative; z-index: 1; }

@keyframes spin { to { transform: rotate(360deg); } }

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #93c5fd;
    background: rgba(59,130,246,0.10);
    border: 1px solid rgba(59,130,246,0.25);
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    margin-bottom: 1.1rem;
}
.hero-eyebrow .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22c55e; box-shadow: 0 0 12px #22c55e;
    animation: pulse 1.6s ease-in-out infinite;
}
@keyframes pulse { 50% { opacity: 0.4; } }

.hero-title {
    font-size: clamp(2.1rem, 4.2vw, 3.2rem);
    font-weight: 900;
    line-height: 1.05;
    letter-spacing: -0.02em;
    background: linear-gradient(120deg, #ffffff 0%, #c7d2fe 45%, #6ee7b7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.6rem 0;
}

.hero-sub {
    font-size: 1.05rem;
    color: #cbd5e1;
    max-width: 640px;
    line-height: 1.55;
    margin-bottom: 1.5rem;
}

.badge-row { display: flex; flex-wrap: wrap; gap: 0.55rem; }

.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.45rem 0.85rem;
    border-radius: 10px;
    backdrop-filter: blur(8px);
    border: 1px solid transparent;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.badge:hover { transform: translateY(-2px); }

.badge-green {
    color: #6ee7b7;
    background: rgba(16,185,129,0.10);
    border-color: rgba(16,185,129,0.30);
    box-shadow: 0 0 0 1px rgba(16,185,129,0.05), 0 8px 24px -12px rgba(16,185,129,0.6);
}
.badge-blue {
    color: #93c5fd;
    background: rgba(59,130,246,0.10);
    border-color: rgba(59,130,246,0.30);
}
.badge-purple {
    color: #d8b4fe;
    background: rgba(168,85,247,0.10);
    border-color: rgba(168,85,247,0.30);
}
</style>

<div class="hero-card">
    <div class="hero-eyebrow"><span class="dot"></span> AI · Neuroimaging · Live</div>
    <div class="hero-title">🧠 Synaptiq AI</div>
    <div class="hero-sub">Synaptiq is a memory assistance and cognitive support platform for older adults and caregivers, with machine learning, Generative AI, voice interaction and culturally relevant activities.</div>
    <div class="badge-row">
        <span class="badge badge-green">✓ ML Decision Support · 96%</span>
        <span class="badge badge-blue">◆ GEN AI Project</span>
    </div>
</div>
""", unsafe_allow_html=True)
    st.markdown("""<div class="old-user-note"><strong>👋 Welcome!</strong> Use the large tabs on the left for memories, reminders, voice help, games and caregiver support.</div>""", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (val, label) in zip([c1,c2,c3,c4,c5], [
        ("96.00%","Best Accuracy"), ("0.9629","Precision"),
        ("0.9600","Recall"),        ("0.9599","F1-Score"), ("373","Clinical Records")
    ]):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown('<div class="section-header">System Architecture</div>', unsafe_allow_html=True)
        for icon, title, desc in [
            ("📋","Clinical Input",       "Age · MMSE · CDR · nWBV · Gender · EDUC · SES · eTIV · ASF"),
            ("⚙️","Preprocessing",        "Missing value imputation · Gender encoding · Z-score normalization"),
            ("🔬","Feature Engineering",  "Cognitive Index · CDR-MMSE Product · Brain Volume Ratio · 6 derived features"),
            ("⚖️","SMOTE Balancing",      "Synthetic minority oversampling on training set only"),
            ("🌲","ML Models",            "Random Forest · SVM (Linear/RBF/Poly) · Voting Ensemble"),
            ("📊","Evaluation",           "Accuracy · Precision · Recall · F1 · ROC/AUC · Confusion Matrix"),
            ("💡","Explainability",       "Feature importance ranking · Per-patient risk breakdown"),
        ]:
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;gap:0.8rem;
                        background:{t['card']};border:1px solid {t['border2']};
                        border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.45rem;'>
                <div style='font-size:1.2rem;min-width:26px;'>{icon}</div>
                <div>
                    <div style='color:{t['txt']};font-weight:600;font-size:0.88rem;'>{title}</div>
                    <div style='color:{t['txt2']};font-size:0.77rem;margin-top:0.1rem;'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
        models_d = ['Voting Ensemble','Random Forest','SVM Linear','SVM Polynomial','SVM RBF']
        accs_d   = [96.00, 94.67, 94.67, 94.67, 90.67]
        colors_d = [t['blue'],t['green'],t['purple'],t['orange'],t['pink']]
        fig = go.Figure(go.Bar(
            x=accs_d, y=models_d, orientation='h',
            marker=dict(color=colors_d),
            text=[f"{v:.1f}%" for v in accs_d],
            textposition='outside', textfont=dict(color=t['txt'], size=11),
        ))
        fig.add_vline(x=93, line_dash="dash", line_color=t['orange'],
                      annotation_text="Clinical Threshold 93%",
                      annotation_font_color=t['orange'], annotation_font_size=10)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=t['txt'], family='Inter'),
            xaxis=dict(range=[80,100], gridcolor=t['grid'], ticksuffix='%'),
            yaxis=dict(gridcolor=t['grid']),
            margin=dict(l=10,r=60,t=10,b=10), height=260, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

      
    
# ══════════════════════════════════════════════════════════════
# PAGE 2 — PATIENT DIAGNOSIS
# ══════════════════════════════════════════════════════════════
elif page == "📝 Cognitive Questionnaire":
    st.markdown("## 📝 Cognitive Assessment Questionnaire")
    friendly_card(
        "MMSE / CDR assessment assistant",
        "This page provides an original cognitive screening exercise and a caregiver functional questionnaire. ",
        "🧠"
    )
    st.info(
        "Important: the official MMSE is a copyrighted 30-point instrument, and CDR is a licensed "
        "clinician/informant-based staging instrument."
        "The results below are therefore labelled screening/provisional results."
    )

    st.markdown("### 1️⃣ Original 30-point cognitive screening")
    st.caption("These questions are original screening tasks organized around cognitive domains commonly assessed in brief cognitive screening.")

    with st.form("cognitive_screening_form"):
        st.markdown("#### Orientation — 10 points")
        q1, q2 = st.columns(2)
        with q1:
            year_ok = st.radio("What year is it?", ["Correct", "Incorrect"], key="q_year")
            month_ok = st.radio("What month is it?", ["Correct", "Incorrect"], key="q_month")
            day_ok = st.radio("What day/date is it?", ["Correct", "Incorrect"], key="q_day")
            season_ok = st.radio("Which season are we in?", ["Correct", "Incorrect"], key="q_season")
            country_ok = st.radio("Which country are we in?", ["Correct", "Incorrect"], key="q_country")
        with q2:
            state_ok = st.radio("Which state/region are we in?", ["Correct", "Incorrect"], key="q_state")
            city_ok = st.radio("Which city/town are we in?", ["Correct", "Incorrect"], key="q_city")
            place_ok = st.radio("What kind of place are you in right now?", ["Correct", "Incorrect"], key="q_place")
            floor_ok = st.radio("Which floor/area are you on?", ["Correct", "Incorrect"], key="q_floor")
            familiar_ok = st.radio("Who is the person accompanying you?", ["Correct", "Incorrect"], key="q_person")

        st.markdown("#### Immediate learning — 3 points")
        st.write("Say three unrelated words aloud to the patient: **mango — chair — river**. Ask the patient to repeat them immediately.")
        reg = st.multiselect("Which words were repeated correctly?", ["mango", "chair", "river"], key="q_reg")

        st.markdown("#### Attention — 5 points")
        st.write("Ask the patient to count backward from 30 by 3, stopping after five answers.")
        attention = st.number_input("Number of correct steps (0–5)", min_value=0, max_value=5, value=0, step=1, key="q_attention")

        st.markdown("#### Delayed recall — 3 points")
        recall = st.multiselect("After a short delay, which of the three words can the patient recall?", ["mango", "chair", "river"], key="q_recall")

        st.markdown("#### Language / comprehension / construction — 9 points")
        l1 = st.radio("Names two everyday objects correctly?", ["Yes", "No"], key="q_name_objects")
        l2 = st.radio("Follows a simple one-step instruction correctly?", ["Yes", "No"], key="q_one_step")
        l3 = st.radio("Follows a second, different one-step instruction correctly?", ["Yes", "No"], key="q_second_step")
        l4 = st.radio("Produces a meaningful spoken sentence?", ["Yes", "No"], key="q_sentence")
        l5 = st.radio("Reads and follows a short, caregiver-created instruction?", ["Yes", "No"], key="q_read")
        l6 = st.radio("Copies a simple geometric drawing shown by the assessor?", ["Yes", "No"], key="q_copy")
        l7 = st.radio("Repeats a short, unfamiliar phrase accurately?", ["Yes", "No"], key="q_repeat")
        l8 = st.radio("Understands a simple two-part instruction?", ["Yes", "No"], key="q_two_part")
        l9 = st.radio("Can write a short meaningful sentence?", ["Yes", "No"], key="q_write")

        calculate_screen = st.form_submit_button("🧮 Calculate Screening Score", use_container_width=True)

    if calculate_screen:
        orientation = sum(x == "Correct" for x in [year_ok, month_ok, day_ok, season_ok, country_ok, state_ok, city_ok, place_ok, floor_ok, familiar_ok])
        registration = len(reg)
        attention_score = int(attention)
        recall_score = len(recall)
        language_score = sum(x == "Yes" for x in [l1,l2,l3,l4,l5,l6,l7,l8,l9])
        screening_score = orientation + registration + attention_score + recall_score + language_score
        st.session_state.screening_mmse_score = int(min(30, screening_score))
        st.success(f"🧠 Original cognitive screening score: **{screening_score}/30**")
        st.write(f"Orientation: {orientation}/10 · Immediate learning: {registration}/3 · Attention: {attention_score}/5 · Recall: {recall_score}/3 · Language/comprehension: {language_score}/9")

    st.markdown("---")
    st.markdown("### 2️⃣ Caregiver functional questionnaire for a provisional CDR-style profile")
    st.caption("Answer based on change from the person's usual functioning. These are original questions covering the six CDR domains.")

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
        calculate_cdr = st.form_submit_button("📊 Calculate Provisional CDR-Style Profile", use_container_width=True)

    if calculate_cdr:
        domain_scores = {d: rating_options[a] for d, a in cdr_answers.items()}
        # This is intentionally a transparent provisional estimate, not the licensed CDR algorithm.
        provisional = float(np.median(list(domain_scores.values())))
        st.session_state.provisional_cdr = provisional
        st.success(f"📊 Provisional CDR-style rating: **{provisional:g}**")
        st.write("Domain profile:", domain_scores)

    

    st.markdown("### 🔗 Current values available to Patient Diagnosis")
    v1, v2 = st.columns(2)
    v1.metric("Screening score", "—" if st.session_state.screening_mmse_score is None else f"{st.session_state.screening_mmse_score}/30")
    v2.metric("Provisional CDR-style rating", "—" if st.session_state.provisional_cdr is None else f"{st.session_state.provisional_cdr:g}")

elif page == "🔬 Patient Diagnosis":

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{t['hero1']},{t['hero2']});
                border:1px solid {t['border']};border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;'>
        <div style='font-size:1.6rem;font-weight:700;color:{t['txt']};'>🔬 Patient Diagnosis</div>
        <div style='color:{t['txt2']};margin-top:0.3rem;'>Review the questionnaire scores automatically and enter the remaining clinical measurements for an ML-based risk estimate.</div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1.1])

    with left:
        st.markdown('<div class="section-header">Clinical Parameters</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age (years)", 60, 98, 75)

            # MMSE-style screening score comes directly from the Cognitive Questionnaire tab.
            if st.session_state.screening_mmse_score is None:
                st.warning("⚠️ Complete the Cognitive Questionnaire first to calculate the MMSE-style score.")
                mmse = 0
            else:
                mmse = int(st.session_state.screening_mmse_score)
                st.metric("🧠 MMSE-style Score", f"{mmse}/30")

            educ = st.slider("Education (years)", 6, 23, 14)
        with c2:
            # Provisional CDR-style rating comes directly from the Cognitive Questionnaire tab.
            if st.session_state.provisional_cdr is None:
                st.warning("⚠️ Complete the Cognitive Questionnaire first to calculate the CDR-style rating.")
                cdr = 0.0
            else:
                cdr = float(st.session_state.provisional_cdr)
                st.metric("📊 CDR-style Rating", f"{cdr:g}")

            ses = st.selectbox("Socioeconomic Status", [1,2,3,4,5], index=1)
            gender = st.selectbox("Gender", ["Male","Female"])
        c3, c4 = st.columns(2)
        with c3:
            etiv = st.number_input("eTIV (mm³)", 1100, 2000, 1480,
                                   help="Estimated Total Intracranial Volume")
            nwbv = st.number_input("nWBV", 0.64, 0.84, 0.73, step=0.001, format="%.3f",
                                   help="Normalized Whole Brain Volume")
        with c4:
            asf  = st.number_input("ASF", 0.88, 1.59, 1.20, step=0.001, format="%.3f",
                                   help="Atlas Scaling Factor")

        st.markdown("<br>", unsafe_allow_html=True)
        scores_ready = (
            st.session_state.screening_mmse_score is not None
            and st.session_state.provisional_cdr is not None
        )
        predict_btn = st.button(
            "🔍  Run Diagnosis",
            use_container_width=True,
            disabled=not scores_ready
        )

        st.markdown('<div class="section-header">Clinical Reference</div>', unsafe_allow_html=True)
        for name, desc in [
            ("MMSE Score", "27–30 = Normal · 21–26 = Mild · 10–20 = Moderate · <10 = Severe"),
            ("CDR Rating", "Official CDR uses 0, 0.5, 1, 2, 3. Use a licensed CDR assessment for the official rating."),
            ("nWBV",       "Normal: 0.72–0.84 · Lower values indicate brain atrophy"),
        ]:
            st.markdown(f'<div class="info-box"><h4>{name}</h4><p>{desc}</p></div>',
                        unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">Diagnosis Result</div>', unsafe_allow_html=True)

        if predict_btn:
            from sklearn.impute import SimpleImputer
            gender_enc = 1 if gender == "Male" else 0
            cog_idx  = mmse * (1 - cdr * 0.5)
            cdr_mmse = cdr * (30 - mmse)
            mmse_diff= mmse - (cdr * 10)
            age_mmse = age / (mmse + 1)
            bvr      = nwbv / (etiv / 1000)
            age_nwbv = age * (1 - nwbv)

            feat_arr = np.array([gender_enc, age, educ, ses, mmse, cdr, etiv, nwbv, asf,
                                 cog_idx, cdr_mmse, mmse_diff, age_mmse, bvr, age_nwbv]).reshape(1,-1)
            imp = SimpleImputer(strategy='median')
            imp.fit(pipe['X_train'])
            feat_scaled = pipe['scaler'].transform(imp.transform(feat_arr))

            model    = pipe['ensemble']
            proba    = model.predict_proba(feat_scaled)[0]
            pred_idx = int(np.argmax(proba))
            pred_cls = pipe['class_names'][pred_idx]
            # STORE RESULTS FOR TIMELINE PAGE
            conf     = proba[pred_idx] * 100
           # SAVE RESULT FOR TIMELINE PAGE
            st.session_state["prediction_done"] = True
            st.session_state["risk_confidence"] = conf
            st.session_state["prediction_class"] = pred_cls
            st.session_state["mmse_value"] = mmse
            st.session_state["cdr_value"] = cdr
            if pred_cls == 'Demented':
                st.markdown(f"""
                <div class="result-demented">
                    <div style='font-size:1.6rem;font-weight:700;color:{t['red']};'>⚠️ Alzheimer's Risk Detected</div>
                    <div style='font-size:2.8rem;font-weight:800;color:{t['red']};margin:0.5rem 0;'>{conf:.1f}%</div>
                    <div style='color:{t['txt2']};'>Confidence · Voting Ensemble</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-nondemented">
                    <div style='font-size:1.6rem;font-weight:700;color:{t['green']};'>✅ No Dementia Detected</div>
                    <div style='font-size:2.8rem;font-weight:800;color:{t['green']};margin:0.5rem 0;'>{conf:.1f}%</div>
                    <div style='color:{t['txt2']};'>Confidence · Voting Ensemble</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[0]*100,
                title={'text':"Dementia Risk %",'font':{'color':t['txt'],'size':13}},
                number={'suffix':'%','font':{'color':t['txt'],'size':26}},
                gauge={
                    'axis':{'range':[0,100],'tickcolor':t['txt2']},
                    'bar':{'color':t['red'] if proba[0]>0.5 else t['green']},
                    'bgcolor':t['gbg'],'bordercolor':t['border'],
                    'steps':[
                        {'range':[0,30],'color':t['glow']},
                        {'range':[30,60],'color':t['gmid']},
                        {'range':[60,100],'color':t['ghigh']},
                    ],
                    'threshold':{'line':{'color':t['orange'],'width':3},'thickness':0.8,'value':50}
                }
            ))
            fig_g.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color=t['txt']),
                height=210, margin=dict(l=20,r=20,t=30,b=10)
            )
            st.plotly_chart(fig_g, use_container_width=True)
            if pred_cls == 'Demented':
                risk_result = "Demented"
            else: 
                risk_result = "Non-Demented"

            prompt = f"""
            You are an expert neurological healthcare AI assistant.

            The patient prediction result is:
            {risk_result}

            Generate:
            1. Explanation of the condition
            2. Possible symptoms
            3. Lifestyle precautions
            4. Recommended next steps

            Keep response simple, professional, and healthcare-oriented.
            """

            if client is None:
                st.error("Groq API key is not configured. Set GROQ_API_KEY in your environment.")
                st.stop()
            chat_completion = client.chat.completions.create(
                messages=[
              {
            "role": "user",
            "content": prompt,
              }
             ],
            model="openai/gpt-oss-20b",
    )

            ai_response = chat_completion.choices[0].message.content

            st.subheader("🧠 AI Clinical Insight")
            st.write(ai_response)
           
            

            # Feature contributions
            st.markdown('<div class="section-header">Key Clinical Factors</div>', unsafe_allow_html=True)
            importances = pipe['rf'].feature_importances_
            feat_names  = pipe['feature_names']
            top_idx     = np.argsort(importances)[::-1][:6]
            for idx in top_idx:
                imp_pct = importances[idx] * 100
                bar_w   = min(int(imp_pct * 3), 100)
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:0.8rem;
                            padding:0.45rem 0.8rem;background:{t['card']};
                            border-radius:8px;margin-bottom:0.3rem;border:1px solid {t['border2']};'>
                    <div style='color:{t['txt']};font-size:0.82rem;min-width:145px;font-weight:500;'>{feat_names[idx]}</div>
                    <div style='flex:1;background:{t['border2']};border-radius:4px;height:7px;'>
                        <div style='width:{bar_w}%;background:linear-gradient(90deg,{t['blue']},{t['purple']});height:7px;border-radius:4px;'></div>
                    </div>
                    <div style='color:{t['blue']};font-size:0.82rem;font-weight:600;min-width:38px;text-align:right;'>{imp_pct:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style='background:{t['card']};border:1px dashed {t['border']};border-radius:12px;
                        padding:3rem;text-align:center;color:{t['txt2']};'>
                <div style='font-size:3rem;margin-bottom:1rem;'>🩺</div>
                <div style='font-size:1rem;font-weight:500;color:{t['txt']};'>Enter Patient Data</div>
                <div style='font-size:0.85rem;margin-top:0.5rem;'>
                    Fill in the clinical parameters on the left<br>and click Run Diagnosis
                </div>
            </div>""", unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════
# PAGE — chatbot
# ══════════════════════════════════════════════════════════════

# =========================
# SESSION STATE INIT
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# LANGUAGE SELECTION
# =========================
elif page == "💬 AI Chatbot":
    st.title("🌍 Synaptiq Multilingual AI Chatbot")

    language = APP_LANGUAGE
    st.caption(f"🌐 {tr('Current app language')}: {APP_LANGUAGE}")


    def get_system_prompt(lang):

        base = """
You are Synaptiq, an AI healthcare assistant specialized in Alzheimer’s disease.
Respond clearly, simply, and medically accurate.
Keep answers short and easy to understand.
"""

        if lang != "English":
            base += f"\nRespond in {lang}."

        return base

# =========================
# DISPLAY CHAT HISTORY
# =========================
    for role, msg in st.session_state.chat_history:

        with st.chat_message(role):
            st.write(msg)

# =========================
# USER INPUT
# =========================
    user_input = st.chat_input("Ask Synaptiq anything about memory, brain health, or symptoms...")

    if user_input:

    # show user message
        st.chat_message("user").write(user_input)
        st.session_state.chat_history.append(("user", user_input))

    # build messages
        messages = [
        {"role": "system", "content": get_system_prompt(language)}
    ]

        for role, msg in st.session_state.chat_history:
            messages.append({"role": role, "content": msg})

    # =========================
    # AI RESPONSE
    # =========================
        with st.chat_message("assistant"):

            with st.spinner("🧠 Thinking..."):

                if client is None:
                    st.error("Groq API key is not configured. Set GROQ_API_KEY in your environment.")
                    st.stop()
                response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages
            )

                reply = response.choices[0].message.content

                st.write(reply)

        st.session_state.chat_history.append(("assistant", reply))
# ══════════════════════════════════════════════════════════════
# PAGE — RISK TIMELINE
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# PAGE — MY MEMORIES
# ══════════════════════════════════════════════════════════════
elif page == "📸 My Memories":
    st.markdown("## 📸 My Memories")
    st.markdown("### A simple personal memory book for familiar people, places and events.")
    friendly_card("Memory Book", "A caregiver can add a photo, person's name, relationship, place and a short story. The information stays in this demo session.", "❤️")

    with st.expander("➕ Add a new memory", expanded=True):
        with st.form("memory_form", clear_on_submit=True):
            photo = st.file_uploader("📷 Photo", type=["png", "jpg", "jpeg"])
            c1, c2 = st.columns(2)
            with c1:
                person = st.text_input("👤 Name", placeholder="e.g. Anita")
                relation = st.text_input("👪 Relationship", placeholder="e.g. Daughter")
            with c2:
                place = st.text_input("📍 Place", placeholder="e.g. Shillong")
                event = st.text_input("📅 Important event", placeholder="e.g. Family trip in 2024")
            story = st.text_area("💭 Short story", placeholder="Write one or two familiar sentences...")
            submitted = st.form_submit_button("💾 Save Memory", use_container_width=True)
            if submitted:
                if person.strip():
                    image_bytes = photo.getvalue() if photo else None
                    st.session_state.memories.append({
                        "name": person.strip(), "relation": relation.strip(),
                        "place": place.strip(), "event": event.strip(),
                        "story": story.strip(), "image": image_bytes
                    })
                    st.success("✅ Memory saved.")
                else:
                    st.warning("Please enter a name before saving.")

    st.markdown("### 🖼️ Familiar Memories")
    if not st.session_state.memories:
        friendly_card("No memories added yet", "Use the Add a new memory section to create the first family memory.", "🌱")
    else:
        cols = st.columns(3)
        for i, mem in enumerate(st.session_state.memories):
            with cols[i % 3]:
                with st.container(border=True):
                    if mem.get("image"):
                        st.image(mem["image"], use_container_width=True)
                    st.markdown(f"### 👤 {mem['name']}")
                    if mem['relation']: st.write(f"**Relationship:** {mem['relation']}")
                    if mem['place']: st.write(f"**Place:** {mem['place']}")
                    if mem['event']: st.write(f"**Event:** {mem['event']}")
                    if mem['story']: st.write(mem['story'])
                    if st.button("🔊 Read this memory", key=f"speak_mem_{i}"):
                        browser_speak(f"This is {mem['name']}. {mem['relation']}. {mem['story']}")

    st.markdown("### 🧠 Remember Who This Is")
    if st.session_state.memories:
        q_index = st.selectbox("Choose a memory for a gentle recall question", range(len(st.session_state.memories)), format_func=lambda i: st.session_state.memories[i]["name"], key="memory_question")
        q_mem = st.session_state.memories[q_index]
        if q_mem.get("image"):
            st.image(q_mem["image"], width=240)
        st.write("**Who is this person?**")
        answer = st.text_input("Your answer", key="memory_answer")
        if st.button("Check Answer", use_container_width=True):
            if answer.strip().lower() == q_mem["name"].lower():
                st.success("🎉 Wonderful! You remembered correctly.")
            else:
                st.info(f"That's okay. This is {q_mem['name']}, your {q_mem['relation'] or 'family member'}.")

# ══════════════════════════════════════════════════════════════
# PAGE — CAREGIVER PORTAL
# ══════════════════════════════════════════════════════════════
elif page == "👨‍👩‍👧 Caregiver Portal":
    st.markdown("## 👨‍👩‍👧 Family & Caregiver Portal")
    friendly_card("Caregiver view", "Only the caregiver account should add the patient’s memories, visited-place photos and trusted contacts. The patient account can use the saved memories and call buttons.", "🤝")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Memories", len(st.session_state.memories))
    c2.metric("Reminders", len(st.session_state.reminders))
    c3.metric("Games completed", len(st.session_state.game_history))
    c4.metric("Contacts", len(st.session_state.important_contacts))

    st.markdown("### 👤 Patient Information")
    with st.form("patient_profile"):
        c1, c2 = st.columns(2)
        with c1:
            patient_name = st.text_input("Patient name", value=st.session_state.get("patient_name", "My Family Member"))
            patient_age = st.number_input("Age", 50, 110, int(st.session_state.get("patient_age", 70)))
        with c2:
            caregiver_name = st.text_input("Primary caregiver", value=st.session_state.get("caregiver_name", "Anita"))
            home_place = st.text_input("Home / familiar place", value=st.session_state.get("home_place", "Home"))
        if st.form_submit_button("💾 Save Patient Profile", use_container_width=True):
            st.session_state.patient_name = patient_name
            st.session_state.patient_age = patient_age
            st.session_state.caregiver_name = caregiver_name
            st.session_state.home_place = home_place
            st.success("Profile updated.")

    st.markdown("### 📋 Recent Activity")
    if st.session_state.game_history:
        st.dataframe(pd.DataFrame(st.session_state.game_history), use_container_width=True, hide_index=True)
    else:
        st.info("No cognitive game sessions recorded yet.")

    st.markdown("### 📞 Important Contacts")
    with st.form("contact_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Name")
        relation = c2.text_input("Relationship")
        phone = c3.text_input("Phone / contact note")
        if st.form_submit_button("➕ Add Contact", use_container_width=True) and name:
            st.session_state.important_contacts.append({"name": name, "relation": relation, "phone": phone})
            st.success("Contact added.")
    for contact in st.session_state.important_contacts:
        friendly_card(contact["name"], f"{contact['relation']} · {contact['phone']}", "📞")

# ══════════════════════════════════════════════════════════════
# PAGE — SMART REMINDERS
# ══════════════════════════════════════════════════════════════
elif page == "⏰ Smart Reminders":
    st.markdown("## ⏰ Smart Reminder System")
    friendly_card(
        "Gentle reminders that actually pop up",
        "Set a date and time. Keep the Synaptiq tab open. At the scheduled time an in-page popup appears, the reminder can be spoken aloud, and browser notifications can also appear after permission is enabled.",
        "🔔"
    )

    with st.form("reminder_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            reminder_text = st.text_input("What should we remember?", placeholder="e.g. Take medicine")
            reminder_type = st.selectbox("Type", ["Medicine", "Meal", "Water", "Doctor appointment", "Family call", "Daily activity", "Sleep", "Other"])
            reminder_date = st.date_input("Date", value=date.today(), min_value=date.today())
        with c2:
            reminder_time = st.time_input("Time", value=dt_time(9, 0))
            reminder_note = st.text_input("Short note", placeholder="e.g. After breakfast")
        if st.form_submit_button("➕ Set Reminder", width='stretch'):
            if reminder_text.strip():
                st.session_state.reminders.append({
                    "id": f"rem-{datetime.now().timestamp()}-{random.randint(1000,9999)}",
                    "date": reminder_date.isoformat(),
                    "time": reminder_time.strftime("%H:%M"),
                    "type": reminder_type,
                    "text": reminder_text.strip(),
                    "note": reminder_note.strip()
                })
                st.success(f"✅ Reminder set for {reminder_date.strftime('%d %B %Y')} at {reminder_time.strftime('%I:%M %p')}.")
                st.rerun()
            else:
                st.warning("Please enter what you want to remember.")

    browser_reminder_scheduler(st.session_state.reminders)

    st.markdown("### 📋 Your Reminders")
    if not st.session_state.reminders:
        st.info("No reminders yet. Add one above.")
    else:
        sorted_reminders = sorted(st.session_state.reminders, key=lambda x: (x.get("date", ""), x.get("time", "")))
        for i, rem in enumerate(sorted_reminders):
            c1, c2, c3 = st.columns([1.5, 3.2, 1])
            display_date = datetime.strptime(rem.get("date", date.today().isoformat()), "%Y-%m-%d").strftime("%d %b %Y")
            c1.markdown(f"## ⏰ {rem['time']}\n**{display_date}**")
            c2.markdown(f"**{rem['type']} — {rem['text']}**  \n{rem.get('note', '')}")
            if c3.button("🗑️ Delete", key=f"delete_rem_{i}"):
                st.session_state.reminders.remove(rem)
                st.rerun()

    st.markdown("### 🗓️ Daily Routine")
    for rt, task in st.session_state.daily_routine:
        friendly_card(rt, task, "🌞")

# ══════════════════════════════════════════════════════════════
# PAGE — NER MEMORY & CULTURE
# ══════════════════════════════════════════════════════════════
elif page == "🌏 NER Memory & Culture":
    st.markdown("## 🌏 North Eastern Region Memory & Culture")
    friendly_card(
        "A meaningful memory-and-culture activity",
        "Caregivers can save familiar places, and patients can practise recognition. The culture section also gives short, friendly learning cards and quizzes instead of only showing a placeholder message.",
        "🌿"
    )

    language = st.selectbox("🗣️ Preferred language", ["English", "Assamese", "Bengali", "Khasi", "Meitei / Manipuri", "Mizo", "Bodo", "Garo", "Nagamese"], key="ner_preferred_language")

    NER_CULTURE = {
        "Assamese": [
            ("🎉 Bihu", "Bihu is a major Assamese festival tradition. Bohag Bihu is associated with spring and the Assamese New Year.", "festival"),
            ("🍚 Pitha", "Pitha are traditional Assamese rice-based foods prepared in many forms, especially around festivals.", "food"),
            ("👗 Mekhela Chador", "Mekhela Chador is a traditional Assamese women's attire, commonly seen during cultural occasions.", "clothing"),
            ("🎵 Pepa", "Pepa is a traditional Assamese wind instrument traditionally associated with Bihu celebrations.", "music"),
        ],
        "Bengali": [
            ("🎉 Durga Puja", "Durga Puja is a major Bengali cultural celebration with community gatherings, art and music.", "festival"),
            ("🍚 Panta Bhat", "Panta bhat is a traditional rice preparation associated with Bengali food culture.", "food"),
            ("👗 Tant Saree", "Tant handloom sarees are an important part of Bengali textile traditions.", "clothing"),
            ("🥁 Dhak", "The dhak is a traditional drum strongly associated with Bengali festive music.", "music"),
        ],
        "Khasi": [
            ("🎉 Nongkrem", "Nongkrem Dance is a major Khasi cultural festival of Meghalaya.", "festival"),
            ("🍲 Jadoh", "Jadoh is a well-known Khasi rice-and-meat dish from Meghalaya.", "food"),
            ("👗 Jainsem", "Jainsem is a traditional Khasi women's garment.", "clothing"),
            ("🎵 Duitara", "The duitara is a traditional string instrument found in Meghalaya's musical traditions.", "music"),
        ],
        "Meitei / Manipuri": [
            ("🎉 Yaoshang", "Yaoshang is a major Manipuri festival celebrated with community activities, sports and cultural traditions.", "festival"),
            ("🍚 Chak-hao", "Chak-hao is a traditional black rice variety from Manipur.", "food"),
            ("👗 Phanek", "Phanek is a traditional Meitei women's wraparound garment.", "clothing"),
            ("🎵 Pena", "Pena is a traditional bowed string instrument of the Meitei cultural tradition.", "music"),
        ],
        "Mizo": [
            ("🎉 Chapchar Kut", "Chapchar Kut is a well-known Mizo spring festival featuring music, dance and community celebration.", "festival"),
            ("🍚 Bai", "Bai is a traditional Mizo vegetable-based dish with many local variations.", "food"),
            ("👗 Puan", "Puan is a traditional Mizo woven textile used in clothing and cultural occasions.", "clothing"),
            ("🎵 Khuang", "Khuang is a traditional Mizo drum used in music and dance.", "music"),
        ],
        "Bodo": [
            ("🎉 Bwisagu", "Bwisagu is a major Bodo festival associated with spring and the beginning of a new year.", "festival"),
            ("🍚 Narzi", "Narzi is a traditional Bodo preparation made with dried jute leaves.", "food"),
            ("👗 Dokhona", "Dokhona is a traditional Bodo women's garment.", "clothing"),
            ("🎵 Kham", "Kham is a traditional drum used in Bodo musical traditions.", "music"),
        ],
        "Garo": [
            ("🎉 Wangala", "Wangala is the well-known harvest festival of the Garo people, featuring drums and traditional dances.", "festival"),
            ("🍚 Wak Pura", "Wak pura is a pork-and-rice preparation found in Garo food traditions.", "food"),
            ("👗 Dakmanda", "Dakmanda is a traditional Garo women's wraparound garment.", "clothing"),
            ("🥁 Nagra", "Traditional Garo celebrations use drums and other instruments to accompany dance.", "music"),
        ],
        "Nagamese": [
            ("🌏 Community culture", "Nagamese is widely used as a link language in Nagaland. Cultural traditions vary across Naga communities.", "culture"),
            ("🍚 Local food", "Nagaland has diverse food traditions, with recipes differing across Naga communities.", "food"),
            ("👗 Traditional textiles", "Naga communities have distinctive shawls and woven textiles with community-specific patterns.", "clothing"),
            ("🥁 Festival music", "Traditional Naga festivals include community singing, dancing and drum-based music.", "music"),
        ],
        "English": [
            ("🌏 Explore the Northeast", "Choose a community from the list to explore short cultural memory cards and take a gentle quiz.", "culture"),
        ],
    }

    culture_cards = NER_CULTURE.get(language, NER_CULTURE["English"])
    st.info(f"Selected language: {language}. These are educational memory prompts; local community members should verify content before clinical deployment.")

    st.markdown("### 🧭 Choose a community")
    community_options = ["Assamese", "Bengali", "Khasi", "Meitei / Manipuri", "Mizo", "Bodo", "Garo", "Nagamese"]
    community = st.selectbox("Community / tradition", community_options, key="ner_community")
    cards = NER_CULTURE[community]

    # ------------------------------------------------------------------
    # CULTURE MEMORY CARD GAME
    # ------------------------------------------------------------------
    # Keep card state stable across Streamlit reruns.
    if "ner_card_community" not in st.session_state or st.session_state.ner_card_community != community:
        st.session_state.ner_card_community = community
        st.session_state.ner_card_order = list(range(len(cards)))
        st.session_state.ner_revealed_cards = set()

    if "ner_card_order" not in st.session_state or len(st.session_state.ner_card_order) != len(cards):
        st.session_state.ner_card_order = list(range(len(cards)))

    if "ner_revealed_cards" not in st.session_state:
        st.session_state.ner_revealed_cards = set()

    st.markdown(f"### 🃏 {community} Culture Memory Cards")
    st.caption("Tap a card to reveal it. Try to remember the cultural item, then hide the cards and test yourself again.")

    game_c1, game_c2 = st.columns(2)
    with game_c1:
        if st.button("🔀 Shuffle Cards", key="ner_shuffle_cards", width='stretch'):
            st.session_state.ner_card_order = random.sample(range(len(cards)), len(cards))
            st.session_state.ner_revealed_cards = set()
            st.rerun()
    with game_c2:
        if st.button("🙈 Hide All Cards", key="ner_hide_cards", width='stretch'):
            st.session_state.ner_revealed_cards = set()
            st.rerun()

    category_icons = {
        "festival": "🎉",
        "food": "🍚",
        "clothing": "👗",
        "music": "🎵",
        "culture": "🌏",
    }

    st.markdown("""
    <style>
    .ner-card {
        min-height: 235px;
        border-radius: 22px;
        padding: 1.25rem;
        margin: 0 0 0.55rem 0;
        border: 1px solid rgba(100, 116, 139, 0.22);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.10);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        text-align: center;
    }
    .ner-card-hidden {
        background: linear-gradient(135deg, #dbeafe, #ede9fe);
    }
    .ner-card-revealed {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
    }
    .ner-card-top {
        font-size: 2.5rem;
        margin-top: 0.15rem;
    }
    .ner-card-title {
        font-size: 1.22rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }
    .ner-card-category {
        display: inline-block;
        margin-top: 0.35rem;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        background: rgba(99, 102, 241, 0.10);
        font-size: 0.78rem;
        font-weight: 600;
    }
    .ner-card-description {
        margin-top: 0.75rem;
        font-size: 0.92rem;
        line-height: 1.5;
        text-align: left;
    }
    .ner-card-hint {
        font-size: 0.82rem;
        opacity: 0.78;
        margin-top: 0.7rem;
    }
    </style>
    """, unsafe_allow_html=True)

    card_cols = st.columns(2, gap="large")
    for position, card_index in enumerate(st.session_state.ner_card_order):
        title, description, category = cards[card_index]
        icon = category_icons.get(category, "🌿")
        is_revealed = card_index in st.session_state.ner_revealed_cards
        clean_title = title.split(" ", 1)[1] if " " in title else title

        with card_cols[position % 2]:
            if is_revealed:
                st.markdown(
                    f"""
                    <div class="ner-card ner-card-revealed">
                        <div>
                            <div class="ner-card-top">{icon}</div>
                            <div class="ner-card-title">{clean_title}</div>
                            <div class="ner-card-category">{category.title()}</div>
                            <div class="ner-card-description">{description}</div>
                        </div>
                        <div class="ner-card-hint">🧠 Remember this card</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("🙈 Hide Card", key=f"ner_hide_{community}_{card_index}", width='stretch'):
                    st.session_state.ner_revealed_cards.discard(card_index)
                    st.rerun()
            else:
                st.markdown(
                    f"""
                    <div class="ner-card ner-card-hidden">
                        <div>
                            <div class="ner-card-top">🃏</div>
                            <div class="ner-card-title">Culture Memory Card {position + 1}</div>
                            <div class="ner-card-category">Tap to reveal</div>
                        </div>
                        <div class="ner-card-hint">Can you remember what this card might show?</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("👀 Reveal Card", key=f"ner_reveal_{community}_{card_index}", width='stretch'):
                    st.session_state.ner_revealed_cards.add(card_index)
                    st.rerun()

    st.markdown("### 🧠 Gentle Culture Quiz")
    quiz_bank = [
        {"q": f"Which of these is associated with {community}?", "options": [c[0].split(" ", 1)[1] if " " in c[0] else c[0] for c in cards], "answer": 0},
        {"q": "What is the best way to use these cards with a person with memory difficulties?", "options": ["Use familiar images and gentle repetition", "Force the person to remember quickly", "Correct every mistake harshly", "Give a long test without breaks"], "answer": 0},
        {"q": "Who should verify local cultural content before wider deployment?", "options": ["Local community members and cultural sources", "Only an automated model", "Random internet comments", "Nobody"], "answer": 0},
    ]
    q_index = st.selectbox("Quiz question", range(len(quiz_bank)), format_func=lambda x: f"Question {x+1}", key="ner_quiz_index")
    q = quiz_bank[q_index]
    st.markdown(f"**{q['q']}**")
    answer = st.radio("Choose an answer", q["options"], key=f"ner_culture_answer_{q_index}")
    if st.button("✅ Check Answer", key="ner_culture_check", width='stretch'):
        if answer == q["options"][q["answer"]]:
            st.success("🎉 Correct! Great job.")
        else:
            st.info(f"That's okay. The helpful answer is: **{q['options'][q['answer']]}**")

    st.markdown("### 👨‍👩‍👧 Caregiver: Upload Visited Places")
    if st.session_state.get("username") == "caregiver":
        with st.form("ner_place_upload_form", clear_on_submit=True):
            uploaded_image = st.file_uploader("Upload a photo of a place the patient has visited", type=["png", "jpg", "jpeg", "webp"])
            c1, c2 = st.columns(2)
            with c1:
                place_name = st.text_input("Place name", placeholder="e.g. Shillong")
                place_region = st.text_input("State / region", placeholder="e.g. Meghalaya")
            with c2:
                memory_hint = st.text_input("Memory hint", placeholder="e.g. We visited here during your birthday trip")
                quiz_question = st.text_input("Quiz question", value="Do you remember where this is?")
            if st.form_submit_button("📸 Save Place Memory", width='stretch'):
                if uploaded_image and place_name.strip():
                    st.session_state.ner_uploaded_places.append({
                        "name": place_name.strip(), "region": place_region.strip(),
                        "hint": memory_hint.strip(), "question": quiz_question.strip() or "Do you remember where this is?",
                        "image": uploaded_image.getvalue()
                    })
                    st.success(f"📸 {place_name.strip()} was added to the patient's place memories.")
                else:
                    st.warning("Please upload an image and enter the place name.")
    else:
        st.info("Caregiver login is required to upload or manage place memories. The patient can still use the culture cards and quiz above.")

    if st.session_state.ner_uploaded_places:
        st.markdown("### 🖼️ Saved Place Memories")
        cols = st.columns(min(3, len(st.session_state.ner_uploaded_places)))
        for i, place_mem in enumerate(st.session_state.ner_uploaded_places):
            with cols[i % len(cols)]:
                st.image(place_mem["image"], width='stretch')
                st.markdown(f"**{place_mem['name']}**")
                if place_mem.get("region"): st.caption(place_mem["region"])
                if place_mem.get("hint"): st.caption(f"💭 {place_mem['hint']}")

        st.markdown("### 🧠 Patient Place Recall Quiz")
        quiz_mem = random.choice(st.session_state.ner_uploaded_places)
        st.image(quiz_mem["image"], width='stretch')
        st.markdown(f"### ❓ {quiz_mem.get('question') or 'Do you remember where this is?'}")
        names = list(dict.fromkeys(x["name"] for x in st.session_state.ner_uploaded_places))
        if len(names) > 1:
            options = names[:]
            if len(options) >= 4: options = random.sample(options, 4)
            if quiz_mem["name"] not in options: options[-1] = quiz_mem["name"]
            random.shuffle(options)
            selected = st.radio("Choose the place", options, key="ner_place_quiz_answer")
            if st.button("✅ Check Memory", key="ner_place_check", width='stretch'):
                if selected == quiz_mem["name"]:
                    st.success("🎉 Wonderful! You recognized the place.")
                    if quiz_mem.get("hint"): st.info(f"💭 Memory hint: {quiz_mem['hint']}")
                else:
                    st.info(f"That's okay. This photo is **{quiz_mem['name']}**. Gentle repetition can support recall.")
        else:
            answer = st.text_input("What place is shown?", key="ner_quiz_text")
            if st.button("✅ Check Memory", key="ner_place_text_check"):
                if answer.strip().lower() == quiz_mem["name"].lower(): st.success("🎉 Wonderful! You recognized the place.")
                else: st.info(f"That's okay. This photo is **{quiz_mem['name']}**.")

# ══════════════════════════════════════════════════════════════
elif page == "🆘 Safety / SOS":
    st.markdown("## 🆘 Safety & SOS")
    friendly_card(
        "Call a saved caregiver",
        "The caregiver adds trusted contacts in the Caregiver Portal. "
        "The patient can then use a large Call button here.",
        "📞"
    )

    st.warning(
        "This is a prototype safety interface. The Call buttons open the phone's "
        "calling interface when the device/browser supports tel: links. "
        "The SOS button does not automatically contact emergency services."
    )

    st.markdown("### 📞 Saved Caregiver / Emergency Contacts")

    callable_contacts = []
    for contact in st.session_state.important_contacts:
        phone = str(contact.get("phone", "")).strip()
        digits = "".join(ch for ch in phone if ch.isdigit())

        if len(digits) >= 7:
            callable_contacts.append(contact)

            friendly_card(
                contact["name"],
                f"{contact.get('relation', 'Contact')} · {phone}",
                "📞"
            )

            safe_phone = html.escape(phone, quote=True)
            safe_name = html.escape(str(contact["name"]))

            st.markdown(
                f"""
                <a href="tel:{safe_phone}"
                   style="
                     display:block;
                     text-align:center;
                     text-decoration:none;
                     background:#16a34a;
                     color:white;
                     padding:18px;
                     border-radius:16px;
                     font-size:1.25rem;
                     font-weight:800;
                     margin:8px 0 18px 0;
                   ">
                   📞 CALL {safe_name}
                </a>
                """,
                unsafe_allow_html=True
            )
        else:
            st.caption(
                f"⚠️ {contact.get('name', 'Contact')} needs a valid phone number "
                "in the Caregiver Portal before calling is enabled."
            )

    if not callable_contacts:
        st.info(
            "No callable contacts are available yet. "
            "Ask the caregiver to add a trusted person's name, relationship and phone number "
            "in the Caregiver Portal."
        )

    st.markdown("### 🔴 SOS Request")

    if st.button(
        "🆘 I NEED HELP",
        use_container_width=True,
        key="sos_help_button"
    ):
        st.session_state.sos_message = (
            f"SOS requested at {datetime.now().strftime('%d %B %Y, %I:%M %p')}"
        )
        st.error(
            "🚨 Help requested. Please use one of the green Call buttons above "
            "to contact the saved caregiver."
        )
        browser_speak(
            "Help is needed. Please call your saved caregiver."
        )

    if st.session_state.sos_message:
        st.info(f"🕒 {st.session_state.sos_message}")


elif page == "🧠 Risk Timeline":

    # =======================
    # CHECK DATA
    # =======================
    if "prediction_done" not in st.session_state:
        st.warning("⚠ Please complete Patient Diagnosis first.")

    else:

        conf = st.session_state["risk_confidence"]
        mmse = st.session_state["mmse_value"]
        cdr = st.session_state["cdr_value"]

        # =======================
        # STAGE LOGIC
        # =======================
        if mmse >= 27:
            stage = 1
            stage_name = "No Cognitive Decline"
            stage_color = "🟢"

        elif mmse >= 24:
            stage = 2
            stage_name = "Very Mild Cognitive Decline"
            stage_color = "🔵"

        elif mmse >= 20:
            stage = 3
            stage_name = "Mild Cognitive Decline"
            stage_color = "🟡"

        elif mmse >= 15:
            stage = 4
            stage_name = "Moderate Cognitive Decline"
            stage_color = "🟠"

        elif mmse >= 10:
            stage = 5
            stage_name = "Moderately Severe Decline"
            stage_color = "🟠"

        elif mmse >= 5:
            stage = 6
            stage_name = "Severe Cognitive Decline"
            stage_color = "🔴"

        else:
            stage = 7
            stage_name = "Very Severe Cognitive Decline"
            stage_color = "🔴"

        # =======================
        # HEADER
        # =======================
        st.markdown("## 🧠 Cognitive Assessment Report")
        st.subheader(stage_name)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Confidence", f"{conf:.1f}%")
        c2.metric("MMSE Score", mmse)
        c3.metric("CDR Rating", cdr)
        c4.metric("Stage", f"{stage}/7")

        st.divider()

        # =======================
        # SLIDER
        # =======================
        st.markdown("## 📊 Cognitive Progression Axis")

        labels = [
            "No Decline",
            "Very Mild",
            "Mild",
            "Moderate",
            "Moderately Severe",
            "Severe",
            "Very Severe"
        ]

        selected_stage = st.slider(
            "Move through cognitive stages",
            1, 7, stage
        )

        st.progress(selected_stage / 7)

        # =======================
        # AXIS (NO HTML - STREAMLIT ONLY)
        # =======================

        cols = st.columns(7)

        for i in range(7):

            with cols[i]:

                if i + 1 < selected_stage:
                    st.success(f"{i+1}\n{labels[i]}")

                elif i + 1 == selected_stage:
                    st.info(f"{i+1}\n➡ {labels[i]}")

                else:
                    st.write(f"{i+1}\n{labels[i]}")

        # =======================
        # AI INTERPRETATION
        # =======================
        st.markdown("### 🧠 AI Interpretation")

        if selected_stage < stage:
            st.success("Simulated earlier cognitive state than prediction.")

        elif selected_stage == stage:
            st.info("Matches AI predicted stage.")

        else:
            st.warning("Simulated progression into higher risk stage.")

        # =======================
        # SUMMARY
        # =======================
        st.markdown("### 🔬 AI Diagnosis Summary")

        st.write(f"""
The patient is classified as **{stage_name}**.

This interactive axis allows exploration of cognitive decline from Stage 1 to Stage 7,
based on MMSE-driven machine learning prediction.

Recommended: continuous monitoring and early intervention.
""")
# ══════════════════════════════════════════════════════════════
