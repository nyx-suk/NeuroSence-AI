"""
SYNAPTIQ — Interpretable Alzheimer's Detection
Full Streamlit UI — Memory Assistance Platform
"""
   
import tempfile

# Optional voice/audio dependencies. The rest of the app still starts if one is missing.
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
try:
    from streamlit_geolocation import streamlit_geolocation
except ImportError:
    streamlit_geolocation = None
import sqlite3
import math

# Geofence caregiver calling uses the same browser `tel:` mechanism as the SOS page.
# No paid SMS/telephony provider is required.

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

# Caregiver phone numbers are taken from the trusted contacts saved in the SOS/Caregiver Portal.

import base64
import json
import html
import random
warnings.filterwarnings('ignore')

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

    # Migrate databases created by earlier versions.
    existing_columns = {row[1] for row in cur.execute("PRAGMA table_info(cognitive_records)").fetchall()}
    if "game_name" not in existing_columns:
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
        game_score, game_attempts, game_accuracy
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
    st.markdown("## 📊 Cognitive Performance Analytics")
    friendly_card(
        "Track change over time",
        "This dashboard combines your original cognitive screening, diagnosis results, "
        "and memory-game performance. It is a monitoring aid, not a clinical diagnosis.",
        "🧠"
    )

    hist = cognitive_history()
    if hist.empty:
        st.info("No cognitive records yet. Complete the Cognitive Questionnaire, Patient Diagnosis, or Cognitive Game.")
        return

    assessment = hist[hist["record_type"].isin(["screening", "diagnosis"])].copy()
    games = hist[hist["record_type"] == "game"].copy()

    c1, c2, c3, c4 = st.columns(4)
    latest_mmse = assessment["mmse"].dropna().iloc[-1] if assessment["mmse"].notna().any() else "—"
    latest_cdr = assessment["cdr"].dropna().iloc[-1] if assessment["cdr"].notna().any() else "—"
    latest_conf = assessment["confidence"].dropna().iloc[-1] if assessment["confidence"].notna().any() else "—"
    avg_acc = games["game_accuracy"].dropna().mean() if not games.empty else None
    c1.metric("Latest MMSE-style score", f"{latest_mmse:.0f}/30" if isinstance(latest_mmse, (int,float,np.integer,np.floating)) else "—")
    c2.metric("Latest CDR-style rating", f"{latest_cdr:g}" if isinstance(latest_cdr, (int,float,np.integer,np.floating)) else "—")
    c3.metric("Latest model confidence", f"{latest_conf:.1f}%" if isinstance(latest_conf, (int,float,np.integer,np.floating)) else "—")
    c4.metric("Game accuracy", f"{avg_acc:.1f}%" if avg_acc is not None and not pd.isna(avg_acc) else "—")

    if not assessment.empty and assessment["mmse"].notna().any():
        chart_df = assessment[assessment["mmse"].notna()].copy()
        chart_df["recorded_at"] = pd.to_datetime(chart_df["recorded_at"])
        fig = px.line(chart_df, x="recorded_at", y="mmse", markers=True,
                      title="Cognitive Screening / Assessment Trend")
        fig.update_yaxes(range=[0, 30], title="MMSE-style score")
        fig.update_xaxes(title="")
        st.plotly_chart(fig, use_container_width=True)

    if not assessment.empty and assessment["cdr"].notna().any():
        chart_df = assessment[assessment["cdr"].notna()].copy()
        chart_df["recorded_at"] = pd.to_datetime(chart_df["recorded_at"])
        fig = px.line(chart_df, x="recorded_at", y="cdr", markers=True,
                      title="CDR-style Profile Trend")
        fig.update_yaxes(range=[0, 3], title="CDR-style rating")
        fig.update_xaxes(title="")
        st.plotly_chart(fig, use_container_width=True)

    if not games.empty:
        chart_df = games.copy()
        chart_df["recorded_at"] = pd.to_datetime(chart_df["recorded_at"])
        fig = px.line(chart_df, x="recorded_at", y="game_accuracy", markers=True,
                      title="Memory Game Accuracy Trend")
        fig.update_yaxes(range=[0, 100], title="Accuracy (%)")
        fig.update_xaxes(title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Assessment History")
    show_cols = ["recorded_at", "record_type", "game_name", "mmse", "cdr", "confidence", "prediction",
                 "game_score", "game_attempts", "game_accuracy"]
    show_cols = [c for c in show_cols if c in hist.columns]
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
    st.markdown("## 📍 GPS Safety & Geofencing")
    friendly_card(
        "Consent-based location safety",
        "Location is requested by your browser only when you choose to use this feature. "
        "The latest location is stored locally for this demo so the caregiver portal can display it.",
        "🛡️"
    )

    st.warning(
        "GPS is a safety-support feature, not a substitute for supervision or emergency services. "
        "For continuous background tracking after the browser is closed, use a dedicated mobile app or wearable."
    )

    zone = get_safe_zone()
    if zone:
        st.info(
            f"Safe zone: {zone['radius_m']:.0f} m radius around "
            f"({zone['latitude']:.6f}, {zone['longitude']:.6f})"
        )
    else:
        st.info("No safe zone has been configured yet. A caregiver can create one below.")

    st.markdown("### 1️⃣ Get current GPS location")
    if streamlit_geolocation is None:
        st.error("GPS package is not installed. Run: python -m pip install streamlit-geolocation")
        location = None
    else:
        location = streamlit_geolocation()

    if location and "error" in location:
        st.error(f"Location error: {location['error'].get('message', 'Unable to read location.')}")
    elif location and location.get("latitude") is not None:
        lat = float(location["latitude"])
        lon = float(location["longitude"])
        accuracy = location.get("accuracy")
        st.session_state.gps_last_location = location

        status, distance = save_location(lat, lon, accuracy)
        alert_result = process_geofence_alert(lat, lon, status, distance)

        # Use the exact same `tel:` calling approach as the SOS page.
        # A normal browser cannot silently place a phone call without user
        # interaction; the large call button is therefore shown immediately
        # when the geofence is outside.
        if status == "Outside safe zone":
            contact, phone = get_callable_caregiver()
            render_geofence_caregiver_call(contact, phone, distance)

        c1, c2, c3 = st.columns(3)
        c1.metric("Latitude", f"{lat:.6f}")
        c2.metric("Longitude", f"{lon:.6f}")
        c3.metric("Accuracy", f"{float(accuracy):.0f} m" if accuracy else "—")

        if status == "Outside safe zone":
            st.error(f"🚨 OUTSIDE SAFE ZONE — approximately {distance:.0f} m from the safe-zone center.")
        elif status == "Inside safe zone":
            st.success(f"✅ Inside safe zone — {distance:.0f} m from the center.")
        else:
            st.info("Location received. Configure a safe zone to enable geofence alerts.")

        st.map(pd.DataFrame([{"lat": lat, "lon": lon}]), zoom=15)

    st.markdown("### 2️⃣ Configure safe zone")
    st.caption("For the demo, the caregiver can use the patient's last known location as the safe-zone center.")
    latest = latest_location()
    with st.form("safe_zone_form"):
        if latest:
            default_lat = float(latest["latitude"])
            default_lon = float(latest["longitude"])
        else:
            default_lat, default_lon = 0.0, 0.0
        c1, c2, c3 = st.columns(3)
        with c1:
            zone_lat = st.number_input("Center latitude", value=default_lat, format="%.6f")
        with c2:
            zone_lon = st.number_input("Center longitude", value=default_lon, format="%.6f")
        with c3:
            zone_radius = st.number_input("Safe-zone radius (metres)", min_value=50.0,
                                          max_value=10000.0, value=float(zone["radius_m"]) if zone else 500.0,
                                          step=50.0)
        if st.form_submit_button("💾 Save Safe Zone", use_container_width=True):
            if abs(zone_lat) <= 90 and abs(zone_lon) <= 180:
                set_safe_zone(zone_lat, zone_lon, zone_radius)
                st.success("Safe zone saved.")
                st.rerun()
            else:
                st.error("Enter valid latitude/longitude values.")

    st.markdown("### 3️⃣ Recent location history")
    hist = location_history(25)
    if hist.empty:
        st.info("No location updates recorded yet.")
    else:
        display = hist[["recorded_at", "latitude", "longitude", "accuracy", "status", "distance_from_safe_zone"]].copy()
        st.dataframe(display, use_container_width=True, hide_index=True)


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
        "📍 GPS Safety & Geofencing",
        "🧠 Cognitive Performance",
        "🧠 Risk Timeline",
        "💬 AI Chatbot",
        "🎤 Voice Assistant",
        "🧩 Cognitive Games",
        "📊 Model Analytics",
        "📈 Visualizations"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"""
    <div style='padding:0.8rem; background:{t['card']}; border-radius:8px; border:1px solid {t['border2']}; margin-bottom:0.8rem;'>
        <div style='font-size:0.7rem; color:{t['txt2']}; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem;'>Dataset</div>
        <div style='color:{t['txt']}; font-size:0.85rem; font-weight:500;'>OASIS Longitudinal</div>
        <div style='color:{t['txt2']}; font-size:0.75rem;'>373 clinical records</div>
    </div>
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
# SYNAPTIQ - COMPLETE COGNITIVE GAMES MODULE
# ============================================================
#
# ADD THIS ENTIRE BLOCK TO YOUR ourapp.py
#
# Games:
#   1. Quick Tap       - Reaction Speed
#   2. Color Check     - Attention
#   3. Card Memory     - Visual Memory
#   4. Previous Card   - Working Memory
#   5. Memory Maze     - Executive Function
#
# Every game contains:
#   - Demo animation
#   - Instructions
#   - Playable game
#   - Score
#   - Accuracy
#   - Database saving
#
# ============================================================

import streamlit as st
import streamlit.components.v1 as components
import random
import time


# ============================================================
# DATABASE SAVE FUNCTION
# ============================================================

def cg_save_result(game_name, score, attempts, accuracy):
    """
    Safely saves game result to the existing SYNAPTIQ
    cognitive_records database.

    Works with your existing save_cognitive_record()
    function.
    """

    try:

        save_cognitive_record(
            record_type="game",
            game_score=float(score),
            game_attempts=int(attempts),
            game_accuracy=float(accuracy)
        )

    except Exception as e:

        # The game should continue even if analytics
        # database is temporarily unavailable.
        print("Cognitive game database warning:", e)


# ============================================================
# COGNITIVE GAMES CSS
# ============================================================

def cg_apply_style():

    st.markdown(
        """
        <style>

        /* ==============================================
           MAIN TITLE
        ============================================== */

        .cg-main-title {
            text-align:center;
            font-size:44px;
            font-weight:900;
            margin-top:10px;
            margin-bottom:5px;
        }

        .cg-main-subtitle {
            text-align:center;
            font-size:20px;
            margin-bottom:25px;
        }


        /* ==============================================
           WELCOME CARD
        ============================================== */

        .cg-welcome {
            padding:28px;
            border-radius:28px;
            margin:15px 0 25px 0;
            border:2px solid rgba(100,100,100,0.12);
            box-shadow:0 8px 30px rgba(0,0,0,0.08);
            text-align:center;
        }

        .cg-welcome h2 {
            font-size:30px;
            margin-bottom:12px;
        }

        .cg-welcome p {
            font-size:19px;
            line-height:1.7;
        }


        /* ==============================================
           GAME INFORMATION
        ============================================== */

        .cg-info-card {
            padding:25px;
            border-radius:25px;
            margin:15px 0;
            border:2px solid rgba(100,100,100,0.12);
            box-shadow:0 7px 25px rgba(0,0,0,0.07);
        }

        .cg-game-name {
            font-size:30px;
            font-weight:900;
            margin-bottom:8px;
        }

        .cg-game-description {
            font-size:18px;
            line-height:1.7;
        }

        .cg-domain {
            display:inline-block;
            padding:8px 16px;
            border-radius:30px;
            margin-top:12px;
            font-size:16px;
            font-weight:800;
        }


        /* ==============================================
           INSTRUCTION
        ============================================== */

        .cg-instruction {
            text-align:center;
            font-size:23px;
            font-weight:800;
            padding:18px;
            margin:15px 0;
            border-radius:20px;
        }


        /* ==============================================
           RESULT
        ============================================== */

        .cg-result {
            text-align:center;
            padding:28px;
            border-radius:28px;
            margin:20px 0;
            border:2px solid rgba(100,100,100,0.12);
            box-shadow:0 8px 30px rgba(0,0,0,0.08);
        }

        .cg-result-number {
            font-size:42px;
            font-weight:900;
        }

        .cg-result-label {
            font-size:20px;
            font-weight:700;
        }


        /* ==============================================
           GAME CARD
        ============================================== */

        .cg-big-card {
            width:230px;
            height:290px;
            margin:25px auto;
            border-radius:30px;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:85px;
            font-weight:900;
            box-shadow:0 15px 35px rgba(0,0,0,0.18);
        }


        /* ==============================================
           MAZE
        ============================================== */

        .maze-cell {
            height:45px;
            border-radius:9px;
            margin:3px;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:20px;
            font-weight:900;
        }


        /* ==============================================
           ELDER FRIENDLY BUTTONS
        ============================================== */

        div.stButton > button {
            min-height:52px;
            font-size:18px;
            font-weight:800;
            border-radius:16px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DEMO 1 - QUICK TAP
# ============================================================

def cg_demo_quick_tap():

    html = """
    <html>

    <head>

    <style>

    body {
        margin:0;
        font-family:Arial,sans-serif;
        background:linear-gradient(135deg,#e3f2fd,#f3e5f5);
        overflow:hidden;
    }

    .container {
        height:100vh;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
    }

    .title {
        font-size:27px;
        font-weight:bold;
        margin-bottom:20px;
    }

    .screen {
        width:350px;
        height:170px;
        background:white;
        border-radius:25px;
        display:flex;
        justify-content:center;
        align-items:center;
        box-shadow:0 10px 30px rgba(0,0,0,.15);
    }

    .ready {
        font-size:25px;
        font-weight:bold;
    }

    .target {
        width:85px;
        height:85px;
        background:#ef5350;
        border-radius:50%;
        display:none;
        box-shadow:0 0 0 12px rgba(239,83,80,.18);
    }

    .result {
        margin-top:18px;
        font-size:20px;
        font-weight:bold;
    }

    </style>

    </head>

    <body>

    <div class="container">

        <div class="title">
            ⚡ SYNAPTIQ Quick Tap
        </div>

        <div class="screen">

            <div id="ready" class="ready">
                Get Ready...
            </div>

            <div id="target" class="target">
            </div>

        </div>

        <div id="result" class="result">
        </div>

    </div>


    <script>

    setTimeout(function(){

        document.getElementById("ready").style.display="none";

        document.getElementById("target").style.display="block";

        let start = performance.now();

        document.getElementById("target").onclick=function(){

            let reaction =
                Math.round(performance.now()-start);

            document.getElementById("target").style.display="none";

            document.getElementById("result").innerHTML =
                "✓ Target tapped! Response: "
                + reaction + " ms";

        };

    },1800);

    </script>

    </body>
    </html>
    """

    components.html(
        html,
        height=350
    )


# ============================================================
# DEMO 2 - COLOR CHECK
# ============================================================

def cg_demo_color_check():

    html = """
    <html>

    <head>

    <style>

    body {
        margin:0;
        font-family:Arial;
        background:linear-gradient(135deg,#fff3e0,#e8f5e9);
    }

    .container {
        height:100vh;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
    }

    .title {
        font-size:27px;
        font-weight:bold;
        margin-bottom:15px;
    }

    .card {
        width:160px;
        height:210px;
        background:#ef5350;
        border-radius:25px;
        display:flex;
        align-items:center;
        justify-content:center;
        color:white;
        font-size:30px;
        font-weight:bold;
        box-shadow:0 10px 30px rgba(0,0,0,.18);
    }

    .question {
        margin-top:18px;
        font-size:21px;
        font-weight:bold;
    }

    </style>

    </head>

    <body>

    <div class="container">

        <div class="title">
            🎨 SYNAPTIQ Color Check
        </div>

        <div class="card">
            CARD
        </div>

        <div class="question">
            Is the card RED? → YES
        </div>

    </div>

    </body>

    </html>
    """

    components.html(
        html,
        height=350
    )


# ============================================================
# DEMO 3 - CARD MEMORY
# ============================================================

def cg_demo_card_memory():

    html = """
    <html>

    <head>

    <style>

    body {
        margin:0;
        font-family:Arial;
        background:linear-gradient(135deg,#ede7f6,#e3f2fd);
    }

    .container {
        height:100vh;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
    }

    .title {
        font-size:27px;
        font-weight:bold;
        margin-bottom:15px;
    }

    .card {
        width:170px;
        height:210px;
        background:white;
        border-radius:25px;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:80px;
        box-shadow:0 10px 30px rgba(0,0,0,.15);
    }

    .text {
        margin-top:18px;
        font-size:21px;
        font-weight:bold;
    }

    </style>

    </head>

    <body>

    <div class="container">

        <div class="title">
            🃏 SYNAPTIQ Card Memory
        </div>

        <div class="card">
            ★
        </div>

        <div class="text">
            Remember this card...
        </div>

    </div>

    </body>

    </html>
    """

    components.html(
        html,
        height=350
    )


# ============================================================
# DEMO 4 - PREVIOUS CARD
# ============================================================

def cg_demo_previous_card():

    html = """
    <html>

    <head>

    <style>

    body {
        margin:0;
        font-family:Arial;
        background:linear-gradient(135deg,#fce4ec,#fff8e1);
    }

    .container {
        height:100vh;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
    }

    .title {
        font-size:27px;
        font-weight:bold;
        margin-bottom:20px;
    }

    .cards {
        display:flex;
        gap:25px;
    }

    .card {
        width:120px;
        height:155px;
        background:white;
        border-radius:20px;
        display:flex;
        justify-content:center;
        align-items:center;
        font-size:55px;
        box-shadow:0 10px 25px rgba(0,0,0,.15);
    }

    .text {
        margin-top:20px;
        font-size:21px;
        font-weight:bold;
    }

    </style>

    </head>

    <body>

    <div class="container">

        <div class="title">
            🔄 SYNAPTIQ Previous Card
        </div>

        <div class="cards">

            <div class="card">
                ♥
            </div>

            <div class="card">
                ♥
            </div>

        </div>

        <div class="text">
            Are they the same? → YES
        </div>

    </div>

    </body>

    </html>
    """

    components.html(
        html,
        height=350
    )


# ============================================================
# DEMO 5 - MEMORY MAZE
# ============================================================

def cg_demo_memory_maze():

    path = [
        0, 1, 7, 13, 19,
        25, 26, 27, 28, 34,
        35
    ]

    html = """
    <html>

    <head>

    <style>

    body {
        margin:0;
        font-family:Arial;
        background:linear-gradient(135deg,#e8eaf6,#e0f2f1);
    }

    .container {
        height:100vh;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
    }

    .title {
        font-size:26px;
        font-weight:bold;
        margin-bottom:15px;
    }

    .grid {
        display:grid;
        grid-template-columns:repeat(6,40px);
        gap:4px;
    }

    .cell {
        width:40px;
        height:40px;
        background:white;
        border-radius:7px;
        box-shadow:0 2px 5px rgba(0,0,0,.1);
    }

    .path {
        background:#66bb6a;
    }

    .start {
        background:#42a5f5;
    }

    .finish {
        background:#ef5350;
    }

    .text {
        margin-top:15px;
        font-size:20px;
        font-weight:bold;
    }

    </style>

    </head>

    <body>

    <div class="container">

        <div class="title">
            🧩 SYNAPTIQ Memory Maze
        </div>

        <div class="grid">
    """

    for i in range(36):

        if i == path[0]:

            cls = "cell start"

        elif i == path[-1]:

            cls = "cell finish"

        elif i in path:

            cls = "cell path"

        else:

            cls = "cell"

        html += f'<div class="{cls}"></div>'

    html += """
        </div>

        <div class="text">
            Memorize the hidden pathway
        </div>

    </div>

    </body>

    </html>
    """

    components.html(
        html,
        height=350
    )


# ============================================================
# GAME 1 - QUICK TAP
# ============================================================

def cg_game_quick_tap():

    st.markdown(
        """
        <div class="cg-info-card">

            <div class="cg-game-name">
                ⚡ Quick Tap
            </div>

            <div class="cg-game-description">
                Wait for the red target to appear.
                Tap as quickly as you can.
            </div>

            <div class="cg-domain">
                🧠 Reaction Speed
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "💡 Don't click while waiting. "
        "Tap only when the red target appears."
    )

    if "qt_active" not in st.session_state:
        st.session_state.qt_active = False

    if "qt_target_time" not in st.session_state:
        st.session_state.qt_target_time = None

    if "qt_result" not in st.session_state:
        st.session_state.qt_result = None

    if not st.session_state.qt_active:

        if st.button(
            "🚀 START GAME",
            use_container_width=True,
            key="qt_start_game"
        ):

            delay = random.uniform(
                1.2,
                2.8
            )

            st.session_state.qt_target_time = (
                time.perf_counter() + delay
            )

            st.session_state.qt_active = True
            st.session_state.qt_result = None

            st.rerun()

    else:

        now = time.perf_counter()

        if now < st.session_state.qt_target_time:

            remaining = round(
                st.session_state.qt_target_time - now,
                1
            )

            st.markdown(
                f"""
                <div class="cg-instruction">
                    👀 GET READY...<br><br>
                    Target coming in {remaining} seconds
                </div>
                """,
                unsafe_allow_html=True
            )

            # Clicking this makes Streamlit rerun and update
            # the countdown.
            if st.button(
                "⏳ WAIT",
                use_container_width=True,
                key="qt_wait_button"
            ):
                st.rerun()

        else:

            st.markdown(
                """
                <div class="cg-instruction">
                    🔴 TAP NOW!
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "🔴 TAP TARGET",
                use_container_width=True,
                key="qt_target_button"
            ):

                reaction = (
                    time.perf_counter()
                    - st.session_state.qt_target_time
                )

                reaction_ms = round(
                    reaction * 1000
                )

                st.session_state.qt_result = reaction_ms
                st.session_state.qt_active = False

                cg_save_result(
                    "Quick Tap",
                    reaction_ms,
                    1,
                    100
                )

                st.rerun()

    if st.session_state.qt_result is not None:

        st.markdown(
            f"""
            <div class="cg-result">

                <div class="cg-result-number">
                    ⚡ {st.session_state.qt_result} ms
                </div>

                <div class="cg-result-label">
                    Response Time
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "🔄 PLAY AGAIN",
            use_container_width=True,
            key="qt_again"
        ):

            st.session_state.qt_result = None
            st.session_state.qt_active = False

            st.rerun()


# ============================================================
# GAME 2 - COLOR CHECK
# ============================================================

def cg_game_color_check():

    st.markdown(
        """
        <div class="cg-info-card">

            <div class="cg-game-name">
                🎨 Color Check
            </div>

            <div class="cg-game-description">
                Look at the card and decide whether it is red.
                Try to be fast and accurate.
            </div>

            <div class="cg-domain">
                👁️ Attention
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if "cc_active" not in st.session_state:
        st.session_state.cc_active = False

    if "cc_trial" not in st.session_state:
        st.session_state.cc_trial = 0

    if "cc_score" not in st.session_state:
        st.session_state.cc_score = 0

    if "cc_red" not in st.session_state:
        st.session_state.cc_red = True

    if not st.session_state.cc_active:

        st.info(
            "You will see 10 cards. "
            "Choose YES when the card is red."
        )

        if st.button(
            "🎨 START GAME",
            use_container_width=True,
            key="cc_start_game"
        ):

            st.session_state.cc_active = True
            st.session_state.cc_trial = 0
            st.session_state.cc_score = 0

            st.session_state.cc_red = random.choice(
                [True, False]
            )

            st.rerun()

    else:

        if st.session_state.cc_trial < 10:

            red = st.session_state.cc_red

            color = (
                "#ef5350"
                if red
                else "#424242"
            )

            st.markdown(
                f"""
                <div
                    class="cg-big-card"
                    style="background:{color};color:white;"
                >
                    🃏
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="cg-instruction">

                    Is the card RED?

                    <br><br>

                    Trial
                    {st.session_state.cc_trial + 1}
                    / 10

                </div>
                """,
                unsafe_allow_html=True
            )

            yes, no = st.columns(2)

            with yes:

                if st.button(
                    "🔴 YES",
                    use_container_width=True,
                    key=f"cc_yes_{st.session_state.cc_trial}"
                ):

                    if red:
                        st.session_state.cc_score += 1

                    st.session_state.cc_trial += 1

                    st.session_state.cc_red = random.choice(
                        [True, False]
                    )

                    st.rerun()

            with no:

                if st.button(
                    "⚫ NO",
                    use_container_width=True,
                    key=f"cc_no_{st.session_state.cc_trial}"
                ):

                    if not red:
                        st.session_state.cc_score += 1

                    st.session_state.cc_trial += 1

                    st.session_state.cc_red = random.choice(
                        [True, False]
                    )

                    st.rerun()

        else:

            accuracy = (
                st.session_state.cc_score / 10
            ) * 100

            st.markdown(
                f"""
                <div class="cg-result">

                    <div class="cg-result-number">
                        🎉 {st.session_state.cc_score}/10
                    </div>

                    <div class="cg-result-label">
                        Accuracy: {accuracy:.0f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            cg_save_result(
                "Color Check",
                st.session_state.cc_score,
                10,
                accuracy
            )

            if st.button(
                "🔄 PLAY AGAIN",
                use_container_width=True,
                key="cc_again"
            ):

                st.session_state.cc_active = False
                st.session_state.cc_trial = 0
                st.session_state.cc_score = 0

                st.rerun()


# ============================================================
# GAME 3 - CARD MEMORY
# ============================================================

def cg_game_card_memory():

    st.markdown(
        """
        <div class="cg-info-card">

            <div class="cg-game-name">
                🃏 Card Memory
            </div>

            <div class="cg-game-description">
                Remember the symbols you have seen.
                Decide whether the current symbol appeared before.
            </div>

            <div class="cg-domain">
                🧠 Visual Memory
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    symbols = [
        "★",
        "♥",
        "●",
        "◆",
        "▲",
        "☀",
        "☘",
        "♦"
    ]

    if "cm_active" not in st.session_state:
        st.session_state.cm_active = False

    if "cm_trial" not in st.session_state:
        st.session_state.cm_trial = 0

    if "cm_score" not in st.session_state:
        st.session_state.cm_score = 0

    if "cm_seen" not in st.session_state:
        st.session_state.cm_seen = []

    if "cm_current" not in st.session_state:
        st.session_state.cm_current = "★"

    if not st.session_state.cm_active:

        st.info(
            "Remember the symbols that appear. "
            "Then decide whether you have seen them before."
        )

        if st.button(
            "🧠 START GAME",
            use_container_width=True,
            key="cm_start_game"
        ):

            st.session_state.cm_active = True
            st.session_state.cm_trial = 0
            st.session_state.cm_score = 0
            st.session_state.cm_seen = []

            st.session_state.cm_current = random.choice(
                symbols
            )

            st.rerun()

    else:

        if st.session_state.cm_trial < 10:

            current = st.session_state.cm_current

            seen_before = (
                current
                in st.session_state.cm_seen
            )

            st.markdown(
                f"""
                <div
                    class="cg-big-card"
                    style="
                    background:white;
                    color:#3949ab;
                    "
                >
                    {current}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="cg-instruction">

                    Have you seen this card before?

                    <br><br>

                    Trial
                    {st.session_state.cm_trial + 1}
                    / 10

                </div>
                """,
                unsafe_allow_html=True
            )

            yes, no = st.columns(2)

            with yes:

                if st.button(
                    "✅ YES",
                    use_container_width=True,
                    key=f"cm_yes_{st.session_state.cm_trial}"
                ):

                    if seen_before:
                        st.session_state.cm_score += 1

                    st.session_state.cm_seen.append(
                        current
                    )

                    st.session_state.cm_trial += 1

                    st.session_state.cm_current = random.choice(
                        symbols
                    )

                    st.rerun()

            with no:

                if st.button(
                    "❌ NO",
                    use_container_width=True,
                    key=f"cm_no_{st.session_state.cm_trial}"
                ):

                    if not seen_before:
                        st.session_state.cm_score += 1

                    st.session_state.cm_seen.append(
                        current
                    )

                    st.session_state.cm_trial += 1

                    st.session_state.cm_current = random.choice(
                        symbols
                    )

                    st.rerun()

        else:

            accuracy = (
                st.session_state.cm_score / 10
            ) * 100

            st.markdown(
                f"""
                <div class="cg-result">

                    <div class="cg-result-number">
                        🧠 {st.session_state.cm_score}/10
                    </div>

                    <div class="cg-result-label">
                        Accuracy: {accuracy:.0f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            cg_save_result(
                "Card Memory",
                st.session_state.cm_score,
                10,
                accuracy
            )

            if st.button(
                "🔄 PLAY AGAIN",
                use_container_width=True,
                key="cm_again"
            ):

                st.session_state.cm_active = False
                st.session_state.cm_trial = 0
                st.session_state.cm_score = 0
                st.session_state.cm_seen = []

                st.rerun()


# ============================================================
# GAME 4 - PREVIOUS CARD
# ============================================================

def cg_game_previous_card():

    st.markdown(
        """
        <div class="cg-info-card">

            <div class="cg-game-name">
                🔄 Previous Card
            </div>

            <div class="cg-game-description">
                Compare the current card with the previous card.
                Decide whether they are the same.
            </div>

            <div class="cg-domain">
                🧠 Working Memory
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    cards = [
        "♥",
        "♦",
        "♣",
        "♠"
    ]

    if "pc_active" not in st.session_state:
        st.session_state.pc_active = False

    if "pc_trial" not in st.session_state:
        st.session_state.pc_trial = 0

    if "pc_score" not in st.session_state:
        st.session_state.pc_score = 0

    if "pc_previous" not in st.session_state:
        st.session_state.pc_previous = "♥"

    if "pc_current" not in st.session_state:
        st.session_state.pc_current = "♥"

    if not st.session_state.pc_active:

        st.info(
            "Compare the current card with the previous card."
        )

        if st.button(
            "🔄 START GAME",
            use_container_width=True,
            key="pc_start_game"
        ):

            st.session_state.pc_active = True
            st.session_state.pc_trial = 0
            st.session_state.pc_score = 0

            st.session_state.pc_previous = random.choice(
                cards
            )

            st.session_state.pc_current = random.choice(
                cards
            )

            st.rerun()

    else:

        if st.session_state.pc_trial < 10:

            previous = (
                st.session_state.pc_previous
            )

            current = (
                st.session_state.pc_current
            )

            same = previous == current

            left, right = st.columns(2)

            with left:

                st.markdown(
                    f"""
                    <div
                        class="cg-big-card"
                        style="
                        width:180px;
                        height:220px;
                        background:#e3f2fd;
                        color:#1565c0;
                        "
                    >
                        {previous}
                    </div>

                    <div class="cg-instruction">
                        Previous
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with right:

                st.markdown(
                    f"""
                    <div
                        class="cg-big-card"
                        style="
                        width:180px;
                        height:220px;
                        background:#fff3e0;
                        color:#ef6c00;
                        "
                    >
                        {current}
                    </div>

                    <div class="cg-instruction">
                        Current
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                f"""
                <div class="cg-instruction">

                    Are they the same?

                    <br><br>

                    Trial
                    {st.session_state.pc_trial + 1}
                    / 10

                </div>
                """,
                unsafe_allow_html=True
            )

            yes, no = st.columns(2)

            with yes:

                if st.button(
                    "✅ YES",
                    use_container_width=True,
                    key=f"pc_yes_{st.session_state.pc_trial}"
                ):

                    if same:
                        st.session_state.pc_score += 1

                    st.session_state.pc_previous = current

                    st.session_state.pc_current = random.choice(
                        cards
                    )

                    st.session_state.pc_trial += 1

                    st.rerun()

            with no:

                if st.button(
                    "❌ NO",
                    use_container_width=True,
                    key=f"pc_no_{st.session_state.pc_trial}"
                ):

                    if not same:
                        st.session_state.pc_score += 1

                    st.session_state.pc_previous = current

                    st.session_state.pc_current = random.choice(
                        cards
                    )

                    st.session_state.pc_trial += 1

                    st.rerun()

        else:

            accuracy = (
                st.session_state.pc_score / 10
            ) * 100

            st.markdown(
                f"""
                <div class="cg-result">

                    <div class="cg-result-number">
                        🔄 {st.session_state.pc_score}/10
                    </div>

                    <div class="cg-result-label">
                        Accuracy: {accuracy:.0f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            cg_save_result(
                "Previous Card",
                st.session_state.pc_score,
                10,
                accuracy
            )

            if st.button(
                "🔄 PLAY AGAIN",
                use_container_width=True,
                key="pc_again"
            ):

                st.session_state.pc_active = False
                st.session_state.pc_trial = 0
                st.session_state.pc_score = 0

                st.rerun()


# ============================================================
# MEMORY MAZE PATH
# ============================================================

def cg_generate_maze():

    size = 6

    path = [(0, 0)]

    current = (0, 0)

    for _ in range(13):

        r, c = current

        possible = []

        directions = [
            (r + 1, c),
            (r - 1, c),
            (r, c + 1),
            (r, c - 1)
        ]

        random.shuffle(directions)

        for nr, nc in directions:

            if (
                0 <= nr < size
                and 0 <= nc < size
                and (nr, nc) not in path
            ):
                possible.append(
                    (nr, nc)
                )

        if not possible:
            break

        current = random.choice(
            possible
        )

        path.append(current)

    return path


# ============================================================
# GAME 5 - MEMORY MAZE
# ============================================================

def cg_game_memory_maze():

    st.markdown(
        """
        <div class="cg-info-card">

            <div class="cg-game-name">
                🧩 Memory Maze
            </div>

            <div class="cg-game-description">
                Memorize the hidden pathway and reproduce it
                by selecting the tiles in the correct order.
            </div>

            <div class="cg-domain">
                🧠 Executive Function & Spatial Memory
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    size = 6

    if "mz_active" not in st.session_state:
        st.session_state.mz_active = False

    if "mz_show_path" not in st.session_state:
        st.session_state.mz_show_path = False

    if "mz_path" not in st.session_state:
        st.session_state.mz_path = []

    if "mz_user_path" not in st.session_state:
        st.session_state.mz_user_path = []

    if "mz_errors" not in st.session_state:
        st.session_state.mz_errors = 0

    if not st.session_state.mz_active:

        st.info(
            "First memorize the green pathway. "
            "Then reproduce it after hiding the pathway."
        )

        if st.button(
            "🧩 START MEMORY MAZE",
            use_container_width=True,
            key="mz_start_game"
        ):

            st.session_state.mz_path = (
                cg_generate_maze()
            )

            st.session_state.mz_user_path = []

            st.session_state.mz_errors = 0

            st.session_state.mz_active = True

            st.session_state.mz_show_path = True

            st.rerun()

    else:

        path = st.session_state.mz_path

        # ----------------------------------------------------
        # SHOW PATH
        # ----------------------------------------------------

        if st.session_state.mz_show_path:

            st.markdown(
                """
                <div class="cg-instruction">
                    👀 MEMORIZE THE PATH
                </div>
                """,
                unsafe_allow_html=True
            )

            for r in range(size):

                cols = st.columns(size)

                for c in range(size):

                    with cols[c]:

                        cell = (r, c)

                        if cell in path:

                            if cell == path[0]:

                                bg = "#42a5f5"
                                symbol = "▶"

                            elif cell == path[-1]:

                                bg = "#ef5350"
                                symbol = "★"

                            else:

                                bg = "#66bb6a"
                                symbol = "●"

                            st.markdown(
                                f"""
                                <div
                                    class="maze-cell"
                                    style="
                                    background:{bg};
                                    color:white;
                                    "
                                >
                                    {symbol}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        else:

                            st.markdown(
                                """
                                <div
                                    class="maze-cell"
                                    style="
                                    background:#eeeeee;
                                    "
                                >
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

            if st.button(
                "🙈 HIDE PATH & PLAY",
                use_container_width=True,
                key="mz_hide_path"
            ):

                st.session_state.mz_show_path = False

                st.rerun()

        # ----------------------------------------------------
        # PLAY MAZE
        # ----------------------------------------------------

        else:

            st.markdown(
                """
                <div class="cg-instruction">
                    🧠 RECREATE THE PATH
                </div>
                """,
                unsafe_allow_html=True
            )

            current_length = len(
                st.session_state.mz_user_path
            )

            total_length = len(path)

            st.progress(
                min(
                    current_length / total_length,
                    1.0
                )
            )

            st.caption(
                f"Correct steps: "
                f"{current_length} / {total_length} "
                f"   |   Mistakes: "
                f"{st.session_state.mz_errors}"
            )

            # ------------------------------------------------
            # GRID
            # ------------------------------------------------

            for r in range(size):

                cols = st.columns(size)

                for c in range(size):

                    with cols[c]:

                        cell = (r, c)

                        clicked = (
                            cell
                            in st.session_state.mz_user_path
                        )

                        if clicked:

                            st.markdown(
                                """
                                <div
                                    class="maze-cell"
                                    style="
                                    background:#42a5f5;
                                    color:white;
                                    "
                                >
                                    ✓
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        else:

                            if st.button(
                                " ",
                                key=f"maze_{r}_{c}"
                            ):

                                expected_index = len(
                                    st.session_state.mz_user_path
                                )

                                expected = path[
                                    expected_index
                                ]

                                if cell == expected:

                                    st.session_state.mz_user_path.append(
                                        cell
                                    )

                                    st.rerun()

                                else:

                                    st.session_state.mz_errors += 1

                                    st.error(
                                        "❌ Wrong tile. "
                                        "Try again."
                                    )

            # ------------------------------------------------
            # FINISHED
            # ------------------------------------------------

            if len(
                st.session_state.mz_user_path
            ) == len(path):

                errors = (
                    st.session_state.mz_errors
                )

                accuracy = max(
                    0,
                    100 - (errors * 5)
                )

                st.markdown(
                    f"""
                    <div class="cg-result">

                        <div class="cg-result-number">
                            🎉 Maze Complete!
                        </div>

                        <div class="cg-result-label">
                            Mistakes: {errors}
                            <br>
                            Performance: {accuracy}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                cg_save_result(
                    "Memory Maze",
                    accuracy,
                    len(path),
                    accuracy
                )

                if st.button(
                    "🔄 PLAY AGAIN",
                    use_container_width=True,
                    key="mz_again"
                ):

                    st.session_state.mz_active = False

                    st.session_state.mz_show_path = False

                    st.session_state.mz_path = []

                    st.session_state.mz_user_path = []

                    st.session_state.mz_errors = 0

                    st.rerun()


# ============================================================
# MAIN COGNITIVE GAMES PAGE
# ============================================================

def render_cognitive_games():

    # Apply CSS
    cg_apply_style()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="cg-main-title">
            🧠 Cognitive Games
        </div>

        <div class="cg-main-subtitle">
            Play • Think • Remember • Have Fun!
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # WELCOME
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="cg-welcome">

            <h2>
                🌟 Welcome to SYNAPTIQ Brain Play!
            </h2>

            <p>
                Exercise your memory, attention, reaction speed
                and thinking skills through simple interactive games.
            </p>

            <p>
                💡 Watch the demonstration first,
                then try the game yourself!
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # GAME SELECTOR
    # --------------------------------------------------------

    selected_game = st.selectbox(
        "🎮 Choose a Game",
        [
            "⚡ Quick Tap",
            "🎨 Color Check",
            "🃏 Card Memory",
            "🔄 Previous Card",
            "🧩 Memory Maze"
        ],
        key="cg_selected_game"
    )


    st.divider()


    # ========================================================
    # QUICK TAP
    # ========================================================

    if selected_game == "⚡ Quick Tap":

        st.markdown(
            "## ⚡ Quick Tap"
        )

        st.caption(
            "Reaction Speed"
        )

        with st.expander(
            "🎥 WATCH GAME DEMONSTRATION",
            expanded=True
        ):

            cg_demo_quick_tap()

        st.markdown(
            "### 🎮 Your Turn"
        )

        cg_game_quick_tap()


    # ========================================================
    # COLOR CHECK
    # ========================================================

    elif selected_game == "🎨 Color Check":

        st.markdown(
            "## 🎨 Color Check"
        )

        st.caption(
            "Attention"
        )

        with st.expander(
            "🎥 WATCH GAME DEMONSTRATION",
            expanded=True
        ):

            cg_demo_color_check()

        st.markdown(
            "### 🎮 Your Turn"
        )

        cg_game_color_check()


    # ========================================================
    # CARD MEMORY
    # ========================================================

    elif selected_game == "🃏 Card Memory":

        st.markdown(
            "## 🃏 Card Memory"
        )

        st.caption(
            "Visual Memory"
        )

        with st.expander(
            "🎥 WATCH GAME DEMONSTRATION",
            expanded=True
        ):

            cg_demo_card_memory()

        st.markdown(
            "### 🎮 Your Turn"
        )

        cg_game_card_memory()


    # ========================================================
    # PREVIOUS CARD
    # ========================================================

    elif selected_game == "🔄 Previous Card":

        st.markdown(
            "## 🔄 Previous Card"
        )

        st.caption(
            "Working Memory"
        )

        with st.expander(
            "🎥 WATCH GAME DEMONSTRATION",
            expanded=True
        ):

            cg_demo_previous_card()

        st.markdown(
            "### 🎮 Your Turn"
        )

        cg_game_previous_card()


    # ========================================================
    # MEMORY MAZE
    # ========================================================

    elif selected_game == "🧩 Memory Maze":

        st.markdown(
            "## 🧩 Memory Maze"
        )

        st.caption(
            "Executive Function & Spatial Memory"
        )

        with st.expander(
            "🎥 WATCH GAME DEMONSTRATION",
            expanded=True
        ):

            cg_demo_memory_maze()

        st.markdown(
            "### 🎮 Your Turn"
        )

        cg_game_memory_maze()


    # ========================================================
    # BOTTOM INFORMATION
    # ========================================================

    st.divider()

    st.markdown(
        """
        <div class="cg-info-card">

            <h3>
                💜 About SYNAPTIQ Cognitive Games
            </h3>

            <p style="font-size:16px;line-height:1.7;">

            These activities are designed as interactive
            cognitive exercises and monitoring activities.

            They explore different areas such as reaction speed,
            attention, visual memory, working memory and
            spatial/executive skills.

            </p>

            <p style="font-size:15px;">

            ⚠️ These games are not a substitute for a clinical
            cognitive assessment or medical diagnosis.

            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SYNAPTIQ - COGNITIVE GAMES
# COMPLETE DROP-IN CODE
# ============================================================

import random
import time


# ============================================================
# SAVE GAME RESULT
# ============================================================

def synaptiq_save_game(game_name, score, attempts, accuracy):
    """Save one completed game result without duplicate Streamlit reruns."""
    try:
        conn = safety_db()
        row = conn.execute("""
            SELECT id, recorded_at FROM cognitive_records
            WHERE patient_id = ? AND record_type = 'game'
              AND game_name = ? AND game_score = ?
              AND game_attempts = ? AND game_accuracy = ?
            ORDER BY id DESC LIMIT 1
        """, (patient_id(), str(game_name), float(score), int(attempts), float(accuracy))).fetchone()
        conn.close()

        if row:
            try:
                previous = datetime.fromisoformat(row["recorded_at"])
                if (datetime.now() - previous).total_seconds() < 30:
                    return False
            except Exception:
                return False

        save_cognitive_record(
            record_type="game",
            game_score=float(score),
            game_attempts=int(attempts),
            game_accuracy=float(accuracy),
            game_name=str(game_name)
        )
        return True
    except Exception as e:
        print("Cognitive game save warning:", e)
        return False


# ============================================================
# COGNITIVE GAMES STYLE
# ============================================================

def synaptiq_cognitive_css():

    st.markdown(
        """
        <style>

        .syn-cg-title {
            text-align: center;
            font-size: 44px;
            font-weight: 900;
            margin-top: 10px;
            margin-bottom: 5px;
        }

        .syn-cg-subtitle {
            text-align: center;
            font-size: 20px;
            margin-bottom: 25px;
        }

        .syn-cg-welcome {
            padding: 28px;
            border-radius: 28px;
            margin: 15px 0 25px 0;
            border: 2px solid rgba(120,120,120,0.15);
            box-shadow: 0 8px 30px rgba(0,0,0,0.08);
            text-align: center;
        }

        .syn-cg-welcome h2 {
            font-size: 30px;
            margin-bottom: 10px;
        }

        .syn-cg-welcome p {
            font-size: 18px;
            line-height: 1.7;
        }

        .syn-cg-card {
            padding: 25px;
            border-radius: 25px;
            margin: 15px 0;
            border: 2px solid rgba(120,120,120,0.15);
            box-shadow: 0 7px 25px rgba(0,0,0,0.08);
        }

        .syn-cg-game-title {
            font-size: 30px;
            font-weight: 900;
            margin-bottom: 8px;
        }

        .syn-cg-description {
            font-size: 18px;
            line-height: 1.7;
        }

        .syn-cg-domain {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 30px;
            margin-top: 12px;
            font-size: 16px;
            font-weight: 800;
        }

        .syn-cg-instruction {
            text-align: center;
            font-size: 23px;
            font-weight: 800;
            padding: 18px;
            margin: 15px 0;
            border-radius: 20px;
        }

        .syn-cg-result {
            text-align: center;
            padding: 28px;
            border-radius: 28px;
            margin: 20px 0;
            border: 2px solid rgba(120,120,120,0.15);
            box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        }

        .syn-cg-result-number {
            font-size: 42px;
            font-weight: 900;
        }

        .syn-cg-result-label {
            font-size: 20px;
            font-weight: 700;
        }

        .syn-cg-big-card {
            width: 220px;
            height: 280px;
            margin: 25px auto;
            border-radius: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 85px;
            font-weight: 900;
            box-shadow: 0 15px 35px rgba(0,0,0,0.18);
        }

        .syn-cg-maze-cell {
            height: 45px;
            border-radius: 9px;
            margin: 3px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            font-weight: 900;
        }

        div.stButton > button {
            min-height: 52px;
            font-size: 18px;
            font-weight: 800;
            border-radius: 16px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DEMO VIDEO 1 - QUICK TAP
# ============================================================

def synaptiq_demo_quick_tap():

    html = """
    <html>
    <head>

    <style>

    body {
        margin: 0;
        font-family: Arial;
        background: linear-gradient(135deg,#e3f2fd,#f3e5f5);
        overflow: hidden;
    }

    .container {
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .title {
        font-size: 27px;
        font-weight: bold;
        margin-bottom: 20px;
    }

    .screen {
        width: 360px;
        height: 180px;
        background: white;
        border-radius: 25px;
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: 0 10px 30px rgba(0,0,0,.15);
    }

    .ready {
        font-size: 25px;
        font-weight: bold;
    }

    .target {
        width: 90px;
        height: 90px;
        background: #ef5350;
        border-radius: 50%;
        display: none;
        box-shadow: 0 0 0 12px rgba(239,83,80,.2);
    }

    .result {
        margin-top: 18px;
        font-size: 20px;
        font-weight: bold;
    }

    </style>

    </head>

    <body>

    <div class="container">

        <div class="title">
            ⚡ SYNAPTIQ Quick Tap
        </div>

        <div class="screen">

            <div id="ready" class="ready">
                Get Ready...
            </div>

            <div id="target" class="target"></div>

        </div>

        <div id="result" class="result"></div>

    </div>

    <script>

    setTimeout(function(){

        document.getElementById("ready").style.display = "none";

        document.getElementById("target").style.display = "block";

        let start = performance.now();

        document.getElementById("target").onclick = function(){

            let reaction =
                Math.round(performance.now() - start);

            document.getElementById("target").style.display = "none";

            document.getElementById("result").innerHTML =
                "✓ Target detected! " + reaction + " ms";

        };

    },1800);

    </script>

    </body>
    </html>
    """

    components.html(html, height=350)


# ============================================================
# DEMO VIDEO 2 - COLOR CHECK
# ============================================================

def synaptiq_demo_color_check():

    html = """
    <html>

    <head>

    <style>

    body {
        margin: 0;
        font-family: Arial;
        background: linear-gradient(135deg,#fff3e0,#e8f5e9);
    }

    .container {
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .title {
        font-size: 27px;
        font-weight: bold;
        margin-bottom: 15px;
    }

    .card {
        width: 160px;
        height: 210px;
        background: #ef5350;
        border-radius: 25px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 30px;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(0,0,0,.18);
    }

    .question {
        margin-top: 18px;
        font-size: 21px;
        font-weight: bold;
    }

    </style>

    </head>

    <body>

    <div class="container">

        <div class="title">
            🎨 SYNAPTIQ Color Check
        </div>

        <div class="card">
            🃏
        </div>

        <div class="question">
            Is the card RED? → YES
        </div>

    </div>

    </body>

    </html>
    """

    components.html(html, height=350)


# ============================================================
# DEMO VIDEO 3 - CARD MEMORY
# ============================================================

def synaptiq_demo_card_memory():

    html = """
    <html>

    <head>

    <style>

    body {
        margin: 0;
        font-family: Arial;
        background: linear-gradient(135deg,#ede7f6,#e3f2fd);
    }

    .container {
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .title {
        font-size: 27px;
        font-weight: bold;
        margin-bottom: 15px;
    }

    .card {
        width: 170px;
        height: 210px;
        background: white;
        border-radius: 25px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 80px;
        box-shadow: 0 10px 30px rgba(0,0,0,.15);
    }

    .text {
        margin-top: 18px;
        font-size: 21px;
        font-weight: bold;
    }

    </style>

    </head>

    <body>

    <div class="container">

        <div class="title">
            🃏 SYNAPTIQ Card Memory
        </div>

        <div class="card">
            ★
        </div>

        <div class="text">
            Remember this card...
        </div>

    </div>

    </body>

    </html>
    """

    components.html(html, height=350)


# ============================================================
# DEMO VIDEO 4 - PREVIOUS CARD
# ============================================================

def synaptiq_demo_previous_card():

    html = """
    <html>

    <head>

    <style>

    body {
        margin: 0;
        font-family: Arial;
        background: linear-gradient(135deg,#fce4ec,#fff8e1);
    }

    .container {
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .title {
        font-size: 27px;
        font-weight: bold;
        margin-bottom: 20px;
    }

    .cards {
        display: flex;
        gap: 25px;
    }

    .card {
        width: 120px;
        height: 155px;
        background: white;
        border-radius: 20px;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 55px;
        box-shadow: 0 10px 25px rgba(0,0,0,.15);
    }

    .text {
        margin-top: 20px;
        font-size: 21px;
        font-weight: bold;
    }

    </style>

    </head>

    <body>

    <div class="container">

        <div class="title">
            🔄 SYNAPTIQ Previous Card
        </div>

        <div class="cards">

            <div class="card">♥</div>

            <div class="card">♥</div>

        </div>

        <div class="text">
            Are they the same? → YES
        </div>

    </div>

    </body>

    </html>
    """

    components.html(html, height=350)


# ============================================================
# DEMO VIDEO 5 - MEMORY MAZE
# ============================================================

def synaptiq_demo_memory_maze():

    path = [
        0, 1, 7, 13, 19,
        25, 26, 27, 28, 34, 35
    ]

    html = """
    <html>

    <head>

    <style>

    body {
        margin: 0;
        font-family: Arial;
        background: linear-gradient(135deg,#e8eaf6,#e0f2f1);
    }

    .container {
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .title {
        font-size: 26px;
        font-weight: bold;
        margin-bottom: 15px;
    }

    .grid {
        display: grid;
        grid-template-columns: repeat(6,40px);
        gap: 4px;
    }

    .cell {
        width: 40px;
        height: 40px;
        background: white;
        border-radius: 7px;
        box-shadow: 0 2px 5px rgba(0,0,0,.1);
    }

    .path {
        background: #66bb6a;
    }

    .start {
        background: #42a5f5;
    }

    .finish {
        background: #ef5350;
    }

    .text {
        margin-top: 15px;
        font-size: 20px;
        font-weight: bold;
    }

    </style>

    </head>

    <body>

    <div class="container">

        <div class="title">
            🧩 SYNAPTIQ Memory Maze
        </div>

        <div class="grid">
    """

    for i in range(36):

        if i == path[0]:

            cls = "cell start"

        elif i == path[-1]:

            cls = "cell finish"

        elif i in path:

            cls = "cell path"

        else:

            cls = "cell"

        html += f'<div class="{cls}"></div>'

    html += """
        </div>

        <div class="text">
            Memorize the hidden pathway
        </div>

    </div>

    </body>

    </html>
    """

    components.html(html, height=350)


# ============================================================
# GAME 1 - QUICK TAP
# ============================================================

def synaptiq_game_quick_tap():

    st.markdown(
        """
        <div class="syn-cg-card">

            <div class="syn-cg-game-title">
                ⚡ Quick Tap
            </div>

            <div class="syn-cg-description">
                Wait for the red target to appear.
                Then tap as quickly as possible.
            </div>

            <div class="syn-cg-domain">
                🧠 Reaction Speed
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Wait for the target. Don't press anything until it appears."
    )

    if "syn_qt_active" not in st.session_state:
        st.session_state.syn_qt_active = False

    if "syn_qt_target" not in st.session_state:
        st.session_state.syn_qt_target = None

    if "syn_qt_result" not in st.session_state:
        st.session_state.syn_qt_result = None

    if not st.session_state.syn_qt_active:

        if st.button(
            "🚀 START QUICK TAP",
            use_container_width=True,
            key="syn_qt_start"
        ):

            delay = random.uniform(1.0, 2.5)

            st.session_state.syn_qt_target = (
                time.perf_counter() + delay
            )

            st.session_state.syn_qt_active = True

            st.session_state.syn_qt_result = None

            st.rerun()

    else:

        now = time.perf_counter()

        if now < st.session_state.syn_qt_target:

            remaining = round(
                st.session_state.syn_qt_target - now,
                1
            )

            st.markdown(
                f"""
                <div class="syn-cg-instruction">
                    👀 GET READY...<br><br>
                    {remaining} seconds
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "⏳ WAIT",
                use_container_width=True,
                key="syn_qt_wait"
            ):

                st.rerun()

        else:

            st.markdown(
                """
                <div class="syn-cg-instruction">
                    🔴 TAP NOW!
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "🔴 TAP TARGET",
                use_container_width=True,
                key="syn_qt_target"
            ):

                reaction = (
                    time.perf_counter()
                    - st.session_state.syn_qt_target
                )

                reaction_ms = round(
                    reaction * 1000
                )

                st.session_state.syn_qt_result = reaction_ms

                st.session_state.syn_qt_active = False

                synaptiq_save_game(
                    "Quick Tap",
                    reaction_ms,
                    1,
                    100
                )

                st.rerun()

    if st.session_state.syn_qt_result is not None:

        st.markdown(
            f"""
            <div class="syn-cg-result">

                <div class="syn-cg-result-number">
                    ⚡ {st.session_state.syn_qt_result} ms
                </div>

                <div class="syn-cg-result-label">
                    Your Response Time
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "🔄 PLAY AGAIN",
            use_container_width=True,
            key="syn_qt_again"
        ):

            st.session_state.syn_qt_result = None
            st.session_state.syn_qt_active = False

            st.rerun()


# ============================================================
# GAME 2 - COLOR CHECK
# ============================================================

def synaptiq_game_color_check():

    st.markdown(
        """
        <div class="syn-cg-card">

            <div class="syn-cg-game-title">
                🎨 Color Check
            </div>

            <div class="syn-cg-description">
                Look at each card and decide whether it is red.
            </div>

            <div class="syn-cg-domain">
                👁️ Attention
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if "syn_cc_active" not in st.session_state:
        st.session_state.syn_cc_active = False

    if "syn_cc_trial" not in st.session_state:
        st.session_state.syn_cc_trial = 0

    if "syn_cc_score" not in st.session_state:
        st.session_state.syn_cc_score = 0

    if "syn_cc_red" not in st.session_state:
        st.session_state.syn_cc_red = True

    if not st.session_state.syn_cc_active:

        st.info(
            "10 cards will appear. Choose YES if the card is red."
        )

        if st.button(
            "🎨 START COLOR CHECK",
            use_container_width=True,
            key="syn_cc_start"
        ):

            st.session_state.syn_cc_active = True
            st.session_state.syn_cc_trial = 0
            st.session_state.syn_cc_score = 0

            st.session_state.syn_cc_red = random.choice(
                [True, False]
            )

            st.rerun()

    else:

        if st.session_state.syn_cc_trial < 10:

            red = st.session_state.syn_cc_red

            bg = "#ef5350" if red else "#424242"

            st.markdown(
                f"""
                <div
                    class="syn-cg-big-card"
                    style="
                    background:{bg};
                    color:white;
                    "
                >
                    🃏
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="syn-cg-instruction">

                    Is the card RED?

                    <br><br>

                    Trial
                    {st.session_state.syn_cc_trial + 1}
                    / 10

                </div>
                """,
                unsafe_allow_html=True
            )

            yes, no = st.columns(2)

            with yes:

                if st.button(
                    "🔴 YES",
                    use_container_width=True,
                    key=f"syn_cc_yes_{st.session_state.syn_cc_trial}"
                ):

                    if red:
                        st.session_state.syn_cc_score += 1

                    st.session_state.syn_cc_trial += 1

                    st.session_state.syn_cc_red = random.choice(
                        [True, False]
                    )

                    st.rerun()

            with no:

                if st.button(
                    "⚫ NO",
                    use_container_width=True,
                    key=f"syn_cc_no_{st.session_state.syn_cc_trial}"
                ):

                    if not red:
                        st.session_state.syn_cc_score += 1

                    st.session_state.syn_cc_trial += 1

                    st.session_state.syn_cc_red = random.choice(
                        [True, False]
                    )

                    st.rerun()

        else:

            accuracy = (
                st.session_state.syn_cc_score / 10
            ) * 100

            st.markdown(
                f"""
                <div class="syn-cg-result">

                    <div class="syn-cg-result-number">
                        🎉 {st.session_state.syn_cc_score}/10
                    </div>

                    <div class="syn-cg-result-label">
                        Accuracy: {accuracy:.0f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            synaptiq_save_game(
                "Color Check",
                st.session_state.syn_cc_score,
                10,
                accuracy
            )

            if st.button(
                "🔄 PLAY AGAIN",
                use_container_width=True,
                key="syn_cc_again"
            ):

                st.session_state.syn_cc_active = False
                st.session_state.syn_cc_trial = 0
                st.session_state.syn_cc_score = 0

                st.rerun()


# ============================================================
# GAME 3 - CARD MEMORY
# ============================================================

def synaptiq_game_card_memory():

    st.markdown(
        """
        <div class="syn-cg-card">

            <div class="syn-cg-game-title">
                🃏 Card Memory
            </div>

            <div class="syn-cg-description">
                Remember the symbols you have already seen.
                Decide whether the current symbol appeared earlier.
            </div>

            <div class="syn-cg-domain">
                🧠 Visual Memory
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    symbols = [
        "★",
        "♥",
        "●",
        "◆",
        "▲",
        "☀",
        "☘",
        "♦"
    ]

    if "syn_cm_active" not in st.session_state:
        st.session_state.syn_cm_active = False

    if "syn_cm_trial" not in st.session_state:
        st.session_state.syn_cm_trial = 0

    if "syn_cm_score" not in st.session_state:
        st.session_state.syn_cm_score = 0

    if "syn_cm_seen" not in st.session_state:
        st.session_state.syn_cm_seen = []

    if "syn_cm_current" not in st.session_state:
        st.session_state.syn_cm_current = "★"

    if not st.session_state.syn_cm_active:

        st.info(
            "Remember the symbols and answer whether "
            "you have seen the current one before."
        )

        if st.button(
            "🧠 START CARD MEMORY",
            use_container_width=True,
            key="syn_cm_start"
        ):

            st.session_state.syn_cm_active = True
            st.session_state.syn_cm_trial = 0
            st.session_state.syn_cm_score = 0
            st.session_state.syn_cm_seen = []

            st.session_state.syn_cm_current = random.choice(
                symbols
            )

            st.rerun()

    else:

        if st.session_state.syn_cm_trial < 10:

            current = (
                st.session_state.syn_cm_current
            )

            seen = (
                current
                in st.session_state.syn_cm_seen
            )

            st.markdown(
                f"""
                <div
                    class="syn-cg-big-card"
                    style="
                    background:white;
                    color:#3949ab;
                    "
                >
                    {current}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="syn-cg-instruction">

                    Have you seen this card before?

                    <br><br>

                    Trial
                    {st.session_state.syn_cm_trial + 1}
                    / 10

                </div>
                """,
                unsafe_allow_html=True
            )

            yes, no = st.columns(2)

            with yes:

                if st.button(
                    "✅ YES",
                    use_container_width=True,
                    key=f"syn_cm_yes_{st.session_state.syn_cm_trial}"
                ):

                    if seen:
                        st.session_state.syn_cm_score += 1

                    st.session_state.syn_cm_seen.append(
                        current
                    )

                    st.session_state.syn_cm_trial += 1

                    st.session_state.syn_cm_current = random.choice(
                        symbols
                    )

                    st.rerun()

            with no:

                if st.button(
                    "❌ NO",
                    use_container_width=True,
                    key=f"syn_cm_no_{st.session_state.syn_cm_trial}"
                ):

                    if not seen:
                        st.session_state.syn_cm_score += 1

                    st.session_state.syn_cm_seen.append(
                        current
                    )

                    st.session_state.syn_cm_trial += 1

                    st.session_state.syn_cm_current = random.choice(
                        symbols
                    )

                    st.rerun()

        else:

            accuracy = (
                st.session_state.syn_cm_score / 10
            ) * 100

            st.markdown(
                f"""
                <div class="syn-cg-result">

                    <div class="syn-cg-result-number">
                        🧠 {st.session_state.syn_cm_score}/10
                    </div>

                    <div class="syn-cg-result-label">
                        Accuracy: {accuracy:.0f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            synaptiq_save_game(
                "Card Memory",
                st.session_state.syn_cm_score,
                10,
                accuracy
            )

            if st.button(
                "🔄 PLAY AGAIN",
                use_container_width=True,
                key="syn_cm_again"
            ):

                st.session_state.syn_cm_active = False
                st.session_state.syn_cm_trial = 0
                st.session_state.syn_cm_score = 0
                st.session_state.syn_cm_seen = []

                st.rerun()


# ============================================================
# GAME 4 - PREVIOUS CARD
# ============================================================

def synaptiq_game_previous_card():

    st.markdown(
        """
        <div class="syn-cg-card">

            <div class="syn-cg-game-title">
                🔄 Previous Card
            </div>

            <div class="syn-cg-description">
                Compare the current card with the previous card.
                Decide whether they are the same.
            </div>

            <div class="syn-cg-domain">
                🧠 Working Memory
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    cards = [
        "♥",
        "♦",
        "♣",
        "♠"
    ]

    if "syn_pc_active" not in st.session_state:
        st.session_state.syn_pc_active = False

    if "syn_pc_trial" not in st.session_state:
        st.session_state.syn_pc_trial = 0

    if "syn_pc_score" not in st.session_state:
        st.session_state.syn_pc_score = 0

    if "syn_pc_previous" not in st.session_state:
        st.session_state.syn_pc_previous = "♥"

    if "syn_pc_current" not in st.session_state:
        st.session_state.syn_pc_current = "♥"

    if not st.session_state.syn_pc_active:

        st.info(
            "Compare the previous card with the current card."
        )

        if st.button(
            "🔄 START PREVIOUS CARD",
            use_container_width=True,
            key="syn_pc_start"
        ):

            st.session_state.syn_pc_active = True
            st.session_state.syn_pc_trial = 0
            st.session_state.syn_pc_score = 0

            st.session_state.syn_pc_previous = random.choice(
                cards
            )

            st.session_state.syn_pc_current = random.choice(
                cards
            )

            st.rerun()

    else:

        if st.session_state.syn_pc_trial < 10:

            previous = (
                st.session_state.syn_pc_previous
            )

            current = (
                st.session_state.syn_pc_current
            )

            same = previous == current

            left, right = st.columns(2)

            with left:

                st.markdown(
                    f"""
                    <div
                        class="syn-cg-big-card"
                        style="
                        width:180px;
                        height:220px;
                        background:#e3f2fd;
                        color:#1565c0;
                        "
                    >
                        {previous}
                    </div>

                    <div class="syn-cg-instruction">
                        Previous
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with right:

                st.markdown(
                    f"""
                    <div
                        class="syn-cg-big-card"
                        style="
                        width:180px;
                        height:220px;
                        background:#fff3e0;
                        color:#ef6c00;
                        "
                    >
                        {current}
                    </div>

                    <div class="syn-cg-instruction">
                        Current
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                f"""
                <div class="syn-cg-instruction">

                    Are they the same?

                    <br><br>

                    Trial
                    {st.session_state.syn_pc_trial + 1}
                    / 10

                </div>
                """,
                unsafe_allow_html=True
            )

            yes, no = st.columns(2)

            with yes:

                if st.button(
                    "✅ YES",
                    use_container_width=True,
                    key=f"syn_pc_yes_{st.session_state.syn_pc_trial}"
                ):

                    if same:
                        st.session_state.syn_pc_score += 1

                    st.session_state.syn_pc_previous = current

                    st.session_state.syn_pc_current = random.choice(
                        cards
                    )

                    st.session_state.syn_pc_trial += 1

                    st.rerun()

            with no:

                if st.button(
                    "❌ NO",
                    use_container_width=True,
                    key=f"syn_pc_no_{st.session_state.syn_pc_trial}"
                ):

                    if not same:
                        st.session_state.syn_pc_score += 1

                    st.session_state.syn_pc_previous = current

                    st.session_state.syn_pc_current = random.choice(
                        cards
                    )

                    st.session_state.syn_pc_trial += 1

                    st.rerun()

        else:

            accuracy = (
                st.session_state.syn_pc_score / 10
            ) * 100

            st.markdown(
                f"""
                <div class="syn-cg-result">

                    <div class="syn-cg-result-number">
                        🔄 {st.session_state.syn_pc_score}/10
                    </div>

                    <div class="syn-cg-result-label">
                        Accuracy: {accuracy:.0f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            synaptiq_save_game(
                "Previous Card",
                st.session_state.syn_pc_score,
                10,
                accuracy
            )

            if st.button(
                "🔄 PLAY AGAIN",
                use_container_width=True,
                key="syn_pc_again"
            ):

                st.session_state.syn_pc_active = False
                st.session_state.syn_pc_trial = 0
                st.session_state.syn_pc_score = 0

                st.rerun()


# ============================================================
# CREATE MEMORY MAZE
# ============================================================

def synaptiq_create_maze():

    size = 6

    path = [(0, 0)]

    current = (0, 0)

    for _ in range(14):

        r, c = current

        possible = []

        directions = [
            (r + 1, c),
            (r - 1, c),
            (r, c + 1),
            (r, c - 1)
        ]

        random.shuffle(directions)

        for nr, nc in directions:

            if (
                0 <= nr < size
                and 0 <= nc < size
                and (nr, nc) not in path
            ):

                possible.append(
                    (nr, nc)
                )

        if not possible:
            break

        current = random.choice(
            possible
        )

        path.append(current)

    return path


# ============================================================
# GAME 5 - MEMORY MAZE
# ============================================================

def synaptiq_game_memory_maze():

    st.markdown(
        """
        <div class="syn-cg-card">

            <div class="syn-cg-game-title">
                🧩 Memory Maze
            </div>

            <div class="syn-cg-description">
                Memorize the hidden pathway and reproduce it
                by clicking the tiles in the correct order.
            </div>

            <div class="syn-cg-domain">
                🧠 Executive Function & Spatial Memory
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    size = 6

    if "syn_mz_active" not in st.session_state:
        st.session_state.syn_mz_active = False

    if "syn_mz_show" not in st.session_state:
        st.session_state.syn_mz_show = False

    if "syn_mz_path" not in st.session_state:
        st.session_state.syn_mz_path = []

    if "syn_mz_user" not in st.session_state:
        st.session_state.syn_mz_user = []

    if "syn_mz_errors" not in st.session_state:
        st.session_state.syn_mz_errors = 0

    if not st.session_state.syn_mz_active:

        st.info(
            "First memorize the green pathway. "
            "Then reproduce it after hiding the pathway."
        )

        if st.button(
            "🧩 START MEMORY MAZE",
            use_container_width=True,
            key="syn_mz_start"
        ):

            st.session_state.syn_mz_path = (
                synaptiq_create_maze()
            )

            st.session_state.syn_mz_user = []

            st.session_state.syn_mz_errors = 0

            st.session_state.syn_mz_active = True

            st.session_state.syn_mz_show = True

            st.rerun()

    else:

        path = st.session_state.syn_mz_path

        # ----------------------------------------------------
        # SHOW PATH
        # ----------------------------------------------------

        if st.session_state.syn_mz_show:

            st.markdown(
                """
                <div class="syn-cg-instruction">
                    👀 MEMORIZE THE PATH
                </div>
                """,
                unsafe_allow_html=True
            )

            for r in range(size):

                cols = st.columns(size)

                for c in range(size):

                    with cols[c]:

                        cell = (r, c)

                        if cell in path:

                            if cell == path[0]:

                                bg = "#42a5f5"
                                symbol = "▶"

                            elif cell == path[-1]:

                                bg = "#ef5350"
                                symbol = "★"

                            else:

                                bg = "#66bb6a"
                                symbol = "●"

                            st.markdown(
                                f"""
                                <div
                                    class="syn-cg-maze-cell"
                                    style="
                                    background:{bg};
                                    color:white;
                                    "
                                >
                                    {symbol}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        else:

                            st.markdown(
                                """
                                <div
                                    class="syn-cg-maze-cell"
                                    style="
                                    background:#eeeeee;
                                    "
                                >
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

            if st.button(
                "🙈 HIDE PATH & PLAY",
                use_container_width=True,
                key="syn_mz_hide"
            ):

                st.session_state.syn_mz_show = False

                st.rerun()

        # ----------------------------------------------------
        # PLAY
        # ----------------------------------------------------

        else:

            st.markdown(
                """
                <div class="syn-cg-instruction">
                    🧠 RECREATE THE PATH
                </div>
                """,
                unsafe_allow_html=True
            )

            progress = (
                len(st.session_state.syn_mz_user)
                /
                len(path)
            )

            st.progress(
                min(progress, 1.0)
            )

            st.caption(
                f"Correct steps: "
                f"{len(st.session_state.syn_mz_user)} / "
                f"{len(path)}"
                f"     |     "
                f"Mistakes: "
                f"{st.session_state.syn_mz_errors}"
            )

            for r in range(size):

                cols = st.columns(size)

                for c in range(size):

                    with cols[c]:

                        cell = (r, c)

                        if cell in st.session_state.syn_mz_user:

                            st.markdown(
                                """
                                <div
                                    class="syn-cg-maze-cell"
                                    style="
                                    background:#42a5f5;
                                    color:white;
                                    "
                                >
                                    ✓
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        else:

                            if st.button(
                                " ",
                                key=f"syn_mz_{r}_{c}"
                            ):

                                expected_index = len(
                                    st.session_state.syn_mz_user
                                )

                                expected = path[
                                    expected_index
                                ]

                                if cell == expected:

                                    st.session_state.syn_mz_user.append(
                                        cell
                                    )

                                    st.rerun()

                                else:

                                    st.session_state.syn_mz_errors += 1

                                    st.error(
                                        "❌ Wrong tile. "
                                        "Try to remember the path."
                                    )

            # ------------------------------------------------
            # COMPLETE
            # ------------------------------------------------

            if (
                len(st.session_state.syn_mz_user)
                ==
                len(path)
            ):

                errors = (
                    st.session_state.syn_mz_errors
                )

                accuracy = max(
                    0,
                    100 - errors * 5
                )

                st.markdown(
                    f"""
                    <div class="syn-cg-result">

                        <div class="syn-cg-result-number">
                            🎉 MAZE COMPLETE!
                        </div>

                        <div class="syn-cg-result-label">

                            Mistakes: {errors}

                            <br>

                            Performance: {accuracy}%

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                synaptiq_save_game(
                    "Memory Maze",
                    accuracy,
                    len(path),
                    accuracy
                )

                if st.button(
                    "🔄 PLAY AGAIN",
                    use_container_width=True,
                    key="syn_mz_again"
                ):

                    st.session_state.syn_mz_active = False
                    st.session_state.syn_mz_show = False
                    st.session_state.syn_mz_path = []
                    st.session_state.syn_mz_user = []
                    st.session_state.syn_mz_errors = 0

                    st.rerun()


# ============================================================
# MAIN COGNITIVE GAMES PAGE
# ============================================================

def render_cognitive_games():

    # Apply styling
    synaptiq_cognitive_css()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="syn-cg-title">
            🧠 Cognitive Games
        </div>

        <div class="syn-cg-subtitle">
            Play • Think • Remember • Have Fun!
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # WELCOME
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="syn-cg-welcome">

            <h2>
                🌟 Welcome to SYNAPTIQ Brain Play!
            </h2>

            <p>
                Explore fun and simple activities that exercise
                reaction speed, attention, memory and thinking skills.
            </p>

            <p>
                🎥 Watch the demonstration first,
                then try the game yourself!
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # SELECT GAME
    # --------------------------------------------------------

    selected = st.selectbox(
        "🎮 Choose a Cognitive Game",
        [
            "⚡ Quick Tap",
            "🎨 Color Check",
            "🃏 Card Memory",
            "🔄 Previous Card",
            "🧩 Memory Maze"
        ],
        key="synaptiq_cognitive_game_selector"
    )


    st.divider()


    # ========================================================
    # QUICK TAP
    # ========================================================

    if selected == "⚡ Quick Tap":

        st.markdown(
            "## ⚡ Quick Tap"
        )

        st.caption(
            "Reaction Speed"
        )

        with st.expander(
            "🎥 WATCH DEMONSTRATION",
            expanded=True
        ):

            synaptiq_demo_quick_tap()

        st.markdown(
            "### 🎮 PLAY"
        )

        synaptiq_game_quick_tap()


    # ========================================================
    # COLOR CHECK
    # ========================================================

    elif selected == "🎨 Color Check":

        st.markdown(
            "## 🎨 Color Check"
        )

        st.caption(
            "Attention"
        )

        with st.expander(
            "🎥 WATCH DEMONSTRATION",
            expanded=True
        ):

            synaptiq_demo_color_check()

        st.markdown(
            "### 🎮 PLAY"
        )

        synaptiq_game_color_check()


    # ========================================================
    # CARD MEMORY
    # ========================================================

    elif selected == "🃏 Card Memory":

        st.markdown(
            "## 🃏 Card Memory"
        )

        st.caption(
            "Visual Memory"
        )

        with st.expander(
            "🎥 WATCH DEMONSTRATION",
            expanded=True
        ):

            synaptiq_demo_card_memory()

        st.markdown(
            "### 🎮 PLAY"
        )

        synaptiq_game_card_memory()


    # ========================================================
    # PREVIOUS CARD
    # ========================================================

    elif selected == "🔄 Previous Card":

        st.markdown(
            "## 🔄 Previous Card"
        )

        st.caption(
            "Working Memory"
        )

        with st.expander(
            "🎥 WATCH DEMONSTRATION",
            expanded=True
        ):

            synaptiq_demo_previous_card()

        st.markdown(
            "### 🎮 PLAY"
        )

        synaptiq_game_previous_card()


    # ========================================================
    # MEMORY MAZE
    # ========================================================

    elif selected == "🧩 Memory Maze":

        st.markdown(
            "## 🧩 Memory Maze"
        )

        st.caption(
            "Executive Function & Spatial Memory"
        )

        with st.expander(
            "🎥 WATCH DEMONSTRATION",
            expanded=True
        ):

            synaptiq_demo_memory_maze()

        st.markdown(
            "### 🎮 PLAY"
        )

        synaptiq_game_memory_maze()


    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()

    st.markdown(
        """
        <div class="syn-cg-card">

            <h3>💜 About SYNAPTIQ Cognitive Games</h3>

            <p style="font-size:16px;line-height:1.7;">

            These interactive activities are designed for
            cognitive exercise and monitoring. They explore
            areas such as reaction speed, attention,
            visual memory, working memory and executive skills.

            </p>

            <p style="font-size:15px;">

            ⚠️ These activities are not a substitute for a
            clinical cognitive assessment or medical diagnosis.

            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# END OF COGNITIVE GAMES MODULE
# ============================================================

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
        <span class="badge badge-purple">❖ OASIS Dataset</span>
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

        st.markdown('<div class="section-header">CI2018 Units Covered</div>', unsafe_allow_html=True)
        for unit, desc in [
            ("Unit I",  "ML Basics, Data Types, Train/Test Split"),
            ("Unit II", "Normalization, K-Fold CV, PCA, LDA"),
            ("Unit III","Confusion Matrix, ROC/AUC, Precision, Recall, F1"),
            ("Unit IV", "Gradient Descent, Bias-Variance, Regularization"),
            ("Unit V",  "SVM (3 kernels), Random Forest, Feature Importance"),
        ]:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:0.45rem 0.8rem;background:{t['card']};border-radius:8px;
                        margin-bottom:0.3rem;border:1px solid {t['border2']};'>
                <span style='color:{t['blue']};font-weight:600;font-size:0.82rem;'>{unit}</span>
                <span style='color:{t['txt2']};font-size:0.77rem;'>{desc}</span>
            </div>""", unsafe_allow_html=True)
    
# ══════════════════════════════════════════════════════════════
# PAGE 2 — PATIENT DIAGNOSIS
# ══════════════════════════════════════════════════════════════
elif page == "📝 Cognitive Questionnaire":
    st.markdown("## 📝 Cognitive Assessment Questionnaire")
    friendly_card(
        "MMSE / CDR assessment assistant",
        "This page provides an original cognitive screening exercise and a caregiver functional questionnaire. "
        "It does NOT reproduce the licensed MMSE or CDR forms. Use licensed official forms for a clinical MMSE/CDR result.",
        "🧠"
    )
    st.info(
        "Important: the official MMSE is a copyrighted 30-point instrument, and CDR is a licensed "
        "clinician/informant-based staging instrument. A custom questionnaire cannot honestly be called an official MMSE or CDR. "
        "The results below are therefore labelled screening/provisional results."
    )

    st.markdown("### 1️⃣ Original 30-point cognitive screening")
    st.caption("These questions are original screening tasks organized around cognitive domains commonly assessed in brief cognitive screening. They are not official MMSE items.")

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
        save_cognitive_record("screening", mmse=float(st.session_state.screening_mmse_score))
        st.success(f"🧠 Original cognitive screening score: **{screening_score}/30**")
        st.caption("This is a screening score, not an official MMSE score. Do not use it as a diagnosis.")
        st.write(f"Orientation: {orientation}/10 · Immediate learning: {registration}/3 · Attention: {attention_score}/5 · Recall: {recall_score}/3 · Language/comprehension: {language_score}/9")

    st.markdown("---")
    st.markdown("### 2️⃣ Caregiver functional questionnaire for a provisional CDR-style profile")
    st.caption("Answer based on change from the person's usual functioning. These are original questions covering the six CDR domains; they are not the licensed CDR worksheet.")

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
        save_cognitive_record("screening", mmse=st.session_state.screening_mmse_score,
                             cdr=float(provisional))
        st.success(f"📊 Provisional CDR-style rating: **{provisional:g}**")
        st.write("Domain profile:", domain_scores)
        st.caption("This median-based estimate is NOT the official CDR scoring algorithm. Official CDR scoring requires the licensed protocol and trained clinical judgment.")

    st.markdown("---")
    st.markdown("### 3️⃣ Licensed official scores")
    st.write("If a clinician has administered the official instruments, enter the official scores here. These values can then be used by the Patient Diagnosis tab.")
    oc1, oc2 = st.columns(2)
    with oc1:
        official_mmse = st.number_input("Official MMSE score (0–30)", 0, 30, value=int(st.session_state.official_mmse_score or 0), key="official_mmse_entry")
        if st.button("Use official MMSE score", use_container_width=True):
            st.session_state.official_mmse_score = int(official_mmse)
            st.success("Official MMSE score saved for Patient Diagnosis.")
    with oc2:
        cdr_options = [0.0, 0.5, 1.0, 2.0, 3.0]
        saved_cdr = st.session_state.official_cdr_score
        cdr_index = 0 if saved_cdr is None else cdr_options.index(float(saved_cdr))
        official_cdr = st.selectbox("Official CDR global rating", cdr_options, index=cdr_index, key="official_cdr_entry")
        if st.button("Use official CDR rating", use_container_width=True):
            st.session_state.official_cdr_score = float(official_cdr)
            st.success("Official CDR rating saved for Patient Diagnosis.")

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
            save_cognitive_record("diagnosis", mmse=float(mmse), cdr=float(cdr),
                                 confidence=float(conf), prediction=str(pred_cls))
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
# PAGE — GPS SAFETY
# ══════════════════════════════════════════════════════════════
elif page == "📍 GPS Safety & Geofencing":
    render_gps_safety()

# ══════════════════════════════════════════════════════════════
# PAGE — COGNITIVE PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "🧠 Cognitive Performance":
    render_cognitive_analytics()

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
        st.image(q_mem["image"], width=240) if q_mem.get("image") else None
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

    st.markdown("---")
    render_caregiver_location_monitor()

# ══════════════════════════════════════════════════════════════
# PAGE — SMART REMINDERS
# ══════════════════════════════════════════════════════════════
elif page == "⏰ Smart Reminders":
    st.markdown("## ⏰ Smart Reminder System")
    friendly_card("Date & time pop-up reminders", "Set the exact date and time. When that time arrives, Synaptiq shows a large on-screen pop-up, speaks the reminder, and can also show a browser notification when permission is enabled.", "🔔")

    with st.form("reminder_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            reminder_text = st.text_input("What should we remember?", placeholder="e.g. Take medicine")
            reminder_type = st.selectbox("Type", ["Medicine", "Meal", "Water", "Doctor appointment", "Family call", "Daily activity", "Sleep", "Other"])
            reminder_date = st.date_input("Date", value=date.today(), min_value=date.today())
        with c2:
            reminder_time = st.time_input("Time", value=dt_time(9, 0))
            reminder_note = st.text_input("Short note", placeholder="e.g. After breakfast")
        if st.form_submit_button("➕ Set Reminder", use_container_width=True):
            if reminder_text:
                st.session_state.reminders.append({
                    "id": f"rem-{datetime.now().timestamp()}",
                    "date": reminder_date.isoformat(),
                    "time": reminder_time.strftime("%H:%M"),
                    "type": reminder_type,
                    "text": reminder_text,
                    "note": reminder_note
                })
                st.success(f"✅ Reminder set for {reminder_date.strftime('%d %B %Y')} at {reminder_time.strftime('%I:%M %p')}.")
            else:
                st.warning("Please enter what you want to remember.")

    browser_reminder_scheduler(st.session_state.reminders, APP_LANG_CODE)

    now = datetime.now()
    st.markdown(f"### 🕒 Today — {now.strftime('%A, %d %B %Y')}")
    if not st.session_state.reminders:
        st.info("No reminders yet. Add one above.")
    for i, rem in enumerate(sorted(st.session_state.reminders, key=lambda x: (x.get("date", ""), x.get("time", "")))):
        c1, c2, c3 = st.columns([1.5, 3, 1])
        display_date = datetime.strptime(rem.get("date", date.today().isoformat()), "%Y-%m-%d").strftime("%d %b %Y")
        c1.markdown(f"## ⏰ {rem['time']}\n**{display_date}**")
        c2.markdown(f"**{rem['type']} — {rem['text']}**  \n{rem.get('note', '')}")
        if c3.button("🔊 Speak", key=f"speak_rem_{i}"):
            browser_speak(f"Reminder for {display_date} at {rem['time']}. {rem['text']}. {rem.get('note', '')}")

    st.markdown("### 🗓️ Daily Routine")
    for rt, task in st.session_state.daily_routine:
        friendly_card(rt, task, "🌞")

# ══════════════════════════════════════════════════════════════
# PAGE — NER MEMORY & CULTURE
# ══════════════════════════════════════════════════════════════
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
# PAGE 3 — MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# PAGE 4 — VOICE ASSISTANT
# ══════════════════════════════════════════════════════════════

elif page == "🎤 Voice Assistant":

    import tempfile

    # =======================
    # HEADER
    # =======================
    st.markdown("""
    <div style="
        padding:22px;
        border-radius:18px;
        background:linear-gradient(135deg,#0f172a,#111827);
        border:1px solid rgba(255,255,255,0.08);
        margin-bottom:20px;
    ">
        <h2 style="color:white;margin:0;">🎤 Synaptiq Voice Assistant</h2>
        <p style="color:#94a3b8;margin-top:8px;">
            Speak naturally — AI will transcribe and analyze your input in real time
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =======================
    # RECORDING
    # =======================
    st.markdown("### 🎙️ Voice Input")

    audio = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=True
    )

    status = st.empty()

    # =======================
    # PROCESS AUDIO
    # =======================
    if audio:

        status.info("⏳ Processing voice input...")

        # Save audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio["bytes"])
            temp_path = f.name

        # =======================
        # TRANSCRIPTION
        # =======================
        with open(temp_path, "rb") as file:

            transcription = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3"
            )

        voice_text = transcription.text

        status.success("✅ Speech successfully recognized")

        # =======================
        # TRANSCRIPT BOX (HIGH CONTRAST)
        # =======================
        st.markdown("### 📝 Recognized Speech")

        st.markdown(f"""
        <div style="
            padding:18px;
            border-radius:16px;
            background:#0b1220;
            border:1px solid rgba(255,255,255,0.15);
            color:#e5e7eb;
            font-size:1.05rem;
            line-height:1.7;
        ">
            {voice_text}
        </div>
        """, unsafe_allow_html=True)

        # =======================
        # AI PROCESSING
        # =======================
        with st.spinner("🧠 Synaptiq is analyzing your query..."):

            prompt = f"""
You are Synaptiq, an AI healthcare assistant specialized in Alzheimer's disease.

Respond clearly, professionally, and in simple medical language.
Respond in {APP_LANGUAGE}.

{voice_text}
"""

            if client is None:
                st.error("Groq API key is not configured. Set GROQ_API_KEY in your environment.")
                st.stop()
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-20b",
            )

            ai_response = chat_completion.choices[0].message.content

        # =======================
        # AI RESPONSE BOX (CLEAR + STRONG CONTRAST)
        # =======================
        st.markdown("### 🧠 AI Response")

        if st.button("🔊 Read AI response aloud", use_container_width=True):
            browser_speak(ai_response, APP_LANG_CODE)

        st.markdown(f"""
        <div style="
            padding:20px;
            border-radius:18px;
            background:linear-gradient(135deg,#0f172a,#111827);
            border:1px solid rgba(255,255,255,0.12);
            color:#f1f5f9;
            font-size:1.05rem;
            line-height:1.8;
            box-shadow:0 0 25px rgba(0,0,0,0.35);
        ">
            {ai_response}
        </div>
        """, unsafe_allow_html=True)

    else:

        st.info("🎧 Click Start Recording to begin interacting with Synaptiq AI Assistant")
#==============game=============
# ══════════════════════════════════════════════════════════════
# PAGE — AI COGNITIVE GAMES
# ══════════════════════════════════════════════════════════════


# ============================================================
# COGNITIVE GAMES PAGE ROUTE
# ============================================================
elif page == "🧩 Cognitive Games":
    render_cognitive_games()

# PAGE 4 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
elif page == "📈  Visualizations":

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{t['hero1']},{t['hero2']});
                border:1px solid {t['border']};border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;'>
        <div style='font-size:1.6rem;font-weight:700;color:{t['txt']};'>📈 Visualizations</div>
        <div style='color:{t['txt2']};margin-top:0.3rem;'>Feature importance, PCA, LDA and dataset distribution analysis.</div>
    </div>
    """, unsafe_allow_html=True)

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

    st.plotly_chart(fig_mmse, use_container_width=True, key="mmse_chart")