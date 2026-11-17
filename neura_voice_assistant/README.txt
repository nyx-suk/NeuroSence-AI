HEY NEURO — SYNAPTIQ VOICE ASSISTANT
=====================================

This package contains your existing SYNAPTIQ project with the Hey Neuro
hands-free voice assistant added to the existing Streamlit app.

IMPORTANT:
- Your existing project files are preserved.
- Only `ourapp.py` was extended to load the new voice assistant.
- A new folder `neura_voice_assistant/` was added beside `ourapp.py`.
- The wake phrase accepts both "Hey Neuro" and "Hey Neura" variants.

RUN:
1. Extract this ZIP.
2. Open PowerShell in:
   sih\NeuroSense-AI
3. Run:
   python -m streamlit run ourapp.py
4. In the browser, click "Enable Voice" once and allow microphone access.
5. After that say "Hey Neuro" without clicking the microphone.

SUPPORTED COMMANDS:
- Hey Neuro, go to games
- Hey Neuro, open games
- Hey Neuro, go home
- Hey Neuro, show location
- Hey Neuro, where is the patient
- Hey Neuro, show alerts
- Hey Neuro, show medications
- Hey Neuro, show activity
- Hey Neuro, show safe zones
- Hey Neuro, call caregiver
- Hey Neuro, help

CALL SAFETY:
Calling the caregiver is treated as a sensitive action. The assistant asks
for a spoken YES/NO confirmation before using the saved caregiver number.

BROWSER NOTE:
The Web Speech API is browser-dependent. Current Chrome/Edge are recommended.
The page must remain open/active for hands-free wake-word listening; a normal
website cannot guarantee always-on microphone listening after the browser
suspends a tab.

If voice recognition works but a spoken navigation command does not change the
Streamlit page, the likely issue is browser iframe navigation restrictions.
The assistant uses a safe query-parameter bridge for Streamlit navigation.
