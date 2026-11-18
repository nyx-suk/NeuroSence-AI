# SYNAPTIQ — Memory Assistance Platform

> **Notice:** Research prototype. Not a diagnostic tool.

SYNAPTIQ is a Streamlit prototype for cognitive screening, memory support, and caregiver safety. It combines interactive cognitive activities and assessment tools with multilingual AI assistance. It is intended for demonstration and supportive use; it is not a diagnostic system or a substitute for assessment by a qualified clinician.

## What’s in the app

- **Cognitive dashboard and assessments** for recording cognitive screening results and viewing history and analytics.
- **Cognitive games** such as card memory, working memory, and memory maze activities.
- **Multilingual AI chatbot** powered by Groq, with language options including Northeast Indian languages.
- **Voice assistant** for spoken interaction, subject to optional audio dependencies and browser support.
- **Caregiver portal and memory support** for trusted contacts and personal memory prompts.
- **Safety features** including SOS calling and GPS safe-zone/geofencing tools. Location access depends on browser permissions and device support.
- **ML research pipeline** in `run_ml_pipeline.py`, separate from the Streamlit app. It trains and evaluates classifiers using OASIS data and writes visualizations to `outputs/`.

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- `ffmpeg` (optional, for audio format conversion)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/nyx-suk/NeuroSense-AI.git
cd NeuroSense-AI

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
Install all pinned dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and set your `GROQ_API_KEY`:
```env
GROQ_API_KEY=your_groq_api_key_here
```
*(Get a free key at [console.groq.com](https://console.groq.com/keys). If omitted, the app starts with offline core features like games and assessments; only the chatbot and voice Q&A will be disabled).*

### 5. Run the Application
Launch the Streamlit app:
```bash
streamlit run ourapp.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

## Demo accounts

The app currently defines these demo credentials in code:

| Role | Username | Password |
|---|---|---|
| Doctor | `doctor` | `1234` |
| Admin | `admin` | `admin123` |
| Patient | `patient` | `test` |
| Caregiver | `caregiver` | `care123` |

These are demonstration credentials, not production authentication. Do not deploy the app with them or use it to store real patient information without replacing the authentication and data-protection approach.

## ML pipeline

`run_ml_pipeline.py` runs a separate supervised-learning workflow: it loads OASIS longitudinal data, preprocesses features, trains Random Forest, SVM, and voting-ensemble models, evaluates them, and saves plots under `outputs/`.

Install the additional packages used by the pipeline:

```bash
python -m pip install scikit-learn imbalanced-learn matplotlib seaborn
```

Then run:

```bash
python run_ml_pipeline.py
```

The pipeline may download OASIS data if `data/oasis_longitudinal.csv` is missing. If that download fails, its loader can fall back to generated synthetic data. Check the dataset source and actual run output before interpreting metrics. Results from this experimental pipeline are not evidence of clinical performance.

## Repository map

```text
ourapp.py                  Current consolidated Streamlit application
requirements.txt           Pinned environment dependencies
theme.py                   Shared UI theme
ml/                        Data preparation, models, evaluation, explainability
run_ml_pipeline.py         Standalone model-training pipeline
data/                      Dataset directory
cognitive_demo_videos/     Demonstration videos for cognitive activities
neura_voice_assistant/     Voice-assistant interface assets
.env.example               Environment-variable template
```

The app stores safety and cognitive records in local SQLite databases. Treat these files as sensitive if they contain real information, and make sure the runtime environment provides appropriate access controls and backups.

## Limitations

- **Research prototype. Not a diagnostic tool.** All screenings and scores are experimental.
- AI output depends on an external Groq service and should not be treated as medical advice.
- Cognitive activities and scores are supportive prototypes, not validated diagnoses.
- Browser location, speech recognition, and audio conversion depend on the user’s device, browser permissions, and installed packages.
- The demo login and local SQLite storage are not designed for production healthcare deployment.
