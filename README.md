# 🛡️ Clinical Risk Monitor: AI-Powered Pharmacovigilance System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://clinicalriskmonitor-ai.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini%202.0-green)

## 📋 Overview
The **Clinical Risk Monitor** is a real-time clinical decision support system (CDSS) designed to bridge the gap between raw patient data and actionable ICU insights. 

Unlike standard dashboards, this application combines **Deterministic Logic** (Clinical Rules like qSOFA) with **Probabilistic Machine Learning** (XGBoost & Generative AI) to predict patient deterioration before it happens.

### 🚀 Live Demo
**[Click here to launch the application](https://clinicalriskmonitor-ai.streamlit.app)**

---

## ⚙️ Key Features

### 1. 🧠 AI Clinical Consultant (Generative AI)
- Integrated **Google Gemini 2.0 Flash** to act as an automated medical resident.
- **Provider Mode:** Analyzes calculated risk scores (AKI, Bleeding) and generates a differential diagnosis and treatment plan.
- **Patient Mode:** A triage symptom checker that translates layperson terms into medical terminology.

### 2. 🩸 Bleeding Risk Prediction (Machine Learning)
- Deployed a pre-trained **XGBoost Regressor** (`bleeding_risk_model.json`).
- Predicts the probability of hemorrhage based on INR, Anticoagulant use, Age, and Comorbidities.
- Trained on synthetic acute care data.

### 3. ⚡ Real-Time Protocol Monitors (Clinical Logic)
- **Sepsis Watch:** Auto-calculates **qSOFA** scores based on Vitals (BP, Resp Rate, Mental Status).
- **AKI Monitor:** Tracks Creatinine spikes according to **KDIGO** guidelines.
- **Hypoglycemia Alert:** Flags critical glucose levels in diabetic patients.

### 4. 💊 Drug-Drug Interaction Checker
- Features a backend database of **High-Alert Medications** (Warfarin, Amiodarone, NSAIDs, etc.).
- Flags **Critical** (Contraindicated) and **Major** (Monitor closely) interactions instantly to prevent adverse drug events (ADEs).

---

## 🛠️ Technical Architecture

The application follows a modular **Model-View-Controller (MVC)** pattern:

* **Frontend (`app.py`):** Built with **Streamlit**. Handles UI rendering, session state, and input validation.
* **Backend (`backend.py`):** Pure Python logic layer. Handles:
    * **SQL Database:** SQLite integration for storing patient history.
    * **ML Inference:** Loading and querying the XGBoost model.
    * **API Gateway:** Secure connection to Google Gemini via Streamlit Secrets.
* **Data Visualization:** Used **Altair** for interactive vitals telemetry charts.

### 📂 Project Structure
```bash
├── app.py                 # Frontend Interface (Streamlit)
├── backend.py             # Logic & AI Controller (Python)
├── bleeding_risk_model.json # Pre-trained XGBoost Model
├── clinical_data.db       # SQLite Database (Patient History)
├── requirements.txt       # Dependencies
└── README.md              # Documentation

## 💻 How to Run Locally

If you want to run this on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Pravanith/ClinicalRiskApp.git](https://github.com/Pravanith/ClinicalRiskApp.git)
    cd ClinicalRiskApp
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Keys:**
    * Create a `.streamlit/secrets.toml` file.
    * Add your Google Gemini API Key: `GEMINI_API_KEY = "YOUR_KEY_HERE"`

4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

---

## 👨‍⚕️ About the Author
**Pravanith** | *Pharm.D. & Health Data Scientist*

This project demonstrates the intersection of **Clinical Expertise** and **Data Engineering**. It showcases the ability to not only analyze health data but to build deployed, scalable tools that improve patient safety.

* **Capstone Project:** Predicting Mental Health Severity from Digital Habits (R/Stats).
* **Engineering Project:** Clinical Risk Monitor (Python/ML).

---

*Disclaimer: This tool is a prototype for portfolio and educational purposes only. It is not a substitute for professional medical advice.*
