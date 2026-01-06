import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import re
import datetime
import joblib
import json
import google.generativeai as genai
import streamlit as st

def redact_pii(text):
    """Sanitizes input to remove potential Patient Identifiers."""
    text = re.sub(r'\b(Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+', '', text)
    text = re.sub(r'\b\d{6,}\b', '', text)
    text = re.sub(r'\d{2}/\d{2}/\d{4}', '', text)
    return text

# Load Interaction DB
try:
    from drug_data import INTERACTION_DB
except ImportError:
    INTERACTION_DB = {}

# --- DATABASE ---
def get_db_connection():
    return sqlite3.connect('clinical_data.db', check_same_thread=False)

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patient_history (
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            age INTEGER, gender TEXT, sbp INTEGER,
            aki_risk_score INTEGER, bleeding_risk_score REAL, status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_patient_to_db(age, gender, sbp, aki, bleed, status):
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO patient_history (age, gender, sbp, aki_risk_score, bleeding_risk_score, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (age, gender, sbp, aki, bleed, status))
        conn.commit()

# --- AI PARSER ---
def parse_clinical_note(text):
    """Uses Gemini to extract clinical values from unstructured text."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        Extract clinical values from this medical note: "{text}"
        Return ONLY a raw JSON object (no markdown, no backticks) with these exact keys: 
        "age", "gender", "weight_input", "sys_bp", "dia_bp", "hr", "creat", "anticoagulant".
        
        Rules:
        - Use 0 for missing numbers.
        - Gender must be "Male" or "Female".
        - anticoagulant must be true or false.
        """
        response = model.generate_content(prompt)
        clean_json = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json)
    except Exception as e:
        return {}

# --- CLINICAL CALCULATORS ---
def calculate_aki_risk(age, diuretic, acei, sys_bp, chemo, creat, nsaid, heart_failure):
    score = 0
    if diuretic: score += 30
    if acei: score += 40
    if nsaid: score += 25
    if age > 75: score += 20
    if sys_bp < 90 and sys_bp > 0: score += 20
    if creat > 1.5: score += 30
    return min(score, 100)

def calculate_sepsis_risk(sys_bp, resp_rate, altered_mental, temp_c):
    qsofa = 0
    if 0 < sys_bp <= 100: qsofa += 1
    if resp_rate >= 22: qsofa += 1
    if altered_mental: qsofa += 1
    return 90 if qsofa >= 2 else (45 if qsofa >= 1 else 0)

# --- AI CONSULTANTS ---
def consult_ai_doctor(role, user_input, patient_context=None):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.0-flash')
        safe_input = redact_pii(user_input)
        prompt = f"Expert Medical Consult ({role}). Patient context: {patient_context}. Query: {safe_input}"
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Error: {str(e)}"
