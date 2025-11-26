import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sqlite3

# ---------------------------------------------------------
# DATABASE ENGINE (SQL)
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect('clinical_data.db')
    c = conn.cursor()
    # Create table if not exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS patient_history (
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            age INTEGER,
            gender TEXT,
            sbp INTEGER,
            aki_risk_score INTEGER,
            bleeding_risk_score REAL,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_patient_to_db(age, gender, sbp, aki, bleed, status):
    conn = sqlite3.connect('clinical_data.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO patient_history (age, gender, sbp, aki_risk_score, bleeding_risk_score, status)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (age, gender, sbp, aki, bleed, status))
    conn.commit()
    conn.close()

# Initialize DB on app start
init_db()
# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Clinical Risk Monitor", 
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. LOAD AI MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_bleeding_model():
    """Loads the pre-trained XGBoost model."""
    model = xgb.XGBRegressor()
    model_file = "bleeding_risk_model.json"
    
    if not os.path.exists(model_file):
        st.error(f"⚠️ CRITICAL ERROR: '{model_file}' not found.")
        st.info("💡 To fix: Run 'train_model.py' locally first.")
        st.stop()
        
    model.load_model(model_file)
    return model

try:
    bleeding_model = load_bleeding_model()
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. CLINICAL LOGIC ENGINES
# ---------------------------------------------------------

# A. AKI Risk (Kidney) - FIXED to accept 8 arguments
def calculate_aki_risk(age, diuretic, acei, high_bp, chemo, creat, nsaid, heart_failure):
    score = 0
    score += 30 if diuretic else 0
    score += 40 if acei else 0
    score += 25 if nsaid else 0       # Matches Calculator
    score += 20 if age > 75 else 0
    score += 15 if heart_failure else 0 # Matches Calculator
    score += 10 if high_bp else 0
    score += 20 if chemo else 0
    
    if creat > 1.5: score += 30
    elif creat > 1.2: score += 15
    return min(score, 100)
# B. Sepsis Screen (qSOFA) - PASTE THIS SECTION
def calculate_sepsis_risk(sys_bp, resp_rate, altered_mental, temp_c):
    qsofa = 0
    # qSOFA Criteria
    if sys_bp > 0 and sys_bp <= 100: qsofa += 1
    if resp_rate >= 22: qsofa += 1
    if altered_mental: qsofa += 1
    
    # Fever check (SIRS criteria)
    if temp_c > 0 and (temp_c > 38.0 or temp_c < 36.0):
        qsofa += 0.5
    
    if qsofa >= 2: return 90  # High Risk
    if qsofa >= 1: return 45  # Moderate Risk
    return 5                  # Low Risk
# C. Hypoglycemic Risk (Blood Sugar)
def calculate_hypoglycemic_risk(insulin, renal, hba1c_high, neuropathy, recent_dka):
    score = 0
    score += 30 if insulin else 0
    score += 45 if renal else 0
    score += 20 if hba1c_high else 0
    score += 10 if neuropathy else 0
    score += 20 if recent_dka else 0 
    return min(score, 100)

# C. Interaction Database (FULL VERSION)
interaction_db = {
    # --- CRITICAL / FATAL ---
    ("sildenafil", "nitroglycerin"): "CRITICAL: Fatal hypotension. Contraindicated.",
    ("tadalafil", "isosorbide mononitrate"): "CRITICAL: Fatal hypotension. Contraindicated.",
    ("methotrexate", "trimethoprim"): "CRITICAL: Bone marrow toxicity. Avoid.",
    ("sertraline", "linezolid"): "CRITICAL: Fatal Serotonin Syndrome risk.",
    ("fentanyl", "midazolam"): "CRITICAL: Severe respiratory depression.",
    ("spironolactone", "trimethoprim"): "CRITICAL: High risk of sudden death from Hyperkalemia.",

    # --- MAJOR ---
    ("warfarin", "amiodarone"): "MAJOR: Amiodarone increases INR significantly. Reduce Warfarin dose.",
    ("warfarin", "ibuprofen"): "MAJOR: NSAIDs increase bleeding risk and damage gastric mucosa.",
    ("lisinopril", "spironolactone"): "MAJOR: Risk of severe hyperkalemia (high potassium).",
    ("simvastatin", "amiodarone"): "MAJOR: Rhabdomyolysis risk. Simvastatin dose limit 20mg.",
    ("ciprofloxacin", "ondansetron"): "MAJOR: QT Prolongation risk. Monitor ECG.",
    ("levofloxacin", "ondansetron"): "MAJOR: QT Prolongation risk. Monitor ECG.",
    ("citalopram", "fluconazole"): "MAJOR: QT Prolongation risk. Avoid high doses.",
    ("warfarin", "ciprofloxacin"): "MAJOR: Antibiotics kill gut flora, spiking INR.",
    ("warfarin", "fluconazole"): "MAJOR: Fluconazole inhibits CYP2C9, dramatically raising INR.",
    ("clopidogrel", "omeprazole"): "MAJOR: Omeprazole blocks CYP2C19, making Plavix ineffective.",
    ("tramadol", "fluoxetine"): "MAJOR: Risk of Serotonin Syndrome and Seizures.",
    ("lithium", "lisinopril"): "MAJOR: ACE Inhibitors reduce Lithium excretion -> Lithium Toxicity.",
    ("lithium", "ibuprofen"): "MAJOR: NSAIDs reduce Lithium excretion -> Lithium Toxicity.",
    ("digoxin", "clarithromycin"): "MAJOR: Macrolides block P-gp -> Digoxin Toxicity.",
    ("phenytoin", "oral contraceptives"): "MAJOR: Phenytoin induces enzymes, causing contraceptive failure.",
    
    # --- MODERATE ---
    ("apixaban", "ibuprofen"): "MODERATE: NSAIDs increase bleeding risk with Apixaban.",
    ("clopidogrel", "aspirin"): "MODERATE: Dual antiplatelet therapy, increases bleed risk.",
    ("warfarin", "acetaminophen"): "MODERATE: High/Chronic Tylenol use can elevate INR.",
    ("levothyroxine", "calcium"): "MODERATE: Calcium blocks Thyroid absorption. Separate by 4 hours.",
    ("doxycycline", "iron"): "MODERATE: Iron binds to antibiotic, causing treatment failure.",
}

def check_interaction(d1, d2):
    d1, d2 = d1.lower().strip(), d2.lower().strip()
    if (d1, d2) in interaction_db: return interaction_db[(d1, d2)]
    if (d2, d1) in interaction_db: return interaction_db[(d2, d1)]
    return "✅ No high-alert interaction found."

# D. Chatbot Logic (FULL VERSION)
def chatbot_response(text):
    text = text.lower()
    
    # Disease Definitions
    conditions = {
        "sepsis": "Life-threatening response to infection. Watch for: Fever, Hypotension, Tachycardia (qSOFA score).",
        "pneumonia": "Lung infection. Symptoms: Cough with phlegm, fever, chills, difficulty breathing.",
        "heart failure": "CHF: Heart doesn't pump well. Watch for: Edema, Dyspnea, Fatigue.",
        "copd": "Chronic Obstructive Pulmonary Disease. Watch for O2 desaturation.",
        "afib": "Atrial Fibrillation: Irregular, rapid heart rate. Stroke risk. Needs anticoagulation.",
        "stroke": "Medical emergency. BE-FAST: Balance, Eyes, Face, Arms, Speech, Time.",
        "dvt": "Deep Vein Thrombosis: Leg clot. Symptoms: Swelling, pain, warmth.",
        "pe": "Pulmonary Embolism: Lung clot. Symptoms: Sudden SOB, chest pain.",
        "anemia": "Low RBCs. Symptoms: Fatigue, weakness, pale skin.",
        "gout": "Arthritis from uric acid. Avoid Thiazide diuretics.",
        "uti": "Urinary Tract Infection. Symptoms: Urgency, burning, cloudy urine.",
        "gerd": "Acid reflux. Symptom: Heartburn.",
        "migraine": "Severe throbbing headache, nausea, light sensitivity.",
        "osteoporosis": "Weak bones. Fracture risk.",
        "cirrhosis": "Liver scarring. Causes: Hepatitis, Alcohol.",
        "pancreatitis": "Pancreas inflammation. Symptoms: Upper abdominal pain, nausea.",
        "cellulitis": "Bacterial skin infection. Red, swollen, painful skin.",
        "hypothyroidism": "Underactive thyroid. Fatigue, weight gain.",
        "hyperthyroidism": "Overactive thyroid. Weight loss, rapid heart rate.",
        "asthma": "Airway inflammation. Wheezing, SOB."
    }
    
    # Drug/Lab Lookups
    if "inr" in text: return "High INR (>3.5) = Bleeding Risk. Target 2.0-3.0."
    if "creatinine" in text: return "Serum Creatinine > 1.2 suggests renal impairment."
    if "metformin" in text: return "Hold Metformin before contrast if eGFR < 30 (Lactic Acidosis)."
    if "hypoglycemia" in text: return "Symptoms: Sweating, confusion. Protocol: 'Rule of 15'."
    if "hyperkalemia" in text: return "High Potassium (>5.5). Arrhythmia risk. Cause: ACEi + Spironolactone."
    if "ibuprofen" in text: return "NSAID: Avoid in CKD, Heart Failure, and with Blood Thinners."
    
    for k, v in conditions.items():
        if k in text: return f"**{k.title()}**: {v}"
        
    return "I didn't recognize that term. Try 'Sepsis', 'Afib', 'INR', or 'Ibuprofen'."

# ---------------------------------------------------------
# 4. SESSION STATE
# ---------------------------------------------------------
if 'patient_data' not in st.session_state:
    st.session_state['patient_data'] = {
        'id': 'Room 101', 'age': 65, 'bleeding_risk': 12.5, 
        'aki_risk': 10, 'hypo_risk': 5, 'status': 'Stable'
    }

if 'entered_app' not in st.session_state:
    st.session_state['entered_app'] = False

# ---------------------------------------------------------
# 5. UI LAYOUT
# ---------------------------------------------------------

# --- COVER PAGE ---
if not st.session_state['entered_app']:
    st.markdown("<h1 style='text-align: center;'>🛡️ Clinical Risk Monitor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-Driven Pharmacovigilance System</p>", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns([1, 2, 1])
    if c2.button("🚀 Launch Dashboard", use_container_width=True, type="primary"):
        st.session_state['entered_app'] = True
        st.rerun()

# --- MAIN APP ---
else:
    with st.sidebar:
        st.title("Navigation")
        menu = st.radio("Select Module", [
            "Risk Calculator", 
            "Patient History (SQL)",  
            "Live Dashboard", 
            "Batch Analysis (CSV)", 
            "Medication Checker", 
            "Clinical Chatbot"
        ])
        st.info("v2.6 - Zero-Base Inputs")
# --- MODULE 1: RISK CALCULATOR (NOW FIRST) ---
    if menu == "Risk Calculator":
        st.subheader("Acute Risk Calculator (Advanced)")
        
        with st.form("risk_form"):
            # 1. Demographics & BMI
            st.markdown("#### 1. Patient Profile")
            c1, c2, c3, c4 = st.columns(4)
            age = c1.number_input("Age", 18, 100, 65)
            gender = c2.selectbox("Gender", ["Male", "Female"])
            height = c3.number_input("Height (cm)", 100, 250, 170)
            
            # Weight with Unit Selector
            with c4:
                w_val, w_unit = st.columns([2, 1]) 
                weight_input = w_val.number_input("Weight", 40.0, 400.0, 70.0)
                weight_scale = w_unit.selectbox("Unit", ["kg", "lbs"], key="w_unit")

            # Logic: Convert Weight for BMI & AI Model
            if weight_scale == "lbs":
                weight_kg = weight_input * 0.453592
            else:
                weight_kg = weight_input

            # Live BMI Calculation
            bmi = weight_kg / ((height/100)**2)
            c4.caption(f"Calculated BMI: {bmi:.1f}")

            # 2. Vitals
            st.markdown("#### 2. Vitals Signs")
            v1, v2, v3, v4 = st.columns(4)
            sys_bp = v1.number_input("Systolic BP", 60, 250, 120)
            dia_bp = v2.number_input("Diastolic BP", 40, 150, 80)
            hr = v3.number_input("Heart Rate", 40, 200, 72)
            resp_rate = v4.number_input("Resp Rate", 8, 50, 16)
            
            # Temperature with Unit Selector
            v5, v6, v7 = st.columns(3)
            with v5:
                t_val, t_unit = st.columns([2, 1]) 
                temp_input = t_val.number_input("Temperature", 30.0, 110.0, 37.0, step=0.1)
                temp_scale = t_unit.selectbox("Unit", ["°C", "°F"], key="t_unit")
            
            o2_sat = v6.number_input("O2 Sat (%)", 80, 100, 98)
            altered_mental = v7.checkbox("Altered Mental Status?")

            # 3. Labs
            st.markdown("#### 3. Laboratory Values")
            l1, l2, l3, l4 = st.columns(4)
            creat = l1.number_input("Creatinine", 0.5, 10.0, 1.0)
            inr = l2.number_input("INR", 0.0, 10.0, 1.0)
            potassium = l3.number_input("Potassium (K+)", 2.0, 8.0, 4.0) 
            platelets = l4.number_input("Platelets (10^9/L)", 10, 500, 250)

            # 4. Medications & History
            st.markdown("#### 4. Meds & History")
            m1, m2, m3, m4 = st.columns(4)
            anticoag = m1.checkbox("Anticoagulant")
            nsaid = m2.checkbox("NSAID Use") 
            active_chemo = m3.checkbox("Active Chemo")
            hba1c_high = m4.checkbox("HbA1c > 9.0%?")
            
            m5, m6, m7, m8 = st.columns(4)
            diuretic = m5.checkbox("Diuretic")
            acei = m6.checkbox("ACEi/ARB")
            insulin = m7.checkbox("Insulin")
            gi_bleed = m8.checkbox("Hx GI Bleed") 
            
            h1, h2 = st.columns(2)
            heart_failure = h1.checkbox("Heart Failure") 
            liver_disease = h2.checkbox("Liver Disease")

           submitted = st.form_submit_button("Run Analysis & Update Dashboard", type="primary")
           if submitted:
                # [EXISTING] Logic: Convert Temp to Celsius
                if temp_scale == "°F":
                    final_temp_c = (temp_input - 32) * 5/9
                else:
                    final_temp_c = temp_input

                # [EXISTING] 1. AI Prediction (Bleeding)
                is_high_bp = 1 if sys_bp > 140 else 0
                
                input_df = pd.DataFrame({
                    'age': [age], 'inr': [inr], 
                    'anticoagulant': [1 if anticoag else 0],
                    'gi_bleed': [1 if gi_bleed else 0], 
                    'high_bp': [is_high_bp],
                    'antiplatelet': [0], 
                    'gender_female': [1 if gender == "Female" else 0],
                    'weight': [weight_kg], 
                    'liver_disease': [1 if liver_disease else 0]
                })
                pred_bleeding = bleeding_model.predict(input_df)[0]
                
                # [EXISTING] 2. Rule Prediction (AKI)
                pred_aki = calculate_aki_risk(age, diuretic, acei, sys_bp, active_chemo, creat, nsaid, heart_failure)
                
                # [EXISTING] 3. Rule Prediction (Sepsis)
                pred_sepsis = calculate_sepsis_risk(sys_bp, resp_rate, altered_mental, final_temp_c)

                # [EXISTING] 4. Update Session
                st.session_state['patient_data'] = {
                    'id': 'Calculated Patient', 'age': age,
                    'bleeding_risk': float(pred_bleeding), 
                    'aki_risk': int(pred_aki),
                    'sepsis_risk': int(pred_sepsis),
                    'hypo_risk': 0, 
                    'status': 'Critical' if (pred_bleeding > 50 or pred_aki > 50 or pred_sepsis > 50) else 'Stable'
                }
                
                st.success("Analysis Complete! Live Dashboard Updated.")
                r1, r2, r3 = st.columns(3)
                r1.metric("Bleeding Risk (AI)", f"{pred_bleeding:.1f}%")
                r2.metric("AKI Risk (Rule)", f"{pred_aki}%")
                r3.metric("Sepsis Risk (qSOFA)", f"{pred_sepsis}%")
                
                # [EXISTING] Lab Alerts
                if potassium > 5.5: st.error("⚠️ HYPERKALEMIA ALERT: K+ > 5.5 (Arrhythmia Risk)")
                if platelets < 100: st.error("⚠️ THROMBOCYTOPENIA: Bleeding Risk")

                # ---------------------------------------------------------
                # 👇 INSERT HERE (Make sure it is indented inside 'if submitted:')
                # ---------------------------------------------------------
                save_patient_to_db(
                    age, 
                    gender, 
                    sys_bp, 
                    int(pred_aki), 
                    float(pred_bleeding), 
                    'Critical' if (pred_bleeding > 50 or pred_aki > 50) else 'Stable'
                )
                st.toast("✅ Patient Data Saved to Database!", icon="💾")
    
    #MODULE SQL
    # ---------------------------------------------------------
    elif menu == "Patient History (SQL)":
        st.subheader("🗄️ Patient History Database")
        
        # Load Data from SQL
        conn = sqlite3.connect('clinical_data.db')
        history_df = pd.read_sql("SELECT * FROM patient_history ORDER BY timestamp DESC", conn)
        conn.close()
        
        if not history_df.empty:
            st.dataframe(history_df)
            
            st.markdown("### 📊 Cohort Analytics")
            c1, c2 = st.columns(2)
            c1.bar_chart(history_df['aki_risk_score'])
            c2.line_chart(history_df['sbp'])
            
            if st.button("🗑️ Clear Database"):
                conn = sqlite3.connect('clinical_data.db')
                conn.execute("DELETE FROM patient_history")
                conn.commit()
                conn.close()
                st.rerun()
        else:
            st.info("No patient records found in database yet. Go to 'Risk Calculator' and run an analysis!")
                    
    # --- MODULE 2: LIVE DASHBOARD ---
    elif menu == "Live Dashboard":
        data = st.session_state['patient_data']
        st.subheader(f"🏥 Patient Monitor: {data['id']}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patient Age", f"{data['age']} yrs")
        c2.metric("Bleeding Risk", f"{data['bleeding_risk']:.1f}%", "High" if data['bleeding_risk']>50 else "Normal")
        c3.metric("AKI Risk", f"{data['aki_risk']}%", "Critical" if data['aki_risk']>70 else "Normal")
        c4.metric("Hypo Risk", f"{data['hypo_risk']}%", "High" if data['hypo_risk']>50 else "Low")
        
        st.divider()
        col_main, col_queue = st.columns([2, 1])
        
        with col_main:
            st.markdown("#### 📈 Real-Time Vitals")
            chart_data = pd.DataFrame({
                'Hour': [1,2,3,4,5],
                'Risk Score': [data['bleeding_risk']*0.8, data['bleeding_risk']*0.9, data['bleeding_risk'], data['bleeding_risk']*1.1, data['bleeding_risk']]
            })
            st.line_chart(chart_data.set_index('Hour'))
            
            if data['bleeding_risk'] > 50: st.error("HIGH BLEED RISK: Review Anticoagulants.")
            elif data['aki_risk'] > 50: st.warning("RENAL ALERT: Monitor Urine Output.")
            elif data['hypo_risk'] > 50: st.warning("METABOLIC ALERT: Hypoglycemia Risk.")
            else: st.success("No Critical Alerts Active.")

        with col_queue:
            st.markdown("#### 📋 Patient Queue")
            st.markdown(f"""<div style="background-color:#d4edda; color:#155724; padding:10px; border-radius:5px; margin-bottom:10px;">
                <strong>{data['id']} (Current)</strong><br>Status: {data['status']}</div>""", unsafe_allow_html=True)
            st.markdown("""<div style="background-color:#fff3cd; color:#856404; padding:10px; border-radius:5px; margin-bottom:10px;">
                <strong>Patient B (Room 410)</strong><br>🟠 Risk: 80% (AKI Focus)</div>
            <div style="background-color:#f8d7da; color:#721c24; padding:10px; border-radius:5px;">
                <strong>Patient C (Room 105)</strong><br>🔴 Risk: 92% (Sepsis)</div>""", unsafe_allow_html=True)

   # --- MODULE 3: BATCH ANALYSIS (CSV) ---
    elif menu == "Batch Analysis (CSV)":
        st.subheader("Bulk Patient Processing")
        
        # 1. Download Template Button (Helper for User)
        sample_data = {
            'Age': [65, 80, 45, 72],
            'Creatinine': [1.1, 1.8, 0.9, 2.5],
            'Systolic_BP': [130, 160, 120, 110],
            'Diuretic_Use': [0, 1, 0, 1],
            'ACEI_Use': [1, 1, 0, 0],
            'Chemo_Hx': [0, 0, 1, 0],
            'NSAID_Use': [0, 1, 1, 0],
            'Heart_Failure_Hx': [0, 1, 0, 1]
        }
        df_sample = pd.DataFrame(sample_data)
        
        with st.expander("ℹ️ CSV Format Guide"):
            st.write("Your CSV must contain these columns (0=No, 1=Yes):")
            st.dataframe(df_sample)
            st.download_button("📥 Download Template CSV", 
                               df_sample.to_csv(index=False), 
                               "patient_template.csv", "text/csv")

        # 2. Main Processing Tool
        tab1, tab2 = st.tabs(["📄 CSV Data Processor", "🖼️ Medical Imaging"])
        
        with tab1:
            uploaded_csv = st.file_uploader("Upload Patient Cohort (CSV)", type=["csv"])
            
            if uploaded_csv:
                df = pd.read_csv(uploaded_csv)
                st.write(f"**Loaded {len(df)} patient records.**")
                st.dataframe(df.head(3))

                if st.button("⚡ Run Risk Analysis on All Rows"):
                    try:
                        # PROGRESS BAR
                        progress_bar = st.progress(0)
                        
                        # LOGIC: Iterate via Pandas Apply (Vectorized iteration)
                        # We map the CSV columns to the function arguments
                        def batch_risk(row):
                            return calculate_aki_risk(
                                age=row.get('Age', 60), 
                                diuretic=row.get('Diuretic_Use', 0), 
                                acei=row.get('ACEI_Use', 0), 
                                # Logic: Convert Sys BP to High BP Boolean
                                high_bp=1 if row.get('Systolic_BP', 120) > 140 else 0,
                                chemo=row.get('Chemo_Hx', 0), 
                                creat=row.get('Creatinine', 1.0), 
                                nsaid=row.get('NSAID_Use', 0), 
                                heart_failure=row.get('Heart_Failure_Hx', 0)
                            )

                        # Apply function to create new column
                        df['AKI_Risk_Score'] = df.apply(batch_risk, axis=1)
                        
                        # Add a Risk Label based on score
                        df['Risk_Category'] = df['AKI_Risk_Score'].apply(
                            lambda x: '🔴 CRITICAL' if x > 50 else ('🟠 High' if x > 30 else '🟢 Low')
                        )

                        progress_bar.progress(100)
                        st.success("Batch Processing Complete!")

                        # Display Summary Metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Patients", len(df))
                        m2.metric("High Risk Cases", len(df[df['AKI_Risk_Score'] > 30]))
                        m3.metric("Avg Risk Score", f"{df['AKI_Risk_Score'].mean():.1f}")

                        # Show Result
                        st.dataframe(df.style.map(
                            lambda x: 'color: red; font-weight: bold;' if x == '🔴 CRITICAL' else None, 
                            subset=['Risk_Category']
                        ))

                        # Download Result
                        st.download_button(
                            "📥 Download Analyzed Data",
                            df.to_csv(index=False),
                            "analyzed_patients.csv",
                            "text/csv"
                        )

                    except Exception as e:
                        st.error(f"Error processing CSV: {e}")
                        st.info("Check your column names match the Template above!")

        with tab2:
            uploaded_image = st.file_uploader("Upload Wound/X-Ray (JPG)", type=["jpg"])
            if uploaded_image: st.image(uploaded_image, width=300)
                
   # --- MODULE 4: MEDICATION CHECKER (UPDATED WITH CATEGORIES) ---
    elif menu == "Medication Checker":
        st.subheader("Drug-Drug Interaction Checker")
        
        # 1. Add a Guide so users know what to test
        with st.expander("ℹ️  Supported Drug Categories (Demo Database)"):
            st.markdown("""
            This demo checks for **High-Alert** interactions in these specific categories:
            * **🫀 Cardiac:** Warfarin, Amiodarone, Digoxin, Sildenafil, Nitroglycerin, Spironolactone, Lisinopril.
            * **💊 Pain/NSAIDs:** Ibuprofen, Tramadol, Fentanyl, Methotrexate, Oxycodone.
            * **🧠 Psych/Neuro:** Sertraline, Fluoxetine, Linezolid, Lithium, Citalopram.
            * **🦠 Antibiotics:** Ciprofloxacin, Erythromycin, Trimethoprim (Bactrim), Claritromycin.
            """)
        
        st.caption("Type two drugs below to check for CRITICAL or MAJOR interactions.")

        # 2. Input Fields
        col_d1, col_d2 = st.columns(2)
        d1 = col_d1.text_input("Drug A", placeholder="e.g., Warfarin")
        d2 = col_d2.text_input("Drug B", placeholder="e.g., Ibuprofen")

        # 3. Logic
        if d1 and d2:
            res = check_interaction(d1, d2)
            
            # Dynamic Styling based on Severity
            if "CRITICAL" in res: 
                st.error(f"❌ {res}")
            elif "MAJOR" in res: 
                st.warning(f"⚠️ {res}")
            elif "MODERATE" in res: 
                st.info(f"ℹ️ {res}")
            else: 
                st.success(res)
    # --- MODULE 5: CHATBOT ---
    elif menu == "Clinical Chatbot":
        st.subheader("AI Clinical Assistant")
        q = st.text_input("Ask a clinical question (e.g. 'Sepsis', 'AFib', 'Metformin'):")
        if q:
            with st.chat_message("assistant"):
                st.write(chatbot_response(q))
