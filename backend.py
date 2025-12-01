import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import re

# ==========================================
# 1. DATABASE MANAGEMENT
# ==========================================
def init_db():
    conn = sqlite3.connect('clinical_data.db')
    c = conn.cursor()
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

def fetch_history():
    if not os.path.exists('clinical_data.db'):
        return pd.DataFrame()
    conn = sqlite3.connect('clinical_data.db')
    df = pd.read_sql("SELECT * FROM patient_history ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def clear_history():
    conn = sqlite3.connect('clinical_data.db')
    conn.execute("DELETE FROM patient_history")
    conn.commit()
    conn.close()

# ==========================================
# 2. AI MODEL LOADING
# ==========================================
class DummyModel:
    def predict(self, df):
        return [np.random.randint(10, 30) + (df['age'].values[0] * 0.2)]

def load_bleeding_model():
    model_file = "bleeding_risk_model.json"
    if os.path.exists(model_file):
        model = xgb.XGBRegressor()
        model.load_model(model_file)
        return model
    else:
        return DummyModel()

# ==========================================
# 3. CLINICAL CALCULATORS (LOGIC)
# ==========================================
def calculate_aki_risk(age, diuretic, acei, sys_bp, chemo, creat, nsaid, heart_failure):
    score = 0
    score += 30 if diuretic else 0
    score += 40 if acei else 0
    score += 25 if nsaid else 0
    score += 15 if heart_failure else 0
    score += 20 if chemo else 0
    if age > 0: score += 20 if age > 75 else 0
    if sys_bp > 0:
        score += 10 if sys_bp > 160 else 0
        score += 20 if sys_bp < 90 else 0
    if creat > 0:
        if creat > 1.5: score += 30
        elif creat > 1.2: score += 15
    return min(score, 100)

def calculate_sepsis_risk(sys_bp, resp_rate, altered_mental, temp_c):
    qsofa = 0
    if sys_bp > 0 and sys_bp <= 100: qsofa += 1
    if resp_rate > 0 and resp_rate >= 22: qsofa += 1
    if altered_mental: qsofa += 1
    if temp_c > 0 and (temp_c > 38.0 or temp_c < 36.0): qsofa += 0.5
    
    if qsofa >= 2: return 90
    if qsofa >= 1: return 45
    return 0

def calculate_hypoglycemic_risk(insulin, renal, hba1c_high, recent_dka):
    score = 0
    score += 30 if insulin else 0
    score += 45 if renal else 0
    score += 20 if hba1c_high else 0
    score += 20 if recent_dka else 0 
    return min(score, 100)

def calculate_sirs_score(temp_c, hr, resp_rate, wbc):
    score = 0
    if temp_c > 0 and (temp_c > 38 or temp_c < 36): score += 1
    if hr > 90: score += 1
    if resp_rate > 20: score += 1
    if wbc > 0 and (wbc > 12 or wbc < 4): score += 1
    return score

# ==========================================
# 4. FULL INTERACTION DATABASE
# ==========================================
INTERACTION_DB = {
    ("sildenafil", "nitroglycerin"): "CRITICAL: Severe hypotension and cardiovascular collapse. Contraindicated.",
    ("tadalafil", "nitroglycerin"): "CRITICAL: Severe hypotension. Contraindicated.",
    ("sildenafil", "isosorbide mononitrate"): "CRITICAL: Severe hypotension. Contraindicated.",
    ("methotrexate", "trimethoprim"): "CRITICAL: Fatal bone marrow suppression & pancytopenia. Avoid.",
    ("methotrexate", "bactrim"): "CRITICAL: Fatal bone marrow suppression. Avoid.",
    ("methotrexate", "ibuprofen"): "CRITICAL: NSAIDs block Methotrexate excretion. Risk of Kidney Failure & Toxicity.",
    ("methotrexate", "naproxen"): "CRITICAL: NSAIDs block Methotrexate excretion. Risk of Toxicity.",
    ("sertraline", "linezolid"): "CRITICAL: Fatal Serotonin Syndrome risk (Linezolid is an MAOI).",
    ("fluoxetine", "linezolid"): "CRITICAL: Fatal Serotonin Syndrome risk.",
    ("citalopram", "linezolid"): "CRITICAL: Fatal Serotonin Syndrome risk.",
    ("spironolactone", "trimethoprim"): "CRITICAL: Risk of sudden cardiac death from severe Hyperkalemia.",
    ("spironolactone", "bactrim"): "CRITICAL: Risk of sudden cardiac death from severe Hyperkalemia.",
    ("fentanyl", "benzodiazepine"): "CRITICAL: Profound respiratory depression, coma, and death.",
    ("oxycodone", "alcohol"): "CRITICAL: Respiratory depression and fatal overdose risk.",
    ("tramadol", "alcohol"): "CRITICAL: Seizure threshold lowered + Respiratory depression.",
    ("clarithromycin", "simvastatin"): "CRITICAL: Severe Rhabdomyolysis and kidney failure (CYP3A4 inhibition).",
    ("erythromycin", "simvastatin"): "CRITICAL: Severe Rhabdomyolysis and kidney failure.",
    ("ciprofloxacin", "tizanidine"): "CRITICAL: Severe hypotension and sedation.",
    ("warfarin", "amiodarone"): "MAJOR: Amiodarone inhibits Warfarin metabolism. INR can double. Reduce Warfarin 50%.",
    ("warfarin", "ciprofloxacin"): "MAJOR: Antibiotics kill gut flora, spiking INR. High bleed risk.",
    ("warfarin", "bactrim"): "MAJOR: Significant increase in INR. High bleed risk.",
    ("warfarin", "erythromycin"): "MAJOR: Increases Warfarin levels. Monitor INR closely.",
    ("warfarin", "clarithromycin"): "MAJOR: Increases Warfarin levels. Monitor INR.",
    ("warfarin", "fluconazole"): "MAJOR: Dramatically increases INR. Bleed risk.",
    ("warfarin", "metronidazole"): "MAJOR: Dramatically increases INR. Bleed risk.",
    ("warfarin", "aspirin"): "MAJOR: Significantly increased bleeding risk (Antiplatelet + Anticoagulant).",
    ("warfarin", "ibuprofen"): "MAJOR: Gastrointestinal bleeding risk. NSAIDs damage stomach lining.",
    ("digoxin", "amiodarone"): "MAJOR: Amiodarone doubles Digoxin levels. Risk of toxicity (Halo vision, Nausea).",
    ("digoxin", "clarithromycin"): "MAJOR: Macrolides block P-gp, leading to Digoxin toxicity.",
    ("digoxin", "erythromycin"): "MAJOR: Macrolides block P-gp, leading to Digoxin toxicity.",
    ("digoxin", "spironolactone"): "MAJOR: Spironolactone reduces Digoxin clearance. Monitor levels.",
    ("clopidogrel", "omeprazole"): "MAJOR: Omeprazole blocks CYP2C19, making Plavix ineffective (Stent thrombosis risk).",
    ("sildenafil", "erythromycin"): "MAJOR: Erythromycin increases Sildenafil levels. Risk of hypotension.",
    ("sildenafil", "clarithromycin"): "MAJOR: Clarithromycin increases Sildenafil levels. Risk of hypotension.",
    ("lisinopril", "ibuprofen"): "MAJOR: NSAIDs reduce efficacy of ACEi and cause Acute Kidney Injury.",
    ("lisinopril", "spironolactone"): "MAJOR: High risk of Hyperkalemia. Monitor Potassium.",
    ("lisinopril", "potassium"): "MAJOR: Risk of Hyperkalemia. Avoid supplements.",
    ("lisinopril", "lithium"): "MAJOR: ACEi reduces Lithium excretion -> Lithium Toxicity.",
    ("furosemide", "ibuprofen"): "MAJOR: NSAIDs block diuretic effect and hurt kidneys.",
    ("tramadol", "sertraline"): "MAJOR: Serotonin Syndrome risk (Seizures, Agitation, Hyperthermia).",
    ("tramadol", "fluoxetine"): "MAJOR: Serotonin Syndrome risk.",
    ("tramadol", "citalopram"): "MAJOR: Serotonin Syndrome risk + Seizure risk.",
    ("citalopram", "ondansetron"): "MAJOR: QT Prolongation. Risk of Torsades de Pointes arrhythmia.",
    ("citalopram", "ciprofloxacin"): "MAJOR: QT Prolongation. Risk of Torsades de Pointes.",
    ("citalopram", "erythromycin"): "MAJOR: QT Prolongation. Risk of Torsades de Pointes.",
    ("citalopram", "clarithromycin"): "MAJOR: QT Prolongation. Risk of Torsades de Pointes.",
    ("fluoxetine", "ondansetron"): "MAJOR: QT Prolongation risk.",
    ("lithium", "ibuprofen"): "MAJOR: NSAIDs reduce Lithium excretion -> Lithium Toxicity (Tremors, Confusion).",
    ("lithium", "hydrochlorothiazide"): "MAJOR: Thiazides reduce Lithium excretion -> Lithium Toxicity.",
    ("ciprofloxacin", "theophylline"): "MAJOR: Cipro increases Theophylline levels (Seizures).",
    ("ciprofloxacin", "prednisone"): "MAJOR: Increased risk of tendon rupture (Achilles), especially in elderly.",
    ("levofloxacin", "prednisone"): "MAJOR: Increased risk of tendon rupture.",
    ("doxycycline", "calcium"): "MAJOR: Calcium binds to antibiotic, preventing absorption.",
    ("ciprofloxacin", "calcium"): "MAJOR: Calcium binds to antibiotic, preventing absorption.",
    ("simvastatin", "grapefruit"): "MAJOR: Grapefruit inhibits CYP3A4. Risk of Rhabdomyolysis.",
    ("atorvastatin", "grapefruit"): "MODERATE: Grapefruit increases levels. Use caution.",
    ("warfarin", "alcohol"): "MAJOR: Acute alcohol intake increases bleed risk. Chronic intake decreases Warfarin effect.",
    ("metronidazole", "alcohol"): "MAJOR: Disulfiram-like reaction (Severe vomiting). Avoid alcohol for 72h.",
    ("oxycodone", "grapefruit"): "MODERATE: Grapefruit may increase Opioid levels (Sedation).",
    ("apixaban", "ibuprofen"): "MODERATE: Increased bleeding risk.",
    ("rivaroxaban", "ibuprofen"): "MODERATE: Increased bleeding risk.",
    ("aspirin", "ibuprofen"): "MODERATE: Ibuprofen blocks Aspirin's heart-protective effect if taken together.",
    ("sertraline", "ibuprofen"): "MODERATE: SSRIs + NSAIDs increase GI bleed risk.",
    ("fluoxetine", "ibuprofen"): "MODERATE: SSRIs + NSAIDs increase GI bleed risk.",
    ("citalopram", "ibuprofen"): "MODERATE: SSRIs + NSAIDs increase GI bleed risk.",
    ("metoprolol", "amiodarone"): "MODERATE: Risk of severe Bradycardia (slow heart rate).",
    ("levothyroxine", "calcium"): "MODERATE: Calcium blocks thyroid absorption. Separate by 4 hours.",
    ("levothyroxine", "iron"): "MODERATE: Iron blocks thyroid absorption. Separate by 4 hours.",
    ("allopurinol", "amoxicillin"): "MODERATE: High risk of skin rash.",
}

def check_interaction(d1, d2):
    d1, d2 = d1.lower().strip(), d2.lower().strip()
    if (d1, d2) in INTERACTION_DB: return INTERACTION_DB[(d1, d2)]
    if (d2, d1) in INTERACTION_DB: return INTERACTION_DB[(d2, d1)]
    return "✅ No high-alert interaction found."

# ==========================================
# 5. FULL KNOWLEDGE BASE & CHATBOT LOGIC
# ==========================================
KNOWLEDGE_BASE = {
    "mi": "Myocardial Infarction (Heart Attack). Blockage of blood flow. STEMI is critical. Symptoms: Chest pressure, radiating pain.",
    "heart attack": "Myocardial Infarction (MI). Emergency. Protocol: MONA (Morphine, O2, Nitro, Aspirin).",
    "hypertension": "High Blood Pressure (>130/80). 'The Silent Killer'. Risk of Stroke/MI/Kidney Failure.",
    "hypotension": "Low BP (<90/60). Signs of shock. Causes: Dehydration, Sepsis, Hemorrhage.",
    "afib": "Atrial Fibrillation. Irregular heart rate. Risk: Stroke. Treatment: Beta-blockers + Anticoagulation.",
    "chf": "Congestive Heart Failure. Fluid overload. Symptoms: SOB lying flat (Orthopnea), Edema. Labs: BNP.",
    "cad": "Coronary Artery Disease. Plaque buildup in heart arteries. Risk of MI. Managed with Statins.",
    "angina": "Chest pain due to ischemia (lack of O2). Stable (exertional) vs Unstable (at rest).",
    "aortic stenosis": "Narrowing of aortic valve. Symptoms: SAD (Syncope, Angina, Dyspnea). Murmur heard.",
    "pericarditis": "Inflammation of heart sac. Sharp pain relieved by leaning forward. Friction rub heard.",
    "endocarditis": "Infection of heart valves. IV drug use is risk factor. Fever + New Murmur.",
    "bradycardia": "Slow heart rate (<60). Causes: Beta-blockers, heart block, athletic heart.",
    "tachycardia": "Fast heart rate (>100). Causes: Fever, Pain, Dehydration, PE, Anxiety.",
    "svt": "Supraventricular Tachycardia. Rate >150. Tx: Vagal maneuvers, Adenosine.",
    "vtach": "Ventricular Tachycardia. Life-threatening. Pulse? Cardiovert. No pulse? Defibrillate.",
    "vfib": "Ventricular Fibrillation. Cardiac Arrest. No pulse. CPR + Defibrillation immediately.",
    "asthma": "Airway inflammation. Wheezing. Rescue: Albuterol. Maintenance: Steroids.",
    "copd": "Chronic Obstructive Pulmonary Disease. Air trapping. Smokers. Risk of CO2 retention.",
    "pneumonia": "Lung infection. Fever, cough, consolidation on X-Ray. Antibiotics needed.",
    "pe": "Pulmonary Embolism. Lung clot. Sudden SOB, chest pain, tachycardia. Emergency.",
    "pneumothorax": "Collapsed lung. Air in pleural space. Absent breath sounds. Trauma/Spontaneous.",
    "pleural effusion": "Fluid around lungs. 'Water on lungs'. Causes: CHF, Cancer. Tx: Thoracentesis.",
    "ards": "Acute Respiratory Distress Syndrome. Severe lung failure. 'White out' X-ray.",
    "bronchitis": "Inflammation of bronchi. Productive cough. Usually viral.",
    "tuberculosis": "TB. Bacterial infection. Night sweats, weight loss, bloody cough. Isolation required.",
    "sleep apnea": "Airway collapse during sleep. Snoring, fatigue. Risk: HTN, Stroke. Tx: CPAP.",
    "croup": "Pediatric viral infection. 'Barking seal' cough. Steeple sign on X-ray.",
    "rsv": "Respiratory Syncytial Virus. Bronchiolitis in kids. Watch for hypoxia.",
    "cva": "Cerebrovascular Accident (Stroke). Ischemic vs Hemorrhagic. Time is Brain.",
    "stroke": "Brain attack. BE-FAST: Balance, Eyes, Face, Arms, Speech, Time. CT Head stat.",
    "tia": "Transient Ischemic Attack. Warning stroke. Symptoms resolve <24h. High risk of future stroke.",
    "seizure": "Abnormal electrical activity. Protect head. Benzodiazepines if > 5 mins.",
    "epilepsy": "Recurrent seizures. Meds: Keppra, Depakote, Phenytoin.",
    "migraine": "Severe unilateral headache. Nausea, photophobia. Tx: Triptans.",
    "cluster headache": "Severe eye pain. 'Suicide headache'. Tx: High flow Oxygen.",
    "meningitis": "Brain lining infection. Fever + Stiff Neck + Headache. Emergency.",
    "concussion": "Mild TBI. Confusion, amnesia, headache. Brain rest needed.",
    "parkinson": "Low dopamine. Tremor (resting), Rigidity, Slow movement (Bradykinesia).",
    "alzheimer": "Dementia type. Memory loss, cognitive decline.",
    "ms": "Multiple Sclerosis. Autoimmune nerve damage. Vision loss, weakness.",
    "als": "Lou Gehrig's Disease. Motor neuron death. Paralysis. Sensation intact.",
    "guillain-barre": "Ascending paralysis after infection. Watch breathing.",
    "gerd": "Acid Reflux. Heartburn. Risk of esophageal damage. PPIs (Omeprazole).",
    "pud": "Peptic Ulcer Disease. Stomach ulcers. Pain with food. H. Pylori or NSAIDs.",
    "gi bleed": "Upper (Vomit blood) vs Lower (Bloody stool). Monitor Hemoglobin.",
    "appendicitis": "RLQ pain (McBurney's). Fever, nausea. Surgical emergency.",
    "cholecystitis": "Gallbladder inflammation. RUQ pain after fatty meal.",
    "pancreatitis": "Pancreas inflammation. Epigastric pain to back. High Lipase.",
    "hepatitis": "Liver inflammation. Viral (A/B/C) or Alcohol. Jaundice.",
    "cirrhosis": "End-stage liver scarring. Ascites, confusion (Ammonia), Bleeding risk.",
    "ibs": "Irritable Bowel Syndrome. Functional pain/diarrhea. No organ damage.",
    "ibd": "Crohn's/Ulcerative Colitis. Autoimmune. Bloody diarrhea.",
    "diverticulitis": "Infected colon pouches. LLQ pain, fever. Antibiotics.",
    "c diff": "Antibiotic diarrhea. Contagious spores. Soap & Water wash only.",
    "bowel obstruction": "Blockage. Constipation, vomiting, distension. NPO + NG Tube.",
    "aki": "Acute Kidney Injury. Creatinine spike. Causes: Dehydration, Contrast, NSAIDs.",
    "ckd": "Chronic Kidney Disease. GFR < 60 > 3 months. Diabetes/HTN causes.",
    "esrd": "End Stage Renal Disease. Needs Dialysis or Transplant.",
    "uti": "Urinary Tract Infection. Burning, frequency. E. Coli common.",
    "kidney stone": "Nephrolithiasis. Flank pain, hematuria. Hydration + Pain meds.",
    "bph": "Enlarged prostate. Dribbling, frequency in older men.",
    "rhabdo": "Muscle breakdown. Clogs kidneys. Tea-colored urine. Fluids.",
    "diabetes": "Metabolic disease. High sugar. Causes damage to eyes, kidneys, nerves.",
    "diabetes type 1": "Autoimmune. No insulin. DKA risk. Insulin dependent.",
    "diabetes type 2": "Insulin resistance. Lifestyle + Metformin.",
    "prediabetes": "A1C 5.7-6.4%. Warning sign. Reversible.",
    "dka": "Diabetic Ketoacidosis. Acidosis + Ketones. ICU care.",
    "hhs": "Hyperosmolar Hyperglycemic State. Glucose > 600. Dehydration.",
    "hypoglycemia": "Low Sugar (<70). Sweating, confusion. Glucose needed.",
    "hypothyroidism": "Low Thyroid. Fatigue, weight gain, cold.",
    "hyperthyroidism": "High Thyroid. Weight loss, heat intolerance, fast HR.",
    "addison": "Adrenal insufficiency. Low cortisol. Bronze skin, hypotension.",
    "cushing": "High cortisol. Moon face, buffalo hump, high sugar.",
    "sepsis": "Infection + Organ Failure. qSOFA criteria. Antibiotics ASAP.",
    "septic shock": "Sepsis + Hypotension requiring pressors.",
    "flu": "Influenza. Sudden fever, aches. Tamiflu < 48h.",
    "covid": "SARS-CoV-2. Fever, cough, loss of taste/smell.",
    "hiv": "Virus attacking immune system (CD4). Needs ARV therapy.",
    "mrsa": "Resistant Staph infection. Needs Vancomycin.",
    "cellulitis": "Skin infection. Red, hot, spreading.",
    "abscess": "Pus pocket. Needs drainage (I&D).",
    "osteomyelitis": "Bone infection. Long-term IV antibiotics.",
    "fever": "Temp > 100.4F (38C). Sign of inflammation.",
    "neutropenic fever": "Fever in chemo patient. ONCOLOGIC EMERGENCY.",
    "anemia": "Low Hemoglobin. Fatigue, pallor. Iron deficiency common.",
    "sickle cell": "Genetic. Pain crises. RBCs shape sickle. Fluids/Pain meds.",
    "thrombocytopenia": "Low platelets. Bleeding risk.",
    "leukemia": "Blood cancer. High WBC blasts. Infection risk.",
    "lymphoma": "Lymphatic cancer. Hodgkin vs Non-Hodgkin.",
    "neutropenia": "Low neutrophils. Severe infection risk. Fever is emergency.",
    "dvt": "Deep Vein Thrombosis. Leg clot.",
    "osteoarthritis": "Wear-and-tear. Joint pain. Worse with use.",
    "rheumatoid arthritis": "Autoimmune. Morning stiffness > 30 mins.",
    "gout": "Uric acid crystals. Big toe pain. Colchicine/Allopurinol.",
    "osteoporosis": "Weak bones. Fracture risk.",
    "compartment syndrome": "Muscle pressure. Pain out of proportion. Emergency surgery.",
    "rhabdomyolysis": "Muscle breakdown releasing myoglobin. Kidney damage. Tea-colored urine.",
    "lisinopril": "ACE Inhibitor (BP). Side effects: Cough, High K+.",
    "amlodipine": "Calcium Channel Blocker (BP). Side effect: Leg swelling.",
    "metoprolol": "Beta-blocker. Lowers HR and BP.",
    "furosemide": "Lasix. Diuretic. Monitor Potassium.",
    "spironolactone": "Potassium-sparing diuretic. Risk: High K+.",
    "atorvastatin": "Lipitor. Cholesterol. Watch for muscle pain.",
    "metformin": "Diabetes. Risk: Lactic Acidosis. Hold for contrast.",
    "insulin": "Lowers sugar. High Alert. Hypoglycemia risk.",
    "glipizide": "Sulfonylurea. Stimulates pancreas. Hypo risk.",
    "warfarin": "Coumadin. Anticoagulant. Monitor INR (2-3).",
    "eliquis": "Apixaban. Blood thinner. No INR needed.",
    "plavix": "Clopidogrel. Antiplatelet. Keeps stents open.",
    "aspirin": "Antiplatelet. MI/Stroke prevention. Bleed risk.",
    "ibuprofen": "NSAID. Pain/Fever. Avoid in Kidney/Ulcer/CHF.",
    "naproxen": "Aleve. NSAID.",
    "acetaminophen": "Tylenol. Pain/Fever. Liver toxicity > 4g.",
    "tramadol": "Weak opioid. Serotonin syndrome risk.",
    "oxycodone": "Opioid. Severe pain. Respiratory depression risk.",
    "morphine": "Opioid. Gold standard for MI/Pain.",
    "naloxone": "Narcan. Opioid antidote.",
    "albuterol": "Asthma rescue inhaler. Causes jitters.",
    "prednisone": "Steroid. High sugar, anger, insomnia.",
    "pantoprazole": "Protonix. Acid reflux.",
    "ondansetron": "Zofran. Nausea. QT prolongation.",
    "sertraline": "Zoloft. SSRI. Depression/Anxiety.",
    "fluoxetine": "Prozac. SSRI.",
    "alprazolam": "Xanax. Benzo. Anxiety. Addictive.",
    "lorazepam": "Ativan. Benzo. Seizures.",
    "vancomycin": "Antibiotic (MRSA). Monitor levels. Red Man Syndrome.",
    "piperacillin": "Zosyn. Broad spectrum antibiotic.",
    "ciprofloxacin": "Antibiotic (UTI). Tendon rupture risk.",
    "azithromycin": "Z-Pak. Pneumonia.",
    "diphenhydramine": "Benadryl. Allergy. Sedating.",
    "epinephrine": "Adrenaline. Anaphylaxis/Code Blue.",
    "wbc": "White Blood Cells. High=Infection.",
    "hgb": "Hemoglobin. Low=Anemia. <7 Transfuse.",
    "plt": "Platelets. Low=Bleeding risk.",
    "na": "Sodium. Low=Confusion.",
    "k": "Potassium. Critical for heart rhythm.",
    "bun": "BUN. High=Dehydration/Kidney.",
    "cr": "Creatinine. Best kidney marker.",
    "glucose": "Blood Sugar. 70-100 Fasting.",
    "a1c": "HbA1c. 3-month sugar avg. <5.7 Normal. >6.5 Diabetes.",
    "trop": "Troponin. Heart enzyme. High=Heart Attack.",
    "bnp": "Heart Failure marker. High=Fluid overload.",
    "inr": "Clotting time (Warfarin). Normal 1.0. Goal 2-3.",
    "lactate": "Sepsis marker. >2.0 indicates shock.",
    "ph": "Acidity (7.35-7.45).",
    "bid": "Twice a day.",
    "tid": "Three times a day.",
    "qid": "Four times a day.",
    "qd": "Daily.",
    "prn": "As needed.",
    "ac": "Before meals.",
    "pc": "After meals.",
    "po": "By mouth.",
    "iv": "Intravenous.",
    "im": "Intramuscular.",
    "npo": "Nothing by mouth.",
    "stat": "Immediately.",
    "vs": "Vital Signs.",
    "nkda": "No Known Drug Allergies."
}

def chatbot_response(text):
    text = text.lower().strip()
    
    # 1. SMART TRANSLATOR
    synonyms = {
        "type1": "type 1", "type2": "type 2",
        "high bp": "hypertension", "high blood pressure": "hypertension",
        "low bp": "hypotension", "low blood pressure": "hypotension",
        "heart failure": "chf", "kidney failure": "ckd", "kidney injury": "aki",
        "sugar": "glucose", "blood sugar": "glucose", "clot": "dvt",
        "breathing problem": "copd", "shortness of breath": "sob",
        "belly pain": "appendicitis", "stomach pain": "gerd",
        "heart rate": "tachycardia", "temp": "fever", "temperature": "fever",
        "heart attack": "mi", "stroke": "cva"
    }
    for slang, term in synonyms.items():
        if slang in text:
            text = text.replace(slang, term)
            
    # 2. KNOWLEDGE BASE SEARCH
    sorted_keys = sorted(KNOWLEDGE_BASE.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if re.search(r'\b' + re.escape(key) + r'\b', text):
            return f"**{key.upper()}**: {KNOWLEDGE_BASE[key]}"
            
    return "ℹ️ I didn't recognize that term. Try specific medical terms like 'Sepsis', 'Warfarin', 'INR', or 'Stroke'."

# ==========================================
# 6. AI DIAGNOSTIC ENGINE (UPDATED)
# ==========================================
def consult_ai_doctor(role, user_input, patient_context=None):
    import google.generativeai as genai
    import streamlit as st

    # 1. Securely Load API Key
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        return "⚠️ Error: API Key not found. Please set GEMINI_API_KEY in Streamlit Secrets."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    # 2. Construct Prompt based on Role
    if role == 'patient':
        # ... (Keep your existing patient code here) ...
        prompt = f"""Medical Triage Assistant. User: "{user_input}". Task: Suggest 3 causes & urgency."""

    elif role == 'provider':
        # ... (Keep your existing provider code here) ...
        prompt = f"""Expert Consult. Observation: "{user_input}". Task: Differential diagnosis & plan."""

    # --- NEW ROLE ADDED HERE ---
    elif role == 'risk_assessment':
        # Extract context
        age = patient_context.get('age')
        sbp = patient_context.get('sbp')
        creat = patient_context.get('creat')
        bleed = patient_context.get('bleeding_risk')
        aki = patient_context.get('aki_risk')
        sepsis = patient_context.get('sepsis_risk')
        
        prompt = f"""
        Act as a Senior ICU Consultant. Analyze this patient's risk profile:
        - Age: {age}
        - Systolic BP: {sbp}
        - Creatinine: {creat}
        - Predicted Bleeding Risk (AI): {bleed:.1f}%
        - Kidney Injury Risk (Rule): {aki}%
        - Sepsis Score: {sepsis}
        
        TASK:
        1. Identify the primary clinical threat (The "Headline").
        2. Explain WHY specific values are concerning in combination (e.g. High BP + High Bleed Risk).
        3. Suggest 3 immediate medical actions.
        4. Keep it concise (bullet points).
        """

    # 3. Call Gemini
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ AI Connection Error: {str(e)}"

# ==========================================
# 7. DATA RETRIEVAL (PERSISTENCE)
# ==========================================
def get_latest_patient():
    """Fetches the most recent patient record from the DB to restore state."""
    if not os.path.exists('clinical_data.db'):
        return None
        
    conn = sqlite3.connect('clinical_data.db')
    try:
        df = pd.read_sql("SELECT * FROM patient_history ORDER BY timestamp DESC LIMIT 1", conn)
        conn.close()
        
        if not df.empty:
            row = df.iloc[0]
            return {
                'id': f"Restored-{str(row['timestamp'])[11:16]}", 
                'age': int(row['age']),
                'bleeding_risk': float(row['bleeding_risk_score']),
                'aki_risk': int(row['aki_risk_score']),
                'status': row['status'],
                'hypo_risk': 0, 
                'sepsis_risk': 0
            }
        return None
    except Exception as e:
        return None

# ==========================================
# 8. AI DISCHARGE SUMMARY (NEW)
# ==========================================
def generate_discharge_summary(patient_data):
    import google.generativeai as genai
    import streamlit as st

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        Act as a Hospitalist. Write a formal Hospital Discharge Summary for this patient:
        
        Patient Data:
        - Age: {patient_data.get('age')}
        - Diagnosis: {patient_data.get('status')} Risk Profile
        - Bleeding Risk: {patient_data.get('bleeding_risk', 0):.1f}%
        - AKI Risk: {patient_data.get('aki_risk', 0)}%
        - Sepsis Score: {patient_data.get('sepsis_risk', 0)}
        
        Format as:
        1. Assessment
        2. Hospital Course (Summarize the risks identified)
        3. Discharge Plan (Suggest 3 specific monitoring actions)
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"
