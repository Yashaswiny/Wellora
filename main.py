import ast
import bcrypt
from flask import Flask, flash, redirect, request, render_template, jsonify, session, url_for
import mysql.connector
import numpy as np
import pandas as pd
import pickle
from difflib import get_close_matches
import ast  
import requests

app = Flask(__name__)
app.secret_key = 'medical_app_secret'

# Load datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")
training_df = pd.read_csv("datasets/Training_augmented.csv")  

# Train/load model 
disease_to_specialist = {
    "Diabetes": "Endocrinologist",
    "Skin Infection": "Dermatologist",
    "Hypertension": "Cardiologist",
    "Asthma": "Pulmonologist",
    "Flu": "General Physician",
    "Migraine": "Neurologist",
    "Allergy": "Immunologist",
    "Heart attack": "Cardiologist",
    "Urinary tract infection": "Urologist",
    "Pneumonia": "Pulmonologist",
    "Tuberculosis": "Pulmonologist",
    "Bronchitis": "Pulmonologist",
    "Arthritis": "Rheumatologist",
    "Depression": "Psychiatrist",
    "Anxiety": "Psychiatrist",
    "Stroke": "Neurologist",
    "Appendicitis": "General Surgeon",
    "Gastroenteritis": "Gastroenterologist",
    "Hepatitis": "Hepatologist",
    "Cirrhosis": "Hepatologist",
    "Cholelithiasis": "Gastroenterologist",
    "Thyroid disorders": "Endocrinologist",
    "Obesity": "Endocrinologist",
    "Kidney stones": "Urologist",
    "Nephritis": "Nephrologist",
    "Epilepsy": "Neurologist",
    "Conjunctivitis": "Ophthalmologist",
    "Glaucoma": "Ophthalmologist",
    "Cataract": "Ophthalmologist",
    "Sinusitis": "ENT Specialist",
    "Tonsillitis": "ENT Specialist",
    "Otitis media": "ENT Specialist",
    "Chickenpox": "General Physician",
    "Measles": "General Physician",
    "Dengue": "Infectious Disease Specialist",
    "Malaria": "Infectious Disease Specialist",
    "COVID-19": "Pulmonologist",
    "PCOS": "Gynecologist",
    "Menstrual disorders": "Gynecologist",
    "Pregnancy": "Gynecologist",
    "Infertility": "Reproductive Endocrinologist",
    "Back pain": "Orthopedic",
    "Fracture": "Orthopedic",
    "Sprain": "Orthopedic",
    "Anemia": "Hematologist",
    "Leukemia": "Oncologist",
    "Lung cancer": "Oncologist",
    "Breast cancer": "Oncologist",
    "Prostate cancer": "Oncologist",
    "Paralysis (brain hemorrhage)": "Neurologist",
    "Jaundice": "Hepatologist",
    "GERD": "Gastroenterologist",
    "Peptic ulcer": "Gastroenterologist",
    "Insomnia": "Psychiatrist",
    "Vertigo": "Neurologist",
    "Parkinson's Disease": "Neurologist"
}


SERPAPI_KEY = "1214b0d92578749274218615300efa2c57843a1cb6d938721a725ea8bf4bcb65"

with open("models/xgb_disease_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("models/le.pkl", "rb") as f:
    le = pickle.load(f)
    
with open("models/rf_disease_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
    
# Generate mappings
symptoms_dict = {symptom: idx for idx, symptom in enumerate(training_df.columns[:-1])}
diseases_list = training_df['prognosis'].unique().tolist()

# MySQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="test@123",
    database="medical_system"
)
cursor = db.cursor(buffered=True)

def parse_diet_column(df):
    parsed = []
    for val in df['Diet']:
        if isinstance(val, str):
            try:
                parsed_val = ast.literal_eval(val)
            except Exception:
                parsed_val = [val]  
        else:
            parsed_val = val
        parsed.append(parsed_val)
    df['Diet'] = parsed
    return df

diets = parse_diet_column(diets)

def helper(dis):
    desc_row = description[description['Disease'] == dis]['Description']
    desc = " ".join(desc_row) if not desc_row.empty else \
        "This condition may require further clinical diagnosis. Please consult a physician."

    # Precautions
    pre_df = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = []
    for item in pre_df.values.flatten():
        try:
            parsed = ast.literal_eval(item) if isinstance(item, str) and item.startswith("[") else item
            if isinstance(parsed, list):
                pre.extend([p.strip() for p in parsed if isinstance(p, str) and p.strip()])
            elif isinstance(parsed, str) and parsed.strip():
                pre.append(parsed.strip())
        except Exception:
            continue
            pre = ["Stay hydrated", "Get adequate rest"]


    # Medications
    med_row = medications[medications['Disease'].str.lower() == dis.lower()]
    if not med_row.empty:
        med = med_row.iloc[0].dropna().astype(str).str.strip().tolist()[1:]  # skip disease name
        med = [m for m in med if m != ""]
        if not med:
            med = ["Consult a licensed medical practitioner"]
    else:
        med = ["Consult a licensed medical practitioner"]

    
    # Diets
    dis_clean = dis.strip().lower()
    diet_rows = diets[diets['Disease'].str.lower() == dis_clean]['Diet']
    if not diet_rows.empty:
        die = []
        for diet_list in diet_rows:
            if isinstance(diet_list, list):
                die.extend([d for d in diet_list if isinstance(d, str) and d.strip() != ""])
            else:
                die.append(str(diet_list).strip())
        if not die:
            die = ["Consume balanced meals", "Stay hydrated", "Avoid oily/junk food"]
    else:
        die = ["Consume balanced meals", "Stay hydrated", "Avoid oily/junk food"]

    # Workouts 
    wrkout_row = workout[workout['disease'].str.lower() == dis_clean]['workout']
    if not wrkout_row.empty:
        wrkout = []
        for item in wrkout_row:
            if isinstance(item, str) and item.strip() != "":
                try:
                    workout_list = ast.literal_eval(item)
                    wrkout.extend(workout_list)
                except Exception:
                    wrkout.append(item)
        if len(wrkout) == 0:
            wrkout = ["Light walking", "Basic stretching", "Yoga or breathing exercises"]
    else:
        wrkout = ["Light walking", "Basic stretching", "Yoga or breathing exercises"]

    return desc, pre, med, die, wrkout




# Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    prediction_index = xgb_model.predict([input_vector])[0]
    predicted_disease = le.inverse_transform([prediction_index])[0]
    print("Input vector:", input_vector)
    print("Prediction index:", prediction_index)

    return predicted_disease

def correct_symptom(symptom, valid_symptoms):
    match = get_close_matches(symptom.lower(), valid_symptoms, n=1, cutoff=0.6)
    return match[0] if match else None

def get_nearby_doctors_serpapi(specialist, location):
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_local",
        "q": f"{specialist} near {location}",
        "api_key": SERPAPI_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    doctors = []

    for result in data.get("local_results", []):
        name = result.get("title")
        address = result.get("address", "Address not available")
        phone = result.get("phone", "Phone not available")
        doctors.append({
            "name": name,
            "address": address,
            "phone": phone
        })

    return doctors[:5]



# Routes
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor = db.cursor()
        cursor.execute("SELECT id, name, email, password FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()

        if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
            session['user_name'] = user[1]
            session['user_email'] = user[2]
            return redirect(url_for('index'))  
        else:
            flash('Invalid email or password')
            return render_template('login.html')  
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if not name or not email or not password or not confirm_password:
            flash("All fields are required.")
            return redirect('/signup')
        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect('/signup')
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            flash("Email already registered.")
            return redirect('/signup')
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
        (name, email, hashed_password.decode('utf-8')))
        db.commit()
        flash("Signup successful. Please login.")
        return redirect('/')
    return render_template('signup.html')

@app.route('/index')
def index():
    if 'user_name' in session and 'user_email' in session:
        return render_template('index.html', user_name=session['user_name'], user_email=session['user_email'])
    else:
        return redirect(url_for('login'))


@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect('/')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/history')
def history():
    email = session.get('user_email')
    if not email:
        return redirect(url_for('login'))

    cursor = db.cursor()
    cursor.execute("SELECT symptoms, predicted_disease, timestamp FROM user_history WHERE user_email = %s ORDER BY timestamp DESC", (email,))
    history_data = cursor.fetchall()
    cursor.close()

    return render_template('history.html', email=email, user_history=history_data)

@app.route('/get_disease_info', methods=['POST'])
def get_disease_info():
    user_input = request.form.get('symptoms', "")
    location = request.form.get('location')
    if not user_input:
        return render_template('index.html', error="Please enter symptoms.")
    symptoms = list(symptoms_dict.keys())
    raw_symptoms = [s.strip().lower().replace(" ", "_") for s in user_input.split(",")]
    corrected_symptoms = []
    for s in raw_symptoms:
        match = correct_symptom(s, symptoms)
        if match:
            corrected_symptoms.append(match)

    if not corrected_symptoms:
        return render_template('index.html', error="No valid symptoms recognized. Please check your spelling.")

    input_vector = pd.DataFrame([[1 if symptom in corrected_symptoms else 0 for symptom in symptoms]], columns=symptoms)

    pred_proba = xgb_model.predict_proba(input_vector)[0]
    max_prob = max(pred_proba)
    predicted_index = np.argmax(pred_proba)
    predicted_disease = le.inverse_transform([predicted_index])[0]

    specialist = disease_to_specialist.get(predicted_disease, "General Physician")
    recommended_doctors = get_nearby_doctors_serpapi(specialist, location)
    if max_prob < 0.6:
        prediction_note = "⚠️ Prediction made with low confidence. Please enter more symptoms for better accuracy."
    else:
        prediction_note = "✅ Prediction made confidently."
    try:
        user_email = session.get('user_email')
        if user_email:
            symptoms_str = ", ".join(corrected_symptoms)
            cursor = db.cursor()
            insert_query = """
                INSERT INTO user_history (user_email, symptoms, predicted_disease)
                VALUES (%s, %s, %s)
            """
            cursor.execute(insert_query, (user_email, symptoms_str, predicted_disease))
            db.commit()
            cursor.close()
    except Exception as e:
        print("History insert error:", str(e))

    try:
        desc, pre, med, die, wrkout = helper(predicted_disease)
        return render_template('index.html',
            predicted_disease=predicted_disease,
            dis_des=desc,
            my_precautions=pre[0] if pre else [],
            medications=med,
            workout=wrkout if isinstance(wrkout, list) else list(wrkout),
            my_diet=die,
            doctors=recommended_doctors,
            specialist=specialist,
            prediction_note=prediction_note,
            location=location,
            user_name=session.get('user_name'),
            user_email=session.get('user_email')
        )
    except Exception as e:
        return render_template('index.html', error=f"Something went wrong: {str(e)}")



if __name__ == '__main__':
    app.run(debug=True)