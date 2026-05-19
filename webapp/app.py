from flask import Flask, render_template, request, jsonify, redirect, session, url_for
import os
import logging
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import json
import numpy as np
from dotenv import load_dotenv
# --------------------------------------------------
#  LOG CONFIG
# --------------------------------------------------
# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")

# Absolute path to the database, always relative to this file
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "users.db")


# --------------------------------------------------
#  DATABASE SETUP
# --------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()
# --------------------------------------------------
#  LOAD TRAINED RANDOM FOREST MODEL
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "ml_model", "diabetes_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "ml_model", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "ml_model", "feature_names.json")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

logger.info("✅ Trained RandomForest model loaded successfully.")
# --------------------------------------------------
#  LOGIN PROTECTION (MIDDLEWARE)
# --------------------------------------------------
@app.before_request
def require_login():
    public_routes = ["login", "signup", "static"]
    if request.endpoint not in public_routes and "user" not in session:
        return redirect("/login")

FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# --------------------------------------------------
#  ROUTES
# --------------------------------------------------

# ---------- FORCE LOGIN FIRST ----------
@app.route('/')
def root():
    # Redirect root URL to login page
    return redirect("/login")

# ------------------ LOGIN ------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session["user"] = user[2]
            return redirect("/index")  # after login, go to prediction page

        return render_template("login.html", error="Invalid email or password.")

    return render_template("login.html")

# ------------------ SIGNUP ------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        hashed_pw = generate_password_hash(password)

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                           (name, email, hashed_pw))
            conn.commit()
            conn.close()
        except sqlite3.IntegrityError:
            return render_template("signup.html", error="Email already exists.")

        return redirect("/login")

    return render_template("signup.html")

# ------------------ LOGOUT ------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ------------------ INDEX / PREDICTION PAGE ------------------------
@app.route("/index")
def index_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []
        for feature in FEATURE_NAMES:
            value = request.form.get(feature)
            if not value:
                return render_template('index.html', error=f"Please provide {feature}")
            features.append(float(value))

        validation_errors = validate_input(features)
        if validation_errors:
            return render_template('index.html', error=validation_errors)

        
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        from datetime import datetime

        # Store features as ordered list of [name, value] pairs so JSON
        # serialisation (used by Flask sessions) cannot alphabetise them.
        CLINICAL_ORDER = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        feature_map = dict(zip(FEATURE_NAMES, features))
        ordered_features = [
            [name, int(feature_map[name]) if feature_map[name] == int(feature_map[name]) else feature_map[name]]
            for name in CLINICAL_ORDER if name in feature_map
        ]

        result = {
            'prediction': int(prediction),
            'probability_no_diabetes': round(probability[0] * 100, 2),
            'probability_diabetes': round(probability[1] * 100, 2),
            'features': ordered_features,           # list of [name, value]
            'feature_map': feature_map,             # kept for batch/API routes
            'timestamp': datetime.now().strftime("%d %B %Y, %I:%M %p")
        }

        session['result'] = result

        # Keep a rolling history of the last 2 results for comparison
        history = session.get('result_history', [])
        history.append(result)
        session['result_history'] = history[-2:]   # keep only last 2

        return redirect('/result')

    except ValueError:
        return render_template('index.html', error="Invalid numeric input.")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template('index.html', error="Error processing prediction")

# ------------------ RESULT PAGE ------------------------
@app.route('/result')
def result_page():
    result = session.get('result')
    if not result:
        return redirect('/index')
    history = session.get('result_history', [])
    # previous = the result before the current one (if it exists)
    previous = history[-2] if len(history) >= 2 else None
    return render_template('result.html', result=result, previous=previous)

# ------------------ VALIDATION ------------------------
def validate_input(features):
    ranges = {
        'Pregnancies': (0, 20),
        'Glucose': (0, 300),
        'BloodPressure': (0, 200),
        'SkinThickness': (0, 100),
        'Insulin': (0, 1000),
        'BMI': (0, 100),
        'DiabetesPedigreeFunction': (0, 3),
        'Age': (1, 120)
    }

    for feature, value in zip(FEATURE_NAMES, features):
        min_val, max_val = ranges[feature]
        if not (min_val <= value <= max_val):
            return f"{feature} must be between {min_val} and {max_val}"
    return None
@app.route('/model-info')
def model_info():
    return jsonify({
        "model_type": type(model).__name__,
        "features": FEATURE_NAMES,
        "author": "Mohammed Sinwan",
        "description": "Diabetes risk prediction model (RandomForest)"
    })
# ------------------ HEALTH + API ------------------------
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'diabetes-prediction-api',
        'model_loaded': True
    })
    
@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        samples = data.get("samples", [])

        results = []

        for i, sample in enumerate(samples):
            features = [float(sample.get(f, 0)) for f in FEATURE_NAMES]
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)

            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]

            results.append({
                "sample_id": i,
                "prediction": int(prediction),
                "probability_diabetes": float(probability),
                "risk_level": "High" if probability > 0.6 else "Low"
            })

        return jsonify({
            "predictions": results,
            "total_samples": len(samples),
            "successful_predictions": len(results)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        
        features = [float(data.get(feature, 0)) for feature in FEATURE_NAMES]
        
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        return jsonify({
            'prediction': int(prediction),
            'probability_diabetes': float(probability[1]),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --------------------------------------------------
#  RUN APP
# --------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, use_reloader=False)
