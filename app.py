import os
from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model("anaecheck_model.h5")

# Advice dictionary
advice_map = {
    "Normal": {
        "message": "Normal — Keep maintaining a balanced diet and healthy lifestyle.",
        "do": [
            "Stay hydrated",
            "Eat iron-rich foods occasionally (e.g., lentils, spinach)",
            "Exercise regularly"
        ],
        "dont": [
            "Skip meals",
            "Overconsume caffeine",
            "Ignore fatigue symptoms"
        ]
    },
    "Mild": {
        "message": "Mild — Consider dietary changes and monitor symptoms.",
        "do": [
            "Increase intake of leafy greens and legumes",
            "Include vitamin C with meals to boost iron absorption",
            "Get regular sleep and reduce stress"
        ],
        "dont": [
            "Drink tea or coffee immediately after meals",
            "Ignore dizziness or fatigue",
            "Delay routine health checkups"
        ]
    },
    "Severe": {
        "message": "Severe — Please consult a doctor immediately.",
        "do": [
            "Eat iron-rich foods like spinach, red meat, and fortified cereals",
            "Take prescribed iron supplements",
            "Rest well and avoid strenuous activity"
        ],
        "dont": [
            "Self-medicate without guidance",
            "Consume alcohol or tobacco",
            "Delay medical consultation"
        ]
    }
}

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to read image")

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(processed_img):
    prediction = model.predict(processed_img)
    class_index = np.argmax(prediction)
    labels = ['Normal', 'Mild', 'Severe']
    return labels[class_index]

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Extract form data
    full_name = request.form.get('full_name')
    gender = request.form.get('gender')
    age = request.form.get('age')
    menstrual_history = request.form.get('menstrual_history')
    pregnancy_status = request.form.get('pregnancy_status')
    diet_type = request.form.get('diet_type')
    smoking_status = request.form.get('smoking_status')
    recent_blood_donation = request.form.get('recent_blood_donation')
    past_anaemia_diagnosis = request.form.get('past_anaemia_diagnosis')
    fatigue_frequency = request.form.get('fatigue_frequency')
    dizziness_frequency = request.form.get('dizziness_frequency')
    headache_frequency = request.form.get('headache_frequency')
    cold_sensitivity = request.form.get('cold_sensitivity')
    pica_habit = request.form.get('pica_habit')
    breath_hold_timer = request.form.get('breath_hold_timer')

    image = request.files.get('nail_conjunctiva_image')
    image_filename = None
    prediction = "No image uploaded"
    advice = {
        "message": "No prediction available.",
        "do": [],
        "dont": []
    }

    if image and image.filename != '':
        image_filename = image.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image.save(filepath)

        processed_img = preprocess_image(filepath)
        prediction = predict_image(processed_img)
        advice = advice_map.get(prediction, advice)

    submitted_data = {
        "Full Name": full_name,
        "Gender": gender,
        "Age": age,
        "Menstrual History": menstrual_history,
        "Pregnancy Status": pregnancy_status,
        "Diet Type": diet_type,
        "Smoking Status": smoking_status,
        "Recent Blood Donation": recent_blood_donation,
        "Past Anaemia Diagnosis": past_anaemia_diagnosis,
        "Fatigue Frequency": fatigue_frequency,
        "Dizziness Frequency": dizziness_frequency,
        "Headache Frequency": headache_frequency,
        "Cold Sensitivity": cold_sensitivity,
        "Pica Habit": pica_habit,
        "Breath Hold Timer": breath_hold_timer,
        "Uploaded Image Filename": image_filename,
        "AI Prediction": prediction,
        "Advice Message": advice["message"],
        "Recommended Actions": advice["do"],
        "Avoid These": advice["dont"]
    }

    return render_template('result.html', data=submitted_data)

if __name__ == '__main__':
    app.run(debug=True)
