from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

MODEL_FILE = "accident_model.pkl"
SCALER_FILE = "scaler.pkl"

def train_and_save_model():
    print("Training model...")

    np.random.seed(42)
    data_size = 1000

    data = pd.DataFrame({
        'speed_limit': np.random.randint(20, 120, data_size),
        'number_of_vehicles': np.random.randint(1, 5, data_size),
        'weather_condition': np.random.randint(0, 3, data_size),
        'road_condition': np.random.randint(0, 2, data_size),
        'light_condition': np.random.randint(0, 2, data_size),
        'severity': np.random.randint(0, 3, data_size)
    })

    X = data.drop("severity", axis=1)
    y = data["severity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("Model trained and saved!")

if not os.path.exists(MODEL_FILE):
    train_and_save_model()

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    speed = int(request.form['speed'])
    vehicles = int(request.form['vehicles'])
    weather = int(request.form['weather'])
    road = int(request.form['road'])
    light = int(request.form['light'])

    features = np.array([[speed, vehicles, weather, road, light]])
    features = scaler.transform(features)
    prediction = model.predict(features)[0]

    severity_dict = {
        0: "Minor Accident",
        1: "Serious Accident",
        2: "Fatal Accident"
    }
    result = severity_dict[prediction]
    return render_template("index.html",prediction_text=f"Predicted Severity: {result}")

if __name__ == "__main__":
    app.run(debug=True)