import pickle
import numpy as np

with open("model/hospital_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_readmission(data):
    gender = 0 if data.gender == 'M' else 1
    features = np.array([[data.age, gender, data.days_in_hospital, data.lab_procedures, data.medications, data.visits_last_year]])
    prediction = model.predict(features)
    return int(prediction[0])
