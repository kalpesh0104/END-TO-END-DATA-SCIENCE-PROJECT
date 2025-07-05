from fastapi import FastAPI
from app.schema import PatientData
from app.model import predict_readmission

app = FastAPI()

@app.get("/")
def root():
    return {"message": "🏥 Welcome to the Hospital Readmission Predictor API"}

@app.post("/predict")
def get_prediction(data: PatientData):
    result = predict_readmission(data)
    return {
        "readmitted": bool(result),
        "risk": "High" if result else "Low"
    }
