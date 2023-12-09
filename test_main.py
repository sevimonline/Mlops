from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Heart Disease Predictor" in response.text

def test_logreg_predict():
    payload = {
        "Age": 35,
        "Sex": 1,
        "ChestPainType": 2,
        "RestingBP": 120,
        "Cholesterol": 200,
        "FastingBS": 0,
        "RestingECG": 1,
        "MaxHR": 150,
        "ExerciseAngina": 0,
        "Oldpeak": 1.00,
        "ST_Slope": 1,
    }

    response = client.post("/pridict/model", data=payload)
    assert response.status_code == 200
    assert "Model Prediction Result" in response.text
