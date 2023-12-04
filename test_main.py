from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200

def test_logreg_predict():
    data = {
        "Age": 50,
        "Sex": 0,
        "ChestPainType": 2,
        "RestingBP": 135,
        "Cholesterol": 250,
        "FastingBS": 0,
        "RestingECG": 2,
        "MaxHR": 140,
        "ExerciseAngina": 1,
        "Oldpeak": 1.00,
        "ST_Slope": 1
    }

    response = client.post("/pridict/logreg_model/", json=data)
    assert response.status_code == 200
    
