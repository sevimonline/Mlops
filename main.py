from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Logistic Regression Predictor</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100vh;
                background-image: url('13up-healthlife-superJumbo-v2.gif'); /* Arka plan resmi */
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                color: #fff; /* Yazı rengi beyaz */
            }

            h1 {
                text-align: center;
            }

            form {
                margin-top: 20px;
                text-align: center;
                background-color: rgba(255, 255, 255, 0.8); /* Saydam bir arka plan rengi */
                padding: 20px;
                border-radius: 10px;
            }

            label {
                display: block;
                margin-bottom: 5px;
                color: #333; /* Kutucuk içindeki metin rengi */
            }

            input {
                padding: 8px;
                width: 200px;
                margin-bottom: 10px;
                color: #333; /* Kutucuk içindeki yazı rengi */
            }

            button {
                padding: 8px 15px;
                background-color: #4caf50;
                color: #fff;
                border: none;
                cursor: pointer;
            }

            button:hover {
                background-color: #45a049;
            }

            p {
                margin-top: 20px;
                text-align: center;
                font-size: 18px;
                color: #333; /* Sonuç metni rengi */
            }
        </style>
    </head>
    <body>
        <div>
            <h1>Heart Disease Predictor</h1>
            <form action="/pridict/logreg_model/" method="post">
                <label for="Age">Age:</label>
                <input type="number" name="Age" required>
                
                <label for="Sex">Sex:</label>
                <input type="number" name="Sex" required>
                
                <label for="ChestPainType">Chest Pain Type:</label>
                <input type="number" name="ChestPainType" required>
                
                <label for="RestingBP">Resting Blood Pressure:</label>
                <input type="number" name="RestingBP" required>
                
                <label for="Cholesterol">Cholesterol:</label>
                <input type="number" name="Cholesterol" required>
                
                <label for="FastingBS">Fasting Blood Sugar:</label>
                <input type="number" name="FastingBS" required>
                
                <label for="RestingECG">Resting Electrocardiographic Results:</label>
                <input type="number" name="RestingECG" required>
                
                <label for="MaxHR">Maximum Heart Rate Achieved:</label>
                <input type="number" name="MaxHR" required>
                
                <label for="ExerciseAngina">Exercise Induced Angina:</label>
                <input type="number" name="ExerciseAngina" required>
                
                <label for="Oldpeak">Oldpeak:</label>
                <input type="number" step="any" name="Oldpeak" required>
                
                <label for="ST_Slope">ST Slope:</label>
                <input type="number" name="ST_Slope" required>
                
                <button type="submit">Predict</button>
            </form>
            <p id="result"></p>
        </div>
    </body>
    </html>

    """

class logreg_schema(BaseModel):
    Age: int
    Sex: int
    ChestPainType: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: int
    MaxHR: int
    ExerciseAngina: int
    Oldpeak: float
    ST_Slope: int

@app.post("/pridict/logreg_model")
def knn_predict(
    Age: int = Form(...),
    Sex: int = Form(...),
    ChestPainType: int = Form(...),
    RestingBP: int = Form(...),
    Cholesterol: int = Form(...),
    FastingBS: int = Form(...),
    RestingECG: int = Form(...),
    MaxHR: int = Form(...),
    ExerciseAngina: int = Form(...),
    Oldpeak: float = Form(...),
    ST_Slope: int = Form(...),

):
    try:
        load_model = pickle.load(open('logreg_model.pkl', 'rb'))
        
        # Form verilerini ModelSchema'ya dönüştürme
        predict_values = logreg_schema(
            Age=Age,
            Sex=Sex,
            ChestPainType=ChestPainType,
            RestingBP=RestingBP,
            Cholesterol=Cholesterol,
            FastingBS=FastingBS,
            RestingECG=RestingECG,
            MaxHR=MaxHR,    
            ExerciseAngina=ExerciseAngina,
            Oldpeak=Oldpeak,
            ST_Slope=ST_Slope
        )

        # Convert Pydantic model to DataFrame
        df = pd.DataFrame([predict_values.dict()])

        # Add any necessary preprocessing steps here (e.g., encoding categorical variables)

        # Make predictions
        predict = load_model.predict(df)

        result_html = f"""
        <html>
            <head>
                <title>Model Prediction Result</title>
            </head>
            <body>
                <h1>Model Prediction Result</h1>
                <p>Prediction Result: {int(predict)}</p>
            </body>
        </html>
        """
        
        return HTMLResponse(content=result_html, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


 
