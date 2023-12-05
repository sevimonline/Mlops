from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Bu kısım statik dosyaların servis edilmesi için gerekli
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Logistic Regression Predictor</title>
    <link rel="stylesheet" href="/static/style.css">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            background-image: url('13up-healthlife-superJumbo-v2.gif'); /* Değiştirmeniz gereken yer */
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
            <!-- Buraya HTML formunu ekleyin -->
            <!-- ... -->
            <button type="submit">Predict</button>
        </form>
        <p id="result"></p>
    </div>

    <script>
        document.querySelector('form').addEventListener('submit', async function (e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = value;
            });

            const response = await fetch('/pridict/logreg_model/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams(data),
            });

            const result = await response.json();
            document.getElementById('result').innerText = `Prediction: ${result.Predict}`;
        });
    </script>
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

@app.post("/pridict/logreg_model/", response_class=HTMLResponse)
def logreg_predict(predict_values: logreg_schema):
    load_model = pickle.load(open("logreg_model.pkl", "rb"))

    df = pd.DataFrame([predict_values.dict().values()], columns=predict_values.dict().keys())
    predict = load_model.predict(df)

    return HTMLResponse(content=f"<h2>Prediction: {int(predict[0])}</h2>")
