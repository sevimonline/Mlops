from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel


app = FastAPI()

@app.get("/")
def home():
    return {"Message":"Welcome the Model of API"}



class logreg_schema(BaseModel):
    Age:int
    Sex:int
    ChestPainType:int
    RestingBP:int
    Cholesterol:int
    FastingBS:int
    RestingECG:int
    MaxHR:int
    ExerciseAngina:int
    Oldpeak:float
    ST_Slope:int
    


@app.post("/pridict/logreg_model/")
def logreg_predict(predict_values:logreg_schema):
    load_model = pickle.load(open("logreg_model.pkl","rb"))

    df = pd.DataFrame([predict_values.dict().values()],columns=predict_values.dict().keys())
    #print(df)
    predict = load_model.predict(df)


    return {"Predict":int(predict[0])}

