from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI()

class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract: str              
    InternetService: str       
    PaymentMethod: str   
    SeniorCitizen: int = 0
    Partner: int = 0
    Dependents: int = 0
    PaperlessBilling: int = 1
    gender: int = 1
    PhoneService: int = 1
    MultipleLines: str = "No"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"

@app.post("/predict")
def predict(data: CustomerData):
    input_df = pd.DataFrame([data.dict()])

    multi_category = ['InternetService', 'Contract', 'PaymentMethod', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'MultipleLines']
    input_df = pd.get_dummies(input_df, columns=multi_category)

    model_columns = model.feature_names_in_
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
  

    return {"churn": "Yes" if pred == 1 else "No", "probability": round(float(prob), 2)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=8000)
       