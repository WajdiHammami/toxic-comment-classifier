from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import ToxicCommentsInference, format_json_with_probs as format_output
import os


model_path = "../artifacts/distilbert_model_2.pth"
thresholds_path = "../artifacts/thresholds.npy"

app = FastAPI()
inference = ToxicCommentsInference(
    model_path=os.getenv("MODEL_PATH", model_path),
    thresholds_path=os.getenv("THRESHOLDS_PATH", thresholds_path),
    device=os.getenv("DEVICE", "cpu"),
)




class Input(BaseModel):
    text: list[str]





@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}



@app.post("/predict")
def predict(input: Input):


    predictions = inference.predict(input.text, return_probs=True)
    return {"predictions": format_output(predictions)}