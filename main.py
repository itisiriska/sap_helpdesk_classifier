import pickle

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from config import *


class Text(BaseModel):
    text: str


app = FastAPI()


@app.get("/api/text/{text_id}")
def get_text(text_id: int):
    data = pd.read_csv(DATA_PATH)
    try:
        res = data.loc[text_id]
        text = res["text"]
        label = res["class"]
    except KeyError:
        text = "No text with such index"
        label = None
    return {"text": text, "label": label}


@app.post("/api/predict/")
def predict_for_text(text: Text):
    with open(BEST_MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    text_series = pd.Series(text.text)
    predicted_label = model.predict(text_series)[0]
    return {"text": text_series.loc[0], "predicted_label": predicted_label}


@app.post("/api/predict/{text_id}/")
def predict_for_text_id(text_id: int):
    data = pd.read_csv(DATA_PATH)
    try:
        res = data.loc[text_id]
        text = res["text"]
        original_label = res["class"]
        with open(BEST_MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        text_series = pd.Series(text)
        predicted_label = model.predict(text_series)[0]
    except KeyError:
        text = "No text with such index"
        text_series = pd.Series(text)
        original_label = None
        predicted_label = None
    return {
        "text": text_series.loc[0],
        "model": BEST_MODEL_NAME,
        "predicted_label": predicted_label,
        "original_label": original_label
    }


@app.get("/api/{model}/performance")
def get_model_performance(model: str):
    performance = pd.read_csv(
        model_performance_df_path,
        usecols=['Unnamed: 0', model],
        index_col=0
    )
    return {"model_info": performance}


@app.post("/api/{model}/predict/")
def model_predict_for_text(model: str, text: Text):
    path = model_path_mapping[model]
    with open(path, 'rb') as file:
        model = pickle.load(file)
    text_series = pd.Series(text.text)
    predicted_label = model.predict(text_series)[0]
    return {"text": text_series.loc[0], "predicted_label": predicted_label}


@app.post("/api/{model}/predict/{text_id}/")
def model_predict_for_text_id(model: str, text_id: int):
    data = pd.read_csv(DATA_PATH)
    path = model_path_mapping[model]
    model_name = model_names_reverse_mapping[model]
    try:
        res = data.loc[text_id]
        text = res["text"]
        original_label = res["class"]
        with open(path, 'rb') as file:
            model = pickle.load(file)
        text_series = pd.Series(text)
        predicted_label = model.predict(text_series)[0]
    except KeyError:
        text = "No text with such index"
        text_series = pd.Series(text)
        original_label = None
        predicted_label = None
    return {
        "text": text_series.loc[0],
        "model": model_name,
        "predicted_label": predicted_label,
        "original_label": original_label
    }
