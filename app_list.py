from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import List

from House_pricing.bibliotecas_e_modelos.KNN_teste import KNN_felipe

app = FastAPI()

# carregar dataset apenas uma vez
dataset = pd.read_csv("data/train_data.csv")
variables =  ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "sqft_above", "lat", "long", "house_age"]
K = 5


class House(BaseModel):

    sqft_living: float
    bedrooms: int
    bathrooms: float
    sqft_lot: float
    floors: int
    sqft_above: int
    lat: float
    long: float
    house_age: int


@app.post("/predict")
def predict(data: List[House]):


    sample_df = pd.DataFrame([d.model_dump() for d in data])

    prices = list()
    for i in sample_df.index:
        prices.append(KNN_felipe(t_data = dataset, variables = variables, sample = sample_df.iloc[i], neighborhood= K))

    return {"predicted_price": prices}