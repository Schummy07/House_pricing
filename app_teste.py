from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from KNN_teste import KNN_felipe

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
def predict(data: House):

    sample = data.model_dump()

    sample_df = pd.DataFrame([sample])

    price = KNN_felipe(
        t_data=dataset,
        variables=variables,
        sample=sample_df,
        neighborhood=K
    )

    return {"predicted_price": price}