from fastapi import FastAPI
import json
import pandas as pd
import bibliotecas_e_modelos.ledo as ld

# ===== 1. carregar modelo =====
with open("bibliotecas_e_modelos/modelo_XGB_V2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

modelo = data["model"]
base_score = data["base_score"]

# ===== 2. features =====
features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "house_age"]

# ===== 3. função de predição =====
def predict_xgb(sample_dict):
    sample_series = pd.Series(sample_dict)
    sample_series = sample_series[features]
    return ld.result_func_XGBoos(modelo, base_score, sample_series)

# ===== 4. API =====
app = FastAPI()

@app.get("/")
def home():
    return {"mensagem": "API funcionando"}

@app.post("/predict")
def predict(dados: dict):
    previsao = predict_xgb(dados)
    return {"preco_previsto": previsao}