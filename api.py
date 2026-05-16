"""
MapTree AI - FastAPI Microserviço
Endpoints que o NestJS chama via HTTP para predições de crescimento de árvores.

Rotas:
  POST /predict-height       → estima altura atual da árvore
  POST /predict-wire-risk    → prevê dias até atingir o fio + classificação de risco
  POST /optimize-route       → retorna rota ótima de poda (TSP)
  GET  /health               → healthcheck
"""


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import math
import os
from fastapi.middleware.cors import CORSMiddleware
from generate_data import simular_crescimento_fibonacci

# ─── Importar funções do modelo de crescimento ───────────────────────────────

from generate_data import (
    simular_crescimento_fibonacci,
    soil_modifier,
    climate_modifier,
    generate_dataset,
    WIRE_HEIGHT
)

# ─── Carregar modelos ─────────────────────────────────────────────────────────

MODELS_PATH = "models"

def load_models():
    try:
        return {
            "height":          joblib.load(f"{MODELS_PATH}/model_height.pkl"),
            "wire":            joblib.load(f"{MODELS_PATH}/model_wire_days.pkl"),
            "risk":            joblib.load(f"{MODELS_PATH}/model_risk.pkl"),
            "risk_encoder":    joblib.load(f"{MODELS_PATH}/risk_label_encoder.pkl"),
            "base_features":   joblib.load(f"{MODELS_PATH}/base_features.pkl"),
            "risk_features":   joblib.load(f"{MODELS_PATH}/risk_features.pkl"),
        }
    except FileNotFoundError:
        return None

models = load_models()

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MapTree AI",
    description="Microserviço de IA para análise de crescimento de árvores urbanas",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3500"], # Porta do Vue
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas de entrada ───────────────────────────────────────────────────────

class SoilData(BaseModel):
    depth: float = Field(..., description="Profundidade do solo em metros")
    inclination: float = Field(..., description="Inclinação em graus")
    quality: str = Field(..., description="GOOD, REGULAR ou BAD")
    coverage: float = Field(..., description="Cobertura/impermeabilização 0.0 a 1.0")

class SpeciesData(BaseModel):
    height_average: float = Field(..., description="Altura média da espécie em metros")
    growth_rate_k: Optional[float] = Field(0.12, description="Taxa de crescimento k (padrão 0.12)")

class TreePredictRequest(BaseModel):
    # Planta
    age_years: float = Field(..., description="Idade da árvore em anos")
    pruning_count: int = Field(0, description="Quantidade de podas realizadas")
    status: Optional[str] = Field("NORMAL", description="Status atual da árvore")

    # Espécie
    species: SpeciesData

    # Solo (opcional — usa padrão se não informado)
    soil: Optional[SoilData] = None

    # Clima
    annual_rainfall: Optional[float] = Field(700.0, description="Chuva anual em mm")
    altitude: Optional[float] = Field(300.0, description="Altitude em metros")
    avg_temperature: Optional[float] = Field(27.0, description="Temperatura média em °C")

    # Manejo
    has_fertilization: Optional[bool] = Field(False)
    has_irrigation: Optional[bool] = Field(False)

    # Vizinhança
    nearby_trees_count: Optional[int] = Field(0, description="Árvores vizinhas próximas")
    avg_neighbor_distance: Optional[float] = Field(0.0, description="Distância média em metros")

    # Configuração
    wire_height: Optional[float] = Field(WIRE_HEIGHT, description="Altura do fio em metros")


class TreeLocation(BaseModel):
    tree_id: str
    latitude: float
    longitude: float
    status: str
    priority: Optional[int] = 1


class RouteOptimizeRequest(BaseModel):
    trees: list[TreeLocation]
    start_lat: Optional[float] = None
    start_lng: Optional[float] = None


# ─── Lógica de features ───────────────────────────────────────────────────────

def build_features(req: TreePredictRequest) -> dict:
    """Monta o vetor de features a partir do request."""

    soil = req.soil or SoilData(depth=1.0, inclination=5.0, quality="REGULAR", coverage=0.3)

    fib_mod = fibonacci_growth_modifier(req.pruning_count)
    competition = canopy_competition_modifier(
        req.nearby_trees_count or 0,
        req.avg_neighbor_distance or 999
    )

    return {
        "age_years": req.age_years,
        "species_height_max": req.species.height_average,
        "species_k": req.species.growth_rate_k,
        "pruning_count": req.pruning_count,
        "soil_depth": soil.depth,
        "soil_inclination": soil.inclination,
        "soil_quality_enc": {"GOOD": 2, "REGULAR": 1, "BAD": 0}.get(soil.quality, 1),
        "soil_coverage": soil.coverage,
        "annual_rainfall": req.annual_rainfall,
        "altitude": req.altitude,
        "avg_temperature": req.avg_temperature,
        "has_fertilization": 1 if req.has_fertilization else 0,
        "has_irrigation": 1 if req.has_irrigation else 0,
        "nearby_trees_count": req.nearby_trees_count,
        "avg_neighbor_distance": req.avg_neighbor_distance,
        "fibonacci_modifier": fib_mod,
        "canopy_ratio": competition["canopy_ratio"],
    }


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": models is not None,
        "golden_angle": GOLDEN_ANGLE,
        "wire_height_default": WIRE_HEIGHT,
    }


@app.post("/predict-height")
def predict_height(req: TreePredictRequest):
    """
    Estima a altura atual da árvore com base em todos os fatores.
    """
    if not models:
        raise HTTPException(503, "Modelos não carregados. Execute train_models.py primeiro.")

    features = build_features(req)
    feature_names = models["base_features"]
    X = np.array([[features[f] for f in feature_names]])

    height = float(models["height"].predict(X)[0])

    return {
        "estimated_height_m": round(height, 2),
        "canopy_ratio": features["canopy_ratio"],
        "canopy_shape": _canopy_shape_label(features["canopy_ratio"]),
        "fibonacci_modifier": round(features["fibonacci_modifier"], 4),
        "modifiers_used": {
            "fibonacci": round(features["fibonacci_modifier"], 3),
            "canopy_ratio": round(features["canopy_ratio"], 3),
        }
    }


@app.post("/predict-wire-risk")
def predict_wire_risk(req: TreePredictRequest):
    """
    Predição completa: altura estimada, dias até o fio e classificação de risco.
    Este é o endpoint principal que o NestJS deve usar.
    """
    if not models:
        raise HTTPException(503, "Modelos não carregados. Execute train_models.py primeiro.")

    features = build_features(req)
    feature_names = models["base_features"]
    X_base = np.array([[features[f] for f in feature_names]])

    # 1. Estimar altura
    estimated_height = float(models["height"].predict(X_base)[0])

    # 2. Prever dias até o fio
    if estimated_height >= req.wire_height:
        days_to_wire = 0
        will_reach_wire = True
    else:
        X_wire = X_base.copy()
        predicted_days = float(models["wire"].predict(X_wire)[0])
        days_to_wire = max(0, predicted_days)
        will_reach_wire = estimated_height < req.species.height_average * 0.95

    # 3. Classificar risco
    risk_features = models["risk_features"]
    features_with_height = {**features, "estimated_height": estimated_height}
    X_risk = np.array([[features_with_height[f] for f in risk_features]])
    risk_encoded = models["risk"].predict(X_risk)[0]
    risk_label = models["risk_encoder"].inverse_transform([risk_encoded])[0]

    # 4. Alerta
    alert_message = None
    if days_to_wire == 0:
        alert_message = "⚠️ Árvore já atingiu ou ultrapassou a altura do fio! Poda urgente."
    elif will_reach_wire and days_to_wire < 180:
        alert_message = f"⚠️ Árvore atingirá o fio em ~{int(days_to_wire/30)} meses. Agendar poda."
    elif will_reach_wire and days_to_wire < 365:
        alert_message = f"🔍 Árvore atingirá o fio em ~{int(days_to_wire/30)} meses. Monitorar."
        
    simulacao_fib = simular_crescimento_fibonacci(
        altura_atual=estimated_height,
        altura_fio=req.wire_height, 
        taxa_k=req.species.growth_rate_k
    )

    return {
        "months_to_wire_ai": round(days_to_wire / 30, 1) if will_reach_wire else None,
        "months_to_wire_mathematical": simulacao_fib["meses_ate_o_fio"],
        "months_to_wire_simulated": simulacao_fib["meses_ate_o_fio"],
        "estimated_height_m": round(estimated_height, 2),
        "wire_height_m": req.wire_height,
        "will_reach_wire": will_reach_wire,
        "days_to_wire": round(days_to_wire) if will_reach_wire else None,
        "months_to_wire": round(days_to_wire / 30, 1) if will_reach_wire else None,
        "risk_status": risk_label,
        "alert": alert_message,
        "canopy": {
            "shape": _canopy_shape_label(features["canopy_ratio"]),
            "ratio_width_height": features["canopy_ratio"],
            "nearby_competition": req.nearby_trees_count > 2,
        },
        "fibonacci_info": {
            "pruning_cycle_index": req.pruning_count % 11,
            "growth_modifier": round(features["fibonacci_modifier"], 4),
            "golden_angle_deg": GOLDEN_ANGLE,
        }
    }


@app.post("/optimize-route")
def optimize_route(req: RouteOptimizeRequest):
    """
    Calcula a rota ótima de visita para poda (algoritmo guloso simples).
    Para casos maiores (50+ árvores), usar TSP completo.
    """
    trees_to_prune = [t for t in req.trees if t.status in ("TO_PRUNE", "UNDER_OBSERVATION", "NORMAL")]

    if not trees_to_prune:
        return {"message": "Nenhuma árvore para poda encontrada", "route": []}

    # Ponto de partida
    start_lat = req.start_lat or trees_to_prune[0].latitude
    start_lng = req.start_lng or trees_to_prune[0].longitude

    # Algoritmo nearest neighbor (greedy)
    remaining = trees_to_prune.copy()
    route = []
    current_lat, current_lng = start_lat, start_lng

    while remaining:
        # Encontra a árvore mais próxima com maior prioridade
        best = min(remaining, key=lambda t: (
            -t.priority,  # maior prioridade primeiro
            _haversine(current_lat, current_lng, t.latitude, t.longitude)
        ))
        distance = _haversine(current_lat, current_lng, best.latitude, best.longitude)
        route.append({
            "tree_id": best.tree_id,
            "latitude": best.latitude,
            "longitude": best.longitude,
            "status": best.status,
            "distance_from_prev_m": round(distance, 1),
        })
        current_lat, current_lng = best.latitude, best.longitude
        remaining.remove(best)

    total_dist = sum(s["distance_from_prev_m"] for s in route)

    return {
        "total_trees": len(route),
        "total_distance_m": round(total_dist),
        "total_distance_km": round(total_dist / 1000, 2),
        "route": route
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _canopy_shape_label(ratio: float) -> str:
    if ratio > 1.2:
        return "WIDE"       # copa larga — árvore crescendo livre
    elif ratio >= 0.9:
        return "SQUARE"     # copa equilibrada
    else:
        return "TALL"       # copa estreita e alta — competição lateral


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Distância em metros entre dois pontos geográficos."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
