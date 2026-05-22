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
import math
import os
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# ─── Importar funções do modelo de crescimento ───────────────────────────────

from generate_data import (
    GOLDEN_ANGLE,
    simular_crescimento_fibonacci,
    fibonacci_growth_modifier,
    canopy_competition_modifier,
    classify_risk,
    WIRE_HEIGHT
)

# ─── Carregar modelos ─────────────────────────────────────────────────────────

MODELS_PATH = "models"

def load_models():
    """Carrega modelos treinados e as listas de features salvas pelo treino."""
    try:
        return {
            "height":          joblib.load(f"{MODELS_PATH}/model_height.pkl"),
            "annual_growth":   joblib.load(f"{MODELS_PATH}/model_annual_growth.pkl"),
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
    """Dados de solo usados para ajustar crescimento e vigor."""
    depth: float = Field(..., description="Profundidade do solo em metros")
    inclination: float = Field(..., description="Inclinação em graus")
    quality: str = Field(..., description="GOOD, REGULAR ou BAD")
    coverage: float = Field(..., description="Cobertura/impermeabilização 0.0 a 1.0")

class SpeciesData(BaseModel):
    """Dados botanicos minimos da especie usados pelo modelo."""
    height_average: float = Field(..., description="Altura média da espécie em metros")
    growth_rate_k: Optional[float] = Field(0.12, description="Taxa de crescimento k (padrão 0.12)")

class TreePredictRequest(BaseModel):
    """Payload principal enviado pelo backend/NestJS para prever crescimento."""
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
    avg_neighbor_distance: Optional[float] = Field(10.0, description="Distância média em metros")

    # Configuração
    wire_height: Optional[float] = Field(WIRE_HEIGHT, description="Altura do fio em metros")


class TreeLocation(BaseModel):
    """Arvore candidata para rota de poda."""
    tree_id: str
    latitude: float
    longitude: float
    status: str
    priority: Optional[int] = None


class RouteOptimizeRequest(BaseModel):
    """Lista de arvores e ponto inicial opcional para otimizar rota."""
    trees: list[TreeLocation]
    start_lat: Optional[float] = None
    start_lng: Optional[float] = None


# ─── Lógica de features ───────────────────────────────────────────────────────

def build_features(req: TreePredictRequest) -> dict:
    """
    Monta o vetor de features exatamente na ordem esperada pelos modelos.

    Se BASE_FEATURES mudar no train_models.py, esta funcao precisa produzir os
    mesmos campos. Caso contrario a API pode prever com dados trocados.
    """

    soil = req.soil or SoilData(depth=1.0, inclination=5.0, quality="REGULAR", coverage=0.3)

    fib_mod = fibonacci_growth_modifier(req.pruning_count)
    neighbor_count = req.nearby_trees_count or 0
    neighbor_distance = req.avg_neighbor_distance if neighbor_count > 0 else 999
    competition = canopy_competition_modifier(neighbor_count, neighbor_distance or 999)

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
        "wire_height": req.wire_height,
        "fibonacci_modifier": fib_mod,
        "canopy_ratio": competition["canopy_ratio"],
    }


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Retorna status basico da API e confirma se os modelos foram carregados."""
    return {
        "status": "ok",
        "models_loaded": models is not None,
        "golden_angle": GOLDEN_ANGLE,
        "wire_height_default": WIRE_HEIGHT,
    }


@app.post("/predict-height")
def predict_height(req: TreePredictRequest):
    """
    Estima altura atual, crescimento anual e altura projetada em 1 ano.

    Use este endpoint quando o front/backend so precisa da estimativa botanica,
    sem calcular risco contra fio.
    """
    if not models:
        raise HTTPException(503, "Modelos não carregados. Execute train_models.py primeiro.")

    features = build_features(req)
    feature_names = models["base_features"]
    X = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names)

    height = float(models["height"].predict(X)[0])
    annual_growth = max(0.0, float(models["annual_growth"].predict(X)[0]))

    return {
        "estimated_height_m": round(height, 2),
        "annual_growth_m": round(annual_growth, 3),
        "height_next_year_m": round(height + annual_growth, 2),
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
    Predicao completa: altura, crescimento anual, prazo ate fio e risco.

    Este e o endpoint principal para operacao. Ele combina modelo treinado com
    regra deterministica de seguranca para evitar que a IA rebaixe risco de uma
    arvore que ja esta perto demais ou acima do fio.
    """
    if not models:
        raise HTTPException(503, "Modelos não carregados. Execute train_models.py primeiro.")

    features = build_features(req)
    feature_names = models["base_features"]
    X_base = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names)

    # 1. Estimar altura
    estimated_height = float(models["height"].predict(X_base)[0])
    annual_growth = max(0.0, float(models["annual_growth"].predict(X_base)[0]))

    # 2. Prever tempo ate o fio. A simulacao matematica define se a especie
    # ainda tem potencial biologico para chegar no fio; o modelo estima o prazo.
    simulacao_fib = simular_crescimento_fibonacci(
        altura_atual=estimated_height,
        altura_fio=req.wire_height,
        taxa_k=req.species.growth_rate_k,
        altura_max=req.species.height_average,
        modifier=features["fibonacci_modifier"],
    )

    if estimated_height >= req.wire_height:
        days_to_wire = 0
        will_reach_wire = True
    elif simulacao_fib["meses_ate_o_fio"] is None:
        days_to_wire = None
        will_reach_wire = False
    else:
        X_wire = X_base.copy()
        predicted_days = float(models["wire"].predict(X_wire)[0])
        simulated_days = simulacao_fib["meses_ate_o_fio"] * 30
        days_to_wire = max(0, min(predicted_days, simulated_days * 1.5))
        will_reach_wire = True

    # 3. Classificar risco
    risk_features = models["risk_features"]
    features_with_height = {**features, "estimated_height": estimated_height}
    X_risk = pd.DataFrame([[features_with_height[f] for f in risk_features]], columns=risk_features)
    risk_encoded = models["risk"].predict(X_risk)[0]
    model_risk_label = models["risk_encoder"].inverse_transform([risk_encoded])[0]
    risk_label = classify_risk(
        estimated_height,
        req.wire_height,
        -1 if days_to_wire is None else days_to_wire,
        1 if will_reach_wire else 0,
    )

    # 4. Alerta
    alert_message = None
    if risk_label == "CRITICAL":
        alert_message = "⚠️ Árvore já atingiu ou ultrapassou a altura do fio! Poda urgente."
    elif will_reach_wire and days_to_wire is not None and days_to_wire < 180:
        alert_message = f"⚠️ Árvore atingirá o fio em ~{int(days_to_wire/30)} meses. Agendar poda."
    elif will_reach_wire and days_to_wire is not None and days_to_wire < 365:
        alert_message = f"🔍 Árvore atingirá o fio em ~{int(days_to_wire/30)} meses. Monitorar."

    return {
        "months_to_wire_ai": round(days_to_wire / 30, 1) if will_reach_wire else None,
        "months_to_wire_mathematical": simulacao_fib["meses_ate_o_fio"],
        "months_to_wire_simulated": simulacao_fib["meses_ate_o_fio"],
        "estimated_height_m": round(estimated_height, 2),
        "annual_growth_m": round(annual_growth, 3),
        "height_next_year_m": round(estimated_height + annual_growth, 2),
        "wire_height_m": req.wire_height,
        "will_reach_wire": will_reach_wire,
        "days_to_wire": round(days_to_wire) if will_reach_wire else None,
        "months_to_wire": round(days_to_wire / 30, 1) if will_reach_wire else None,
        "risk_status": risk_label,
        "risk_status_model": model_risk_label,
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
    Calcula rota de visita para poda por heuristica gulosa.

    A rota prioriza CRITICAL, TO_PRUNE e UNDER_OBSERVATION. Arvores NORMAL nao
    entram na rota. Para muitas arvores, trocar por OR-Tools/TSP com janelas.
    """
    status_priority = {"CRITICAL": 4, "TO_PRUNE": 3, "UNDER_OBSERVATION": 2}
    trees_to_prune = [t for t in req.trees if t.status in status_priority]

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
            -(t.priority if t.priority is not None else status_priority.get(t.status, 1)),
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
    """Converte canopy_ratio numerico em etiqueta legivel para a UI."""
    if ratio > 1.2:
        return "WIDE"       # copa larga — árvore crescendo livre
    elif ratio >= 0.9:
        return "SQUARE"     # copa equilibrada
    else:
        return "TALL"       # copa estreita e alta — competição lateral


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Calcula distancia em metros entre dois pontos geograficos."""
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
