"""
MapTree AI - FastAPI Microserviço
Endpoints que o NestJS chama via HTTP para predições de crescimento de árvores.

Motor de predição: modelo botânico de Von Bertalanffy (determinístico).
Não usa modelos de ML treinados. Com os dados reais atuais (altura é um alvo
CALCULADO, não medido em campo), um modelo de ML só imitaria a fórmula e
adicionaria ruído. A fórmula é estável, explicável e roda em qualquer máquina
sem etapa de treino. Quando houver altura medida em campo, dá para reavaliar ML.

Rotas:
  POST /predict-height       → estima altura atual da árvore
  POST /predict-wire-risk    → prevê prazo até atingir o fio + classificação de risco
  POST /optimize-route       → retorna rota de poda (heurística do vizinho mais próximo)
  GET  /health               → healthcheck
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import math

# ─── Funções do motor de crescimento (determinístico) ────────────────────────

from generate_data import (
    GOLDEN_ANGLE,
    simular_crescimento_mensal,
    pruning_vigor_modifier,
    phyllotaxis_canopy,
    canopy_competition_modifier,
    classify_risk,
    soil_modifier,
    climate_modifier,
    management_modifier,
    estimate_height,
    estimate_annual_growth,
    WIRE_HEIGHT,
)

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MapTree AI",
    description="Microserviço de IA para análise de crescimento de árvores urbanas",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3500", "http://76.13.227.139:4000/api", "https://maptree.vercel.app"],  # Porta do Vue
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Horizonte operacional para poda (meses). Acima disso, tratamos como longo prazo.
OPERATIONAL_HORIZON_MONTHS = 120

# ─── Schemas de entrada ───────────────────────────────────────────────────────

class SoilData(BaseModel):
    """Dados de solo usados para ajustar crescimento e vigor."""
    depth: float = Field(..., description="Profundidade do solo em metros")
    inclination: float = Field(..., description="Inclinação em graus")
    quality: str = Field(..., description="GOOD, REGULAR ou BAD")
    coverage: float = Field(..., description="Cobertura/impermeabilização 0.0 a 1.0")


class SpeciesData(BaseModel):
    """Dados botânicos mínimos da espécie usados pelo modelo."""
    height_average: float = Field(..., description="Altura média/máxima da espécie em metros")
    growth_rate_k: Optional[float] = Field(0.12, description="Taxa de crescimento k (padrão 0.12)")


class TreePredictRequest(BaseModel):
    """Payload principal enviado pelo backend/NestJS para prever crescimento."""
    # Planta
    age_years: float = Field(..., description="Idade da árvore em anos")
    pruning_count: int = Field(0, description="Quantidade de podas realizadas")
    status: Optional[str] = Field("NORMAL", description="Status atual da árvore")
    branch_count: Optional[int] = Field(
        None, description="Número de galhos observados (opcional, para formato de copa)"
    )

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
    """Árvore candidata para rota de poda."""
    tree_id: str
    latitude: float
    longitude: float
    status: str
    priority: Optional[int] = None


class RouteOptimizeRequest(BaseModel):
    """Lista de árvores e ponto inicial opcional para otimizar rota."""
    trees: list[TreeLocation]
    start_lat: Optional[float] = None
    start_lng: Optional[float] = None


# ─── Núcleo de predição (determinístico) ──────────────────────────────────────

def compute_growth(req: TreePredictRequest) -> dict:
    """
    Calcula altura atual, crescimento anual, modificadores e prazo até o fio
    usando o modelo de Von Bertalanffy. Função pura e testável: mesma entrada,
    mesma saída.
    """
    soil = req.soil or SoilData(depth=1.0, inclination=5.0, quality="REGULAR", coverage=0.3)

    k_species = float(req.species.growth_rate_k or 0.12)

    # Modificadores de crescimento (cada um documentado em generate_data.py).
    s_mod = soil_modifier(soil.depth, soil.inclination, soil.quality, soil.coverage)
    c_mod = climate_modifier(req.annual_rainfall, req.altitude, req.avg_temperature)
    m_mod = management_modifier(bool(req.has_fertilization), bool(req.has_irrigation))
    pruning_mod = pruning_vigor_modifier(req.pruning_count)

    neighbor_count = int(req.nearby_trees_count or 0)
    neighbor_distance = float(req.avg_neighbor_distance) if neighbor_count > 0 else 999.0
    comp = canopy_competition_modifier(neighbor_count, neighbor_distance)

    total_modifier = s_mod * c_mod * m_mod * pruning_mod * comp["growth_modifier"]

    estimated_height = estimate_height(
        req.age_years, req.species.height_average, k_species, total_modifier
    )
    annual_growth = max(
        0.0,
        estimate_annual_growth(estimated_height, req.species.height_average, k_species, total_modifier),
    )

    # Filotaxia: formato/arranjo da copa (ângulo áureo). Separado da altura.
    canopy_shape = phyllotaxis_canopy(req.branch_count, req.pruning_count)

    return {
        "estimated_height": estimated_height,
        "annual_growth": annual_growth,
        "k_species": k_species,
        "modifiers": {
            "soil": s_mod,
            "climate": c_mod,
            "management": m_mod,
            "pruning": pruning_mod,
            "competition": comp["growth_modifier"],
            "total": round(total_modifier, 4),
        },
        "canopy_competition": comp,
        "phyllotaxis": canopy_shape,
    }


def compute_wire_timeline(estimated_height: float, req: TreePredictRequest, total_modifier: float) -> dict:
    """
    Decide se e quando a árvore atinge o fio, usando a simulação mensal de
    Von Bertalanffy. Aplica horizonte operacional de poda.
    """
    k_species = float(req.species.growth_rate_k or 0.12)

    if estimated_height >= req.wire_height:
        return {"days_to_wire": 0, "will_reach_wire": True, "months_math": 0}

    sim = simular_crescimento_mensal(
        altura_atual=estimated_height,
        altura_fio=req.wire_height,
        taxa_k=k_species,
        altura_max=req.species.height_average,
        modifier=total_modifier,
    )
    months = sim["meses_ate_o_fio"]

    if months is None or months > OPERATIONAL_HORIZON_MONTHS:
        return {"days_to_wire": None, "will_reach_wire": False, "months_math": months}

    return {"days_to_wire": months * 30, "will_reach_wire": True, "months_math": months}


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Retorna status básico da API."""
    return {
        "status": "ok",
        "engine": "von_bertalanffy_deterministic",
        "golden_angle": GOLDEN_ANGLE,
        "wire_height_default": WIRE_HEIGHT,
    }


@app.post("/predict-height")
def predict_height(req: TreePredictRequest):
    """
    Estima altura atual, crescimento anual e altura projetada em 1 ano.

    Use quando o front/backend só precisa da estimativa botânica, sem risco de fio.
    """
    g = compute_growth(req)
    height = g["estimated_height"]
    annual_growth = g["annual_growth"]
    canopy = g["canopy_competition"]
    phyllo = g["phyllotaxis"]

    return {
        "estimated_height_m": round(height, 2),
        "annual_growth_m": round(annual_growth, 3),
        "height_next_year_m": round(min(req.species.height_average * 1.03, height + annual_growth), 2),
        "canopy_ratio": canopy["canopy_ratio"],
        "canopy_shape": _canopy_shape_label(canopy["canopy_ratio"]),
        "modifiers_used": g["modifiers"],
        "phyllotaxis": phyllo,
    }


@app.post("/predict-wire-risk")
def predict_wire_risk(req: TreePredictRequest):
    """
    Predição completa: altura, crescimento anual, prazo até fio e risco.

    Endpoint principal para operação. Combina a estimativa botânica com a regra
    determinística de segurança `classify_risk`, que não rebaixa risco de árvore
    já próxima ou acima do fio.
    """
    g = compute_growth(req)
    estimated_height = g["estimated_height"]
    annual_growth = g["annual_growth"]
    total_modifier = g["modifiers"]["total"]
    canopy = g["canopy_competition"]
    phyllo = g["phyllotaxis"]

    wire = compute_wire_timeline(estimated_height, req, total_modifier)
    days_to_wire = wire["days_to_wire"]
    will_reach_wire = wire["will_reach_wire"]

    risk_label = classify_risk(
        estimated_height,
        req.wire_height,
        -1 if days_to_wire is None else days_to_wire,
        1 if will_reach_wire else 0,
    )

    # Alerta
    alert_message = None
    if risk_label == "CRITICAL":
        alert_message = "⚠️ Árvore já atingiu ou ultrapassou a altura do fio! Poda urgente."
    elif will_reach_wire and days_to_wire is not None and days_to_wire < 180:
        alert_message = f"⚠️ Árvore atingirá o fio em ~{int(days_to_wire / 30)} meses. Agendar poda."
    elif will_reach_wire and days_to_wire is not None and days_to_wire < 365:
        alert_message = f"🔍 Árvore atingirá o fio em ~{int(days_to_wire / 30)} meses. Monitorar."

    months_to_wire = round(days_to_wire / 30, 1) if (will_reach_wire and days_to_wire is not None) else None

    return {
        "estimated_height_m": round(estimated_height, 2),
        "annual_growth_m": round(annual_growth, 3),
        "height_next_year_m": round(min(req.species.height_average * 1.03, estimated_height + annual_growth), 2),
        "wire_height_m": req.wire_height,
        "will_reach_wire": will_reach_wire,
        "days_to_wire": int(round(days_to_wire)) if (will_reach_wire and days_to_wire is not None) else None,
        "months_to_wire": months_to_wire,
        "risk_status": risk_label,
        "alert": alert_message,
        "canopy": {
            "shape": _canopy_shape_label(canopy["canopy_ratio"]),
            "ratio_width_height": canopy["canopy_ratio"],
            "nearby_competition": int(req.nearby_trees_count or 0) > 2,
        },
        "phyllotaxis": {
            "golden_angle_deg": phyllo["golden_angle_deg"],
            "branch_whorls": phyllo["branch_whorls"],
            "fibonacci_stage": phyllo["fibonacci_stage"],
            "symmetry_index": phyllo["symmetry_index"],
        },
        "modifiers_used": g["modifiers"],
    }


@app.post("/optimize-route")
def optimize_route(req: RouteOptimizeRequest):
    """
    Calcula rota de visita para poda por heurística gulosa (vizinho mais próximo).

    Prioriza CRITICAL, TO_PRUNE e UNDER_OBSERVATION. Árvores NORMAL não entram.
    Para muitas árvores, considerar OR-Tools/TSP com janelas de tempo.
    """
    status_priority = {"CRITICAL": 4, "TO_PRUNE": 3, "UNDER_OBSERVATION": 2}
    trees_to_prune = [t for t in req.trees if t.status in status_priority]

    if not trees_to_prune:
        return {"message": "Nenhuma árvore para poda encontrada", "route": []}

    start_lat = req.start_lat or trees_to_prune[0].latitude
    start_lng = req.start_lng or trees_to_prune[0].longitude

    remaining = trees_to_prune.copy()
    route = []
    current_lat, current_lng = start_lat, start_lng

    while remaining:
        best = min(remaining, key=lambda t: (
            -(t.priority if t.priority is not None else status_priority.get(t.status, 1)),
            _haversine(current_lat, current_lng, t.latitude, t.longitude),
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
        "route": route,
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _canopy_shape_label(ratio: float) -> str:
    """Converte canopy_ratio numérico em etiqueta legível para a UI."""
    if ratio > 1.2:
        return "WIDE"       # copa larga — árvore crescendo livre
    elif ratio >= 0.9:
        return "SQUARE"     # copa equilibrada
    else:
        return "TALL"       # copa estreita e alta — competição lateral


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Calcula distância em metros entre dois pontos geográficos."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
