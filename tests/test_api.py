"""
MapTree IA - Testes de fumaça (smoke tests) da API FastAPI.

Verificam que os endpoints respondem e devolvem um contrato coerente.
Rodar: pytest -q
"""

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _payload(**overrides):
    base = {
        "age_years": 8,
        "pruning_count": 2,
        "species": {"height_average": 10.0, "growth_rate_k": 0.12},
        "soil": {"depth": 1.2, "inclination": 5, "quality": "REGULAR", "coverage": 0.4},
        "annual_rainfall": 650,
        "altitude": 280,
        "avg_temperature": 28,
        "has_fertilization": False,
        "has_irrigation": False,
        "nearby_trees_count": 2,
        "avg_neighbor_distance": 8.0,
    }
    base.update(overrides)
    return base


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"] == "von_bertalanffy_deterministic"


def test_predict_height_contrato():
    r = client.post("/predict-height", json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["estimated_height_m"] <= 10.3
    assert body["annual_growth_m"] >= 0
    assert body["canopy_shape"] in ("WIDE", "SQUARE", "TALL")
    assert body["phyllotaxis"]["golden_angle_deg"] == 137.5


def test_predict_wire_risk_contrato():
    r = client.post("/predict-wire-risk", json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["risk_status"] in ("NORMAL", "UNDER_OBSERVATION", "TO_PRUNE", "CRITICAL")
    assert body["estimated_height_m"] <= body["wire_height_m"] * 2
    assert isinstance(body["will_reach_wire"], bool)


def test_determinismo_mesma_entrada_mesma_saida():
    p = _payload()
    r1 = client.post("/predict-wire-risk", json=p).json()
    r2 = client.post("/predict-wire-risk", json=p).json()
    assert r1 == r2


def test_arvore_acima_do_fio_e_critica():
    p = _payload(age_years=60, species={"height_average": 15.0, "growth_rate_k": 0.3}, wire_height=6.5)
    body = client.post("/predict-wire-risk", json=p).json()
    assert body["risk_status"] == "CRITICAL"
    assert body["will_reach_wire"] is True


def test_optimize_route_prioriza_poda():
    payload = {
        "trees": [
            {"tree_id": "a", "latitude": -3.7, "longitude": -40.0, "status": "NORMAL"},
            {"tree_id": "b", "latitude": -3.71, "longitude": -40.01, "status": "CRITICAL"},
            {"tree_id": "c", "latitude": -3.72, "longitude": -40.02, "status": "TO_PRUNE"},
        ]
    }
    body = client.post("/optimize-route", json=payload).json()
    ids = [s["tree_id"] for s in body["route"]]
    assert "a" not in ids  # NORMAL não entra na rota
    assert body["route"][0]["tree_id"] == "b"  # CRITICAL primeiro
