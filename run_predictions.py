"""
MapTree IA - Rodar predição em lote sobre dados reais da pasta.

Lê um arquivo de árvores reais (padrão: arvores_mapeadas.json), roda o motor
determinístico (Von Bertalanffy) em cada uma e salva um JSON novo com as
predições ATUALIZADAS (sem o antigo fibonacci_info, com phyllotaxis separada).

Serve para:
  - validar a IA sobre os seus próprios dados, sem precisar do backend;
  - gerar um arquivo pronto para o front/backend consumir.

Uso:
    python run_predictions.py
    python run_predictions.py --input arvores_mapeadas.json --output arvores_preditas.json
    python run_predictions.py --wire-height 8.0

Observação: solo e clima vêm do catálogo real (enriched_trees.csv) quando a
espécie/local existe lá; caso contrário o script usa valores neutros e sinaliza
isso em "data_quality", para você saber que aquela árvore precisa de coleta.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_data import (
    WIRE_HEIGHT,
    ALLOWED_TREE_GENERA,
    choose_wire_context,
    get_species_profile,
    soil_modifier,
    climate_modifier,
    management_modifier,
    pruning_vigor_modifier,
    canopy_competition_modifier,
    phyllotaxis_canopy,
    estimate_height,
    estimate_annual_growth,
    simular_crescimento_mensal,
    classify_risk,
    _build_catalog,
    _norm_text,
    _env_key,
)

BASE_DIR = Path(__file__).resolve().parent

# Ambiente neutro: usado quando não há dado real de solo/clima para a árvore.
NEUTRAL_ENV = {
    "annual_rainfall": 700.0,
    "avg_temperature": 27.0,
    "altitude": 300.0,
    "soil_depth": 1.0,
    "soil_inclination": 5.0,
    "soil_quality": "REGULAR",
    "soil_coverage": 0.3,
}

OPERATIONAL_HORIZON_MONTHS = 120


def lookup_environment(scientific: str | None, family: str | None, genus: str | None,
                       lat: float | None, lng: float | None) -> tuple[dict, str]:
    """Procura solo/clima reais no catálogo. Retorna (env, origem)."""
    catalog = _build_catalog()
    species_key = _norm_text(scientific)
    genus_key = _norm_text(genus) or (_norm_text(str(scientific or "").split()[0]) if species_key else "")
    family_key = _norm_text(family)
    location_key = _env_key(lat, lng)

    for key, bucket, origem in (
        (location_key, catalog.env_by_location, "local_real"),
        (species_key, catalog.env_by_species, "especie_real"),
        (genus_key, catalog.env_by_genus, "genero_real"),
        (family_key, catalog.env_by_family, "familia_real"),
    ):
        if key and key in bucket:
            return bucket[key], origem

    return NEUTRAL_ENV, "neutro_estimado"


def predict_one(tree: dict, wire_height_default: float) -> dict:
    """Roda o motor determinístico em uma árvore e devolve a predição."""
    scientific = tree.get("name") or tree.get("scientific")
    family = tree.get("family")
    genus = tree.get("genus")
    lat, lng = tree.get("lat"), tree.get("lng")
    age_years = tree.get("age_years")
    locality = tree.get("city") or ""

    # Curadoria arbórea. O gênero é o critério confiável: famílias como Fabaceae
    # contêm tanto árvores (Senna, Bauhinia) quanto ervas/forrageiras rasteiras
    # (Centrosema, Stylosanthes). Por isso exigimos gênero na lista curada; a
    # família sozinha NÃO é suficiente para tratar a planta como árvore.
    genus_key = _norm_text(genus) or _norm_text(str(scientific or "").split()[0])
    is_tree = genus_key in ALLOWED_TREE_GENERA
    if not is_tree:
        return {
            "data_quality": "nao_arborea",
            "note": "Gênero/família fora da curadoria arbórea (provável erva/arbusto). Sem predição de altura.",
        }

    # Perfil botânico real (altura máxima + taxa k).
    profile = get_species_profile(scientific, family, genus)
    if profile is None or age_years is None:
        return {
            "data_quality": "sem_perfil_ou_idade",
            "note": "Espécie sem perfil validado ou sem idade. Precisa de coleta de campo.",
        }

    env, env_source = lookup_environment(scientific, family, genus, lat, lng)
    context, wire_from_context = choose_wire_context(locality)
    wire_height = wire_height_default or wire_from_context

    # Modificadores
    s_mod = soil_modifier(env["soil_depth"], env["soil_inclination"], env["soil_quality"], env["soil_coverage"])
    c_mod = climate_modifier(env["annual_rainfall"], env["altitude"], env["avg_temperature"])
    m_mod = management_modifier(False, False)      # sem dado de manejo na base
    pruning_mod = pruning_vigor_modifier(0)        # sem histórico de poda na base
    comp = canopy_competition_modifier(0, 999.0)   # sem dado de vizinhança
    total_modifier = s_mod * c_mod * m_mod * pruning_mod * comp["growth_modifier"]

    h_max, k = profile["height_max"], profile["k"]
    estimated_height = estimate_height(age_years, h_max, k, total_modifier)
    annual_growth = max(0.0, estimate_annual_growth(estimated_height, h_max, k, total_modifier))

    # Prazo até o fio
    if estimated_height >= wire_height:
        days_to_wire, will_reach, months = 0, True, 0
    else:
        sim = simular_crescimento_mensal(estimated_height, wire_height, k, altura_max=h_max, modifier=total_modifier)
        months = sim["meses_ate_o_fio"]
        if months is None or months > OPERATIONAL_HORIZON_MONTHS:
            days_to_wire, will_reach = None, False
        else:
            days_to_wire, will_reach = months * 30, True

    risk = classify_risk(estimated_height, wire_height,
                         -1 if days_to_wire is None else days_to_wire,
                         1 if will_reach else 0)

    phyllo = phyllotaxis_canopy(None, 0)

    return {
        "data_quality": "ok" if env_source != "neutro_estimado" else "ambiente_estimado",
        "environment_source": env_source,
        "wire_context": context,
        "estimated_height_m": round(estimated_height, 2),
        "annual_growth_m": round(annual_growth, 3),
        "height_next_year_m": round(min(h_max * 1.03, estimated_height + annual_growth), 2),
        "wire_height_m": wire_height,
        "will_reach_wire": will_reach,
        "days_to_wire": int(round(days_to_wire)) if (will_reach and days_to_wire is not None) else None,
        "months_to_wire": round(days_to_wire / 30, 1) if (will_reach and days_to_wire is not None) else None,
        "risk_status": risk,
        "species_profile": {"height_max_m": round(h_max, 2), "growth_rate_k": round(k, 4)},
        "phyllotaxis": phyllo,
        "modifiers_used": {
            "soil": s_mod, "climate": c_mod, "management": m_mod,
            "pruning": pruning_mod, "competition": comp["growth_modifier"],
            "total": round(total_modifier, 4),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Roda a IA MapTree sobre dados reais.")
    parser.add_argument("--input", default="arvores_mapeadas.json")
    parser.add_argument("--output", default="arvores_preditas.json")
    parser.add_argument("--wire-height", type=float, default=WIRE_HEIGHT,
                        help=f"Altura do fio em metros (padrão {WIRE_HEIGHT}).")
    args = parser.parse_args()

    input_path = BASE_DIR / args.input
    if not input_path.exists():
        raise SystemExit(f"Arquivo não encontrado: {input_path}")

    trees = json.loads(input_path.read_text(encoding="utf-8"))
    print(f"📂 {len(trees)} árvores em {args.input}")

    results = []
    stats = {"ok": 0, "ambiente_estimado": 0, "sem_perfil_ou_idade": 0, "nao_arborea": 0}
    risk_counter = {}

    for tree in trees:
        pred = predict_one(tree, args.wire_height)
        quality = pred.get("data_quality", "ok")
        stats[quality] = stats.get(quality, 0) + 1
        if "risk_status" in pred:
            risk_counter[pred["risk_status"]] = risk_counter.get(pred["risk_status"], 0) + 1

        results.append({
            "id": tree.get("id"),
            "name": tree.get("name"),
            "family": tree.get("family"),
            "genus": tree.get("genus"),
            "lat": tree.get("lat"),
            "lng": tree.get("lng"),
            "city": tree.get("city"),
            "state": tree.get("state"),
            "year_collected": tree.get("year_collected"),
            "age_years": tree.get("age_years"),
            "prediction": pred,
        })

    output_path = BASE_DIR / args.output
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Predições salvas em {args.output}")
    print(f"   Altura do fio usada: {args.wire_height} m")
    print(f"   Qualidade dos dados: {stats}")
    print(f"   Distribuição de risco: {risk_counter}")
    predicted = [r for r in results if r["prediction"].get("data_quality") in ("ok", "ambiente_estimado")]
    if predicted:
        print("\n   Exemplos:")
        for r in predicted[:5]:
            p = r["prediction"]
            print(f"    - {r['name']} ({r['age_years']}a): {p['estimated_height_m']}m "
                  f"| fio {p['wire_height_m']}m | risco {p['risk_status']} "
                  f"| atinge fio: {p['will_reach_wire']}")


if __name__ == "__main__":
    main()
