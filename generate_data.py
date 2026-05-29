"""
MapTree - Geração real de dataset e lógica de crescimento.

Regras deste módulo:
- Não inventar dados de árvore, solo ou clima quando houver fonte real disponível.
- Usar apenas:
  1) registros locais reais (enriched_trees.csv, dados_tabuleiro.csv, arvores_plantae.csv)
  2) APIs reais quando necessário e quando disponíveis
  3) agregações derivadas de dados reais já coletados (média por espécie/gênero/família)
- Se algo essencial não puder ser obtido de forma real, o registro é ignorado.

Este arquivo também expõe as funções usadas pela API FastAPI.
"""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from enrich_soil_climate import fetch_climate, fetch_soil
except Exception:  # pragma: no cover
    fetch_climate = None
    fetch_soil = None

BASE_DIR = Path(__file__).resolve().parent
CURRENT_YEAR = datetime.now().year

GOLDEN_ANGLE = 137.5
WIRE_HEIGHT = 6.5
WIRE_HEIGHT_BY_CONTEXT = {
    "URBAN_LOW_VOLTAGE": 5.5,
    "URBAN_MEDIUM_VOLTAGE": 6.0,
    "RURAL_DISTRIBUTION": 7.0,
    "ROAD_CROSSING": 8.0,
}
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
PHI = (1 + math.sqrt(5)) / 2

# Famílias priorizadas para o caso de uso de arborização/poda.
# Evita incluir ocorrências de Plantae que não são árvore (ervas, gramíneas, etc.).
ALLOWED_TREE_FAMILIES = {
    "fabaceae",
    "euphorbiaceae",
    "boraginaceae",
    "rubiaceae",
    "combretaceae",
    "arecaceae",
    "bignoniaceae",
    "anacardiaceae",
    "myrtaceae",
}

# Gêneros arbóreos aceitos para treino (curadoria inicial).
ALLOWED_TREE_GENERA = {
    "croton",
    "bauhinia",
    "piptadenia",
    "senna",
    "celtis",
    "auxemma",
    "combretum",
    "cenostigma",
    "geoffroea",
    "apuleia",
    "varronia",
    "erythroxylum",
    "copernicia",
    "tabebuia",
    "anadenanthera",
    "mimosa",
    "ziziphus",
    "prosopis",
    "licania",
    "azadirachta",
}

def _read_csv(path: Path, sep: str = ",") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep=sep)


def _is_valid_number(value: Any) -> bool:
    return value is not None and pd.notna(value)


def _norm_text(value: Any) -> str:
    return str(value).strip().lower() if _is_valid_number(value) else ""


def fibonacci_growth_modifier(pruning_count: int) -> float:
    """Multiplicador suave baseado no número de podas registradas."""
    pruning_count = max(0, int(pruning_count or 0))
    if pruning_count == 0:
        return 1.0

    cycle = pruning_count % len(FIBONACCI)
    ratio = FIBONACCI[cycle] / FIBONACCI[cycle - 1] if cycle > 0 else PHI
    balance = 1.0 + ((ratio / PHI) - 1.0) * 0.08
    pruning_penalty = max(0.72, 1.0 - pruning_count * 0.055)
    return round(min(1.08, max(0.65, pruning_penalty * balance)), 4)


def soil_modifier(depth: float, inclination: float, quality: str, coverage: float) -> float:
    quality_map = {"GOOD": 1.12, "REGULAR": 1.0, "BAD": 0.74}
    q = quality_map.get(str(quality).upper(), 1.0)
    depth_mod = min(max(float(depth), 0.2) / 1.4, 1.08)
    slope_mod = max(0.75, 1.0 - (max(float(inclination), 0) / 90.0) * 0.35)
    coverage_mod = max(0.58, 1.0 - max(0.0, min(float(coverage), 1.0)) * 0.42)
    return round(q * depth_mod * slope_mod * coverage_mod, 4)


def climate_modifier(annual_rainfall: float, altitude: float, avg_temperature: float) -> float:
    rain = float(annual_rainfall)
    temp = float(avg_temperature)
    alt = float(altitude)

    if rain < 350:
        rain_mod = 0.72
    elif rain > 1300:
        rain_mod = 0.9
    else:
        rain_mod = 0.9 + min(rain, 900) / 900 * 0.18

    temp_mod = 0.78 if temp < 18 or temp > 35 else 1.0
    alt_mod = 1.0 if alt < 600 else 0.92
    return round(rain_mod * temp_mod * alt_mod, 4)


def management_modifier(has_fertilization: bool, has_irrigation: bool) -> float:
    mod = 1.0
    if has_fertilization:
        mod += 0.08
    if has_irrigation:
        mod += 0.10
    return round(mod, 4)


def canopy_competition_modifier(nearby_trees_count: int, avg_neighbor_distance: float) -> dict:
    count = max(0, int(nearby_trees_count or 0))
    distance = max(float(avg_neighbor_distance or 0), 0.1)

    if count == 0:
        pressure = 0.0
    else:
        pressure = min(1.0, (count / 8.0) * max(0.25, (10.0 - min(distance, 10.0)) / 10.0))

    canopy_ratio = round(max(0.72, min(1.28, 1.22 - pressure * 0.45)), 4)
    growth_modifier = round(max(0.78, 1.0 - pressure * 0.18), 4)
    return {"canopy_ratio": canopy_ratio, "growth_modifier": growth_modifier}


def estimate_height(age_years: float, height_max: float, species_k: float, total_modifier: float) -> float:
    age = max(0.0, float(age_years))
    adjusted_k = max(0.01, float(species_k) * float(total_modifier))
    height = float(height_max) * (1.0 - math.exp(-adjusted_k * age))
    return round(min(height, float(height_max) * 1.03), 2)


def estimate_annual_growth(current_height: float, height_max: float, species_k: float, total_modifier: float) -> float:
    current_height = max(0.0, float(current_height))
    remaining = max(0.0, float(height_max) - current_height)
    yearly_k = max(0.01, float(species_k) * float(total_modifier))
    return round(remaining * (1.0 - math.exp(-yearly_k)), 3)


def classify_risk(estimated_height: float, wire_height: float, days_to_wire: float, will_reach_wire: int) -> str:
    ratio = estimated_height / wire_height if wire_height else 0
    if ratio >= 1.0 or days_to_wire == 0:
        return "CRITICAL"
    if will_reach_wire and days_to_wire <= 180:
        return "TO_PRUNE"
    if ratio >= 0.85 or (will_reach_wire and days_to_wire <= 365):
        return "UNDER_OBSERVATION"
    if ratio >= 0.72:
        return "TO_PRUNE"
    return "NORMAL"


def simular_crescimento_fibonacci(
    altura_atual: float,
    altura_fio: float,
    taxa_k: float,
    altura_max: float | None = None,
    modifier: float = 1.0,
    max_months: int = 1200,
) -> dict:
    """Simula crescimento mensal até o fio usando parâmetros reais/derivados."""
    if altura_atual >= altura_fio:
        return {"meses_ate_o_fio": 0, "altura_final": round(altura_atual, 2), "historico": []}

    effective_max = altura_max if altura_max is not None else max(altura_fio * 1.2, altura_atual)
    if effective_max <= altura_fio:
        return {"meses_ate_o_fio": None, "altura_final": round(altura_atual, 2), "historico": []}

    meses = 0
    altura = float(altura_atual)
    historico = []
    fib_mod = fibonacci_growth_modifier(0)
    monthly_k = max(0.001, float(taxa_k) * float(modifier) * fib_mod) / 12.0

    while altura < altura_fio and meses < max_months:
        remaining = max(effective_max - altura, 0)
        if remaining <= 0.001:
            break

        altura += remaining * (1.0 - math.exp(-monthly_k))
        meses += 1

        if meses <= 60 or meses % 12 == 0:
            historico.append({"mes": meses, "altura": round(altura, 3)})

    if altura < altura_fio:
        return {"meses_ate_o_fio": None, "altura_final": round(altura, 2), "historico": historico}

    return {"meses_ate_o_fio": meses, "altura_final": round(altura, 2), "historico": historico}


class RealCatalog:
    def __init__(self) -> None:
        self.exact: dict[str, dict[str, float]] = {}
        self.genus: dict[str, dict[str, float]] = {}
        self.family: dict[str, dict[str, float]] = {}
        self.env_by_species: dict[str, dict[str, float]] = {}
        self.env_by_genus: dict[str, dict[str, float]] = {}
        self.env_by_family: dict[str, dict[str, float]] = {}
        self.env_by_location: dict[str, dict[str, float]] = {}


_CATALOG: RealCatalog | None = None


def _catalog_key(scientific: str | None = None, family: str | None = None, genus: str | None = None) -> tuple[str, str, str]:
    return (_norm_text(scientific), _norm_text(family), _norm_text(genus))


def _env_key(lat: float | None, lon: float | None) -> str | None:
    if not _is_valid_number(lat) or not _is_valid_number(lon):
        return None
    return f"{round(float(lat), 4)}:{round(float(lon), 4)}"


def _build_catalog() -> RealCatalog:
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG

    catalog = RealCatalog()
    enriched_path = BASE_DIR / "enriched_trees.csv"
    df = _read_csv(enriched_path)

    if not df.empty:
        df = df.copy()
        # Garantir colunas esperadas
        for col in ["scientificName", "family", "latitude", "longitude", "heightAverage", "growthRateK", "annualRainfall", "avgTemperature", "altitude", "soil_depth", "soil_inclination", "soil_quality", "soil_coverage"]:
            if col not in df.columns:
                df[col] = np.nan

        df["_species_key"] = df["scientificName"].map(_norm_text)
        df["_family_key"] = df["family"].map(_norm_text)
        df["_genus_key"] = df["scientificName"].fillna("").astype(str).map(lambda s: _norm_text(s.split()[0]) if s else "")
        df["_location_key"] = df.apply(lambda r: _env_key(r.get("latitude"), r.get("longitude")), axis=1)

        # Profiles exatos por espécie
        for species_key, group in df.groupby("_species_key"):
            if not species_key:
                continue
            height_vals = group["heightAverage"].dropna().astype(float)
            k_vals = group["growthRateK"].dropna().astype(float)
            if height_vals.empty or k_vals.empty:
                continue
            catalog.exact[species_key] = {
                "height_max": float(height_vals.mean()),
                "k": float(k_vals.mean()),
            }

            env_rows = group.dropna(subset=["annualRainfall", "avgTemperature", "altitude", "soil_depth", "soil_inclination", "soil_quality", "soil_coverage"])
            if not env_rows.empty:
                catalog.env_by_species[species_key] = {
                    "annual_rainfall": float(env_rows["annualRainfall"].astype(float).mean()),
                    "avg_temperature": float(env_rows["avgTemperature"].astype(float).mean()),
                    "altitude": float(env_rows["altitude"].astype(float).mean()),
                    "soil_depth": float(env_rows["soil_depth"].astype(float).mean()),
                    "soil_inclination": float(env_rows["soil_inclination"].astype(float).mean()),
                    "soil_coverage": float(env_rows["soil_coverage"].astype(float).mean()),
                    "soil_quality": str(env_rows["soil_quality"].mode().iloc[0]).upper(),
                }

        # Agregações por gênero e família a partir dos dados reais
        for label, target in (("_genus_key", catalog.genus), ("_family_key", catalog.family)):
            for key, group in df.groupby(label):
                if not key:
                    continue
                height_vals = group["heightAverage"].dropna().astype(float)
                k_vals = group["growthRateK"].dropna().astype(float)
                if height_vals.empty or k_vals.empty:
                    continue
                target[key] = {
                    "height_max": float(height_vals.mean()),
                    "k": float(k_vals.mean()),
                }

        # Ambiente por coordenada, espécie, gênero e família
        for _, row in df.iterrows():
            env = {
                "annual_rainfall": row.get("annualRainfall"),
                "avg_temperature": row.get("avgTemperature"),
                "altitude": row.get("altitude"),
                "soil_depth": row.get("soil_depth"),
                "soil_inclination": row.get("soil_inclination"),
                "soil_quality": row.get("soil_quality"),
                "soil_coverage": row.get("soil_coverage"),
            }
            if all(_is_valid_number(env[k]) for k in ["annual_rainfall", "avg_temperature", "altitude", "soil_depth", "soil_inclination", "soil_coverage"]) and _is_valid_number(env["soil_quality"]):
                skey = _norm_text(row.get("scientificName"))
                gkey = _norm_text(str(row.get("scientificName", "")).split()[0] if _is_valid_number(row.get("scientificName")) else "")
                fkey = _norm_text(row.get("family"))
                lkey = _env_key(row.get("latitude"), row.get("longitude"))
                if skey:
                    catalog.env_by_species.setdefault(skey, env)
                if gkey:
                    catalog.env_by_genus.setdefault(gkey, env)
                if fkey:
                    catalog.env_by_family.setdefault(fkey, env)
                if lkey:
                    catalog.env_by_location.setdefault(lkey, env)

    _CATALOG = catalog
    return catalog


def get_species_profile(scientific_name: str | None, family: str | None = None, genus: str | None = None) -> Optional[dict]:
    """
    Busca perfil de espécie real com foco em qualidade.

    Regra de segurança: não usar fallback por FAMÍLIA para evitar "forçar"
    espécies herbáceas/arbustivas a perfis arbóreos por coincidência taxonômica.
    """
    catalog = _build_catalog()
    scientific_key = _norm_text(scientific_name)
    genus_key = _norm_text(genus) or (_norm_text(str(scientific_name).split()[0]) if scientific_key else "")

    if scientific_key and scientific_key in catalog.exact:
        return catalog.exact[scientific_key]
    if genus_key and genus_key in catalog.genus:
        return catalog.genus[genus_key]
    return None


def choose_wire_context(locality: str | None = None) -> tuple[str, float]:
    text = _norm_text(locality)
    if any(token in text for token in ("rodovia", "br ", "estrada", "highway")):
        context = "ROAD_CROSSING"
    elif any(token in text for token in ("rural", "fazenda", "sitio", "sítio", "serra", "zona rural")):
        context = "RURAL_DISTRIBUTION"
    else:
        context = "URBAN_LOW_VOLTAGE"
    return context, WIRE_HEIGHT_BY_CONTEXT[context]


def normalize_location_key(lat: float | None, lon: float | None, locality: str | None = None, precision: int = 5) -> str | None:
    if lat is not None and lon is not None and pd.notna(lat) and pd.notna(lon):
        return f"{round(float(lat), precision)}:{round(float(lon), precision)}"
    text = str(locality or "").strip().lower()
    return text or None


def _estimate_age_years(year_value: Any) -> Optional[float]:
    if not _is_valid_number(year_value):
        return None
    try:
        year_int = int(float(year_value))
        return max(0.5, float(CURRENT_YEAR - year_int))
    except Exception:
        return None


def _load_real_seed_records() -> list[dict]:
    """Carrega apenas registros reais locais, sem fabricar árvores novas."""
    records: list[dict] = []

    # 1) enriched_trees.csv: base mais confiável para perfil/ambiente real
    enriched = _read_csv(BASE_DIR / "enriched_trees.csv")
    if not enriched.empty:
        for _, row in enriched.iterrows():
            lat = row.get("latitude")
            lon = row.get("longitude")
            if not _is_valid_number(lat) or not _is_valid_number(lon):
                continue
            records.append({
                "source": "enriched_trees",
                "scientific": row.get("scientificName"),
                "family": row.get("family"),
                "genus": str(row.get("scientificName", "")).split()[0] if _is_valid_number(row.get("scientificName")) else "",
                "lat": float(lat),
                "lng": float(lon),
                "year": None,
                "observed_year": None,
                "locality": "",
                "soil_depth": row.get("soil_depth"),
                "soil_inclination": row.get("soil_inclination"),
                "soil_quality": row.get("soil_quality"),
                "soil_coverage": row.get("soil_coverage"),
                "annual_rainfall": row.get("annualRainfall"),
                "avg_temperature": row.get("avgTemperature"),
                "altitude": row.get("altitude"),
                "height_max": row.get("heightAverage"),
                "species_k": row.get("growthRateK"),
                "pruning_count": row.get("pruning_count"),
                "has_fertilization": row.get("has_fertilization"),
                "has_irrigation": row.get("has_irrigation"),
                "nearby_trees_count": row.get("nearby_trees_count"),
                "avg_neighbor_distance": row.get("avg_neighbor_distance"),
            })

    # 2) dados_tabuleiro.csv: ocorrências locais reais
    tab = _read_csv(BASE_DIR / "dados_tabuleiro.csv", sep="\t")
    if not tab.empty:
        for _, row in tab.iterrows():
            lat = row.get("latitude")
            lon = row.get("longitude")
            if not _is_valid_number(lat) or not _is_valid_number(lon):
                continue
            records.append({
                "source": "tabuleiro",
                "scientific": row.get("scientificname"),
                "family": row.get("family"),
                "genus": row.get("genus"),
                "lat": float(lat),
                "lng": float(lon),
                "year": row.get("yearcollected"),
                "observed_year": row.get("yearcollected"),
                "locality": row.get("locality") or row.get("county") or "",
                "pruning_count": row.get("pruning_count"),
                "has_fertilization": row.get("has_fertilization"),
                "has_irrigation": row.get("has_irrigation"),
                "nearby_trees_count": row.get("nearby_trees_count"),
                "avg_neighbor_distance": row.get("avg_neighbor_distance"),
            })

    # 3) arvores_plantae.csv: ocorrências GBIF reais com Plantae
    gbif = _read_csv(BASE_DIR / "arvores_plantae.csv", sep=",")
    if not gbif.empty:
        gbif = gbif[gbif.get("kingdom").astype(str) == "Plantae"].copy()
        for _, row in gbif.iterrows():
            lat = row.get("decimalLatitude")
            lon = row.get("decimalLongitude")
            if not _is_valid_number(lat) or not _is_valid_number(lon):
                continue
            records.append({
                "source": "gbif_plantae",
                "scientific": row.get("scientificName") or row.get("species"),
                "family": row.get("family"),
                "genus": row.get("genus"),
                "lat": float(lat),
                "lng": float(lon),
                "year": row.get("year"),
                "observed_year": row.get("year"),
                "locality": row.get("locality") or row.get("municipality") or "",
                "pruning_count": row.get("pruning_count"),
                "has_fertilization": row.get("has_fertilization"),
                "has_irrigation": row.get("has_irrigation"),
                "nearby_trees_count": row.get("nearby_trees_count"),
                "avg_neighbor_distance": row.get("avg_neighbor_distance"),
            })

    # Deduplicação simples por coordenada + espécie + origem
    deduped = []
    seen = set()
    for rec in records:
        key = (rec["source"], round(rec["lat"], 5), round(rec["lng"], 5), _norm_text(rec.get("scientific")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)
    return deduped


def _lookup_environment(record: dict) -> Optional[dict]:
    catalog = _build_catalog()
    species_key = _norm_text(record.get("scientific"))
    genus_key = _norm_text(record.get("genus")) or (_norm_text(str(record.get("scientific", "")).split()[0]) if species_key else "")
    family_key = _norm_text(record.get("family"))
    location_key = _env_key(record.get("lat"), record.get("lng"))

    for key, bucket in ((location_key, catalog.env_by_location), (species_key, catalog.env_by_species), (genus_key, catalog.env_by_genus), (family_key, catalog.env_by_family)):
        if key and key in bucket:
            return bucket[key]

    # Para registros sem ambiente local suficiente, tentar APIs reais apenas quando disponíveis.
    if fetch_soil is None or fetch_climate is None:
        return None

    soil = fetch_soil(record["lat"], record["lng"])
    climate = fetch_climate(record["lat"], record["lng"])
    if not soil or not climate:
        return None

    if soil.get("depth") is None or soil.get("quality") is None:
        return None
    if climate.get("annual_rainfall") is None or climate.get("avg_temperature") is None or climate.get("altitude") is None:
        return None

    return {
        "annual_rainfall": climate["annual_rainfall"],
        "avg_temperature": climate["avg_temperature"],
        "altitude": climate["altitude"],
        "soil_depth": soil["depth"],
        "soil_inclination": soil.get("inclination", 5.0),
        "soil_quality": soil["quality"],
        "soil_coverage": soil.get("coverage", 0.2),
    }


def generate_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """
    Gera o dataset de treino somente com base em dados reais e agregações reais.

    n_samples é mantido por compatibilidade, mas o dataset resultante depende da
    quantidade de registros reais válidos disponíveis.
    """
    records = _load_real_seed_records()
    rows = []

    for sample_id, rec in enumerate(records):
        family_key = _norm_text(rec.get("family"))
        genus_key = _norm_text(rec.get("genus")) or _norm_text(str(rec.get("scientific") or "").split()[0])

        if family_key and family_key not in ALLOWED_TREE_FAMILIES:
            continue
        if genus_key and genus_key not in ALLOWED_TREE_GENERA:
            continue

        # 1) Primeiro valida se a espécie/gênero possui perfil real conhecido.
        # Evita chamadas de API caras para registros que já serão descartados.
        sp = get_species_profile(rec.get("scientific"), rec.get("family"), rec.get("genus"))
        if not sp:
            continue

        # 2) Só então resolve ambiente (local/agregado/API).
        env = _lookup_environment(rec)
        if not env:
            continue

        age_years = _estimate_age_years(rec.get("year"))
        if age_years is None:
            # Sem uma data real minimamente confiável, o registro não entra no treino.
            continue

        # Sem dados reais de manejo/vizinhança na base atual:
        # usamos modificadores neutros (não aumenta nem reduz crescimento).
        pruning_count = 0
        has_fertilization = 0
        has_irrigation = 0
        nearby_trees_count = 0
        avg_neighbor_distance = 999.0
        wire_context, wire_height = choose_wire_context(rec.get("locality"))

        fib_mod = fibonacci_growth_modifier(pruning_count)
        s_mod = soil_modifier(env["soil_depth"], env["soil_inclination"], env["soil_quality"], env["soil_coverage"])
        c_mod = climate_modifier(env["annual_rainfall"], env["altitude"], env["avg_temperature"])
        m_mod = management_modifier(bool(has_fertilization), bool(has_irrigation))
        canopy = canopy_competition_modifier(nearby_trees_count, avg_neighbor_distance)
        total_modifier = round(s_mod * c_mod * m_mod * fib_mod * canopy["growth_modifier"], 4)

        estimated_height = estimate_height(age_years, sp["height_max"], sp["k"], total_modifier)
        annual_growth_m = estimate_annual_growth(estimated_height, sp["height_max"], sp["k"], total_modifier)
        height_next_year = round(min(sp["height_max"] * 1.03, estimated_height + annual_growth_m), 2)
        future = simular_crescimento_fibonacci(
            estimated_height,
            wire_height,
            sp["k"],
            altura_max=sp["height_max"] * min(1.05, max(0.75, total_modifier)),
            modifier=total_modifier,
        )
        months_to_wire = future["meses_ate_o_fio"]
        will_reach_wire = 1 if months_to_wire is not None else 0
        days_to_wire = 0 if estimated_height >= wire_height else (-1 if months_to_wire is None else months_to_wire * 30)
        risk_status = classify_risk(estimated_height, wire_height, days_to_wire, will_reach_wire)

        rows.append({
            "sample_id": sample_id,
            "species_common_name": str(rec.get("genus") or rec.get("scientific") or "").split()[0],
            "species_scientific_name": rec.get("scientific") or "",
            "species_height_max": sp["height_max"],
            "species_k": sp["k"],
            "age_years": age_years,
            "pruning_count": pruning_count,
            "soil_depth": env["soil_depth"],
            "soil_inclination": env["soil_inclination"],
            "soil_quality": env["soil_quality"],
            "soil_coverage": env["soil_coverage"],
            "annual_rainfall": env["annual_rainfall"],
            "altitude": env["altitude"],
            "avg_temperature": env["avg_temperature"],
            "has_fertilization": has_fertilization,
            "has_irrigation": has_irrigation,
            "nearby_trees_count": nearby_trees_count,
            "avg_neighbor_distance": avg_neighbor_distance,
            "wire_context": wire_context,
            "wire_height": wire_height,
            "fibonacci_modifier": fib_mod,
            "canopy_ratio": canopy["canopy_ratio"],
            "soil_modifier": s_mod,
            "climate_modifier": c_mod,
            "management_modifier": m_mod,
            "total_modifier": total_modifier,
            "estimated_height": estimated_height,
            "annual_growth_m": annual_growth_m,
            "height_next_year": height_next_year,
            "days_to_wire": days_to_wire,
            "risk_status": risk_status,
            "will_reach_wire": will_reach_wire,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("MapTree: dataset real-only")
    df = generate_dataset()
    if df.empty:
        raise SystemExit("Nenhum registro real válido encontrado para treinar.")

    out = BASE_DIR / "tree_dataset.csv"
    df.to_csv(out, index=False)
    print(f"Dataset salvo em {out}")
    print(df.head())
