import math
import os
import random

import numpy as np
import pandas as pd


# Este modulo centraliza a logica de dominio usada para gerar o dataset de treino.
# A API importa varias funcoes daqui, entao qualquer mudanca nas formulas abaixo
# deve ser acompanhada por novo treino em train_models.py.

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


SPECIES_DATA = [
    {"name": "Algaroba", "scientific": "Prosopis juliflora", "height_max": 12.0, "k": 0.18},
    {"name": "Catingueira", "scientific": "Caesalpinia pyramidalis", "height_max": 8.0, "k": 0.15},
    {"name": "Juazeiro", "scientific": "Ziziphus joazeiro", "height_max": 10.0, "k": 0.12},
    {"name": "Oiticica", "scientific": "Licania rigida", "height_max": 15.0, "k": 0.10},
    {"name": "Craibeira", "scientific": "Tabebuia aurea", "height_max": 14.0, "k": 0.11},
    {"name": "Sabiá", "scientific": "Mimosa caesalpiniifolia", "height_max": 9.0, "k": 0.20},
    {"name": "Angico", "scientific": "Anadenanthera colubrina", "height_max": 18.0, "k": 0.09},
    {"name": "Nim", "scientific": "Azadirachta indica", "height_max": 13.0, "k": 0.16},
    {"name": "Carnauba", "scientific": "Copernicia prunifera", "height_max": 15.0, "k": 0.08},
    {"name": "Croton", "scientific": "Croton blanchetianus", "height_max": 5.0, "k": 0.25},
    {"name": "Bauhinia", "scientific": "Bauhinia", "height_max": 8.0, "k": 0.14},
    {"name": "Piptadenia", "scientific": "Piptadenia moniliformis", "height_max": 7.0, "k": 0.18},
    {"name": "Cenostigma", "scientific": "Cenostigma pyramidale", "height_max": 7.0, "k": 0.16},
    {"name": "Geoffroea", "scientific": "Geoffroea spinosa", "height_max": 10.0, "k": 0.12},
    {"name": "Apuleia", "scientific": "Apuleia leiocarpa", "height_max": 20.0, "k": 0.08},
]

SPECIES_BY_NAME = {sp["scientific"].lower(): sp for sp in SPECIES_DATA}
SPECIES_BY_GENUS = {sp["scientific"].split()[0].lower(): sp for sp in SPECIES_DATA}
FAMILY_DEFAULTS = {
    "Arecaceae": {"name": "Palmeira", "scientific": "Arecaceae sp.", "height_max": 14.0, "k": 0.08},
    "Fabaceae": {"name": "Fabaceae", "scientific": "Fabaceae sp.", "height_max": 10.0, "k": 0.13},
    "Euphorbiaceae": {"name": "Euphorbiaceae", "scientific": "Euphorbiaceae sp.", "height_max": 6.0, "k": 0.20},
    "Combretaceae": {"name": "Combretaceae", "scientific": "Combretaceae sp.", "height_max": 7.0, "k": 0.16},
    "Boraginaceae": {"name": "Boraginaceae", "scientific": "Boraginaceae sp.", "height_max": 8.0, "k": 0.13},
}


def fibonacci_growth_modifier(pruning_count: int) -> float:
    """
    Calcula o efeito das podas sobre o crescimento.

    A sequencia de Fibonacci e usada apenas como um ajuste moderado de ciclo,
    nao como crescimento livre. O peso principal e uma penalidade acumulada:
    muitas podas reduzem vigor esperado e evitam previsoes irreais.
    Retorno: multiplicador entre 0.65 e 1.08.
    """
    pruning_count = max(0, int(pruning_count or 0))
    if pruning_count == 0:
        return 1.0

    cycle = pruning_count % len(FIBONACCI)
    ratio = FIBONACCI[cycle] / FIBONACCI[cycle - 1] if cycle > 0 else PHI
    balance = 1.0 + ((ratio / PHI) - 1.0) * 0.08
    pruning_penalty = max(0.72, 1.0 - pruning_count * 0.055)
    return round(min(1.08, max(0.65, pruning_penalty * balance)), 4)


def soil_modifier(depth: float, inclination: float, quality: str, coverage: float) -> float:
    """
    Converte condicoes de solo em multiplicador de crescimento.

    Campos:
    - depth: profundidade util do solo em metros.
    - inclination: inclinacao do terreno em graus.
    - quality: GOOD, REGULAR ou BAD.
    - coverage: impermeabilizacao/cobertura do solo de 0.0 a 1.0.
    """
    quality_map = {"GOOD": 1.12, "REGULAR": 1.0, "BAD": 0.74}
    q = quality_map.get(str(quality).upper(), 1.0)
    depth_mod = min(max(depth, 0.2) / 1.4, 1.08)
    slope_mod = max(0.75, 1.0 - (max(inclination, 0) / 90.0) * 0.35)
    coverage_mod = max(0.58, 1.0 - max(0.0, min(coverage, 1.0)) * 0.42)
    return round(q * depth_mod * slope_mod * coverage_mod, 4)


def climate_modifier(annual_rainfall: float, altitude: float, avg_temperature: float) -> float:
    """
    Converte clima em multiplicador de crescimento.

    A regra privilegia faixa de chuva e temperatura compativel com especies
    urbanas/caatinga. Valores extremos reduzem o crescimento esperado.
    """
    if annual_rainfall < 350:
        rain_mod = 0.72
    elif annual_rainfall > 1300:
        rain_mod = 0.9
    else:
        rain_mod = 0.9 + min(annual_rainfall, 900) / 900 * 0.18

    temp_mod = 0.78 if avg_temperature < 18 or avg_temperature > 35 else 1.0
    alt_mod = 1.0 if altitude < 600 else 0.92
    return round(rain_mod * temp_mod * alt_mod, 4)


def management_modifier(has_fertilization: bool, has_irrigation: bool) -> float:
    """Aplica bonus simples quando ha adubacao e/ou irrigacao registradas."""
    mod = 1.0
    if has_fertilization:
        mod += 0.08
    if has_irrigation:
        mod += 0.10
    return round(mod, 4)


def canopy_competition_modifier(nearby_trees_count: int, avg_neighbor_distance: float) -> dict:
    """
    Estima competicao por copa entre arvores proximas.

    Retorna:
    - canopy_ratio: relacao largura/altura esperada da copa.
    - growth_modifier: reducao de crescimento por competicao lateral.
    """
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
    """
    Estima altura atual por curva assintotica.

    A arvore cresce rapido quando jovem e desacelera ao se aproximar da altura
    maxima da especie. Isso evita que idade alta gere altura infinita.
    """
    age = max(0.0, float(age_years))
    adjusted_k = max(0.01, species_k * total_modifier)
    height = height_max * (1.0 - math.exp(-adjusted_k * age))
    return round(min(height, height_max * 1.03), 2)


def estimate_annual_growth(current_height: float, height_max: float, species_k: float, total_modifier: float) -> float:
    """Crescimento esperado nos proximos 12 meses, limitado pela altura maxima."""
    current_height = max(0.0, float(current_height))
    remaining = max(0.0, height_max - current_height)
    yearly_k = max(0.01, species_k * total_modifier)
    return round(remaining * (1.0 - math.exp(-yearly_k)), 3)


def get_species_profile(scientific_name: str | None, family: str | None = None) -> dict:
    """
    Busca perfil de especie por nome cientifico, genero ou familia.

    Quando nao ha match, usa fallback de familia ou uma especie conhecida.
    Este ponto deve ser melhorado com uma tabela botanica validada pelo time.
    """
    name = str(scientific_name or "").strip()
    if name:
        exact = SPECIES_BY_NAME.get(name.lower())
        if exact:
            return exact
        genus = name.split()[0].lower()
        if genus in SPECIES_BY_GENUS:
            return SPECIES_BY_GENUS[genus]
    return FAMILY_DEFAULTS.get(str(family or "").strip(), random.choice(SPECIES_DATA))


def choose_wire_context(locality: str | None = None) -> tuple[str, float]:
    """
    Escolhe altura de fio por contexto aproximado da localidade.

    Os valores atuais sao parametros operacionais para treino inicial, nao laudo
    normativo. Substituir por norma da distribuidora/ABNT quando disponivel.
    """
    text = str(locality or "").lower()
    if any(token in text for token in ("rodovia", "br ", "estrada", "highway")):
        context = "ROAD_CROSSING"
    elif any(token in text for token in ("rural", "fazenda", "sitio", "sítio", "serra")):
        context = "RURAL_DISTRIBUTION"
    else:
        context = random.choices(
            ["URBAN_LOW_VOLTAGE", "URBAN_MEDIUM_VOLTAGE", "RURAL_DISTRIBUTION"],
            weights=[0.62, 0.28, 0.10],
        )[0]
    return context, WIRE_HEIGHT_BY_CONTEXT[context]


def normalize_location_key(lat: float | None, lon: float | None, locality: str | None = None, precision: int = 5) -> str | None:
    """
    Gera chave estavel para detectar duplicidade de localidade.

    precision=5 agrupa coordenadas dentro de aproximadamente 1 metro. Essa
    tolerancia remove repeticoes de coleta/importacao sem juntar quarteiroes
    diferentes. Quando nao ha coordenada, usa a localidade textual normalizada.
    """
    if lat is not None and lon is not None and pd.notna(lat) and pd.notna(lon):
        return f"{round(float(lat), precision)}:{round(float(lon), precision)}"
    text = str(locality or "").strip().lower()
    return text or None


def _load_real_seed_records() -> list[dict]:
    """
    Carrega sementes reais locais para o dataset.

    Fontes atuais:
    - enriched_trees.csv: dados ja enriquecidos com solo/clima.
    - arvores_plantae.csv: ocorrencias botanicas exportadas do GBIF/Plantae.

    Registros duplicados por coordenada/localidade sao ignorados para reduzir
    vies de repeticao no treino.
    """
    records = []
    seen_locations = set()
    if os.path.exists("enriched_trees.csv"):
        df = pd.read_csv("enriched_trees.csv")
        for _, row in df.iterrows():
            loc_key = normalize_location_key(row.get("latitude"), row.get("longitude"), row.get("scientificName"))
            if loc_key and loc_key in seen_locations:
                continue
            if loc_key:
                seen_locations.add(loc_key)
            records.append({
                "scientific": row.get("scientificName"),
                "family": row.get("family"),
                "height_max": row.get("heightAverage"),
                "k": row.get("growthRateK"),
                "soil_depth": row.get("soil_depth"),
                "soil_inclination": row.get("soil_inclination"),
                "soil_quality": row.get("soil_quality"),
                "soil_coverage": row.get("soil_coverage"),
                "annual_rainfall": row.get("annualRainfall"),
                "altitude": row.get("altitude"),
                "avg_temperature": row.get("avgTemperature"),
            })

    if os.path.exists("arvores_plantae.csv"):
        df = pd.read_csv("arvores_plantae.csv", nrows=1200)
        for _, row in df.dropna(subset=["scientificName"]).iterrows():
            loc_key = normalize_location_key(
                row.get("decimalLatitude"),
                row.get("decimalLongitude"),
                row.get("locality") or row.get("municipality") or row.get("scientificName"),
            )
            if loc_key and loc_key in seen_locations:
                continue
            if loc_key:
                seen_locations.add(loc_key)
            sp = get_species_profile(row.get("scientificName"), row.get("family"))
            records.append({
                "scientific": row.get("scientificName"),
                "family": row.get("family"),
                "height_max": sp["height_max"],
                "k": sp["k"],
                "year": row.get("year"),
                "locality": row.get("locality") or row.get("municipality"),
            })
    return records


def simular_crescimento_fibonacci(
    altura_atual: float,
    altura_fio: float,
    taxa_k: float,
    altura_max: float | None = None,
    modifier: float = 1.0,
    max_months: int = 1200,
) -> dict:
    """
    Simula crescimento mensal ate a altura do fio.

    Usa crescimento assintotico: quanto mais perto da altura maxima da especie,
    menor o ganho mensal. Se a altura maxima ajustada nao chega ao fio, retorna
    meses_ate_o_fio=None.
    """
    if altura_atual >= altura_fio:
        return {"meses_ate_o_fio": 0, "altura_final": round(altura_atual, 2), "historico": []}

    effective_max = altura_max if altura_max is not None else max(altura_fio * 1.2, altura_atual)
    if effective_max <= altura_fio:
        return {"meses_ate_o_fio": None, "altura_final": round(altura_atual, 2), "historico": []}

    meses = 0
    altura = float(altura_atual)
    historico = []
    fib_mod = fibonacci_growth_modifier(0)
    monthly_k = max(0.001, taxa_k * modifier * fib_mod) / 12.0

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


def classify_risk(estimated_height: float, wire_height: float, days_to_wire: float, will_reach_wire: int) -> str:
    """
    Classifica risco operacional para poda.

    CRITICAL: ja atingiu/ultrapassou o fio.
    TO_PRUNE: precisa entrar em rota de poda.
    UNDER_OBSERVATION: monitorar porque esta proxima ou atingira em ate 1 ano.
    NORMAL: sem acao imediata.
    """
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


def generate_dataset(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Gera dataset de treino com mistura de dados reais e simulacao controlada.

    Aproximadamente 45% das amostras usam registros reais locais como semente
    quando os CSVs existem. O restante cobre cenarios sinteticos para dar
    variabilidade ao modelo. Sempre que este arquivo mudar, rode:
    python3 generate_data.py
    python3 train_models.py
    """
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    real_records = _load_real_seed_records()

    for sample_id in range(n_samples):
        real = random.choice(real_records) if real_records and random.random() < 0.45 else None
        sp = get_species_profile(real.get("scientific"), real.get("family")) if real else random.choice(SPECIES_DATA)
        height_max = float(real.get("height_max") if real and pd.notna(real.get("height_max")) else sp["height_max"])
        species_k = float(real.get("k") if real and pd.notna(real.get("k")) else sp["k"])

        if real and pd.notna(real.get("year")):
            age_years = round(max(0.5, 2026 - int(real["year"]) + random.uniform(-0.5, 0.5)), 2)
        else:
            age_years = round(random.uniform(0.5, 45.0), 2)

        pruning_count = random.randint(0, 8)
        soil_depth = round(float(real.get("soil_depth")) if real and pd.notna(real.get("soil_depth")) else random.uniform(0.3, 2.0), 2)
        soil_inclination = round(float(real.get("soil_inclination")) if real and pd.notna(real.get("soil_inclination")) else random.uniform(0, 45), 2)
        soil_quality = str(real.get("soil_quality")) if real and pd.notna(real.get("soil_quality")) else random.choice(["GOOD", "REGULAR", "BAD"])
        soil_coverage = round(float(real.get("soil_coverage")) if real and pd.notna(real.get("soil_coverage")) else random.uniform(0, 0.9), 2)
        annual_rainfall = round(float(real.get("annual_rainfall")) if real and pd.notna(real.get("annual_rainfall")) else max(250, random.normalvariate(700, 210)), 1)
        altitude = round(float(real.get("altitude")) if real and pd.notna(real.get("altitude")) else random.uniform(20, 900), 1)
        avg_temperature = round(float(real.get("avg_temperature")) if real and pd.notna(real.get("avg_temperature")) else random.normalvariate(27, 3), 1)
        has_fertilization = random.randint(0, 1)
        has_irrigation = random.randint(0, 1)
        nearby_trees_count = random.randint(0, 10)
        avg_neighbor_distance = round(random.uniform(1.5, 18.0), 1)
        wire_context, wire_height = choose_wire_context(real.get("locality") if real else None)

        fib_mod = fibonacci_growth_modifier(pruning_count)
        s_mod = soil_modifier(soil_depth, soil_inclination, soil_quality, soil_coverage)
        c_mod = climate_modifier(annual_rainfall, altitude, avg_temperature)
        m_mod = management_modifier(bool(has_fertilization), bool(has_irrigation))
        canopy = canopy_competition_modifier(nearby_trees_count, avg_neighbor_distance)
        total_modifier = round(s_mod * c_mod * m_mod * fib_mod * canopy["growth_modifier"], 4)

        estimated_height = estimate_height(age_years, height_max, species_k, total_modifier)
        annual_growth_m = estimate_annual_growth(estimated_height, height_max, species_k, total_modifier)
        height_next_year = round(min(height_max * 1.03, estimated_height + annual_growth_m), 2)
        future = simular_crescimento_fibonacci(
            estimated_height,
            wire_height,
            species_k,
            altura_max=height_max * min(1.05, max(0.75, total_modifier)),
            modifier=total_modifier,
        )
        months_to_wire = future["meses_ate_o_fio"]
        will_reach_wire = 1 if months_to_wire is not None else 0
        days_to_wire = 0 if estimated_height >= wire_height else (-1 if months_to_wire is None else months_to_wire * 30)
        risk_status = classify_risk(estimated_height, wire_height, days_to_wire, will_reach_wire)

        rows.append({
            "sample_id": sample_id,
            "species_common_name": sp["name"],
            "species_scientific_name": sp["scientific"],
            "species_height_max": height_max,
            "species_k": species_k,
            "age_years": age_years,
            "pruning_count": pruning_count,
            "soil_depth": soil_depth,
            "soil_inclination": soil_inclination,
            "soil_quality": soil_quality,
            "soil_coverage": soil_coverage,
            "annual_rainfall": annual_rainfall,
            "altitude": altitude,
            "avg_temperature": avg_temperature,
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
    print("MapTree: validando logica de crescimento")
    test = simular_crescimento_fibonacci(2.0, 5.0, 0.15, altura_max=8.0)
    assert test["meses_ate_o_fio"] is not None
    assert test["altura_final"] >= 5.0

    df = generate_dataset(n_samples=3000)
    df.to_csv("tree_dataset.csv", index=False)
    print("Dataset salvo em tree_dataset.csv")
    print(df.head())
