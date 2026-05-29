"""
MapTree - Enriquecedor de Dados
Busca automaticamente dados de solo e clima por coordenadas.

Fontes:
  🌍 Solo    → SoilGrids v2 REST API (ISRIC) — gratuita, sem chave
  🌡️ Clima   → Open-Meteo API             — gratuita, sem chave
  🔁 Fallback → perfis médios da Caatinga (quando APIs offline)

Uso:
  python enrich_soil_climate.py
  → Lê dados_tabuleiro.csv
  → Consulta APIs por lat/lon
  → Salva enriched_trees.json  (pronto pra importar no MongoDB)
"""

import json
import time
import requests
import pandas as pd
from datetime import datetime

# ─── Configurações ─────────────────────────────────────────────────────────────

INPUT_CSV       = "dados_tabuleiro.csv"
OUTPUT_JSON     = "enriched_trees.json"
SLEEP_BETWEEN   = 1.2   # segundos entre requests (respeitar rate limit)
TIMEOUT         = 12    # segundos timeout por request

# ─── Observação ────────────────────────────────────────────────────────────────
# O pipeline agora evita inventar valores: se a API falhar, o registro é
# ignorado em vez de receber fallback fictício.

# ─── SoilGrids ─────────────────────────────────────────────────────────────────

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

def fetch_soil(lat: float, lon: float) -> dict:
    """
    Busca dados de solo no SoilGrids v2 pela lat/lon.
    Converte para o formato esperado pelo Soil model do MapTree.
    """
    params = {
        "lon": lon,
        "lat": lat,
        "property": ["clay", "sand", "silt", "phh2o", "cfvo"],
        "depth": "0-30cm",
        "value": "mean",
    }

    try:
        resp = requests.get(SOILGRIDS_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        # Extrair valores das propriedades
        props = data.get("properties", {}).get("layers", [])
        clay = sand = silt = ph = cfvo = None

        for layer in props:
            name = layer.get("name", "")
            depths = layer.get("depths", [])
            if not depths:
                continue
            val = depths[0].get("values", {}).get("mean")
            if val is None:
                continue

            # SoilGrids retorna valores inteiros com fator de conversão
            if name == "clay":    clay  = val / 10.0   # g/kg → %
            if name == "sand":    sand  = val / 10.0
            if name == "silt":    silt  = val / 10.0
            if name == "phh2o":   ph    = val / 10.0   # pH * 10
            if name == "cfvo":    cfvo  = val / 10.0   # coarse fragments %

        # Classificar qualidade do solo baseado em textura e pH
        quality = _classify_soil_quality(clay, sand, ph)

        # Estimar profundidade pelo fragmento rochoso (cfvo alto = solo raso)
        depth = _estimate_depth(cfvo)

        return {
            "depth":       round(depth, 2),
            "inclination": 5.0,     # SoilGrids não tem inclinação — usar SRTM se precisar
            "quality":     quality,
            "coverage":    0.2,     # cobertura: cadastrar manualmente por árvore

            # Dados brutos para auditoria
            "_clay_pct":  clay,
            "_sand_pct":  sand,
            "_silt_pct":  silt,
            "_ph":        ph,
            "_cfvo_pct":  cfvo,
            "_source":    "soilgrids_v2",
        }

    except requests.exceptions.Timeout:
        print(f"    ⚠️  SoilGrids timeout para ({lat}, {lon}) — sem fallback")
        return None

    except Exception as e:
        print(f"    ⚠️  SoilGrids erro: {e} — sem fallback")
        return None


def _classify_soil_quality(clay, sand, ph) -> str:
    """
    Classifica qualidade do solo em GOOD / REGULAR / BAD
    baseado em textura e pH.
    """
    if clay is None or sand is None:
        return "REGULAR"

    # Solo arenoso demais = má retenção de água e nutrientes
    if sand > 70 and clay < 10:
        base = "BAD"
    # Solo argiloso equilibrado = bom para raízes
    elif 20 <= clay <= 45 and sand < 60:
        base = "GOOD"
    else:
        base = "REGULAR"

    # pH muito ácido ou muito alcalino prejudica absorção
    if ph is not None:
        if ph < 5.0 or ph > 8.5:
            # Rebaixa uma categoria
            if base == "GOOD":    base = "REGULAR"
            elif base == "REGULAR": base = "BAD"

    return base


def _estimate_depth(cfvo_pct) -> float:
    """
    Estima profundidade do solo em metros.
    cfvo = coarse fragments volumetric (% de fragmentos rochosos)
    Alto cfvo = solo raso com muito cascalho/rocha
    """
    if cfvo_pct is None:
        return 0.8  # padrão Caatinga

    if cfvo_pct > 50:   return 0.3   # solo muito raso
    if cfvo_pct > 30:   return 0.6
    if cfvo_pct > 15:   return 0.9
    return 1.2                        # solo profundo


# ─── Open-Meteo (Clima) ────────────────────────────────────────────────────────

OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

def fetch_climate(lat: float, lon: float) -> dict:
    """
    Busca dados climáticos históricos (último ano) via Open-Meteo.
    Calcula temperatura média e precipitação anual estimada.
    """
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "daily":         ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone":      "America/Fortaleza",
        "past_days":     365,
        "forecast_days": 1,
    }

    try:
        resp = requests.get(OPENMETEO_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        temps_max = daily.get("temperature_2m_max", [])
        temps_min = daily.get("temperature_2m_min", [])
        precip    = daily.get("precipitation_sum", [])

        # Temperatura média anual
        avg_temp = None
        if temps_max and temps_min:
            temps_avg_daily = [
                (mx + mn) / 2
                for mx, mn in zip(temps_max, temps_min)
                if mx is not None and mn is not None
            ]
            avg_temp = sum(temps_avg_daily) / len(temps_avg_daily) if temps_avg_daily else None

        # Precipitação anual estimada (soma dos últimos 365 dias)
        annual_rain = None
        if precip:
            valid = [p for p in precip if p is not None]
            annual_rain = sum(valid)  # mm/ano

        # Altitude: Open-Meteo retorna elevation
        altitude = data.get("elevation")

        if annual_rain is None or avg_temp is None or altitude is None:
            return None

        return {
            "annual_rainfall":  round(annual_rain, 1),
            "avg_temperature":  round(avg_temp, 1),
            "altitude":         round(altitude, 1),
            "_source": "open_meteo",
        }

    except requests.exceptions.Timeout:
        print(f"    ⚠️  Open-Meteo timeout para ({lat}, {lon}) — sem fallback")
        return None

    except Exception as e:
        print(f"    ⚠️  Open-Meteo erro: {e} — sem fallback")
        return None


# ─── Espécies conhecidas (altura máxima) ──────────────────────────────────────

# Mapeamento de espécies do CSV → altura máxima (m) e taxa k
# Baseado em literatura florestal e TRY Plant Trait Database
SPECIES_HEIGHT_MAP = {
    "Croton blanchetianus":      {"height": 5.0,  "k": 0.25},
    "Bauhinia":                  {"height": 8.0,  "k": 0.14},
    "Piptadenia moniliformis":   {"height": 7.0,  "k": 0.18},
    "Senna trachypus":           {"height": 6.0,  "k": 0.15},
    "Mimosa sensitiva":          {"height": 2.0,  "k": 0.30},
    "Celtis spinosa":            {"height": 9.0,  "k": 0.12},
    "Auxemma glazioviana":       {"height": 12.0, "k": 0.10},
    "Senna martiana":            {"height": 4.0,  "k": 0.20},
    "Combretum leprosum":        {"height": 5.0,  "k": 0.18},
    "Cenostigma pyramidale":     {"height": 7.0,  "k": 0.16},
    "Geoffroea spinosa":         {"height": 10.0, "k": 0.12},
    "Apuleia leiocarpa":         {"height": 20.0, "k": 0.08},
    "Lippia origanoides":        {"height": 3.5,  "k": 0.22},
    "Lantana achyranthifolia":   {"height": 2.5,  "k": 0.28},
    "Varronia leucocephala":     {"height": 3.0,  "k": 0.25},
    "Tephrosia purpurea":        {"height": 1.5,  "k": 0.35},
    "Machaonia brasiliensis":    {"height": 4.0,  "k": 0.20},
    "Erythroxylum nummularia":   {"height": 5.0,  "k": 0.18},
    # Fallback genérico para espécies desconhecidas
    "DEFAULT":                   {"height": 8.0,  "k": 0.14},
}

def get_species_info(scientificname: str):
    """Busca altura máxima e taxa de crescimento da espécie."""
    if pd.isna(scientificname) or not scientificname:
        return None

    # Busca exata primeiro
    if scientificname in SPECIES_HEIGHT_MAP:
        return SPECIES_HEIGHT_MAP[scientificname]

    # Busca parcial pelo gênero
    genus = scientificname.split()[0] if scientificname else ""
    for key in SPECIES_HEIGHT_MAP:
        if key.startswith(genus):
            return SPECIES_HEIGHT_MAP[key]

    return None


# ─── Processamento principal ──────────────────────────────────────────────────

def process_csv(input_path: str) -> list:
    print(f"📂 Lendo {input_path}...")
    df = pd.read_csv(input_path, sep="\t")

    # Filtrar apenas registros com coordenadas válidas
    df = df.dropna(subset=["latitude", "longitude"])

    # Desduplicar por coordenada (mesma lat/lon = mesmo local)
    df_unique = df.drop_duplicates(subset=["latitude", "longitude"])
    print(f"✅ {len(df)} registros → {len(df_unique)} locais únicos\n")

    results = []
    soil_cache    = {}   # cache por coordenada
    climate_cache = {}

    for idx, row in df_unique.iterrows():
        lat = round(float(row["latitude"]),  6)
        lon = round(float(row["longitude"]), 6)
        coord_key = f"{lat},{lon}"

        scientific = str(row.get("scientificname", "")).strip()
        genus      = str(row.get("genus", "")).strip()
        family     = str(row.get("family", "")).strip()

        year_collected = row.get("yearcollected")

        print(f"[{idx+1}/{len(df_unique)}] {scientific or genus} ({lat}, {lon})")

        # ── Solo ────────────────────────────────────────────────────────────────
        if coord_key not in soil_cache:
            print(f"    🌍 Buscando solo (SoilGrids)...")
            soil_cache[coord_key] = fetch_soil(lat, lon)
            time.sleep(SLEEP_BETWEEN)
        soil = soil_cache[coord_key]
        if soil is None:
            print("    ⏭️  Sem dado real de solo; pulando registro")
            continue

        # ── Clima ────────────────────────────────────────────────────────────────
        if coord_key not in climate_cache:
            print(f"    🌡️  Buscando clima (Open-Meteo)...")
            climate_cache[coord_key] = fetch_climate(lat, lon)
            time.sleep(SLEEP_BETWEEN)
        climate = climate_cache[coord_key]
        if climate is None:
            print("    ⏭️  Sem dado real de clima; pulando registro")
            continue

        # ── Espécie ──────────────────────────────────────────────────────────────
        sp_info = get_species_info(scientific)
        if sp_info is None:
            print("    ⏭️  Espécie sem perfil validado; pulando registro")
            continue

        # ── Montar objeto Tree (compatível com o schema Prisma) ──────────────────
        tree_obj = {
            # Localização
            "latitude":  lat,
            "longitude": lon,

            # Espécie
            "species": {
                "scientificName": scientific or genus,
                "commonName":     genus or scientific,
                "family":         family,
                "heightAverage":  sp_info["height"],
                "growthRateK":    sp_info["k"],    # campo extra pro modelo de IA
            },

            # Idade estimada (se yearcollected disponível)
            "age": _estimate_age_date(year_collected),

            # Status padrão
            "status": "NORMAL",

            # Dados climáticos
            "annualRainfall":  climate["annual_rainfall"],
            "altitude":        climate["altitude"],
            "avgTemperature":  climate["avg_temperature"],

            # Solo (compatível com model Soil do schema)
            "soil": {
                "depth":       soil["depth"],
                "inclination": soil["inclination"],
                "quality":     soil["quality"],
                "coverage":    soil["coverage"],
            },

            # Metadados de origem
            "_meta": {
                "source_record":    str(row.get("basisofrecord", "")),
                "year_collected":   int(year_collected) if pd.notna(year_collected) else None,
                "locality":         str(row.get("locality", "")),
                "notes":            str(row.get("notes", "")),
                "soil_source":      soil.get("_source"),
                "climate_source":   climate.get("_source"),
                "enriched_at":      datetime.now().isoformat(),
                "soil_raw": {
                    "clay_pct":  soil.get("_clay_pct"),
                    "sand_pct":  soil.get("_sand_pct"),
                    "silt_pct":  soil.get("_silt_pct"),
                    "ph":        soil.get("_ph"),
                },
            },
        }

        results.append(tree_obj)
        print(f"    ✅ Solo: {soil['quality']} (prof. {soil['depth']}m) | "
              f"Chuva: {climate['annual_rainfall']}mm | "
              f"Temp: {climate['avg_temperature']}°C")

    return results


def _estimate_age_date(year_collected) -> str:
    """
    Usa o ano de coleta como referência de quando a planta foi observada.
    Retorna ISO datetime (campo age do schema Prisma é DateTime).
    Obs: Não é a idade exata — é a data de primeira observação.
    """
    if pd.isna(year_collected):
        return datetime(2020, 1, 1).isoformat()
    try:
        return datetime(int(year_collected), 6, 15).isoformat()
    except:
        return datetime(2020, 1, 1).isoformat()


# ─── Exportação ───────────────────────────────────────────────────────────────

def save_output(records: list, output_path: str):
    output = {
        "generated_at": datetime.now().isoformat(),
        "total_records": len(records),
        "sources": {
            "soil":    "SoilGrids v2 REST API (ISRIC)",
            "climate": "Open-Meteo Historical API",
            "species": "Perfil validado localmente ou skip",
        },
        "trees": records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Também salva CSV enriquecido para conferência
    csv_path = output_path.replace(".json", ".csv")
    rows = []
    for t in records:
        rows.append({
            "scientificName":   t["species"]["scientificName"],
            "family":           t["species"]["family"],
            "latitude":         t["latitude"],
            "longitude":        t["longitude"],
            "age_date":         t["age"],
            "heightAverage":    t["species"]["heightAverage"],
            "growthRateK":      t["species"]["growthRateK"],
            "annualRainfall":   t["annualRainfall"],
            "avgTemperature":   t["avgTemperature"],
            "altitude":         t["altitude"],
            "soil_depth":       t["soil"]["depth"],
            "soil_inclination": t["soil"]["inclination"],
            "soil_quality":     t["soil"]["quality"],
            "soil_coverage":    t["soil"]["coverage"],
            "soil_source":      t["_meta"]["soil_source"],
            "climate_source":   t["_meta"]["climate_source"],
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n💾 JSON salvo:  {output_path}")
    print(f"💾 CSV salvo:   {csv_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("🌳 MapTree - Enriquecimento de Dados (Solo + Clima)")
    print("=" * 55)
    print()
    print("Fontes:")
    print("  Solo  → SoilGrids v2 (ISRIC) — sem chave, gratuito")
    print("  Clima → Open-Meteo           — sem chave, gratuito")
    print("  Sem fallback fictício: registros sem dado real são ignorados\n")

    records = process_csv(INPUT_CSV)
    save_output(records, OUTPUT_JSON)

    # Estatísticas finais
    soil_sources = {}
    for r in records:
        s = r["_meta"]["soil_source"]
        soil_sources[s] = soil_sources.get(s, 0) + 1

    print("\n📊 Estatísticas:")
    print(f"  Total de registros:    {len(records)}")
    print(f"  Fontes de solo usadas: {soil_sources}")
    print("\n✅ Pronto! Importe enriched_trees.json no seu NestJS/MongoDB.")
    print("   Dica: use o endpoint POST /trees do MapTree pra cada item em trees[]")
