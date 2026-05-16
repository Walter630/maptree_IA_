import pandas as pd
import json
import requests
from typing import Optional

API_URL = "http://localhost:8000"
session = request.Session()

def carregar_modelos_ia():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            print("✅ Conectado à API de IA")
            return True
    except:
        print("⚠️ API não disponível, usando fallback local")
    return False

def estimar_altura_ia(idade: float, altura_max: float, pruning_count: int = 0, 
                       rainfall: float = 700.0, temp: float = 27.0) -> dict:
    req = {
        "age_years": idade,
        "pruning_count": pruning_count,
        "status": "NORMAL",
        "species": {
            "height_average": altura_max,
            "growth_rate_k": 0.12
        },
        "soil": {
            "depth": 1.0,
            "inclination": 5.0,
            "quality": "REGULAR",
            "coverage": 0.3
        },
        "annual_rainfall": rainfall,
        "altitude": 300.0,
        "avg_temperature": temp,
        "has_fertilization": False,
        "has_irrigation": False,
        "nearby_trees_count": 0,
        "avg_neighbor_distance": 10.0,
        "wire_height": 5.5
    }
    
    try:
        resp = session.post(f"{API_URL}/predict-wire-risk", json=req, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        pass
    
    return None

altura_especies = {
    'Croton': 6.0,
    'Bauhinia': 8.0,
    'Myrtaceae': 10.0,
    'Combretum': 7.0,
    'Cordia': 8.0,
    'Annonaceae': 8.0,
    'Sapindaceae': 10.0,
    'Malvaceae': 9.0,
    'Solanum': 5.0,
    'Ipomoea': 4.0,
}

ALTURA_POSTE_URBANO = 6.0
ALTURA_POSTE_RURAL = 8.0


def get_altura_poste(city: str, state: str = "Ceará") -> float:
    areas_rurais = ['rural', 'fazenda', 'sít', 'chácara', 'zona rural']
    city_lower = city.lower() if city else ""
    if any(palavra in city_lower for palavra in areas_rurais):
        return ALTURA_POSTE_RURAL
    return ALTURA_POSTE_URBANO


def calcular_status(altura_estimada: float, altura_poste: float) -> str:
    percentage_reached = (altura_estimada / altura_poste) * 100
    
    if percentage_reached >= 90:
        return "CRITICAL"
    elif percentage_reached >= 80:
        return "UNDER_OBSERVATION"
    elif percentage_reached >= 70:
        return "TO_PRUNE"
    return "NORMAL"


def processar_dados_tabuleiro():
    df = pd.read_csv('dados_tabuleiro.csv', sep='\t')
    
    familias_alvo = ['Fabaceae', 'Euphorbiaceae', 'Boraginaceae', 'Rubiaceae', 'Combretaceae']
    df_arvores = df[df['family'].isin(familias_alvo)].copy()
    
    resultado = []
    for _, row in df_arvores.iterrows():
        genus = row.get('genus', '')
        altura_max = altura_especies.get(genus, 7.0)
        
        idade = 2026 - int(row['yearcollected']) if pd.notna(row.get('yearcollected')) else 5
        
        predicao = estimar_altura_ia(idade, altura_max)
        
        if predicao:
            arvore = {
                "id": f"tab_{row['scientificname'][:8].replace(' ', '_')}_{row.get('yearcollected', 'X')}",
                "source": "tabuleiro",
                "name": row['scientificname'],
                "family": row['family'],
                "genus": genus,
                "lat": float(row['latitude']),
                "lng": float(row['longitude']),
                "city": row.get('county', 'Tabuleiro do Norte'),
                "state": "Ceará",
                "year_collected": int(row['yearcollected']) if pd.notna(row.get('yearcollected')) else None,
                "age_years": idade,
                "prediction": {
                    "estimated_height_m": predicao.get('estimated_height_m'),
                    "wire_height_m": predicao.get('wire_height_m'),
                    "will_reach_wire": predicao.get('will_reach_wire'),
                    "days_to_wire": predicao.get('days_to_wire'),
                    "months_to_wire": predicao.get('months_to_wire'),
                    "risk_status": predicao.get('risk_status'),
                    "alert": predicao.get('alert'),
                },
                "canopy": predicao.get('canopy'),
                "fibonacci_info": predicao.get('fibonacci_info'),
            }
        else:
            altura_estimada = round(idade * 0.5, 2)
            city = row.get('county', 'Tabuleiro do Norte')
            altura_poste = get_altura_poste(city)
            status_risco = calcular_status(altura_estimada, altura_poste)
            arvore = {
                "id": f"tab_{row['scientificname'][:8].replace(' ', '_')}_{row.get('yearcollected', 'X')}",
                "source": "tabuleiro",
                "name": row['scientificname'],
                "family": row['family'],
                "genus": genus,
                "lat": float(row['latitude']),
                "lng": float(row['longitude']),
                "city": city,
                "state": "Ceará",
                "year_collected": int(row['yearcollected']) if pd.notna(row.get('yearcollected')) else None,
                "age_years": idade,
                "prediction": {
                    "estimated_height_m": altura_estimada,
                    "wire_height_m": altura_poste,
                    "will_reach_wire": altura_estimada > altura_poste,
                    "days_to_wire": None,
                    "months_to_wire": None,
                    "risk_status": status_risco,
                    "alert": "⚠️ Árvore já atingiu ou ultrapassou a altura do fio! Poda urgente." if status_risco == "CRITICAL" else None,
                },
            }
        
        resultado.append(arvore)
    
    return resultado

def processar_dados_gbif():
    df = pd.read_csv('arvores_mapeadas_gbif.csv', sep='\t', on_bad_lines='skip')
    
    df_plantae = df[df['kingdom'] == 'Plantae'].copy()
    
    resultado = []
    for _, row in df_plantae.iterrows():
        family = row.get('family', '')
        genus = row.get('genus', '')
        
        altura_max = altura_especies.get(family, altura_especies.get(genus, 6.0))
        
        year = row.get('year')
        if pd.notna(year):
            try:
                idade = 2026 - int(year)
            except:
                idade = 5
        else:
            idade = 5
        
        predicao = estimar_altura_ia(idade, altura_max)
        
        lat = row.get('decimalLatitude')
        lng = row.get('decimalLongitude')
        
        if pd.isna(lat) or pd.isna(lng):
            continue
        
        if predicao:
            arvore = {
                "id": f"gbif_{row['gbifID']}",
                "source": "gbif",
                "name": row.get('scientificName', row.get('species', '')),
                "family": family,
                "genus": genus,
                "lat": float(lat),
                "lng": float(lng),
                "city": row.get('locality', ''),
                "state": row.get('stateProvince', 'Ceará'),
                "year_collected": int(year) if pd.notna(year) else None,
                "age_years": idade,
                "prediction": {
                    "estimated_height_m": predicao.get('estimated_height_m'),
                    "wire_height_m": predicao.get('wire_height_m'),
                    "will_reach_wire": predicao.get('will_reach_wire'),
                    "days_to_wire": predicao.get('days_to_wire'),
                    "months_to_wire": predicao.get('months_to_wire'),
                    "risk_status": predicao.get('risk_status'),
                    "alert": predicao.get('alert'),
                },
                "canopy": predicao.get('canopy'),
                "fibonacci_info": predicao.get('fibonacci_info'),
            }
        else:
            altura_estimada = round(idade * 0.5, 2)
            city = row.get('locality', '')
            state = row.get('stateProvince', 'Ceará')
            altura_poste = get_altura_poste(city, state)
            status_risco = calcular_status(altura_estimada, altura_poste)
            arvore = {
                "id": f"gbif_{row['gbifID']}",
                "source": "gbif",
                "name": row.get('scientificName', row.get('species', '')),
                "family": family,
                "genus": genus,
                "lat": float(lat),
                "lng": float(lng),
                "city": row.get('locality', ''),
                "state": row.get('stateProvince', 'Ceará'),
                "year_collected": int(year) if pd.notna(year) else None,
                "age_years": idade,
                "prediction": {
                    "estimated_height_m": altura_estimada,
                    "wire_height_m": altura_poste,
                    "will_reach_wire": altura_estimada > altura_poste,
                    "days_to_wire": None,
                    "months_to_wire": None,
                    "risk_status": status_risco,
                    "alert": "⚠️ Árvore já atingiu ou ultrapassou a altura do fio! Poda urgente." if status_risco == "CRITICAL" else None,
                },
            }
        
        resultado.append(arvore)
    
    return resultado

def gerar_json_para_front():
    api_disponivel = carregar_modelos_ia()
    
    print("📂 Processando dados do tabuleiro...")
    arvores_tabuleiro = processar_dados_tabuleiro()
    print(f"   → {len(arvores_tabuleiro)} árvores do tabuleiro")
    
    print("📂 Processando dados GBIF...")
    arvores_gbif = processar_dados_gbif()
    print(f"   → {len(arvores_gbif)} árvores do GBIF")
    
    todas_arvores = arvores_tabuleiro + arvores_gbif
    
    with open('arvores_mapeadas.json', 'w', encoding='utf-8') as f:
        json.dump(todas_arvores, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Total: {len(todas_arvores)} árvores processadas em arvores_mapeadas.json")

if __name__ == "__main__":
    gerar_json_para_front()