#!/usr/bin/env python3
"""
Converte dados_tabuleiro.csv para JSON no formato esperado pelo backend.
"""

import csv
import json
from datetime import datetime

CSV_PATH = '/home/walter/maptree_IA/dados_tabuleiro.csv'
JSON_PATH = '/home/walter/maptree_IA/arvores_tabuleiro.json'

def get_risk_status(notes):
    """Determina status de risco baseado nas notas."""
    if not notes:
        return 'NORMAL'
    notes_lower = notes.lower()
    if 'morto' in notes_lower or 'seco' in notes_lower:
        return 'CRITICAL'
    if 'doente' in notes_lower or 'praga' in notes_lower:
        return 'UNDER_OBSERVATION'
    if 'poda' in notes_lower or 'risco' in notes_lower or 'antracnose' in notes_lower:
        return 'TO_PRUNE'
    return 'NORMAL'

def convert():
    """Converte CSV para formato JSON esperado pela API."""
    trees = []

    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for i, row in enumerate(reader):
            try:
                # Colunas longitude e latitude (nesta ordem no CSV)
                lat_str = row.get('latitude', '').strip()
                lng_str = row.get('longitude', '').strip()

                # Pula registros sem coordenadas válidas
                if not lat_str or not lng_str:
                    continue

                lat = float(lat_str)
                lng = float(lng_str)

                # Valida range de coordenadas (Brasil)
                if not (-20 < lat < 10 and -75 < lng < -30):
                    print(f"Linha {i+2}: Coordenadas fora do range válido, pulando...")
                    continue

                # Calcula idade baseada no ano da coleta
                year = row.get('yearcollected', '').strip()
                try:
                    year_int = int(year)
                    age_years = datetime.now().year - year_int
                    if age_years < 1 or age_years > 100:
                        age_years = 10
                except:
                    age_years = 10

                # Nome científico
                scientific_name = row.get('scientificname', '').strip()
                if not scientific_name:
                    genus = row.get('genus', '').strip()
                    species = row.get('species', '').strip()
                    scientific_name = f"{genus} {species}".strip() if genus else 'Unknown'

                family = row.get('family', 'Unknown').strip()
                city = row.get('county', 'Tabuleiro do Norte').strip()
                notes = row.get('notes', '') or ''

                # Limpa notes
                if notes:
                    notes = notes.replace('\n', ' ').replace('\r', ' ').strip()
                    if notes.lower() == 'nan':
                        notes = ''

                tree_data = {
                    'name': scientific_name,
                    'family': family,
                    'lat': lat,
                    'lng': lng,
                    'city': city if city else 'Tabuleiro do Norte',
                    'prediction': {
                        'risk_status': get_risk_status(notes),
                        'age_years': age_years,
                    },
                    'notes': notes if notes else None
                }

                trees.append(tree_data)

            except Exception as e:
                print(f"Erro na linha {i+2}: {e}")
                continue

    return trees

if __name__ == '__main__':
    print("=" * 60)
    print("CONVERSOR dados_tabuleiro.csv -> JSON")
    print("=" * 60)

    trees = convert()

    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(trees, f, indent=2, ensure_ascii=False)

    print(f"\n{len(trees)} árvores salvas em {JSON_PATH}")
    print("\nPara importar, atualize o caminho no trees.controller.ts para:")
    print(f"  const path = '{JSON_PATH}';")
    print("\nOu use o endpoint:")
    print("  curl -X POST http://localhost:4000/api/trees/import-external")
