#!/usr/bin/env python3
"""
Converte tree_dataset.csv para JSON no formato esperado pelo backend
e faz upload via API para importar as árvores.
"""

import csv
import json
import random
from datetime import datetime

# Configuração
CSV_PATH = '/home/walter/maptree_IA/arvores_plantae.csv'
JSON_PATH = '/home/walter/maptree_IA/arvores_mapeadas.json'
API_URL = 'http://localhost:4000/api/trees/import-external'

# Mapeamento de famílias para espécies comuns
FAMILY_SPECIES = {
    'Fabaceae': ('Inga edulis', 'Fabaceae'),
    'Euphorbiaceae': ('Croton blanchetianus', 'Euphorbiaceae'),
    'Vitaceae': ('Clematicissus pruinata', 'Vitaceae'),
    'Rubiaceae': ('Cordiera sessilis', 'Rubiaceae'),
    'Anacardiaceae': ('Anacardium occidentale', 'Anacardiaceae'),
    'Arecaceae': ('Copernicia prunifera', 'Arecaceae'),
    'Cactaceae': ('Cereus jamacaru', 'Cactaceae'),
    'Malvaceae': ('Ceiba pentandra', 'Malvaceae'),
    'Myrtaceae': ('Psidium guajava', 'Myrtaceae'),
    'Sapotaceae': ('Manilkara zapota', 'Sapotaceae'),
}

def get_species(family, genus, species):
    """Retorna nome científico completo ou usa família como base."""
    if species and species.strip():
        return f"{genus or ''} {species}".strip()
    elif genus and genus.strip():
        return genus.strip()
    else:
        # Usa espécie padrão da família
        return FAMILY_SPECIES.get(family, (f"{genus or 'Unknown'} sp.", family))[0]

def get_risk_status(notes):
    """Determina status de risco baseado nas notas."""
    if not notes:
        return 'NORMAL'
    notes_lower = notes.lower()
    if 'morto' in notes_lower or 'seco' in notes_lower:
        return 'CRITICAL'
    if 'doente' in notes_lower or 'praga' in notes_lower:
        return 'UNDER_OBSERVATION'
    if 'poda' in notes_lower or 'risco' in notes_lower:
        return 'TO_PRUNE'
    return 'NORMAL'

def convert_csv_to_json():
    """Converte CSV para formato JSON esperado pela API."""
    trees = []

    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            try:
                lat = float(row.get('latitude', 0))
                lng = float(row.get('longitude', 0))

                # Pula registros sem coordenadas válidas
                if lat == 0 and lng == 0:
                    print(f"Linha {i+2}: Coordenadas inválidas, pulando...")
                    continue

                # Calcula idade baseada no ano da coleta
                year = row.get('yearcollected', '2020')
                try:
                    year_int = int(year)
                    age_years = datetime.now().year - year_int
                except:
                    age_years = random.randint(5, 20)

                family = row.get('family', 'Unknown')
                genus = row.get('genus', '')
                species = row.get('species', '')
                scientific_name = get_species(family, genus, species)
                notes = row.get('notes', '') or ''

                # Limpa notes de caracteres problemáticos
                if notes:
                    notes = notes.replace('\n', ' ').replace('\r', ' ')

                tree_data = {
                    'name': scientific_name,
                    'family': family,
                    'lat': lat,
                    'lng': lng,
                    'city': row.get('county', 'Tabuleiro do Norte'),
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

def save_json(trees):
    """Salva JSON no formato esperado."""
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(trees, f, indent=2, ensure_ascii=False)

    print(f"\n{len(trees)} árvores salvas em {JSON_PATH}")

def import_via_api():
    """Importa árvores via API."""
    import requests

    print(f"\nImportando {JSON_PATH} via API...")
    print(f"URL: {API_URL}")

    try:
        # A API lê o arquivo diretamente do path hardcoded
        # Então só precisamos chamar o endpoint
        response = requests.post(API_URL, timeout=30)

        if response.status_code == 200:
            result = response.json()
            print(f"\nImportação concluída!")
            print(f"  - Importadas: {result.get('imported', 0)}")
            print(f"  - Erros: {result.get('errors', 0)}")
        else:
            print(f"Erro na API: {response.status_code}")
            print(f"Resposta: {response.text}")

    except requests.exceptions.ConnectionError:
        print("Erro: Não foi possível conectar à API.")
        print("Verifique se o backend está rodando em http://localhost:4000")
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == '__main__':
    print("=" * 60)
    print("CONVERSOR CSV -> JSON PARA IMPORTAÇÃO")
    print("=" * 60)

    # Converte CSV para JSON
    trees = convert_csv_to_json()
    save_json(trees)

    # Pergunta se deve importar
    print("\n" + "=" * 60)
    print("Para importar as árvores, execute:")
    print(f"  curl -X POST {API_URL}")
    print("\nOu pressione Enter para tentar importar agora...")
    input()

    import_via_api()
