import numpy as np
import pandas as pd
import math
import random
from datetime import datetime, timedelta

# ─── Constantes ───────────────────────────────────────────────────────────────
GOLDEN_ANGLE = 137.5  
WIRE_HEIGHT = 6.5     
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


SPECIES_DATA = [
    {"name": "Algaroba", "scientific": "Prosopis juliflora", "height_max": 12.0, "k": 0.18},
    {"name": "Catingueira", "scientific": "Caesalpinia pyramidalis", "height_max": 8.0, "k": 0.15},
    {"name": "Juazeiro", "scientific": "Ziziphus joazeiro", "height_max": 10.0, "k": 0.12},
    {"name": "Oiticica", "scientific": "Licania rigida", "height_max": 15.0, "k": 0.10},
    {"name": "Craibeira", "scientific": "Tabebuia aurea", "height_max": 14.0, "k": 0.11},
    {"name": "Sabiá", "scientific": "Mimosa caesalpiniifolia", "height_max": 9.0, "k": 0.20},
    {"name": "Angico", "scientific": "Anadenanthera colubrina", "height_max": 18.0, "k": 0.09},
    {"name": "Nim", "scientific": "Azadirachta indica", "height_max": 13.0, "k": 0.16},
]

# ─── Funções de Modelo ────────────────────────────────────────────────────────

def simular_crescimento_fibonacci(altura_atual: float, altura_fio: float, taxa_k: float) -> dict:
    """
    Simula o crescimento mensal usando a proporção áurea ($\phi$) derivada de Fibonacci.
    """
    if altura_atual >= altura_fio:
        return {"meses_ate_o_fio": 0, "altura_final": altura_atual, "historico": []}

    meses = 0
    altura_simulada = altura_atual
    historico = []
    fib = [1, 1] 

    while altura_simulada < altura_fio:
        fator_vigor = fib[-1] / fib[-2] if len(fib) > 2 else 1.0
        incremento_mensal = (taxa_k / 12) * fator_vigor
        
        assert incremento_mensal > 0, "Erro: O incremento deve ser positivo."
        
        altura_simulada += incremento_mensal
        meses += 1
        fib.append(fib[-1] + fib[-2])
        
        historico.append({"mes": meses, "altura": round(altura_simulada, 3)})
        
        if meses > 1200: break # Safety break (100 anos))

    return {"meses_ate_o_fio": meses, "altura_final": round(altura_simulada, 2), "historico": historico}

def soil_modifier(depth: float, inclination: float, quality: str, coverage: float) -> float:
    quality_map = {"GOOD": 1.2, "REGULAR": 1.0, "BAD": 0.7}
    q = quality_map.get(quality, 1.0)
    depth_mod = min(depth / 1.5, 1.0)
    slope_mod = 1.0 - (inclination / 90.0) * 0.3
    coverage_mod = 1.0 - (coverage * 0.4)
    return q * depth_mod * slope_mod * coverage_mod

def climate_modifier(annual_rainfall: float, altitude: float, avg_temperature: float) -> float:
    rain_mod = 0.6 + (annual_rainfall / 400) * 0.4 if annual_rainfall < 400 else 0.85 if annual_rainfall > 1200 else 1.0
    temp_mod = 0.75 if avg_temperature < 18 or avg_temperature > 35 else 1.0
    alt_mod = 1.0 if altitude < 600 else 0.9
    return rain_mod * temp_mod * alt_mod

# ─── Gerador de Dataset ───────────────────────────────────────────────────────

def generate_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    rows = []

    for i in range(n_samples):
        sp = random.choice(SPECIES_DATA)
        
        # Simulação de ambiente
        depth = random.uniform(0.3, 2.0)
        inclination = random.uniform(0, 45)
        quality = random.choice(["GOOD", "REGULAR", "BAD"])
        coverage = random.uniform(0, 0.9)
        
        # Modificadores
        s_mod = soil_modifier(depth, inclination, quality, coverage)
        c_mod = climate_modifier(random.normalvariate(700, 200), random.uniform(50, 900), random.normalvariate(27, 3))
        
        # Aplicamos os modificadores na taxa K original da espécie
        taxa_k_efetiva = sp["k"] * s_mod * c_mod
        
        # Altura inicial aleatória (árvore já plantada)
        altura_inicial = random.uniform(0.5, 4.0)
        
        # CHAMADA DA SUA NOVA FUNÇÃO DE FIBONACCI
        res_fib = simular_crescimento_fibonacci(altura_inicial, WIRE_HEIGHT, taxa_k_efetiva)

        rows.append({
            "sample_id": i,
            "species": sp["name"],
            "altura_inicial": round(altura_inicial, 2),
            "taxa_k_efetiva": round(taxa_k_efetiva, 4),
            "meses_ate_o_fio": res_fib["meses_ate_o_fio"],
            "altura_final": res_fib["altura_final"],
            "soil_quality": quality,
            "risk_status": "URGENT" if res_fib["meses_ate_o_fio"] < 6 else "NORMAL"
        })

    return pd.DataFrame(rows)

# ─── Execução Única ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🌳 MapTree: Validando lógica de Fibonacci...")
    # Teste rápido de sanidade
    teste = simular_crescimento_fibonacci(2.0, 5.0, 0.15)
    assert teste["altura_final"] >= 5.0
    print(f"✅ Teste OK: {teste['meses_ate_o_fio']} meses para atingir o fio.")

    print("\n📊 Gerando dataset sintético...")
    df = generate_dataset(n_samples=1000)
    df.to_csv("tree_dataset_fibonacci.csv", index=False)
    print("💾 Salvo em tree_dataset_fibonacci.csv")
    print(df.head())