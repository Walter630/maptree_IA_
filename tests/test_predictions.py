"""
MapTree IA - Testes de coerência da predição.

Como não há altura MEDIDA em campo ainda (o alvo histórico é calculado), não dá
para medir erro contra a realidade. Então validamos PROPRIEDADES FÍSICAS que a
predição sempre tem que respeitar, mais alguns casos conhecidos de sanity check.

Rodar:
    pytest -q

Quando houver altura medida em campo, adicionar aqui um teste que compara
predição vs. medido e reporta MAE (erro médio em metros).
"""

import math
import pytest

from generate_data import (
    estimate_height,
    estimate_annual_growth,
    simular_crescimento_mensal,
    classify_risk,
    soil_modifier,
    climate_modifier,
    pruning_vigor_modifier,
    phyllotaxis_canopy,
)


# ─── Coerência física: altura (Von Bertalanffy) ──────────────────────────────

def test_altura_nunca_passa_do_maximo_da_especie():
    """A altura estimada nunca deve ultrapassar ~o máximo da espécie."""
    h_max = 10.0
    for idade in range(0, 200, 5):
        h = estimate_height(idade, h_max, species_k=0.12, total_modifier=1.0)
        assert h <= h_max * 1.03 + 1e-6, f"idade {idade}: {h} > {h_max}"


def test_altura_cresce_com_a_idade():
    """Mais idade → não pode diminuir a altura (monotônico)."""
    h_max, k = 12.0, 0.1
    anterior = -1.0
    for idade in range(0, 80, 2):
        h = estimate_height(idade, h_max, k, 1.0)
        assert h >= anterior - 1e-9, f"altura caiu na idade {idade}"
        anterior = h


def test_arvore_recem_nascida_tem_altura_quase_zero():
    assert estimate_height(0, 10.0, 0.12, 1.0) == pytest.approx(0.0, abs=0.05)


def test_crescimento_desacelera_com_a_idade():
    """Crescimento anual deve diminuir conforme a árvore se aproxima do máximo."""
    h_max, k = 10.0, 0.15
    h_jovem = estimate_height(3, h_max, k, 1.0)
    h_velha = estimate_height(40, h_max, k, 1.0)
    g_jovem = estimate_annual_growth(h_jovem, h_max, k, 1.0)
    g_velha = estimate_annual_growth(h_velha, h_max, k, 1.0)
    assert g_jovem > g_velha


# ─── Coerência física: modificadores de ambiente ─────────────────────────────

def test_solo_bom_cresce_mais_que_solo_ruim():
    bom = soil_modifier(depth=1.5, inclination=2, quality="GOOD", coverage=0.1)
    ruim = soil_modifier(depth=0.3, inclination=30, quality="BAD", coverage=0.9)
    assert bom > ruim


def test_clima_favoravel_cresce_mais_que_seca():
    favoravel = climate_modifier(annual_rainfall=1000, altitude=200, avg_temperature=26)
    seca = climate_modifier(annual_rainfall=200, altitude=200, avg_temperature=26)
    assert favoravel > seca


def test_poda_reduz_vigor_e_nunca_aumenta():
    """Modificador de poda: sem poda = 1.0; mais podas = menor; limite inferior 0.72."""
    assert pruning_vigor_modifier(0) == 1.0
    assert pruning_vigor_modifier(1) < 1.0
    assert pruning_vigor_modifier(2) <= pruning_vigor_modifier(1)
    assert pruning_vigor_modifier(50) >= 0.72


def test_solo_melhor_gera_arvore_mais_alta_na_mesma_idade():
    h_max, k, idade = 12.0, 0.12, 10
    mod_bom = soil_modifier(1.5, 2, "GOOD", 0.1) * climate_modifier(1000, 200, 26)
    mod_ruim = soil_modifier(0.3, 30, "BAD", 0.9) * climate_modifier(200, 200, 26)
    assert estimate_height(idade, h_max, k, mod_bom) > estimate_height(idade, h_max, k, mod_ruim)


# ─── Coerência física: prazo até o fio ────────────────────────────────────────

def test_arvore_acima_do_fio_atinge_em_zero_meses():
    sim = simular_crescimento_mensal(altura_atual=7.0, altura_fio=6.5, taxa_k=0.1, altura_max=10.0)
    assert sim["meses_ate_o_fio"] == 0


def test_especie_baixa_nunca_atinge_fio_alto():
    """Espécie que não chega perto da altura do fio → None (não atinge)."""
    sim = simular_crescimento_mensal(altura_atual=1.0, altura_fio=6.5, taxa_k=0.2, altura_max=3.0)
    assert sim["meses_ate_o_fio"] is None


def test_arvore_alta_e_rapida_atinge_o_fio_em_tempo_finito():
    sim = simular_crescimento_mensal(altura_atual=4.0, altura_fio=6.5, taxa_k=0.2, altura_max=12.0)
    assert sim["meses_ate_o_fio"] is not None
    assert sim["meses_ate_o_fio"] > 0


def test_solo_melhor_atinge_fio_mais_cedo():
    base = dict(altura_atual=4.0, altura_fio=6.5, taxa_k=0.15, altura_max=12.0)
    rapido = simular_crescimento_mensal(**base, modifier=1.2)["meses_ate_o_fio"]
    lento = simular_crescimento_mensal(**base, modifier=0.7)["meses_ate_o_fio"]
    assert rapido is not None and lento is not None
    assert rapido <= lento


# ─── Coerência física: classificação de risco ────────────────────────────────

def test_risco_critico_quando_acima_do_fio():
    assert classify_risk(estimated_height=7.0, wire_height=6.5, days_to_wire=0, will_reach_wire=1) == "CRITICAL"


def test_risco_normal_quando_longe_do_fio():
    assert classify_risk(estimated_height=2.0, wire_height=6.5, days_to_wire=-1, will_reach_wire=0) == "NORMAL"


def test_risco_to_prune_quando_atinge_em_breve():
    r = classify_risk(estimated_height=5.0, wire_height=6.5, days_to_wire=90, will_reach_wire=1)
    assert r in ("TO_PRUNE", "UNDER_OBSERVATION", "CRITICAL")


# ─── Filotaxia (separada da altura) ───────────────────────────────────────────

def test_filotaxia_usa_angulo_aureo():
    canopy = phyllotaxis_canopy(branch_count=5)
    assert canopy["golden_angle_deg"] == 137.5
    assert canopy["fibonacci_stage"] == 5  # 5 é termo de Fibonacci


def test_filotaxia_nao_afeta_altura():
    """A altura não deve depender de filotaxia/galhos — só idade, espécie, ambiente."""
    h1 = estimate_height(10, 10.0, 0.12, 1.0)
    # phyllotaxis_canopy não entra em estimate_height; valor independe de galhos.
    _ = phyllotaxis_canopy(branch_count=13)
    h2 = estimate_height(10, 10.0, 0.12, 1.0)
    assert h1 == h2


# ─── Casos conhecidos (sanity check para o especialista revisar) ──────────────

CASOS_CONHECIDOS = [
    # (descrição, idade, h_max, k, modifier, faixa_esperada_metros)
    ("Croton jovem (5 anos, rápido)", 5, 5.0, 0.25, 1.0, (2.5, 4.5)),
    ("Celtis maduro (25 anos, lento)", 25, 9.0, 0.12, 1.0, (7.0, 9.0)),
    ("Apuleia grande e lenta (30 anos)", 30, 20.0, 0.08, 1.0, (11.0, 19.0)),
]


@pytest.mark.parametrize("desc,idade,h_max,k,mod,faixa", CASOS_CONHECIDOS)
def test_casos_conhecidos_dentro_da_faixa_esperada(desc, idade, h_max, k, mod, faixa):
    h = estimate_height(idade, h_max, k, mod)
    lo, hi = faixa
    assert lo <= h <= hi, f"{desc}: altura {h} fora da faixa {faixa}"
