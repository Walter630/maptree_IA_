# MapTree IA - Guia do Modelo

Este documento explica como o modelo calcula altura, crescimento, risco de fio e rota de poda.

## Arquivos principais

- `generate_data.py`: regras de dominio e geracao do `tree_dataset.csv`.
- `train_models.py`: treino dos modelos em `models/`.
- `api.py`: FastAPI usada pelo backend para previsoes.
- `enrich_soil_climate.py`: enriquecimento de dados reais com solo e clima.
- `parser_maptree.py`: converte bases locais/GBIF para JSON de importacao.

## Campos do dataset

- `species_common_name`: nome comum usado internamente.
- `species_scientific_name`: nome cientifico usado para buscar perfil botanico.
- `species_height_max`: altura maxima/media esperada da especie em metros.
- `species_k`: taxa de crescimento da especie; maior valor cresce mais rapido.
- `age_years`: idade estimada da arvore em anos.
- `pruning_count`: quantidade de podas ja feitas.
- `soil_depth`: profundidade util do solo em metros.
- `soil_inclination`: inclinacao do terreno em graus.
- `soil_quality`: `GOOD`, `REGULAR` ou `BAD`.
- `soil_coverage`: impermeabilizacao/cobertura do solo, de `0.0` a `1.0`.
- `annual_rainfall`: chuva anual em milimetros.
- `altitude`: altitude em metros.
- `avg_temperature`: temperatura media em graus Celsius.
- `has_fertilization`: `1` quando ha adubacao registrada.
- `has_irrigation`: `1` quando ha irrigacao registrada.
- `nearby_trees_count`: quantidade de arvores vizinhas proximas.
- `avg_neighbor_distance`: distancia media ate arvores vizinhas em metros.
- `wire_context`: tipo aproximado de rede/localidade.
- `wire_height`: altura do fio usada no calculo, em metros.
- `pruning_modifier`: ajuste de vigor apos podas (antes chamado `fibonacci_modifier`).
- `canopy_ratio`: relacao largura/altura da copa.
- `soil_modifier`: multiplicador calculado a partir do solo.
- `climate_modifier`: multiplicador calculado a partir do clima.
- `management_modifier`: multiplicador de manejo.
- `total_modifier`: produto final dos modificadores de crescimento.
- `estimated_height`: altura atual estimada em metros.
- `annual_growth_m`: crescimento esperado nos proximos 12 meses.
- `height_next_year`: altura projetada para daqui a 1 ano.
- `days_to_wire`: dias estimados ate atingir o fio; `-1` indica que nao deve atingir.
- `risk_status`: `NORMAL`, `UNDER_OBSERVATION`, `TO_PRUNE` ou `CRITICAL`.
- `will_reach_wire`: `1` se a arvore pode atingir o fio; `0` caso contrario.

## Motor de predicao (deterministico)

A versao 2.0 **nao usa modelos de ML treinados** (`.pkl`). A predicao usa o
modelo botanico de Von Bertalanffy, calculado em tempo real. Isso porque o alvo
historico (`estimated_height`) e CALCULADO, nao medido em campo: treinar um ML
para imitar uma formula so adicionava ruido e resultados fora de contexto.

Funcoes principais (em `generate_data.py`):

- `estimate_height()`: altura atual pela curva de Von Bertalanffy.
- `estimate_annual_growth()`: crescimento esperado nos proximos 12 meses.
- `simular_crescimento_mensal()`: simula mes a mes ate o fio.
- `classify_risk()`: regra deterministica de risco (trava de seguranca).
- `soil_modifier()`, `climate_modifier()`, `management_modifier()`,
  `pruning_vigor_modifier()`, `canopy_competition_modifier()`: modificadores.
- `phyllotaxis_canopy()`: filotaxia (arranjo de galhos / angulo aureo),
  usada so para formato de copa, SEPARADA da altura.

## Como rodar

1. Atualize ou gere os dados reais (opcional, requer internet):

```bash
python3 enrich_soil_climate.py
```

2. Gere o dataset de analise (opcional):

```bash
python3 generate_data.py
```

3. Suba a API:

```bash
python3 api.py
```

Depois abra `http://localhost:8000/docs`.

4. Rode os testes de coerencia:

```bash
pytest -q
```

## Como a previsao funciona

1. A API recebe idade, especie, solo, clima, manejo, vizinhanca e altura do fio.
2. `compute_growth()` calcula os modificadores e a altura atual (Von Bertalanffy).
3. `compute_wire_timeline()` simula mes a mes se/quando a arvore atinge o fio,
   com horizonte operacional de 120 meses.
4. `classify_risk()` aplica a trava de seguranca de risco.
5. `phyllotaxis_canopy()` descreve o formato da copa, sem afetar a altura.

## Sobre a filotaxia (Fibonacci / angulo aureo)

A sequencia de Fibonacci e o angulo aureo (137.5 graus) descrevem o ARRANJO
ESPACIAL dos galhos (filotaxia) — um fenomeno real de botanica. Isso explica o
formato/densidade da copa, nao a velocidade de crescimento em altura. Por isso,
na versao 2.0 a filotaxia virou `phyllotaxis_canopy()` e saiu da formula de
altura. O antigo `fibonacci_growth_modifier` na verdade so penalizava o vigor
conforme o numero de podas; isso agora e `pruning_vigor_modifier()`.

## Validacao de localidades e duplicidade

`generate_data.py` usa `normalize_location_key()` para remover repeticoes em dados reais:

- Com coordenadas: arredonda latitude/longitude em 5 casas decimais, cerca de 1 metro.
- Sem coordenadas: usa localidade textual normalizada.

Isso evita treinar varias vezes a mesma arvore ou o mesmo ponto de coleta. Para importacao em producao, o backend tambem deve validar duplicidade antes de salvar:

- bloquear coordenadas invalidas;
- rejeitar latitude/longitude fora da area atendida;
- procurar arvore existente num raio de 1 a 3 metros;
- comparar especie, ano de coleta e fonte;
- manter `source_id` original do GBIF/coleta para rastreabilidade.

## Fontes de dados recomendadas

- GBIF Occurrence API: https://techdocs.gbif.org/en/openapi/v1/occurrence
- GBIF Downloads API: https://techdocs.gbif.org/en/data-use/api-downloads
- ISRIC SoilGrids: https://www.isric.org/explore/soilgrids
- Open-Meteo: https://open-meteo.com/en/features
- ANEEL PRODIST: https://www.gov.br/aneel/pt-br/centrais-de-conteudos/procedimentos-regulatorios/prodist
- Normas da distribuidora local: usar para substituir `WIRE_HEIGHT_BY_CONTEXT`.
- Inventario municipal de arborizacao: melhor fonte para altura real, DAP, poda, conflito com rede e localizacao.

## O que ainda precisa melhorar

- **PRIORIDADE: coletar altura MEDIDA em campo** (~50-100 arvores). Hoje a
  validacao e so de coerencia fisica (ver `tests/`). Com altura real da para
  medir o erro em metros (MAE) e, so entao, avaliar se um modelo de ML supera a
  formula. Enquanto o alvo for calculado, ML nao ajuda.
- Validar `species_height_max` e `species_k` com bibliografia por especie.
- Trocar alturas de fio aproximadas por norma local da distribuidora/ABNT.
- Adicionar campo de tensao/tipo de rede real: baixa tensao, media tensao, ramal, travessia.
- Coletar altura medida em campo para treinar com alvo real, nao apenas alvo calculado.
- Coletar DAP, diametro de copa, estado fitossanitario e ultima data de poda.
- Separar treino por bioma/regiao quando houver volume suficiente.
- Usar validacao espacial: treinar em alguns municipios e testar em outros.
- Salvar relatorio de metricas por especie, nao apenas media global.
- Testar desbalanceamento de risco: `TO_PRUNE` e `UNDER_OBSERVATION` ainda tendem a ter menos amostras.

## Cuidados

Este modelo apoia decisao operacional, mas nao substitui vistoria tecnica. Para poda perto de rede eletrica, a decisao final deve seguir norma da distribuidora, regra municipal e avaliacao de campo.
