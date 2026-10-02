# MapTree IA — Dados usados e por que importam para a predição

Documento para a equipe. Explica **quais dados a IA usa, de onde vêm e por que
cada um afeta a predição** de crescimento e de quando a árvore atinge o fio.

A predição usa o modelo de Von Bertalanffy:

```
altura = altura_máxima_da_espécie × (1 − e^(−k × idade × modificadores))
```

Ou seja, tudo gira em torno de: **quanto a espécie pode crescer**, **quão rápido**,
**há quanto tempo** e **o ambiente ajuda ou atrapalha**. Cada dado abaixo entra
em um desses pontos.

---

## 1. Fontes de dados (arquivos na pasta)

| Arquivo | O que é | Papel na predição |
|---------|---------|-------------------|
| `arvores_mapeadas.json` | 397 árvores reais mapeadas (Tabuleiro do Norte/CE e região) com espécie, local, idade. | Entrada principal do lote (`run_predictions.py`). |
| `dados_tabuleiro.csv` | Ocorrências locais reais (coleta de campo, formato GBIF). | Fonte de espécie, local e ano de coleta. |
| `arvores_plantae.csv` | Ocorrências GBIF (reino Plantae) da região. | Amplia a base de espécies/locais reais. |
| `enriched_trees.csv` / `.json` | Árvores já enriquecidas com **solo e clima reais** (via APIs). | Fonte do perfil de espécie e do ambiente por local. |
| `arvores_mapeadas_gbif.csv` | Base bruta GBIF. | Matéria-prima para o parser/importação. |
| `tree_dataset.csv` | Dataset derivado (gerado por `generate_data.py`). | Análise/auditoria; **não** é usado para treinar (não há mais ML). |
| `arvores_preditas.json` | Saída de `run_predictions.py`. | Resultado pronto para o front/backend. |

**Fontes externas** (quando o enriquecimento roda, em `enrich_soil_climate.py`):

- 🌍 **Solo** → SoilGrids v2 (ISRIC) — grátis, sem chave.
- 🌡️ **Clima** → Open-Meteo — grátis, sem chave.
- 🌳 **Espécie** → perfis validados localmente (altura máxima e taxa k por espécie).

Regra do projeto: **não inventar dado**. Se solo, clima ou perfil de espécie não
puderem ser obtidos de forma real, o registro é ignorado ou marcado como de
baixa qualidade, em vez de receber um valor fictício.

---

## 2. Os campos e por que cada um importa

### 2.1 Espécie — o teto e a velocidade do crescimento

| Campo | O que é | Por que importa |
|-------|---------|-----------------|
| `species_scientific_name` | Nome científico. | Chave para buscar o perfil botânico real da espécie. |
| `species_height_max` (`heightAverage`) | Altura máxima/média que a espécie atinge (m). | **É o teto da predição.** Uma espécie de 5 m nunca vai encostar num fio de 8 m — isso sozinho já decide muitos casos de risco. |
| `species_k` (`growthRateK`) | Taxa de crescimento da espécie. | **Define a velocidade.** `k` alto = cresce rápido e atinge o fio cedo; `k` baixo = demora anos. Entra direto no expoente da fórmula. |
| `family` / `genus` | Família e gênero botânicos. | Usados na **curadoria arbórea**: só gênero de árvore recebe predição de altura (evita tratar erva/forrageira como árvore). |

> Sem espécie confiável, não há predição: ela ancora o teto e a velocidade.

### 2.2 Idade — onde a árvore está na curva de crescimento

| Campo | O que é | Por que importa |
|-------|---------|-----------------|
| `age_years` | Idade estimada (anos), derivada do ano de coleta. | **Posição na curva.** A mesma espécie tem alturas muito diferentes aos 3, 15 ou 40 anos. É o eixo do tempo da fórmula; o crescimento desacelera conforme a árvore envelhece (saturação). |

### 2.3 Solo — o quanto o terreno sustenta o crescimento

| Campo | O que é | Por que importa |
|-------|---------|-----------------|
| `soil_depth` | Profundidade útil do solo (m). | Solo raso limita a raiz e reduz o crescimento; solo profundo favorece. |
| `soil_quality` | GOOD / REGULAR / BAD (textura, pH). | Solo bom acelera; solo pobre (muito arenoso ou pH extremo) desacelera. |
| `soil_inclination` | Inclinação do terreno (graus). | Terreno muito inclinado perde água/nutrientes e reduz o vigor. |
| `soil_coverage` | Impermeabilização ao redor (0 a 1). | Calçada/asfalto sufocando a base reduz a infiltração de água e o crescimento. |

> Esses quatro campos entram no `soil_modifier`, que multiplica a velocidade de
> crescimento. Solo bom → atinge o fio mais cedo; solo ruim → mais tarde ou nunca.

### 2.4 Clima — chuva, temperatura e altitude

| Campo | O que é | Por que importa |
|-------|---------|-----------------|
| `annual_rainfall` | Chuva anual (mm). | Água é o principal motor de crescimento. Seca (típica da Caatinga) segura o crescimento; chuva adequada acelera. |
| `avg_temperature` | Temperatura média (°C). | Fora da faixa ideal (muito frio ou muito quente) o crescimento cai. |
| `altitude` | Altitude (m). | Altitude alta tende a reduzir o ritmo de crescimento. |

> Entram no `climate_modifier`. Explicam por que a mesma espécie cresce diferente
> em regiões diferentes.

### 2.5 Manejo — intervenção humana

| Campo | O que é | Por que importa |
|-------|---------|-----------------|
| `has_fertilization` | Houve adubação? | Adubação aumenta o vigor → cresce um pouco mais rápido. |
| `has_irrigation` | Há irrigação? | Irrigação compensa a seca → acelera o crescimento. |
| `pruning_count` | Nº de podas já feitas. | Poda repetida reduz a área foliar e o vigor → desacelera. Entra no `pruning_vigor_modifier` (antes chamado, incorretamente, de "fibonacci"). |

### 2.6 Vizinhança — competição por luz e espaço

| Campo | O que é | Por que importa |
|-------|---------|-----------------|
| `nearby_trees_count` | Quantas árvores vizinhas próximas. | Mais vizinhos = mais competição por luz/água → crescimento lateral menor. |
| `avg_neighbor_distance` | Distância média aos vizinhos (m). | Vizinhos muito perto aumentam a competição; longe, quase não afeta. |

> Entram no `canopy_competition_modifier`: afetam o **formato da copa**
> (`canopy_ratio`) e, de forma leve, a velocidade de crescimento.

### 2.7 Fio / rede — o alvo do risco

| Campo | O que é | Por que importa |
|-------|---------|-----------------|
| `wire_height` | Altura do fio ao solo (m). | **É a linha de corte do risco.** A IA compara a altura projetada da árvore com essa altura para dizer se, e quando, haverá conflito. Padrão 8 m (travessia/rodovia da cidade); configurável por norma. |
| `wire_context` | Tipo de via (urbana, rural, rodovia…). | Define a altura típica do fio quando ela não é informada manualmente. |

### 2.8 Localização — para dados reais e rota

| Campo | O que é | Por que importa |
|-------|---------|-----------------|
| `latitude` / `longitude` | Coordenadas. | Buscam solo/clima reais por ponto e evitam duplicar a mesma árvore; usadas também na rota de poda. |
| `city` / `locality` | Cidade/localidade. | Contexto, deduplicação e escolha do tipo de fio. |

---

## 3. Dados que saem da predição (resultado)

| Campo | Significado |
|-------|-------------|
| `estimated_height_m` | Altura atual estimada da árvore. |
| `annual_growth_m` | Quanto deve crescer nos próximos 12 meses. |
| `height_next_year_m` | Altura projetada em 1 ano. |
| `will_reach_wire` | Se a árvore tende a alcançar o fio dentro do horizonte operacional. |
| `days_to_wire` / `months_to_wire` | Prazo estimado até o fio. |
| `risk_status` | NORMAL / UNDER_OBSERVATION / TO_PRUNE / CRITICAL. |
| `phyllotaxis` | Arranjo dos galhos (ângulo áureo / Fibonacci) — formato de copa, **não** afeta a altura. |
| `data_quality` | `ok`, `ambiente_estimado`, `sem_perfil_ou_idade` ou `nao_arborea` — diz o quanto confiar naquela predição. |

---

## 4. Qualidade dos dados — o que ainda falta (e por quê)

A predição é tão boa quanto os dados de entrada. Hoje:

- A **altura é calculada**, não medida em campo. Por isso só dá para validar
  coerência, não erro em metros. **Coletar altura real** é o maior ganho possível.
- `species_height_max` e `species_k` são estimados por bibliografia/perfis; valem
  revisar por espécie (TRY, Flora do Brasil).
- `pruning_count`, manejo e vizinhança quase não existem na base atual — por isso
  entram neutros em muitas árvores. Cadastrar esses dados melhora a precisão.
- Altura do fio deve seguir a **norma da distribuidora local** (ABNT NBR
  15688/16615, PRODIST/ANEEL), não só o padrão de 8 m.

Prioridade de coleta (maior impacto na predição primeiro):
1. **Altura medida em campo** + DAP (diâmetro) + data da última poda.
2. Espécie confirmada por botânico (corrige teto e velocidade).
3. Tipo real de rede e altura do fio por ponto.
4. Vizinhança e manejo por árvore.
