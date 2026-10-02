# MapTree IA — Registro de mudanças (v1 → v2)

Documento para a equipe estudar. Explica **o que mudou, por quê e como validar**.
Resumo em uma frase: tiramos o ML que só imitava a fórmula, separamos a
filotaxia (Fibonacci) da predição de altura, modernizamos as dependências e
adicionamos testes. A predição ficou mais estável, coerente e portável.

---

## 1. Por que a predição estava "sempre errada ou fora de contexto"

O projeto treinava 4 modelos de Machine Learning (`GradientBoosting`,
`RandomForest`) para prever altura, crescimento, dias-até-o-fio e risco. Mas:

- O **alvo** desses modelos (`estimated_height`, `days_to_wire`, etc.) era
  **calculado por fórmula** no `generate_data.py`, não medido em campo.
- O dataset tinha **~89 linhas** e risco totalmente desbalanceado
  (`NORMAL=85, TO_PRUNE=3, UNDER_OBSERVATION=1, CRITICAL=0`).

Resultado: o ML tentava **imitar uma fórmula matemática** com pouquíssimos
dados. Isso só adiciona ruído e "patamares" (valores repetidos), produzindo as
predições fora de contexto. Na prática, a própria API já ignorava o ML e usava a
fórmula — o ML só pesava e exigia os arquivos `.pkl`.

**Regra de ouro:** ML só supera uma fórmula quando você tem **dados reais
medidos** em quantidade. Enquanto o alvo for calculado, a fórmula vence.

---

## 2. O "Fibonacci": o que estava errado e o que fizemos

### O que a filotaxia realmente é
A sequência de Fibonacci e o ângulo áureo (137.5°) descrevem a **filotaxia**: o
**arranjo espacial** dos galhos/folhas ao redor do tronco. É botânica real e
descreve o **formato da copa**.

### O que o código fazia
A função `fibonacci_growth_modifier(pruning_count)` usava Fibonacci como função
do **número de podas**, não do arranjo de galhos. Ela devolvia um número entre
0.65 e 1.08 que **multiplicava a altura**. Matematicamente, a parte "áurea" era
quase neutra (≈1), e o que mexia de verdade era a penalidade por poda. Ou seja:
emprestava o nome "Fibonacci" para uma penalidade de poda, misturando um
conceito **espacial** (arranjo) com uma previsão **temporal** (altura no tempo).
Isso distorcia a predição.

### O que fizemos
- Separamos em duas coisas, cada uma no lugar certo:
  - **`pruning_vigor_modifier(pruning_count)`** → o efeito real da poda sobre o
    vigor (poda repetida reduz crescimento). Entra na fórmula de altura.
  - **`phyllotaxis_canopy(branch_count, pruning_count)`** → filotaxia de verdade
    (ângulo áureo + termo de Fibonacci + índice de simetria). Descreve o formato
    da copa e **não** entra no cálculo de altura.
- Mantivemos aliases com os nomes antigos (`fibonacci_growth_modifier`,
  `simular_crescimento_fibonacci`) para não quebrar imports antigos.

**Isso melhora a predição?** Sim. A altura passa a ser 100% Von Bertalanffy
(científico, monótono, saturante — nunca passa do máximo da espécie, nunca pula
patamares). A filotaxia continua existindo e honra o que o especialista explicou,
mas no campo `phyllotaxis` da resposta, como informação de copa.

---

## 3. Mudanças por arquivo

### `requirements.txt`
- Removido `scikit-learn` e `joblib` (não há mais ML).
- Adicionado `requests` — **estava faltando** e `enrich_soil_climate.py` usa.
  Esse era um dos motivos de "não funciona em outro computador".
- Adicionado `pytest` e `httpx` (testes).
- Troca de pins `==` antigos por faixas `>=,<` compatíveis, para instalar em
  Python 3.10+ e em Linux/Windows/macOS sem quebrar.

### `generate_data.py`
- `fibonacci_growth_modifier` → **`pruning_vigor_modifier`** (lógica limpa, só
  efeito de poda). Alias antigo mantido.
- Nova função **`phyllotaxis_canopy()`** (filotaxia de verdade).
- `simular_crescimento_fibonacci` → **`simular_crescimento_mensal`** (o nome
  antigo enganava: a simulação é Von Bertalanffy, sem Fibonacci). Alias mantido.
  Removido um `fib_mod` que era sempre 1.0 (no-op).
- Coluna do dataset `fibonacci_modifier` → **`pruning_modifier`**.
- Removidos imports não usados (`random`, `defaultdict`).

### `api.py`
- **Removida toda a dependência de modelos `.pkl`** e do `joblib`.
- Novo núcleo determinístico: `compute_growth()` e `compute_wire_timeline()`
  (funções puras, fáceis de testar).
- Resposta agora traz `phyllotaxis` (formato de copa) e `modifiers_used`
  (transparência dos multiplicadores). Saiu o antigo `fibonacci_info`.
- Novo campo opcional de entrada: `branch_count` (galhos observados, p/ copa).
- `version` da API: `1.0.0` → `2.0.0`. `/health` agora informa
  `engine: von_bertalanffy_deterministic`.

### Removidos
- **`train_models.py`** — treino de ML que só imitava a fórmula.
- **pasta `models/`** (todos os `.pkl`) — não são mais necessários.
- **`__pycache__/`** — cache.

### `run_pipeline.sh` e `start_api.sh`
- Tinham `cd /home/walter/maptree_IA` **fixo** (caminho de outra máquina) — outro
  motivo de não rodar em outro computador. Agora usam `cd "$(dirname "$0")"`,
  então rodam a partir da própria pasta, em qualquer máquina.
- `run_pipeline.sh` não treina mais ML (passo removido).

### `README.md` e `MODEL_GUIDE.md`
- Atualizados: sem passo de treino, tabela/fórmula corrigidas (poda no lugar de
  Fibonacci), exemplo de resposta novo, seção de filotaxia e instruções de teste.

### Novos: `tests/`
- `tests/test_predictions.py` — coerência física do motor.
- `tests/test_api.py` — smoke tests dos endpoints.

---

## 4. Como validar que a predição está correta

Como **ainda não há altura medida em campo**, não dá para medir erro contra a
realidade. Então validamos de 3 formas:

### (a) Testes automáticos de coerência física — `pytest -q`
Verificam propriedades que **têm** que ser verdade, por exemplo:
- altura nunca passa do máximo da espécie;
- mais idade → mais altura (monótono);
- solo/clima melhor → cresce mais;
- árvore acima do fio → risco `CRITICAL` e dias = 0;
- espécie baixa nunca atinge fio alto;
- crescimento desacelera com a idade;
- filotaxia **não** altera a altura.

Status atual: **26 testes passando.**

### (b) Casos conhecidos (sanity check)
Em `tests/test_predictions.py` há casos plausíveis (Croton jovem, Celtis maduro,
Apuleia grande) com faixas esperadas. A equipe e o especialista revisam se os
números batem com o esperado em campo e ajustam as faixas se necessário.

### (c) Validação real (próximo passo, recomendado)
Coletar **altura medida em campo de ~50-100 árvores reais**. Com isso:
1. adicionamos um teste que compara predição × medido e reporta o **MAE** (erro
   médio em metros);
2. calibramos `species_k`/`height_max` por espécie;
3. **só então** faz sentido reavaliar se um modelo de ML supera a fórmula.

---

## 5. Como rodar (resumo)

```bash
cd maptree_IA
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

pytest -q                          # roda os testes
python api.py                      # sobe a API em :8000
```

API: http://localhost:8000 · Docs: http://localhost:8000/docs

---

## 6. Recomendações adicionais (não aplicadas, para decidir em equipe)

1. **Coletar dados de campo** (altura, DAP, última poda) — maior ganho de todos.
2. **Dois ambientes virtuais** na pasta (`.venv` e `venv`). Manter só um
   (sugiro `.venv`) e apagar o outro para evitar confusão.
3. **Altura do fio por norma** da distribuidora/ABNT, em vez dos valores
   aproximados em `WIRE_HEIGHT_BY_CONTEXT`.
4. **Tipo de rede real** (baixa/média tensão, ramal, travessia) como entrada.
5. **CI** (GitHub Actions) rodando `pytest` a cada push, para não regredir.
6. **Dockerfile** para padronizar o ambiente entre máquinas e no deploy.
7. Validar `species_k`/`height_max` por espécie com bibliografia (TRY, Flora do
   Brasil) — hoje alguns valores são estimados.

---

# Rodada 2 — normas de fio, dados reais, Docker e instalação

## 7. Altura do fio alinhada às normas

- `WIRE_HEIGHT_BY_CONTEXT` revisado para refletir os valores de referência da
  prática brasileira (ABNT NBR 15688/16615 e PRODIST/ANEEL):
  - `PEDESTRIAN_ONLY`: 5.5 m (sem tráfego de veículos)
  - `URBAN_LOW_VOLTAGE`: 6.0 m (ruas/vias urbanas)
  - `URBAN_MEDIUM_VOLTAGE`: 6.5 m (travessia de ruas)
  - `RURAL_DISTRIBUTION`: 7.0 m
  - `ROAD_CROSSING`: 8.0 m (rodovias/travessias)
- **Padrão agora é 8 m** (caso da cidade atendida — travessia/BR 222), e é
  configurável pela variável de ambiente `MAPTREE_WIRE_HEIGHT`, ou pelo campo
  `wire_height` em cada requisição.
- **Fonte**: valores de referência consolidados; a norma oficial da ANEEL/ABNT
  deve prevalecer. Confirmar com a norma da distribuidora local.

## 8. Ambiente virtual

- Havia **dois** venvs na pasta (`.venv` com pacotes e `venv` vazio/quebrado).
  Mantivemos **só o `.venv`** e removemos o `venv`.
- Adicionado `.gitignore` para não versionar venv, cache e logs.

## 9. Rodar a IA sobre os dados reais da pasta — `run_predictions.py` (novo)

- Script que aplica o motor determinístico às 397 árvores reais de
  `arvores_mapeadas.json` e gera `arvores_preditas.json`, sem precisar do backend.
- **Curadoria arbórea corrigida**: só espécies de **gênero arbóreo curado**
  recebem predição de altura. Antes, ervas/forrageiras da família Fabaceae
  (ex.: *Centrosema*, *Stylosanthes*) recebiam o perfil DEFAULT de 8 m de árvore
  — um erro de dado. Agora são marcadas como `nao_arborea`. A família sozinha
  não basta (Fabaceae tem árvores e capim); o gênero é o critério confiável.
- Cada predição vem com `data_quality` (`ok`, `ambiente_estimado`,
  `sem_perfil_ou_idade`, `nao_arborea`) para a equipe saber em quais árvores
  confiar e quais precisam de coleta de campo.
- Resultado nos seus dados: 7 árvores com predição completa, risco todo NORMAL
  (coerente: espécies da Caatinga são baixas e o fio está a 8 m, então não há
  conflito previsto). O resto é não-arbóreo ou sem idade.

## 10. Docker (novo)

- `Dockerfile` + `.dockerignore` para rodar a API igual em qualquer máquina,
  sem instalar Python/venv. Imagem baseada em `python:3.12-slim`. Testada: build
  OK e API responde. Uso em `INSTALL.md`.

## 11. Guia de instalação — `INSTALL.md` (novo)

- Passo a passo **Linux** e **Windows** (PowerShell e CMD), as **3 formas de
  testar** (pytest, Swagger no navegador, lote nos seus dados), uso com Docker,
  e **como conectar o backend NestJS** (CORS, URL, altura de fio, contrato).

---

# Perguntas respondidas

**Qual IA está sendo usada? Tem rede neural / modelo treinado?**
Não há (mais) modelo de ML treinado. O "cérebro" é um **modelo matemático
determinístico** de crescimento de árvores: **Von Bertalanffy**, o padrão
científico em dendrometria/florestal. Ele calcula altura, crescimento e prazo
até o fio a partir de espécie, idade, solo, clima, manejo e competição. É
"inteligência" baseada em modelo botânico, não em rede neural. A versão antiga
tinha modelos de ML (`scikit-learn`), mas eles só imitavam essa fórmula com
poucos dados e pioravam o resultado — por isso foram removidos.

**Python é a melhor escolha para isso?**
Sim, para este caso. Python é o padrão de fato para ciência de dados e IA
(bibliotecas, integração com APIs de solo/clima, FastAPI para o microserviço) e
conversa bem com um backend NestJS por HTTP. Não há motivo técnico para trocar.
Quando houver dados medidos em campo e eventualmente um modelo de ML de verdade,
Python continua sendo a melhor opção (scikit-learn, LightGBM, etc.).

**Quando vale a pena voltar a usar ML?**
Só quando houver **altura medida em campo** de muitas árvores (centenas). Aí o
ML pode capturar padrões que a fórmula não vê. Enquanto o alvo for calculado,
ML não ajuda. O caminho está documentado em `MODEL_GUIDE.md`.

---

# Rodada 3 — documentação dos dados

## 12. `DADOS.md` (novo)

- Documento dedicado explicando **quais dados a IA usa, de onde vêm e por que
  cada um afeta a predição**: espécie (teto e velocidade), idade (posição na
  curva), solo, clima, manejo, vizinhança, fio/rede e localização.
- Inclui tabela de fontes (arquivos da pasta + APIs de solo/clima), os campos de
  saída da predição, e a prioridade de coleta de dados para melhorar a precisão.
- `README.md` agora tem uma seção **Documentação** que lista todos os MDs.
