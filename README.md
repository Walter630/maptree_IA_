# MapTree AI - Microserviço de Crescimento de Árvores

API em FastAPI para estimar crescimento de árvores, risco de alcançar fiação e apoio a rotas de poda.

## Documentação

- `INSTALL.md` — instalação passo a passo (Windows/Linux), como testar e conectar o backend.
- `DADOS.md` — quais dados a IA usa e por que cada um importa para a predição.
- `MODEL_GUIDE.md` — como o modelo calcula altura, crescimento e risco.
- `CHANGES.md` — histórico de mudanças (v1 → v2) para a equipe estudar.

## Requisitos

- Python 3.10 ou superior
- `pip`
- Git, se for clonar o projeto

> Observação: evite instalar dependências direto no Python do sistema. Use ambiente virtual (`venv`) para funcionar bem no Linux, Windows e macOS.

## Como Rodar

### 1. Entrar na pasta do projeto

Se você clonou o repositório:

```bash
git clone <url-do-repositorio>
cd maptree_IA
```

Se a pasta já existe na sua máquina, apenas entre nela:

```bash
cd maptree_IA
```

### 2. Criar e ativar ambiente virtual

#### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativação do ambiente virtual, execute uma vez:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Depois rode novamente:

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows CMD

```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Instalar dependências

Com o ambiente virtual ativo:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Linux: não use `--break-system-packages` se estiver com `.venv` ativo. Esse parâmetro só deve ser usado em casos específicos e pode causar problemas no Python do sistema.

### 4. (Opcional) Gerar dataset de análise

```bash
python generate_data.py
```

Cria/atualiza `tree_dataset.csv` a partir dos dados reais. **Não é mais
necessário treinar modelos**: a predição usa o motor botânico determinístico
(Von Bertalanffy), que não precisa de etapa de treino. O dataset serve apenas
para análise/auditoria.

Para rodar a IA sobre os seus dados reais já existentes na pasta e gerar um
arquivo com predições atualizadas:

```bash
python run_predictions.py                 # usa arvores_mapeadas.json → arvores_preditas.json
python run_predictions.py --wire-height 6.0
```

> **Altura do fio**: o padrão é **8 m** (travessia/rodovia, caso da cidade
> atendida). Dá para mudar por requisição (`wire_height` no corpo), por variável
> de ambiente `MAPTREE_WIRE_HEIGHT`, ou por contexto automático em
> `WIRE_HEIGHT_BY_CONTEXT` (pedestre 5.5 m, urbana 6.0 m, travessia 6.5 m,
> rural 7.0 m, rodovia 8.0 m). Ajuste conforme a norma da sua distribuidora.

### 5. Subir a API

```bash
python api.py
```

Acesse:

- API: http://localhost:8000
- Swagger/docs: http://localhost:8000/docs
- Healthcheck: http://localhost:8000/health

## Comandos Rápidos por Sistema

### Linux/macOS

```bash
cd maptree_IA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python api.py
```

### Windows PowerShell

```powershell
cd maptree_IA
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python api.py
```

### Windows CMD

```bat
cd maptree_IA
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python api.py
```

### Rodar os testes

```bash
pytest -q
```

## Problemas Comuns

### `python` não é reconhecido no Windows

Use `py` no lugar de `python`:

```powershell
py --version
py -m pip install -r requirements.txt
py api.py
```

### Erro ao ativar `.venv` no PowerShell

Execute:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Feche e abra o terminal, ou rode novamente:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Porta 8000 em uso

Rode a API com outra porta:

```bash
uvicorn api:app --host 0.0.0.0 --port 8001
```

Depois acesse:

- http://localhost:8001
- http://localhost:8001/docs

### Como sair do ambiente virtual

```bash
deactivate
```

---

## Endpoints

### POST /predict-wire-risk
Endpoint principal — use este no NestJS.

**Request:**
```json
{
  "age_years": 8,
  "pruning_count": 2,
  "species": {
    "height_average": 10.0,
    "growth_rate_k": 0.12
  },
  "soil": {
    "depth": 1.2,
    "inclination": 5,
    "quality": "REGULAR",
    "coverage": 0.4
  },
  "annual_rainfall": 650,
  "altitude": 280,
  "avg_temperature": 28,
  "has_fertilization": false,
  "has_irrigation": false,
  "nearby_trees_count": 2,
  "avg_neighbor_distance": 8.0
}
```

**Response:**
```json
{
  "estimated_height_m": 5.42,
  "wire_height_m": 6.5,
  "will_reach_wire": true,
  "days_to_wire": 312,
  "months_to_wire": 10.4,
  "risk_status": "UNDER_OBSERVATION",
  "alert": "🔍 Árvore atingirá o fio em ~10 meses. Monitorar.",
  "canopy": {
    "shape": "SQUARE",
    "ratio_width_height": 1.0,
    "nearby_competition": false
  },
  "phyllotaxis": {
    "golden_angle_deg": 137.5,
    "branch_whorls": 3,
    "fibonacci_stage": 3,
    "symmetry_index": 0.7639
  },
  "modifiers_used": {
    "soil": 0.70,
    "climate": 1.03,
    "management": 1.0,
    "pruning": 0.89,
    "competition": 0.99,
    "total": 0.63
  }
}
```

> `phyllotaxis` descreve o **arranjo espacial dos galhos** (ângulo áureo de
> 137.5° e sequência de Fibonacci). É informação de formato de copa e **não**
> entra no cálculo da altura. A altura usa só idade, espécie, solo, clima,
> manejo e competição.

### POST /optimize-route
Retorna rota ótima para equipe de poda.

### GET /health
Healthcheck da API (confirma que o motor determinístico está ativo).

---

## Integração NestJS

```typescript
// No seu service NestJS:
async predictWireRisk(treeData: any) {
  const response = await this.httpService.post(
    'http://localhost:8000/predict-wire-risk',
    treeData
  ).toPromise();
  return response.data;
}
```

---

## Modelo de crescimento

Baseado nos pilares do Jarbas (Fito):

| Fator | Variáveis |
|-------|-----------|
| **Solo** | profundidade, inclinação, qualidade, cobertura |
| **Planta** | espécie, idade, podas realizadas |
| **Manejo** | adubação, irrigação |
| **Clima** | chuva anual, altitude, temperatura média |
| **Competição** | árvores vizinhas → formato da copa |
| **Filotaxia** | arranjo de galhos (ângulo áureo 137.5°) — só formato de copa, **separado da altura** |

### Fórmula base (Von Bertalanffy modificada):
```
h = h_max × (1 - e^(-k × idade)) × mod_solo × mod_clima × mod_manejo × mod_poda × mod_competição
```

A filotaxia (Fibonacci / ângulo áureo) descreve o **arranjo espacial dos
galhos** e é reportada à parte, no campo `phyllotaxis`. Ela não multiplica a
altura, porque é um conceito espacial, não temporal.
