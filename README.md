# MapTree AI - Microserviço de Crescimento de Árvores

## Como rodar (Pop!_OS)

### 1. Instalar dependências
```bash
cd maptree-ai
pip install -r requirements.txt --break-system-packages
```

### 2. Gerar dataset e treinar modelos
```bash
python generate_data.py   # cria tree_dataset.csv
python train_models.py    # treina e salva os modelos em ./models/
```

### 3. Subir a API
```bash
python api.py
# Acessa em: http://localhost:8000
# Docs:       http://localhost:8000/docs
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
  "fibonacci_info": {
    "pruning_cycle_index": 2,
    "growth_modifier": 0.8571,
    "golden_angle_deg": 137.5
  }
}
```

### POST /optimize-route
Retorna rota ótima para equipe de poda.

### GET /health
Healthcheck da API e dos modelos.

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
| **Fibonacci** | padrão de ramificação (ângulo áureo 137.5°) |
| **Competição** | árvores vizinhas → formato da copa |

### Fórmula base (Von Bertalanffy modificada):
```
h = h_max × (1 - e^(-k × idade)) × mod_solo × mod_clima × mod_manejo × mod_fibonacci × mod_competição
```
