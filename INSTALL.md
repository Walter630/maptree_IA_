# MapTree IA — Guia de instalação, execução e teste

Passo a passo para rodar a IA no **Linux** e no **Windows**, testar sem o
backend e conectar depois. Leitura rápida para quem nunca abriu o projeto.

> Pré-requisito: **Python 3.10 ou superior**. Confira com `python3 --version`
> (Linux/macOS) ou `py --version` (Windows).

---

## 1. Linux (e macOS)

Abra o terminal na pasta do projeto e rode, uma linha por vez:

```bash
cd maptree_IA

# 1. Criar o ambiente virtual (isola as dependências)
python3 -m venv .venv

# 2. Ativar o ambiente
source .venv/bin/activate

# 3. Instalar as dependências
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Subir a API
python api.py
```

A API sobe em `http://localhost:8000`. Deixe esse terminal aberto (é o servidor).

Para **sair** do ambiente virtual depois: `deactivate`.

---

## 2. Windows

### PowerShell

```powershell
cd maptree_IA

py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python api.py
```

Se o PowerShell bloquear a ativação do ambiente, rode uma vez e tente de novo:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### CMD (Prompt de Comando)

```bat
cd maptree_IA

py -m venv .venv
.venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python api.py
```

> No Windows, se `python` não for reconhecido, use `py` no lugar.

---

## 3. Como testar — 3 formas

### Forma A — Testes automáticos (não precisa do backend)

Com o ambiente ativado, na pasta do projeto:

```bash
pytest -q
```

Deve aparecer algo como `26 passed`. Isso valida a coerência da IA
(altura nunca passa do máximo da espécie, árvore acima do fio = risco crítico,
etc.). **É aqui que você confirma que a lógica está certa.**

### Forma B — Testar a API no navegador (Swagger)

1. Com a API rodando (`python api.py`), abra no navegador:
   `http://localhost:8000/docs`
2. Clique em **POST /predict-wire-risk** → **Try it out**.
3. Cole um exemplo no corpo e clique em **Execute**:

```json
{
  "age_years": 16,
  "pruning_count": 0,
  "species": { "height_average": 9.0, "growth_rate_k": 0.12 },
  "soil": { "depth": 1.0, "inclination": 5, "quality": "REGULAR", "coverage": 0.3 },
  "annual_rainfall": 700,
  "altitude": 120,
  "avg_temperature": 27,
  "wire_height": 8.0
}
```

A resposta mostra altura estimada, se atinge o fio, prazo e risco.

### Forma C — Rodar a IA nos SEUS dados (lote, sem backend)

Para ver a IA aplicada aos dados reais que já estão na pasta
(`arvores_mapeadas.json`):

```bash
python run_predictions.py
```

Isso gera `arvores_preditas.json` com as predições atualizadas e imprime um
resumo (quantas árvores previstas, distribuição de risco). Opções:

```bash
python run_predictions.py --input arvores_mapeadas.json --output saida.json
python run_predictions.py --wire-height 6.0
```

> Observação: só espécies **arbóreas** curadas recebem predição de altura.
> Ervas e forrageiras (ex.: Centrosema, Stylosanthes) são marcadas como
> `nao_arborea` e ficam sem predição — isso é proposital, para não tratar
> capim como árvore.

---

## 4. Rodar com Docker (opcional, mais portável)

Se a equipe usa Docker, não precisa instalar Python nem venv:

```bash
# Construir a imagem
docker build -t maptree-ia .

# Rodar a API na porta 8000
docker run -p 8000:8000 maptree-ia

# Rodar com outra altura de fio (ex.: 6 m)
docker run -p 8000:8000 -e MAPTREE_WIRE_HEIGHT=6.0 maptree-ia
```

Acesse `http://localhost:8000/docs`.

---

## 5. Conectar com o backend (NestJS)

A API é só HTTP. O backend chama o endpoint principal `POST /predict-wire-risk`.

```typescript
// Exemplo em um service NestJS
async predictWireRisk(treeData: any) {
  const { data } = await firstValueFrom(
    this.httpService.post('http://localhost:8000/predict-wire-risk', treeData),
  );
  return data;
}
```

Pontos de atenção para a integração:

- **CORS**: hoje a API libera `http://localhost:3500` (porta do Vue). Se o
  front/back rodar em outra porta, ajuste `allow_origins` em `api.py`.
- **URL da IA**: em produção, use a URL do container/host, não `localhost`.
  Coloque isso numa variável de ambiente no backend (ex.: `MAPTREE_IA_URL`).
- **Altura do fio**: envie `wire_height` no corpo (ex.: `8.0`) ou confie no
  padrão da IA (`MAPTREE_WIRE_HEIGHT`, hoje 8 m).
- **Contrato de resposta**: ver `README.md` (seção Endpoints) para o formato.

---

## 6. Problemas comuns

| Problema | Solução |
|----------|---------|
| `python` não reconhecido (Windows) | Use `py` no lugar de `python`. |
| Erro ao ativar `.venv` no PowerShell | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e reabra o terminal. |
| Porta 8000 em uso | `uvicorn api:app --port 8001` e acesse `:8001`. |
| `ModuleNotFoundError` | Ambiente não ativado ou deps não instaladas. Repita o passo 3. |
| Front não conecta (CORS) | Ajuste `allow_origins` em `api.py`. |
