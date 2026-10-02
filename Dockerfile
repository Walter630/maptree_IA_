# MapTree IA - imagem para rodar a API em qualquer maquina de forma identica.
FROM python:3.12-slim

# Nao gerar .pyc e nao bufferizar logs (melhor para containers).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependencias primeiro (cache de camada).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o codigo.
COPY . .

# Porta da API.
EXPOSE 8000

# Altura de fio padrao da cidade (sobrescreva com -e MAPTREE_WIRE_HEIGHT=6.0).
ENV MAPTREE_WIRE_HEIGHT=8.0

# Sobe a API.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
