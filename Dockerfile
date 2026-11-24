FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py ./
COPY adain ./adain
COPY models/encoder.onnx models/decoder.onnx ./models/
COPY examples ./examples
COPY .streamlit ./.streamlit

RUN useradd --create-home app
USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s CMD \
    python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
