FROM python:3.11-slim

RUN useradd -m sandboxuser
USER sandboxuser

RUN pip install --no-cache-dir numpy scipy

WORKDIR /home/sandboxuser

COPY eval.py .

ENTRYPOINT ["python", "eval.py"]
