FROM python:3.12

RUN pip install --no-cache-dir --upgrade mlflow

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "8080"]
