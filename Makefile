.PHONY: mlflow

mlflow:
	uv run mlflow server --port 5050 --host 0.0.0.0 --allowed-hosts host.docker.internal:5050
