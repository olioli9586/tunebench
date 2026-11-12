# The serving image: FastAPI + ONNX Runtime + tokenizers, no PyTorch.
# The model is not baked in: mount a directory written by tunebench.export at /model
#   docker build -t tunebench .
#   docker run -p 8000:8000 -v "$PWD/artifacts/onnx/modernbert-seed1:/model:ro" tunebench
FROM python:3.13-slim AS build
RUN pip install --no-cache-dir uv==0.13.7
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-default-groups --group serve --no-install-project
COPY src ./src
RUN uv sync --locked --no-default-groups --group serve --no-editable

FROM python:3.13-slim
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH MODEL_DIR=/model MODEL_FILE=model.int8.onnx
RUN useradd --system --no-create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "tunebench.serve:app", "--host", "0.0.0.0", "--port", "8000"]
