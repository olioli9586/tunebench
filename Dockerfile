# The serving image: FastAPI + ONNX Runtime + tokenizers, no PyTorch.
#
# Mount a directory written by tunebench.export at /model:
#   docker build -t tunebench .
#   docker run -p 8000:8000 -v "$PWD/artifacts/onnx/modernbert-seed1:/model:ro" tunebench
# or bake the published model in (the Hugging Face Space does this):
#   docker build --build-arg MODEL_REPO=<user>/<model repo> -t tunebench .
FROM python:3.13-slim AS build
RUN pip install --no-cache-dir uv==0.13.7 huggingface_hub==2.4.0
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-default-groups --group serve --no-install-project
COPY src ./src
RUN uv sync --locked --no-default-groups --group serve --no-editable
ARG MODEL_REPO=""
RUN mkdir /model && if [ -n "$MODEL_REPO" ]; then \
      python -c "import sys; from huggingface_hub import snapshot_download; \
snapshot_download(sys.argv[1], allow_patterns=['onnx/model.int8.onnx', 'onnx/tokenizer.json', 'onnx/labels.json'], local_dir='/tmp/m')" "$MODEL_REPO" \
      && mv /tmp/m/onnx/* /model/; fi

FROM python:3.13-slim
COPY --from=build /app/.venv /app/.venv
COPY --from=build /model /model
ENV PATH=/app/.venv/bin:$PATH MODEL_DIR=/model MODEL_FILE=model.int8.onnx
RUN useradd --uid 1000 --no-create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "tunebench.serve:app", "--host", "0.0.0.0", "--port", "8000"]
