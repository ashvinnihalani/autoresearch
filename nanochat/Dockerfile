FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

LABEL autoresearch.workstream="nanochat" \
      autoresearch.role="dev"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    VIRTUAL_ENV=/app/.venv \
    MAX_JOBS=4

ENV PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        ninja-build \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/

COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-install-project \
    && uv pip install setuptools wheel

WORKDIR /workspace

CMD ["bash"]
