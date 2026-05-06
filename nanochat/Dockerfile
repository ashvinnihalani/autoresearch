FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/

COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-install-project

COPY prepare.py train.py program.md ./

CMD ["uv", "run", "train.py"]
