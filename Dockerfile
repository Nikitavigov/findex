# syntax=docker/dockerfile:1

# ==============================================================================
# Етап 1: Збірка залежностей через uv (Builder)
# ==============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Кешування встановлення сторонніх пакетів
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Копіюємо вихідний код і встановлюємо проєкт findex
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ==============================================================================
# Етап 2: Фінальний мінімальний runtime образ
# ==============================================================================
FROM python:3.12-slim-bookworm AS runtime

# Встановлюємо curl для перевірки стану (HEALTHCHECK)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Створюємо системного непривілейованого користувача (non-root)
RUN groupadd -r findex && useradd -r -g findex -d /app -s /sbin/nologin findex

WORKDIR /app

# Копіюємо готовий проєкт із віртуальним оточенням із builder
COPY --from=builder --chown=findex:findex /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH="/app/data/index.bin" \
    HOST="0.0.0.0" \
    PORT=8000

USER findex

EXPOSE 8000

# Перевірка працездатності контейнера платформою деплою
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["findex", "serve", "--host", "0.0.0.0", "--port", "8000"]
