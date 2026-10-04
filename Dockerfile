# ---- build stage: install dependencies into a virtualenv ----
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# ---- runtime stage: small image, non-root user ----
FROM python:3.12-slim
ENV PATH="/venv/bin:$PATH" PYTHONUNBUFFERED=1 \
    SUPPORT_DATA_DIR=/app/data SUPPORT_TICKET_DB_PATH=/app/var/tickets.db
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=build /venv /venv
COPY data ./data
RUN mkdir -p /app/var && chown appuser /app/var
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "support_agent.api:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
