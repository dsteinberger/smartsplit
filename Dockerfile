FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY smartsplit/ smartsplit/
RUN uv pip install --system --no-cache .

RUN useradd -m -u 1000 smartsplit
USER smartsplit

ENV PYTHONUNBUFFERED=1
EXPOSE 8420

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:8420/health').raise_for_status()" || exit 1

ENTRYPOINT ["python", "-m", "smartsplit"]
CMD ["--host", "0.0.0.0", "--port", "8420"]
