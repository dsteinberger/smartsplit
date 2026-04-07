FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY smartsplit/ smartsplit/
RUN uv pip install --system --no-cache .

RUN useradd -m -u 1000 smartsplit
USER smartsplit

ENV PYTHONUNBUFFERED=1
EXPOSE 8420

ENTRYPOINT ["python", "-m", "smartsplit"]
CMD ["--host", "0.0.0.0", "--port", "8420"]
