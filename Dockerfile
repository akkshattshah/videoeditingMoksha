# The autodetected Python image has no ffmpeg, and src/vre/probe.py resolves
# ffprobe/ffmpeg off PATH via shutil.which() — so every upload fails with
# "ffprobe not found on PATH" without this layer.
#
# 3.14 matches the version requirements.txt is confirmed against; opencv and
# numpy pins here are recent enough that older images fall back to building
# from source.
FROM python:3.14-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ui/server.py adds src/ to sys.path itself, so no install step is needed.
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn ui.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
