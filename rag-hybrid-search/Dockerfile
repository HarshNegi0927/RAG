# Not live-tested in this build sandbox (no docker CLI available there) --
# the underlying FastAPI app it runs (src/api.py) IS live-tested, see
# README. Written carefully against standard practice; build it yourself
# to confirm: docker build -t rag-hybrid-search .

FROM python:3.11-slim

WORKDIR /app

# build-essential is a safety net in case any spacy dependency (murmurhash,
# cymem, preshed, thinc, blis) doesn't have a prebuilt wheel for your
# target platform and falls back to a source build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_md

COPY src/ src/
COPY data/ data/
COPY eval/ eval/

ENV PYTHONPATH=/app/src
EXPOSE 8000

# Cold start is ~15-20s (loading the spacy model + embedding the corpus
# happens once at startup, not per-request) -- that's expected, not a hang.
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
