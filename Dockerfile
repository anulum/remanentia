# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — production REST API container

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REMANENTIA_BASE=/data \
    REMANENTIA_API_BODY_LIMIT_BYTES=1048576 \
    REMANENTIA_API_RATE_PER_MINUTE=60 \
    REMANENTIA_API_RATE_BURST=10

WORKDIR /app

RUN groupadd --system --gid 10001 remanentia \
    && useradd --system --uid 10001 --gid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin remanentia \
    && mkdir -p /data \
    && chown -R remanentia:remanentia /data

COPY pyproject.toml README.md ./
COPY data/compiled_seed_facts.jsonl data/compiled_seed_facts.jsonl
COPY *.py ./

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir ".[api]" \
    && python -m compileall -q /app

USER remanentia

EXPOSE 8001
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3).read(); sys.exit(0)"

ENTRYPOINT ["remanentia"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8001", "--require-auth"]
