# =============================================================================
# Multi-stage Dockerfile for ANN Project
# =============================================================================
# Stage 1: Build C++ binaries
# Stage 2: Python runtime with uv + both subprojects
#
# Usage:
#   docker build -t ann-project .
#   docker run --rm -it ann-project bash
# =============================================================================

# ── Stage 1: C++ build ───────────────────────────────────────────────────────

FROM ubuntu:22.04 AS cpp-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY classical-ann-search/ ./classical-ann-search/
RUN make -C classical-ann-search -j"$(nproc)"

# ── Stage 2: Python runtime ──────────────────────────────────────────────────

FROM python:3.11-slim AS runtime

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ncbi-blast+ \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /usr/local/bin/uv

WORKDIR /app

# Copy C++ binaries from build stage
COPY --from=cpp-build /build/classical-ann-search/bin/ ./classical-ann-search/bin/
COPY classical-ann-search/include/ ./classical-ann-search/include/
COPY classical-ann-search/src/ ./classical-ann-search/src/
COPY classical-ann-search/Makefile ./classical-ann-search/

# Copy Python subprojects
COPY neural-lsh/ ./neural-lsh/
COPY protein-similarity-search/ ./protein-similarity-search/

# Install Python dependencies
RUN cd neural-lsh && uv sync --quiet && cd .. \
    && cd protein-similarity-search && uv sync --quiet

# Copy project metadata
COPY README.md LICENSE benchmark.sh ./
COPY docs/ ./docs/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["bash"]
