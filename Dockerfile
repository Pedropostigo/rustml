# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
ARG RUST_CHANNEL=stable

########################################
# base: python + rust toolchain + maturin
########################################
FROM python:${PYTHON_VERSION}-slim AS base
ARG RUST_CHANNEL

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        pkg-config \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --default-toolchain "${RUST_CHANNEL}" --profile minimal \
    && rustup component add rustfmt clippy

RUN pip install --no-cache-dir --upgrade pip maturin

WORKDIR /workspace

########################################
# dev: interactive development target (used by .devcontainer)
########################################
FROM base AS dev

# maturin develop requires an active virtualenv (it refuses to touch the
# system interpreter), so give the dev image one and make it the default.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# /etc/profile unconditionally resets PATH for root login shells (the same
# reason rustup's installer patches ~/.profile), which would otherwise drop
# the venv from PATH in an interactive `docker run ... bash` session.
RUN printf 'export VIRTUAL_ENV=%s\nexport PATH="%s/bin:$PATH"\n' \
        "${VIRTUAL_ENV}" "${VIRTUAL_ENV}" > /etc/profile.d/10-venv.sh

RUN pip install --no-cache-dir maturin pytest numpy

# Source is normally bind-mounted here by the devcontainer; COPY is only a
# fallback so this image is self-contained if run standalone.
COPY . .

CMD ["bash"]

########################################
# builder: compiles the release wheel
########################################
FROM base AS builder

COPY . .

RUN maturin build --release --out /workspace/dist

########################################
# runtime: slim image with just the built wheel installed
########################################
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /workspace/dist /tmp/dist
RUN pip install --no-cache-dir /tmp/dist/*.whl && rm -rf /tmp/dist

WORKDIR /workspace
CMD ["python"]
