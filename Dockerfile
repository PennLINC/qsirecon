ARG BASE_IMAGE=pennlinc/qsirecon-base:20260304

FROM ghcr.io/prefix-dev/pixi:0.58.0 AS build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    ca-certificates \
                    build-essential \
                    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pixi config set --global run-post-link-scripts insecure

RUN mkdir /app
COPY pixi.lock pyproject.toml /app
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e qsirecon -e test --frozen --skip qsirecon
RUN --mount=type=cache,target=/root/.npm pixi run --as-is -e qsirecon npm install -g svgo@^3.2.0 bids-validator@1.14.10
RUN pixi shell-hook -e qsirecon --as-is | grep -v PATH > /shell-hook.sh
RUN pixi shell-hook -e test --as-is | grep -v PATH > /test-shell-hook.sh

COPY . /app
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e test --frozen
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e qsirecon --frozen

FROM ${BASE_IMAGE} AS base
WORKDIR /home/qsirecon
ENV HOME="/home/qsirecon"

RUN chmod -R go=u $HOME
WORKDIR /tmp

FROM base AS test
COPY --link --from=build /app/.pixi/envs/test /app/.pixi/envs/test
COPY --link --from=build /test-shell-hook.sh /shell-hook.sh
RUN cat /shell-hook.sh >> $HOME/.bashrc
ENV PATH="/app/.pixi/envs/test/bin:$PATH"
ENV FSLDIR="/app/.pixi/envs/test"

FROM base AS qsirecon
COPY --link --from=build /app/.pixi/envs/qsirecon /app/.pixi/envs/qsirecon
COPY --link --from=build /shell-hook.sh /shell-hook.sh
RUN cat /shell-hook.sh >> $HOME/.bashrc
ENV PATH="/app/.pixi/envs/qsirecon/bin:$PATH"
ENV FSLDIR="/app/.pixi/envs/qsirecon"
ENV IS_DOCKER_8395080871=1

ENTRYPOINT ["/app/.pixi/envs/qsirecon/bin/qsirecon"]
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="qsirecon" \
      org.label-schema.description="qsirecon - q Space Images postprocessing tool" \
      org.label-schema.url="http://qsirecon.readthedocs.io" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/pennlinc/qsirecon" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
