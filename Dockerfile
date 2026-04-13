ARG BASE_IMAGE=pennlinc/qsirecon-base:20260413

FROM ghcr.io/prefix-dev/pixi:0.58.0 AS build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    ca-certificates \
                    build-essential \
                    curl \
                    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pixi config set --global run-post-link-scripts insecure

# Install dependencies before the package itself to leverage caching
RUN mkdir /app
COPY pixi.lock pyproject.toml /app
WORKDIR /app
# First install runs before COPY . so .git is missing.
# Use --skip qsirecon (lockfile name) so pixi skips building the local package; aslprep uses --skip aslprep.
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e qsirecon -e test --frozen --skip qsirecon
RUN --mount=type=cache,target=/root/.npm pixi run --as-is -e qsirecon npm install -g svgo@^3.2.0 bids-validator@1.14.10
RUN pixi shell-hook -e qsirecon --as-is | grep -v PATH > /shell-hook.sh
RUN pixi shell-hook -e test --as-is | grep -v PATH > /test-shell-hook.sh

# Finally, install the package
COPY . /app
# Install test and production environments separately so production does not
# inherit editable-install behavior needed for test workflows.
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e test --frozen
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e qsirecon --frozen
# Ensure qsirecon is installed non-editably in the qsirecon env so the copied env is
# self-contained in the runtime image (lockfile may resolve to editable variant).
# Pixi envs do not include pip; use uv to install into the env's Python.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv pip install --python /app/.pixi/envs/qsirecon/bin/python --no-deps --force-reinstall .

#
# Pre-fetch templates
#
FROM ghcr.io/astral-sh/uv:python3.12-alpine AS templates
ENV TEMPLATEFLOW_HOME="/templateflow"
RUN uv pip install --system templateflow
COPY scripts/fetch_templates.py fetch_templates.py
RUN python fetch_templates.py

FROM ${BASE_IMAGE} AS base
WORKDIR /home/qsirecon
ENV HOME="/home/qsirecon"

COPY --link --from=templates /templateflow /home/qsirecon/.cache/templateflow
RUN chmod -R go=u $HOME

WORKDIR /tmp

FROM base AS amico_cache
COPY --link --from=build /app/.pixi/envs/qsirecon /app/.pixi/envs/qsirecon
COPY scripts/set_up_amico.py set_up_amico.py
RUN mkdir -p ${HOME}/.dipy && /app/.pixi/envs/qsirecon/bin/python set_up_amico.py

FROM base AS test
ARG VCS_REF
LABEL org.opencontainers.image.revision=$VCS_REF \
      org.label-schema.vcs-ref=$VCS_REF
COPY --link --from=build /app/.pixi/envs/test /app/.pixi/envs/test
COPY --link --from=build /test-shell-hook.sh /shell-hook.sh
COPY --link --from=amico_cache /home/qsirecon/.dipy /home/qsirecon/.dipy
RUN cat /shell-hook.sh >> $HOME/.bashrc
ENV PATH="/app/.pixi/envs/test/bin:$PATH"
ENV FSLDIR="/app/.pixi/envs/test"

FROM base AS qsirecon
COPY --link --from=build /app/.pixi/envs/qsirecon /app/.pixi/envs/qsirecon
COPY --link --from=build /shell-hook.sh /shell-hook.sh
COPY --link --from=amico_cache /home/qsirecon/.dipy /home/qsirecon/.dipy
RUN cat /shell-hook.sh >> $HOME/.bashrc
ENV PATH="/app/.pixi/envs/qsirecon/bin:$PATH"
ENV FSLDIR="/app/.pixi/envs/qsirecon"
ENV IS_DOCKER_8395080871=1
# Verify the runtime image can import qsirecon without source tree mounts.
RUN /app/.pixi/envs/qsirecon/bin/python -c "import qsirecon"

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
