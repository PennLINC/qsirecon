
# Build into a wheel in a stage that has git installed
FROM python:slim AS wheelstage
RUN pip install build
RUN apt-get update && \
    apt-get install -y --no-install-recommends git

FROM pennlinc/qsirecon_build:24.9.0

# Install qsirecon
COPY . /src/qsirecon
RUN pip install --no-cache-dir "/src/qsirecon[all]"

# Precaching fonts, set 'Agg' as default backend for matplotlib
RUN python -c "from matplotlib import font_manager" && \
    sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" )

RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} +

RUN ldconfig
WORKDIR /tmp/
ENTRYPOINT ["/opt/conda/envs/qsiprep/bin/qsirecon"]
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
