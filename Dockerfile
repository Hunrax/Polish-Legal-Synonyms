FROM python:3.12.7-bookworm

ENV JAVA_HOME=/opt/java/openjdk
COPY --from=eclipse-temurin:17 $JAVA_HOME $JAVA_HOME
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN \
  adduser \
    --system \
    --uid 999 \
    --group \
      "service" \
  && \
    mkdir \
      "/app" \
  && chown \
      "service:service" \
        "/app"

COPY "requirements.txt" "/app/requirements.txt"

RUN pip3 install \
      --requirement "/app/requirements.txt" \
      --no-cache-dir

COPY "source/" "/app/"

RUN /app/models/fasttext/download_model.sh

WORKDIR "/app"
USER "service:service"
ENTRYPOINT ["uvicorn", "main:app", "--host",  "0.0.0.0", "--port",  "8000"]