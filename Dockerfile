FROM python:3.12.7-bookworm

RUN \
  adduser \
    --system \
    --uid 999 \
    --group \
      "service" \
  && \
    mkdir \
      "/app" \
      "/run/service" \
      "/app/models" \
  && chown \
      "service:service" \
        "/app" \
        "/run/service" \
        "/app/models" 

COPY "requirements.txt" "/app/requirements.txt"
RUN \
  --mount=type=secret,id=build,dst=/etc/build_secrets \
  export $(xargs < /etc/build_secrets) ; \
  pip3 install \
    --requirement "/app/requirements.txt" \
    --no-cache-dir

COPY "source/" "/app/"

WORKDIR "/app"
USER "service:service"
ENTRYPOINT ["uvicorn", "main:app", "--host",  "0.0.0.0", "--port",  "8000"]