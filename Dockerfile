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
  && chown \
      "service:service" \
        "/app" \

COPY "requirements.txt" "/app/requirements.txt"

RUN pip3 install \
      --requirement "/app/requirements.txt" \
      --no-cache-dir 

COPY "source/" "/app/"

WORKDIR "/app"
USER "service:service"
ENTRYPOINT ["uvicorn", "main:app", "--host",  "0.0.0.0", "--port",  "8000"]