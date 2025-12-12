FROM python:3.10

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential curl && \
    apt-get clean

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY docker/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

COPY docker/app /app/

CMD ["python", "inference.py"]