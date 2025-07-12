FROM python:3.13-slim

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD . /app
WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]