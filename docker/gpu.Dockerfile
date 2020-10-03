FROM tensorflow/tensorflow:2.2.0-gpu as base

LABEL org.opencontainers.image.source https://github.com/koralowiec/predict-api

WORKDIR /src

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . ./

CMD [ "uvicorn", "--app-dir", "code", "main:app", "--port", "5000", "--host", "0.0.0.0" ]

# for local development
FROM base as dev

CMD [ "uvicorn", "--app-dir", "code", "main:app", "--reload", "--port", "5000", "--host", "0.0.0.0" ]