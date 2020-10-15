FROM python:3.8.6-buster

RUN pip install poetry==1.0.10
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock /tmp/

WORKDIR /tmp/
RUN poetry install --no-root --no-dev

ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
